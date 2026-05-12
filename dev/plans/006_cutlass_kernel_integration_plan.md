# 006 CUTLASS Kernel 集成 · 可落地实现方案

## 背景

`cutlass_5090_my/test/` 里有 4 个封装好的 `cutlass_profiler` 调用脚本，在 RTX 5090 (sm_120) 上验证可用。本方案基于**相同的调用方式（subprocess 调 cutlass_profiler）**，将其封装成 PyTorch `nn.Module` 替换原有的 fake-quant 权重，实现真实 kernel 路径。

> 不引入 CUTLASS 源码/submodule，只依赖已编译好的 `cutlass_profiler` 二进制。

---

## Kernel 到压缩方法的映射

| 压缩方法 | operation | kernel 过滤串（同 bench 脚本） |
|---|---|---|
| `nvfp4` | `block_scaled_gemm` | `*sm120*bstensorop*ue4m3xe2m1*ue4m3xe2m1*f32_void_f32*cooperative*` |
| `semi_structured_sparse` | `spgemm` | `cutlass_tensorop_s16832spgemm_bf16_64x128_64x6_tn_align8` |
| `nvfp4_semi_structured_sparse` | `block_scaled_gemm` | `*sm120*bssptensorop*ue4m3xe2m1*ue4m3xe2m1*f32_void_f32*` |
| `unstructured_sparse` / `nvfp4_unstructured_sparse` | — | dense fallback，不接 kernel |

---

## 目标目录结构

```
fake/kernels/
├── __init__.py            # is_available(), CUTLASS_PROFILER 路径检查
├── registry.py            # kernel 名 + operation 注册表
├── pack.py                # 权重 pack / 2:4 compress
├── runner.py              # subprocess 调 cutlass_profiler，输入输出走 torch.save/load
├── modules.py             # NVFP4Linear, SemiSparseLinear, NVFP4SemiSparseLinear, Conv1x1AsLinear
└── dispatch.py            # materialize(model, metadata)
```

---

## 各文件实现细节

### `fake/kernels/__init__.py`

```python
import os
from pathlib import Path

CUTLASS_PROFILER = os.environ.get("CUTLASS_PROFILER", "")

def is_available() -> bool:
    return bool(CUTLASS_PROFILER) and Path(CUTLASS_PROFILER).is_file()

def kernel_info() -> dict:
    return {"cutlass_profiler": CUTLASS_PROFILER, "available": is_available()}
```

运行时须设置 `CUTLASS_PROFILER=/path/to/cutlass/build/tools/profiler/cutlass_profiler`。

---

### `fake/kernels/registry.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class KernelEntry:
    method: str        # 压缩方法名，对应 pipeline.py 里的 method 字符串
    operation: str     # --operation 值
    kernels: str       # --kernels 过滤串（从 dry-run CSV 的 Operation 列取精确名后可替换）

REGISTRY: list[KernelEntry] = [
    KernelEntry(
        method="nvfp4",
        operation="block_scaled_gemm",
        kernels="*sm120*bstensorop*ue4m3xe2m1*ue4m3xe2m1*f32_void_f32*cooperative*",
    ),
    KernelEntry(
        method="semi_structured_sparse",
        operation="spgemm",
        kernels="cutlass_tensorop_s16832spgemm_bf16_64x128_64x6_tn_align8",
    ),
    KernelEntry(
        method="nvfp4_semi_structured_sparse",
        operation="block_scaled_gemm",
        kernels="*sm120*bssptensorop*ue4m3xe2m1*ue4m3xe2m1*f32_void_f32*",
    ),
]

DENSE_FALLBACK_METHODS = {"unstructured_sparse", "nvfp4_unstructured_sparse"}

def get_entry(method: str) -> KernelEntry | None:
    for e in REGISTRY:
        if e.method == method:
            return e
    return None
```

---

### `fake/kernels/pack.py`

host 端负责把 fake-quant 路径产出的权重转成 kernel 期望的 layout。

```python
from __future__ import annotations
import torch

FP4_E2M1_MAX = 6.0


def pack_nvfp4_ue4m3(
    weight: torch.Tensor,  # (out, in), fp16/bf16/fp32
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns:
        uint4_packed  : (out, in//2) dtype=torch.uint8  — 2 fp4 values per byte, little-endian nibble
        ue4m3_scales  : (out, in//group_size) dtype=torch.float8_e4m3fn
        alpha         : float32 scalar (=1.0, reserved for per-tensor scale)
    """
    x = weight.detach().float()
    rows, cols = x.shape
    assert cols % group_size == 0, f"K={cols} not divisible by group_size={group_size}"
    grouped = x.reshape(rows, -1, group_size)                        # (out, g, gs)
    scales_f32 = grouped.abs().amax(dim=-1) / FP4_E2M1_MAX          # (out, g)
    scales_f32 = scales_f32.clamp(min=1e-12)
    normalized = grouped / scales_f32.unsqueeze(-1)                  # (out, g, gs)
    q = _cast_to_fp4_int(normalized)                                 # (out, g, gs) int in [0,15]
    # pack two fp4 into one uint8: low nibble = even index, high nibble = odd index
    q_flat = q.reshape(rows, cols)                                   # (out, in)
    uint4_packed = (q_flat[:, 0::2] | (q_flat[:, 1::2] << 4)).to(torch.uint8)
    ue4m3_scales = scales_f32.to(torch.float8_e4m3fn)
    return uint4_packed, ue4m3_scales, 1.0


def compress_24(
    weight: torch.Tensor,  # (out, in), bf16 after pruning (2:4 zeros already applied)
    mask: torch.Tensor,    # (out, in), bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        compressed : (out, in//2)  dtype=weight.dtype — 2 non-zero values per 4
        meta       : (out, in//16) dtype=torch.uint16  — CUTLASS 2:4 metadata

    K must be % 64 == 0 for bf16 spgemm.
    """
    assert weight.shape[-1] % 64 == 0, "K must be divisible by 64 for 2:4 spgemm"
    rows, cols = weight.shape
    w = weight.reshape(rows, -1, 4)    # (out, col//4, 4)
    m = mask.reshape(rows, -1, 4)      # same
    # gather non-zero values (2 per group of 4)
    compressed_vals = []
    meta_bits = []
    for g in range(w.shape[1]):
        vals = w[:, g, :]              # (out, 4)
        ms   = m[:, g, :]             # (out, 4)
        # positions of the two kept values per row
        idx = ms.to(torch.uint8).nonzero(as_tuple=False)  # fallback: simple loop
        # simplified: assume mask is always exactly 2-of-4
        kept = vals[ms].reshape(rows, 2)
        compressed_vals.append(kept)
        # encode positions: 2-bit each → 4-bit meta per group of 4
        pos = ms.to(torch.uint8)       # (out, 4)
        # CUTLASS meta format: 4 groups of 4 packed into uint16
        # here we store per-group 4-bit: pos0 | pos1<<2
        p0 = pos.argmax(dim=-1).to(torch.uint16)  # placeholder, see note below
        meta_bits.append(p0)
    compressed = torch.stack(compressed_vals, dim=1).reshape(rows, cols // 2).contiguous()
    # NOTE: meta encoding above is a placeholder.
    # Real CUTLASS s16832spgemm metadata format must be generated by
    # cutlass::transform::device::StructuredSparseMemcpy or equivalent.
    # Replace this block with a call to the CUTLASS host-side compressor once
    # the compiled library is available (e.g. via ctypes/cffi binding to
    # libcutlass_library.so or a minimal standalone helper executable).
    meta = torch.stack(meta_bits, dim=1).to(torch.uint16)
    return compressed.contiguous(), meta.contiguous()


def k_aligned(cols: int, method: str) -> bool:
    """Whether K dimension satisfies kernel tile alignment."""
    if "nvfp4" in method:
        return cols % 16 == 0
    return cols % 64 == 0  # bf16 spgemm


# ── helpers ──────────────────────────────────────────────────────────

def _cast_to_fp4_int(x: torch.Tensor) -> torch.Tensor:
    """Map float values to e2m1 fp4 code (0-15 unsigned)."""
    sign = (x < 0).to(torch.int32)
    y = x.abs()
    code = torch.zeros_like(y, dtype=torch.int32)
    code[(y > 0.25) & (y < 0.75)]  = 1
    code[(y >= 0.75) & (y <= 1.25)] = 2
    code[(y > 1.25) & (y < 1.75)]  = 3
    code[(y >= 1.75) & (y <= 2.5)]  = 4
    code[(y > 2.5) & (y < 3.5)]    = 5
    code[(y >= 3.5) & (y <= 5.0)]   = 6
    code[y > 5.0]                   = 7
    return code | (sign << 3)  # bit3 = sign
```

> **注意**：`compress_24` 里的 metadata 编码是占位实现。CUTLASS s16832spgemm 的 metadata 格式需要 host-side compressor（库里的 `StructuredSparseMemcpy`）。后续需要写一个小的 standalone 编译单元（一个 `.cu` + `extern "C"` wrapper）导出这个 helper，或者通过 `ctypes` 加载 `libcutlass_library.so`。这是整个方案中**唯一必须编译一点 C++ 代码**的地方。NVFP4 packing 纯 Python 可以实现。

---

### `fake/kernels/runner.py`

仿照 `common.sh` 的参数组织方式，用 subprocess 调 `cutlass_profiler`。用 `torch.save` / `torch.load` 以二进制文件交换张量。

```python
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

import torch

from fake.kernels import CUTLASS_PROFILER, is_available


def run_gemm(
    operation: str,
    kernels: str,
    A: torch.Tensor,
    B: torch.Tensor,
    m: int, n: int, k: int,
    warmup: int = 5,
    iters: int = 20,
) -> torch.Tensor:
    """
    调 cutlass_profiler 跑一次 GEMM，返回结果张量。
    tensors 以临时文件交换（profiler 原生支持 --input/--output-file）。

    实际 A/B 张量的格式由 operation 决定；runner 只负责传参。
    profiler 的 --verification-enabled=false 避免与 cuBLAS 对比。
    """
    if not is_available():
        raise RuntimeError("CUTLASS_PROFILER not set or not found; set env var CUTLASS_PROFILER")

    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = str(Path(tmp) / "result")
        cmd = [
            CUTLASS_PROFILER,
            f"--operation={operation}",
            f"--kernels={kernels}",
            f"--m={m}", f"--n={n}", f"--k={k}",
            "--providers=cutlass",
            "--verification-enabled=false",
            f"--warmup-iterations={warmup}",
            f"--profiling-iterations={iters}",
            f"--output={out_prefix}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"cutlass_profiler failed:\n{result.stderr}")
        # profiler 输出 CSV；这里只读 latency，真实 tensor 输出见下面的说明
        return _parse_best_runtime(out_prefix, operation)


def _parse_best_runtime(out_prefix: str, operation: str) -> float:
    """从 profiler 输出 CSV 读最优 Runtime(ms)。"""
    import csv
    csv_path = f"{out_prefix}.{operation}.csv"
    best = float("inf")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                rt = float(row.get("Runtime", "inf") or "inf")
                if rt < best:
                    best = rt
            except ValueError:
                pass
    return best
```

> **关于 tensor 输出**：`cutlass_profiler` 是 benchmark 工具，不直接返回计算结果 tensor。在 micro-bench（Phase 6）场景下，runner 只需要 latency 数字，上面的实现完全够用。
>
> 若要在 e2e 推理里真正用 CUTLASS kernel 执行 forward，**必须改为 Python C-extension**（用 `torch.utils.cpp_extension` 编译一个最小 `.cu`，内部实例化 CUTLASS kernel operation 并在 CUDA stream 上 launch）。方案如下：
>
> ```
> fake/kernels/csrc/
> ├── nvfp4_gemm.cu          # 实例化 block_scaled_gemm operation
> ├── sparse24_bf16_gemm.cu  # 实例化 spgemm operation
> ├── sparse24_nvfp4_gemm.cu # 实例化 block_scaled sparse gemm operation
> ├── sparse24_meta.cu       # StructuredSparseMemcpy helper（compress_24 需要）
> └── bindings.cpp           # pybind11 入口
> ```
>
> 这些 `.cu` 文件内容结构上和 `cutlass_5090_my/test/bench_01` 等脚本调用的 kernel 完全一致——脚本里通过 `--kernels=` 告诉 profiler 选哪个 operation，`.cu` 里则直接 `#include` 对应的 CUTLASS `gemm_operation.h` 实例。
> Phase 3（micro-bench）只用 runner.py（subprocess），Phase 5（e2e）需要 C-extension。两者可以并行推进。

---

### `fake/kernels/modules.py`

```python
from __future__ import annotations
import torch
import torch.nn as nn
from fake.kernels.pack import pack_nvfp4_ue4m3, compress_24, k_aligned


class NVFP4Linear(nn.Module):
    """Weight-only NVFP4 linear; activation 走 bf16 input。"""
    def __init__(self, uint4_packed, ue4m3_scales, alpha, bias, out_features, in_features):
        super().__init__()
        self.register_buffer("uint4_packed", uint4_packed)
        self.register_buffer("ue4m3_scales", ue4m3_scales)
        self.alpha = alpha
        self.out_features = out_features
        self.in_features = in_features
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "NVFP4Linear":
        uint4, scales, alpha = pack_nvfp4_ue4m3(linear.weight.data)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(uint4, scales, alpha, bias, linear.out_features, linear.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 真实 kernel 路径：调 C-extension（Phase 5 实现）
        # 目前 fallback：dequant 后走 F.linear（精度验证用）
        from fake.kernels._fallback import nvfp4_dequant_linear
        return nvfp4_dequant_linear(x, self.uint4_packed, self.ue4m3_scales, self.alpha, self.bias)


class SemiSparseLinear(nn.Module):
    """2:4 sparse BF16 linear。"""
    def __init__(self, compressed, meta, bias, out_features, in_features):
        super().__init__()
        self.register_buffer("compressed", compressed)
        self.register_buffer("meta", meta)
        self.out_features = out_features
        self.in_features = in_features
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, mask: torch.Tensor) -> "SemiSparseLinear":
        compressed, meta = compress_24(linear.weight.data.to(torch.bfloat16), mask)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(compressed, meta, bias, linear.out_features, linear.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from fake.kernels._fallback import sparse24_bf16_linear
        return sparse24_bf16_linear(x, self.compressed, self.meta, self.bias)


class NVFP4SemiSparseLinear(nn.Module):
    """2:4 sparse + NVFP4 joint kernel。"""
    def __init__(self, uint4_compressed, ue4m3_scales, meta, alpha, bias, out_features, in_features):
        super().__init__()
        self.register_buffer("uint4_compressed", uint4_compressed)
        self.register_buffer("ue4m3_scales", ue4m3_scales)
        self.register_buffer("meta", meta)
        self.alpha = alpha
        self.out_features = out_features
        self.in_features = in_features
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, mask: torch.Tensor) -> "NVFP4SemiSparseLinear":
        # first compress 2:4, then pack uint4 on compressed weight
        compressed_bf16, meta = compress_24(linear.weight.data.to(torch.bfloat16), mask)
        uint4, scales, alpha = pack_nvfp4_ue4m3(compressed_bf16)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(uint4, scales, meta, alpha, bias, linear.out_features, linear.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from fake.kernels._fallback import sparse24_nvfp4_linear
        return sparse24_nvfp4_linear(x, self.uint4_compressed, self.ue4m3_scales, self.meta, self.alpha, self.bias)


class Conv1x1AsLinear(nn.Module):
    """把 1x1 Conv2d 的 forward 路由到一个 Linear-based kernel module。"""
    def __init__(self, linear_module, out_channels, in_channels):
        super().__init__()
        self.inner = linear_module
        self.out_channels = out_channels
        self.in_channels = in_channels

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, kernel_module_cls, **kwargs) -> "Conv1x1AsLinear":
        fake_linear = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
        fake_linear.weight.data = conv.weight.data.squeeze(-1).squeeze(-1)
        if conv.bias is not None:
            fake_linear.bias.data = conv.bias.data
        inner = kernel_module_cls.from_linear(fake_linear, **kwargs)
        return cls(inner, conv.out_channels, conv.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = self.inner(x.permute(0, 2, 3, 1).reshape(B * H * W, C))
        return out.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)
```

---

### `fake/kernels/dispatch.py`

```python
from __future__ import annotations
from typing import Any

import torch.nn as nn

from fake.kernels.registry import get_entry, DENSE_FALLBACK_METHODS
from fake.kernels.modules import NVFP4Linear, SemiSparseLinear, NVFP4SemiSparseLinear, Conv1x1AsLinear
from fake.kernels.pack import k_aligned


def materialize(
    model: nn.Module,
    metadata: dict[str, Any],
    masks: dict[str, Any] | None = None,
) -> nn.Module:
    """
    按 metadata['method'] 把 model 里已压缩的 nn.Linear / pointwise Conv2d
    替换成对应的 kernel module。
    masks: {module_name: bool_mask_tensor}，仅 sparse 方法需要。
    K 对齐失败的层写 kernel_fallback=dense 到 metadata 后原样保留。
    """
    method = metadata.get("method", "")
    if method in DENSE_FALLBACK_METHODS:
        metadata["kernel_path"] = "dense_fallback"
        return model

    entry = get_entry(method)
    if entry is None:
        metadata["kernel_path"] = "dense_fallback"
        return model

    fallback_layers: list[str] = []
    for name, module in list(model.named_modules()):
        cols = _in_cols(module)
        if cols is None:
            continue
        if not k_aligned(cols, method):
            fallback_layers.append(name)
            continue
        mask = (masks or {}).get(name, {}).get("mask")
        replacement = _make_module(method, module, mask)
        if replacement is None:
            fallback_layers.append(name)
            continue
        _set_module(model, name, replacement)

    metadata["kernel_path"] = entry.operation
    metadata["kernel_name"] = entry.kernels
    metadata["kernel_fallback_layers"] = fallback_layers
    return model


# ── helpers ──────────────────────────────────────────────────────────

def _in_cols(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return module.in_features
    if isinstance(module, nn.Conv2d) and tuple(module.kernel_size) == (1, 1) and module.groups == 1:
        return module.in_channels
    return None


def _make_module(method: str, module: nn.Module, mask) -> nn.Module | None:
    if method == "nvfp4":
        cls = NVFP4Linear
        if isinstance(module, nn.Conv2d):
            return Conv1x1AsLinear.from_conv(module, cls)
        return cls.from_linear(module)
    if method == "semi_structured_sparse":
        if mask is None:
            return None
        if isinstance(module, nn.Conv2d):
            return Conv1x1AsLinear.from_conv(module, SemiSparseLinear, mask=mask)
        return SemiSparseLinear.from_linear(module, mask)
    if method == "nvfp4_semi_structured_sparse":
        if mask is None:
            return None
        if isinstance(module, nn.Conv2d):
            return Conv1x1AsLinear.from_conv(module, NVFP4SemiSparseLinear, mask=mask)
        return NVFP4SemiSparseLinear.from_linear(module, mask)
    return None


def _set_module(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
```

---

### `fake/kernels/_fallback.py`（精度验证用，可后期替换为 C-extension）

```python
"""Dequant-based fallback forward，供 modules.py 在 C-extension 就绪前使用。"""
from __future__ import annotations
import torch
import torch.nn.functional as F

FP4_CODEBOOK = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def nvfp4_dequant_linear(x, uint4_packed, ue4m3_scales, alpha, bias):
    weight = _dequant_nvfp4(uint4_packed, ue4m3_scales, alpha)
    return F.linear(x, weight, bias)


def sparse24_bf16_linear(x, compressed, meta, bias):
    # fallback: compressed holds non-zero values; meta unused; reconstruct dense (approx)
    # This is for correctness check only — not a real sparse kernel.
    rows = compressed.shape[0]
    in_features = compressed.shape[1] * 2  # 2 non-zeros per 4 cols
    weight = torch.zeros(rows, in_features, dtype=compressed.dtype, device=compressed.device)
    # simplified restore: not bit-exact, real restore requires meta decode
    weight[:, 0::2] = compressed[:, :compressed.shape[1]//2 * 2:2]  # placeholder
    return F.linear(x.to(weight.dtype), weight, bias)


def sparse24_nvfp4_linear(x, uint4_compressed, ue4m3_scales, meta, alpha, bias):
    weight_bf16 = _dequant_nvfp4(uint4_compressed, ue4m3_scales, alpha)
    return F.linear(x, weight_bf16, bias)


def _dequant_nvfp4(uint4_packed, ue4m3_scales, alpha):
    codebook = FP4_CODEBOOK.to(uint4_packed.device)
    low  = uint4_packed & 0x0F
    high = (uint4_packed >> 4) & 0x0F
    vals = torch.stack([low, high], dim=-1).reshape(uint4_packed.shape[0], -1)  # (out, in)
    sign = (vals >> 3).float() * 2 - 1
    mag  = vals & 0x07
    weight_fp4 = sign * codebook[mag]
    group_size = weight_fp4.shape[1] // ue4m3_scales.shape[1]
    scales = ue4m3_scales.float().unsqueeze(-1).expand(-1, -1, group_size).reshape_as(weight_fp4)
    return (weight_fp4 * scales * alpha).to(torch.bfloat16)
```

---

## 与现有框架的接口改动

### `fake/compression/checkpoint.py`（新增 materialize 调用）

```python
# 在 load_checkpoint_into_model 之后新增
def materialize_from_metadata(
    model: nn.Module,
    metadata: dict,
    masks_path: str | None = None,
) -> nn.Module:
    from fake.kernels.dispatch import materialize
    masks = None
    if masks_path:
        import torch
        payload = torch.load(masks_path, map_location="cpu")
        masks = payload.get("modules", {})
    return materialize(model, metadata, masks)
```

### `scripts/bench_compressed_speed.py` / `eval_compressed_accuracy.sh`

加载 checkpoint 后增加一行：
```python
if args.kernel and metadata.get("kernel_ready"):
    model = materialize_from_metadata(model, metadata, masks_path=args.masks)
```

### `scripts/prepare_compressed_model.py`

加 `--pack-for-kernel` flag，触发时额外调 `pack.py` 产出 packed tensors，在 `metadata.json` 里写 `"kernel_ready": true`。

---

## 分阶段实现路径

| Phase | 产出 | 依赖 | 约时 |
|---|---|---|---|
| **0** 锁定 kernel 名 | profiler dry-run CSV → `registry.py` 填精确名 | 只需 profiler 二进制 | 0.5 天 |
| **1** pack.py + 精度 | `NVFP4Linear.forward` fallback 路径跑通，精度与 fake 一致 | 无编译依赖 | 1 天 |
| **2** compress_24 meta | `SemiSparseLinear.forward` fallback 跑通；meta 占位，需 standalone `.cu` helper | 需编译 1 个 `.cu` | 1.5 天 |
| **3** micro-bench (runner.py) | `kernel_bench.py` 跑 3 条路径，latency 对齐 profiler <5% | profiler + Phase 0 | 1 天 |
| **4** C-extension 替换 forward | `modules.py` forward 真正调 CUTLASS kernel | 需编译 3 个 `.cu` | 3 天 |
| **5** e2e 速度与精度 | `bench_compressed_speed` + `eval_compressed_accuracy` 走 kernel 路径 | Phase 1-4 | 1 天 |

---

## 关键风险

| 风险 | 处置 |
|---|---|
| `compress_24` metadata 格式不对 | Phase 2 的 standalone helper 解决；Phase 1 精度验证先跳过 meta |
| 2:4 sparse + NVFP4 joint kernel 在 sm_120 上不可用 | Phase 0 dry-run 确认；不可用时此方法保留 fallback，不影响其他两条路径 |
| K 对齐失败的层 | dispatch.py 自动回退 dense，metadata 记录 |
| profiler subprocess 延迟高 | micro-bench 只用 runner；e2e 推理必须用 C-extension |
