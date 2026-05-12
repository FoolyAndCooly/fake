## 2026-05-12 - P0/P1 修复：B 卡 e2e 阻塞问题

针对 code review 发现的 P0/P1 问题进行修复，确保 B 卡上能直接 `KERNEL=1 sbatch` 跑通端到端测试。

### P0 修复（阻塞 B 卡 e2e）

1. **`bench_compressed_speed.sh` 缺 `CUTLASS_ROOT` 导出**  
   - 问题：注释写的是 `CUTLASS_PROFILER`（错误），且脚本未导出 `CUTLASS_ROOT`，导致 `KERNEL=1` 时 extension load 失败→静默走 fallback，latency 仍是 fake 数字  
   - 修复：脚本开头加 `export CUTLASS_ROOT="${CUTLASS_ROOT:-/data/home/.../cutlass}"`，注释改为 "JIT-compile via CUTLASS_ROOT"

2. **`NVFP4Linear.from_linear` 未先转 bf16**  
   - 问题：`pack_nvfp4_ue4m3(linear.weight.data, ...)` 直接用原 dtype（可能是 fp32/fp16），与 kernel 的 bf16 假设不一致  
   - 修复：`from_linear` 里先 `.to(torch.bfloat16)`，和 `SemiSparseLinear` 对齐

3. **`sparse_2_4_bf16_gemm.cu` host compressor API 不存在**  
   - 问题：`host_uncompress_and_reorder_meta` 名字错误（CUTLASS 里没有这个函数），大概率编不过→`kernel_ready=False`→所有 sparse bf16 层走 fallback  
   - 修复：完全重写 `compress_2_4_bf16()`，参考 CUTLASS example 15 (`examples/15_ampere_sparse_tensorop_gemm/`)：
     - 手动遍历 dense weight 的 2:4 pattern，提取 compressed + 未重排的 metadata
     - 调用 `cutlass::reorder_meta()` 做 swizzle（CUTLASS 4.x 稳定 API）
     - 返回 CUTLASS-aligned `(compressed, meta)` pair
   - 注意：这个实现依赖 CUTLASS 4.x 的 `cutlass::reorder_meta` 和 `HostTensor`，需要在 B 卡上实跑 nvcc 验证编译通过

4. **smoke 脚本缺 err 阈值 assert**  
   - 问题：`smoke_cutlass_kernels.sh` 打印 `max_err` 但不 assert，layout 错误时不会失败  
   - 修复：加 kernel vs fallback 的 max_err 阈值检查：
     - `NVFP4Linear` / `NVFP4SemiSparseLinear`：`max_err < 5.0`（W4A4 量化误差容忍）
     - `SemiSparseLinear`：`max_err < 1e-2`（bf16 精度，期望近似精确匹配）

### P1 修复（次要问题）

5. **`Conv1x1AsLinear.from_conv` 在 CPU 上构造 `fake_linear`**  
   - 问题：`from_linear` 被调用时 weight 在 CPU→`_maybe_cutlass_compress_24` 返回 `(None,None,False)`→MaxViT 1×1 Conv 走不到 kernel  
   - 修复：`from_conv` 在调用 `kernel_cls.from_linear` 前先 `.to(conv.weight.device)`

6. **`bench_kernel.py` 的 `is_available` 语义错配**  
   - 问题：`is_available()` 检查 extension 能否编译，但 micro-bench 只需要 `cutlass_profiler` 二进制  
   - 修复：改用 `is_profiler_available()`

7. **`compress_24` 的 mask 在 CPU 上导致 gather 失败**  
   - 问题：`Conv1x1AsLinear` 传入 CPU mask→`torch.gather` 报 device mismatch  
   - 修复：`compress_24` 开头加 `mask = mask.to(weight.device)`

8. **`registry.py` kernel 名注释不清晰**  
   - 问题：`kernels` 字段是 CSV label，不控制实际编译的 kernel（容易误导）  
   - 修复：加注释说明 "label only, actual kernel is instantiated in csrc/*.cu"

9. **`dispatch.materialize` 缺 fallback 统计打印**  
   - 问题：端到端跑完不知道哪些层走了 fallback  
   - 修复：`materialize()` 结束后打印 `kernel modules: M matched, N fallback`，fallback 层前 5 个打印名字

10. **`prepare_compressed_model.py` 的 `save_full_masks` 覆盖缺注释**  
    - 问题：强制 sparse method 保存 full masks（DINOv3 7B 上会膨胀到几 GB），但没说明原因  
    - 修复：加注释 "Kernel path (SemiSparseLinear / NVFP4SemiSparseLinear) requires full masks for materialize_from_checkpoint"

### 影响文件

- `scripts/slurm/bench_compressed_speed.sh`：导出 `CUTLASS_ROOT`，注释修正
- `scripts/bench_kernel.py`：`is_available` → `is_profiler_available`
- `fake/kernels/modules.py`：
  - `NVFP4Linear.from_linear` 先转 bf16
  - `Conv1x1AsLinear.from_conv` 保持设备一致
- `fake/kernels/csrc/sparse_2_4_bf16_gemm.cu`：完全重写 `compress_2_4_bf16()` host compressor
- `fake/kernels/pack.py`：`compress_24` 自动移 mask 到 GPU
- `fake/kernels/dispatch.py`：打印 matched/fallback 统计
- `fake/kernels/registry.py`：加 kernel 名注释
- `scripts/slurm/smoke_cutlass_kernels.sh`：加 max_err 阈值 assert
- `scripts/prepare_compressed_model.py`：加 `save_full_masks` 注释

### 验证

- A100 上所有修复通过（fallback 路径）：
  - `NVFP4Linear.from_linear` 接受 fp32 Linear 并正确转 bf16
  - `Conv1x1AsLinear` + CPU mask 不再报 device mismatch
  - `materialize()` 打印 `kernel modules: 4 matched, 0 fallback`
- B 卡验证待 sbatch：
  - `sparse_2_4_bf16_gemm.cu` 的新 compressor 能否编译通过（依赖 CUTLASS 4.x `reorder_meta` API）
  - smoke 脚本的 max_err 阈值是否合理（尤其 NVFP4 的 5.0 是否过松/过紧）

### 后续注意

- `sparse_2_4_bf16_gemm.cu` 的 `cutlass::reorder_meta` 在不同 CUTLASS 版本下签名可能略有差异；若编译失败，对照当前 CUTLASS tag 的 `examples/15_ampere_sparse_tensorop_gemm/` 调整
- NVFP4 kernel 的 layout（uint4 packing 顺序、scale stride）仍需 B 卡 smoke 验证数值正确性；若 max_err 超阈值，检查 `pack.py` 的 nibble 顺序是否和 CUTLASS `bstensorop` 期望一致

## 2026-05-12 - B 卡 2:4 Sparse BF16 + NVFP4 Joint kernel 接入

- 开发目的：把 `cutlass_5090_my/test/bench_03_sparse_f16_bf16.sh` 和 `bench_04_sparse_nvf4_ue4m3.sh` 对应的 kernel 接进 `SemiSparseLinear` / `NVFP4SemiSparseLinear`，集齐三条 kernel 路径。
- 修改内容：
  - 新增 `fake/kernels/csrc/sparse_2_4_bf16_gemm.cu`：基于 `cutlass::gemm::device::SparseGemm`（`s16832spgemm bf16 64x128x64_6stage` 配置），对齐 bench_03 默认 kernel 名；附带 host-side `compress_2_4_bf16(dense) -> (compressed, meta)`，通过 `cutlass::host_uncompress_and_reorder_meta` 产出真实 CUTLASS metadata E
  - 新增 `fake/kernels/csrc/sparse_2_4_nvfp4_gemm.cu`：基于 sm_120 `CollectiveBuilder` + `SparseGemmUniversal` + `OpClassBlockScaledSparseTensorOp`，对齐 bench_04 `*bssptensorop*ue4m3xe2m1*`
  - `fake/kernels/csrc/bindings.cpp`：暴露 `sparse24_gemm_bf16` / `compress_2_4_bf16` / `sparse24_nvfp4_gemm`
  - `fake/kernels/build.py`：sources 加新 `.cu`；改为 ampere+ 即允许编译（sm_80 仅编译 sparse_bf16，sm_120 编译全部），gencode 按设备能力自动选
  - `fake/kernels/modules.py`：
    - `SemiSparseLinear` 现在在 `from_linear` 里优先用 `ext.compress_2_4_bf16` 产出 CUTLASS-aligned `compressed/meta`，并置 `kernel_ready=True`；forward 走 `sparse24_gemm_bf16`，kernel_ready=False 或失败自动回退到 mask 重建路径
    - `NVFP4SemiSparseLinear` forward 加 `_kernel_forward`（activation runtime NVFP4 量化 → 调 `sparse24_nvfp4_gemm`），失败 fallback
    - 两个 module 同样支持任意 rank 输入（2D/3D 都会先 reshape）
  - 新增 `scripts/slurm/smoke_cutlass_kernels.sh`：B 卡 sbatch 跑三条路径 + 打印 `kernel_info()`
- 影响文件：`fake/kernels/csrc/`（+2 .cu，修改 bindings.cpp）、`fake/kernels/build.py`、`fake/kernels/modules.py`、`scripts/slurm/smoke_cutlass_kernels.sh`（新）
- 使用：
  - `CUTLASS_ROOT=/path/to/cutlass sbatch scripts/slurm/smoke_cutlass_kernels.sh`
  - 端到端：`CUTLASS_ROOT=... KERNEL=1 METHOD=semi_structured_sparse sbatch scripts/slurm/bench_compressed_speed.sh`，同理 `METHOD=nvfp4_semi_structured_sparse`
- 验证：A100 上三条路径 fallback forward 均通过（2D/3D 输入），`kernel_info` 正确报告 `CUTLASS_ROOT not set`；sm_120 实跑待 B 卡上 sbatch 验证
- 后续注意：
  - `sparse_2_4_bf16_gemm.cu` 里的 `host_uncompress_and_reorder_meta` 是 CUTLASS util，不同版本签名略有差异；若某个 CUTLASS commit 上编译失败，按其 example 15 (`15_ampere_sparse_tensorop_gemm/`) 或 `cutlass/transform/` 下的具体 compressor 调整
  - bench_03/04 默认 kernel 参数（tile/stage）已写进 `.cu`；如需扫更多形状，可在同 `.cu` 里加多套 `Gemm_*` typedef 并按 shape 选
  - `SemiSparseLinear.kernel_ready=False` 时走 mask 路径（数值正确，速度 = dense bf16），可作为 "kernel不可用 vs 可用" 对照

## 2026-05-12 - B 卡 Dense NVFP4 kernel 接入

- 开发目的：按 `cutlass_5090_my/test/common.sh` 的约定（通过 `CUTLASS_ROOT` 指向已构建的 CUTLASS 源码树），在 `NVFP4Linear.forward` 里接入真实 CUTLASS Blackwell `bstensorop` NVFP4 GEMM，对齐 `bench_01_dense_nvf4_ue4m3.sh` 的 kernel（`*sm120*bstensorop*ue4m3xe2m1*ue4m3xe2m1*f32_void_f32*cooperative*`）。
- 修改内容：
  - 新增 `fake/kernels/csrc/nvfp4_gemm.cu`：基于 `CollectiveBuilder` + `GemmUniversalAdapter` 实例化 sm_120 NVFP4 cooperative kernel，接口 `nvfp4_gemm(a_packed, a_scales, b_packed, b_scales, alpha, m, n, k) -> bf16 (M, N)`
  - 新增 `fake/kernels/csrc/bindings.cpp`：pybind 入口
  - 新增 `fake/kernels/build.py`：JIT 编译逻辑，自动检查 sm_120 和 CUTLASS_ROOT 是否齐备；不满足时 `load()` 返回 None
  - `fake/kernels/__init__.py`：新增 `is_available()` / `is_profiler_available()` / `kernel_info()`，暴露当前环境检测结果
  - `fake/kernels/modules.py`：`NVFP4Linear.forward` 改为两段式——先尝试 `_kernel_forward`（内部做 activation 运行时 NVFP4 量化 + 调 CUTLASS kernel），失败或非 sm_120 回退 `_fallback_forward`；forward 支持任意 rank 输入（reshape 成 2D 后 reshape 回去）
  - `fake/kernels/pack.py`：新增 `pack_nvfp4_ue4m3_activation`（复用 `pack_nvfp4_ue4m3`，语义上标注为 runtime activation 量化入口）
  - 新增 `scripts/slurm/smoke_nvfp4_kernel.sh`：B 卡上 sbatch 跑一次最小 NVFP4Linear forward，打印 `kernel_info()`、kernel vs fallback 误差
- 影响文件：`fake/kernels/__init__.py`、`fake/kernels/build.py`（新）、`fake/kernels/csrc/`（新）、`fake/kernels/modules.py`、`fake/kernels/pack.py`、`scripts/slurm/smoke_nvfp4_kernel.sh`（新）
- 使用：
  - B 卡：`CUTLASS_ROOT=/path/to/cutlass sbatch scripts/slurm/smoke_nvfp4_kernel.sh`
  - 端到端：`CUTLASS_ROOT=... KERNEL=1 METHOD=nvfp4 sbatch scripts/slurm/bench_compressed_speed.sh`
- 验证：A100（sm_80）上 `kernel_info().extension_available=False`、自动走 fallback forward 通过；sm_120 kernel 路径需 B 卡上 sbatch 实跑
- 后续注意：
  - kernel 内部按 activation runtime NVFP4 量化处理（W4A4），与 fake 路径的 W4Adense 存在数值差异，精度回归时以 kernel 路径为准
  - `nvfp4_gemm.cu` 当前只实例化一套 tile/cluster（128x128x128 / 1x1x1 / cooperative），若有形状不满足 `can_implement` 要求，会在 `_kernel_forward` 里抛异常并 fallback；后续可在同一 .cu 加多 heuristic 分支
  - `SemiSparseLinear` / `NVFP4SemiSparseLinear` 的真实 kernel 接入（Phase 4 / Phase 5）按同一模式扩展：在 `csrc/` 加 `.cu`，`build.py` 把新源文件加入 sources，`modules.py` 的 `_kernel_forward` 调对应 ext 方法

## 2026-05-12 - A100 兼容性修复 + GPU 验证

- 开发目的：修复 fallback forward 在 A100（sm_80）上的 dtype 不一致问题，确保代码在 A100 开发环境可运行，B 卡上直接可用。
- 修改内容：
  - `fake/kernels/modules.py`：`NVFP4Linear` / `SemiSparseLinear` / `NVFP4SemiSparseLinear` 的 forward 中 bias cast 统一加 `dtype=weight.dtype`，避免 float32 bias 与 bfloat16 weight 混用导致 `F.linear` 报错
  - `fake/kernels/pack.py`：`compress_24` 位运算改用 int32 中间类型（uint16 不支持 CPU bitshift）
- 验证：A100 上全模块 GPU forward 通过（NVFP4Linear / SemiSparseLinear / NVFP4SemiSparseLinear / Conv1x1AsLinear / materialize）
- 后续注意：B 卡上 Phase 4 替换 forward 时，bias 的 dtype 处理逻辑保持不变，kernel 输出直接是 bf16

## 2026-05-11 - CUTLASS kernel 接入初始实现

- 开发目的：将 `cutlass_5090_my/test/` 验证过的 3 条 kernel 路径（dense NVFP4、2:4 sparse bf16、2:4 sparse NVFP4）接入框架，实现 fake-quant → kernel module 替换的完整链路。
- 修改内容：
  - 新增 `fake/kernels/` 包（`__init__.py` / `registry.py` / `pack.py` / `modules.py` / `dispatch.py`）
  - `fake/kernels/registry.py`：登记 3 条 kernel 的 (operation, kernels_pattern)，与 bench_01/03/04.sh 保持一致
  - `fake/kernels/pack.py`：`pack_nvfp4_ue4m3`（uint4 + ue4m3 scales）、`unpack_nvfp4_to_bf16`（fallback dequant）、`compress_24`（2:4 compressed + metadata 占位）、`k_aligned` 对齐检查
  - `fake/kernels/modules.py`：`NVFP4Linear` / `SemiSparseLinear` / `NVFP4SemiSparseLinear` / `Conv1x1AsLinear`，forward 均为 dequant fallback，注释标记 Phase 4 替换点
  - `fake/kernels/dispatch.py`：`materialize(model, metadata, masks)` 按 method 替换模块，K 对齐失败自动降级 dense
  - `fake/compression/checkpoint.py`：新增 `materialize_from_checkpoint`、`checkpoint_csv_fields` 加 kernel_path/kernel_name 字段
  - `scripts/bench_maxvit_dense_speed.py` / `scripts/bench_dinov3_vit7b16_dense_speed.py`：加 `--kernel` / `--masks` flag
  - `scripts/slurm/bench_compressed_speed.sh`：加 `KERNEL=1` 开关、`MASKS` 路径
  - 新增 `scripts/bench_kernel.py`：单 GEMM latency micro-benchmark，调 cutlass_profiler subprocess
  - 新增 `scripts/slurm/bench_kernel.sh`：Slurm 封装
- 影响文件：`fake/kernels/`（全新）、`fake/compression/checkpoint.py`、两个 bench speed 脚本、两个 slurm 脚本
- 后续注意：
  - `compress_24` 的 metadata 是占位实现，real CUTLASS s16832spgemm metadata 需要 `sparse24_meta.cu` helper（standalone `.cu` + `extern "C"` wrapper）
  - `SemiSparseLinear.forward` 和 `NVFP4SemiSparseLinear.forward` 目前走 mask 重建 dense，Phase 4 换成真实 kernel C-extension 后删掉 `mask` buffer
  - `bench_kernel.py` 需要 `CUTLASS_PROFILER` 环境变量指向已编译的 profiler 二进制
