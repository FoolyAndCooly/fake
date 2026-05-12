from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fake.kernels import build as _kernel_build
from fake.kernels.pack import (
    compress_24,
    pack_nvfp4_ue4m3,
    unpack_nvfp4_to_bf16,
)


class NVFP4Linear(nn.Module):
    """NVFP4-quantized linear layer (W4A4).

    On sm_120 with CUTLASS_ROOT set, forward() calls the CUTLASS bstensorop
    NVFP4 GEMM. Elsewhere it falls back to a bf16 dequant-then-F.linear path,
    which is numerically equivalent to the pre-kernel fake-quant behavior.
    """

    def __init__(
        self,
        uint4_packed: torch.Tensor,
        ue4m3_scales: torch.Tensor,
        alpha: float,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
        group_size: int = 16,
    ) -> None:
        super().__init__()
        self.register_buffer("uint4_packed", uint4_packed)
        self.register_buffer("ue4m3_scales", ue4m3_scales)
        self.alpha = alpha
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 16) -> "NVFP4Linear":
        # Cast weight to bf16 first — the kernel's A/B dtype assumption is bf16
        # activations downcast to fp4 at runtime, weight pre-quantized from bf16.
        w = linear.weight.data.to(torch.bfloat16)
        uint4, scales, alpha = pack_nvfp4_ue4m3(w, group_size)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(uint4, scales, alpha, bias, linear.out_features, linear.in_features, group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()

        out = self._kernel_forward(x2d)
        if out is None:
            out = self._fallback_forward(x2d)

        return out.reshape(*orig_shape[:-1], self.out_features)

    def _kernel_forward(self, x2d: torch.Tensor) -> torch.Tensor | None:
        ext = _kernel_build.load()
        if ext is None or not x2d.is_cuda:
            return None

        # Runtime-quantize activation to NVFP4 (W4A4 path).
        a_packed, a_scales_f8, _ = pack_nvfp4_ue4m3(x2d.to(torch.bfloat16), self.group_size)
        # CUTLASS kernel expects ue4m3 scales as uint8 bit-casts of float8_e4m3fn.
        a_scales = a_scales_f8.view(torch.uint8)
        b_scales = self.ue4m3_scales.view(torch.uint8)

        m = x2d.shape[0]
        k = x2d.shape[1]
        n = self.out_features
        try:
            d = ext.nvfp4_gemm(
                a_packed.contiguous(),
                a_scales.contiguous(),
                self.uint4_packed.contiguous(),
                b_scales.contiguous(),
                float(self.alpha),
                int(m), int(n), int(k),
            )
        except Exception:
            return None

        if self.bias is not None:
            d = d + self.bias.to(dtype=d.dtype, device=d.device)
        return d

    def _fallback_forward(self, x2d: torch.Tensor) -> torch.Tensor:
        weight = unpack_nvfp4_to_bf16(
            self.uint4_packed, self.ue4m3_scales, self.alpha, self.group_size
        ).to(x2d.device)
        bias = self.bias.to(dtype=weight.dtype, device=x2d.device) if self.bias is not None else None
        return F.linear(x2d.to(weight.dtype), weight, bias)


class SemiSparseLinear(nn.Module):
    """2:4 structured-sparse BF16 linear layer.

    On sm_80+ with CUTLASS_ROOT set, forward() calls the CUTLASS s16832spgemm
    kernel. Otherwise falls back to mask-reconstructed dense F.linear.

    The `compressed`/`meta` buffers are CUTLASS-friendly when produced by
    ext.compress_2_4_bf16(); on systems where that path is unavailable, the
    mask-based fallback still produces correct output.
    """

    def __init__(
        self,
        compressed: torch.Tensor,
        meta: torch.Tensor,
        mask: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
        kernel_ready: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("compressed", compressed)
        self.register_buffer("meta", meta)
        self.register_buffer("mask", mask)
        self.out_features = out_features
        self.in_features = in_features
        self.kernel_ready = kernel_ready  # True iff compressed/meta produced by CUTLASS compressor
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, mask: torch.Tensor) -> "SemiSparseLinear":
        w = linear.weight.data.to(torch.bfloat16)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        # Try the CUTLASS host-side compressor first; fall back to Python encoding.
        compressed, meta, kernel_ready = _maybe_cutlass_compress_24(w)
        if compressed is None:
            compressed, meta = compress_24(w, mask)
            kernel_ready = False
        return cls(compressed, meta, mask, bias, linear.out_features, linear.in_features, kernel_ready)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()
        out = self._kernel_forward(x2d)
        if out is None:
            out = self._fallback_forward(x2d)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def _kernel_forward(self, x2d: torch.Tensor) -> torch.Tensor | None:
        if not self.kernel_ready:
            return None
        ext = _kernel_build.load()
        if ext is None or not x2d.is_cuda:
            return None
        m = x2d.shape[0]
        k = x2d.shape[1]
        n = self.out_features
        try:
            d = ext.sparse24_gemm_bf16(
                x2d.to(torch.bfloat16).contiguous(),
                self.compressed.contiguous(),
                self.meta.contiguous(),
                int(m), int(n), int(k),
            )
        except Exception:
            return None
        if self.bias is not None:
            d = d + self.bias.to(dtype=d.dtype, device=d.device)
        return d

    def _fallback_forward(self, x2d: torch.Tensor) -> torch.Tensor:
        dev = x2d.device
        weight = torch.zeros(
            self.out_features, self.in_features,
            dtype=self.compressed.dtype, device=dev,
        )
        weight[self.mask.to(dev)] = self.compressed.to(dev).reshape(-1)
        bias = self.bias.to(dtype=weight.dtype, device=dev) if self.bias is not None else None
        return F.linear(x2d.to(weight.dtype), weight, bias)


class NVFP4SemiSparseLinear(nn.Module):
    """2:4 sparse + NVFP4 joint layer (sm_120 bssptensorop).

    Prune first (2:4), then quantize the compressed weights to NVFP4.
    Kernel path uses ext.sparse24_nvfp4_gemm; fallback dequants compressed
    NVFP4 → bf16 and reconstructs dense via mask.
    """

    def __init__(
        self,
        uint4_compressed: torch.Tensor,
        ue4m3_scales: torch.Tensor,
        meta: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
        bias: torch.Tensor | None,
        out_features: int,
        in_features: int,
        group_size: int = 32,
    ) -> None:
        super().__init__()
        self.register_buffer("uint4_compressed", uint4_compressed)
        self.register_buffer("ue4m3_scales", ue4m3_scales)
        self.register_buffer("meta", meta)
        self.register_buffer("mask", mask)
        self.alpha = alpha
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size
        self.bias = nn.Parameter(bias) if bias is not None else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, mask: torch.Tensor, group_size: int = 32) -> "NVFP4SemiSparseLinear":
        w = linear.weight.data.to(torch.bfloat16)
        compressed_bf16, meta = compress_24(w, mask)
        uint4, scales, alpha = pack_nvfp4_ue4m3(compressed_bf16, group_size)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(uint4, scales, meta, mask, alpha, bias, linear.out_features, linear.in_features, group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()
        out = self._kernel_forward(x2d)
        if out is None:
            out = self._fallback_forward(x2d)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def _kernel_forward(self, x2d: torch.Tensor) -> torch.Tensor | None:
        ext = _kernel_build.load()
        if ext is None or not x2d.is_cuda:
            return None
        # Quantize activation to NVFP4 at runtime.
        a_packed, a_scales_f8, _ = pack_nvfp4_ue4m3(x2d.to(torch.bfloat16), self.group_size)
        a_scales = a_scales_f8.view(torch.uint8)
        b_scales = self.ue4m3_scales.view(torch.uint8)
        m = x2d.shape[0]
        k = x2d.shape[1]
        n = self.out_features
        try:
            d = ext.sparse24_nvfp4_gemm(
                a_packed.contiguous(),
                a_scales.contiguous(),
                self.uint4_compressed.contiguous(),
                b_scales.contiguous(),
                self.meta.contiguous(),
                float(self.alpha),
                int(m), int(n), int(k),
            )
        except Exception:
            return None
        if self.bias is not None:
            d = d + self.bias.to(dtype=d.dtype, device=d.device)
        return d

    def _fallback_forward(self, x2d: torch.Tensor) -> torch.Tensor:
        dev = x2d.device
        compressed_bf16 = unpack_nvfp4_to_bf16(
            self.uint4_compressed, self.ue4m3_scales, self.alpha, self.group_size
        ).to(dev)
        weight = torch.zeros(
            self.out_features, self.in_features,
            dtype=compressed_bf16.dtype, device=dev,
        )
        weight[self.mask.to(dev)] = compressed_bf16.reshape(-1)
        bias = self.bias.to(dtype=weight.dtype, device=dev) if self.bias is not None else None
        return F.linear(x2d.to(weight.dtype), weight, bias)


def _maybe_cutlass_compress_24(w_bf16: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
    """Use the CUTLASS host compressor if the extension is loadable; else return (None, None, False)."""
    if not w_bf16.is_cuda:
        return None, None, False
    ext = _kernel_build.load()
    if ext is None:
        return None, None, False
    try:
        compressed, meta = ext.compress_2_4_bf16(w_bf16.contiguous())
        return compressed, meta, True
    except Exception:
        return None, None, False


class Conv1x1AsLinear(nn.Module):
    """Wraps a 1×1 Conv2d as a Linear-based kernel module.

    Reshapes (B,C,H,W) → (B*H*W, C) before the inner module and back after.
    """

    def __init__(self, inner: nn.Module, out_channels: int, in_channels: int) -> None:
        super().__init__()
        self.inner = inner
        self.out_channels = out_channels
        self.in_channels = in_channels

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, kernel_cls, **kwargs) -> "Conv1x1AsLinear":
        device = conv.weight.device
        fake_linear = nn.Linear(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
        fake_linear.weight.data.copy_(conv.weight.data.squeeze(-1).squeeze(-1))
        if conv.bias is not None:
            fake_linear.bias.data.copy_(conv.bias.data)
        # Keep fake_linear on the same device as the original conv so that any
        # CUTLASS host compressor inside kernel_cls.from_linear sees a CUDA tensor.
        fake_linear = fake_linear.to(device)
        inner = kernel_cls.from_linear(fake_linear, **kwargs)
        return cls(inner, conv.out_channels, conv.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = self.inner(x.permute(0, 2, 3, 1).reshape(B * H * W, C))
        return out.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()