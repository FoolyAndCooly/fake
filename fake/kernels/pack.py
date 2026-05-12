from __future__ import annotations

import torch

FP4_E2M1_MAX = 6.0

# e2m1 fp4 code → float value table (unsigned magnitude, index = code bits [2:0])
_FP4_MAG_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def pack_nvfp4_ue4m3(
    weight: torch.Tensor,
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Quantize and pack a weight matrix to NVFP4 (e2m1) with ue4m3 block scales.

    Returns:
        uint4_packed  (out, in//2)  torch.uint8   — 2 fp4 nibbles per byte (low=even, high=odd)
        ue4m3_scales  (out, in//group_size)  torch.float8_e4m3fn
        alpha         float (per-tensor scale, always 1.0 for now)
    """
    x = weight.detach().float()
    rows, cols = x.shape
    assert cols % group_size == 0, f"in_features={cols} not divisible by group_size={group_size}"

    grouped = x.reshape(rows, -1, group_size)                        # (out, g, gs)
    scales_f32 = grouped.abs().amax(dim=-1) / FP4_E2M1_MAX          # (out, g)
    scales_f32 = scales_f32.clamp(min=1e-12)

    normalized = grouped / scales_f32.unsqueeze(-1)                  # (out, g, gs)
    q_int = _quantize_to_fp4_int(normalized)                         # (out, g, gs) uint8 [0..15]
    q_flat = q_int.reshape(rows, cols)                               # (out, in)

    # pack: even-index nibble in low bits, odd-index nibble in high bits
    uint4_packed = ((q_flat[:, 0::2] & 0x0F) | ((q_flat[:, 1::2] & 0x0F) << 4)).to(torch.uint8)

    ue4m3_scales = scales_f32.to(torch.float8_e4m3fn)
    return uint4_packed, ue4m3_scales, 1.0


def unpack_nvfp4_to_bf16(
    uint4_packed: torch.Tensor,
    ue4m3_scales: torch.Tensor,
    alpha: float,
    group_size: int = 16,
) -> torch.Tensor:
    """Dequantize packed NVFP4 weight back to bf16 (used as fallback forward)."""
    codebook = _FP4_MAG_TABLE.to(uint4_packed.device)
    rows = uint4_packed.shape[0]
    cols = uint4_packed.shape[1] * 2

    low  = (uint4_packed & 0x0F).to(torch.int32)
    high = ((uint4_packed >> 4) & 0x0F).to(torch.int32)
    q_flat = torch.stack([low, high], dim=-1).reshape(rows, cols)    # (out, in)

    sign = torch.where(q_flat >= 8, -1.0, 1.0)
    mag  = q_flat & 0x07
    weight_fp4 = sign * codebook[mag]

    scales_f32 = ue4m3_scales.float()                                # (out, g)
    scales_exp = scales_f32.unsqueeze(-1).expand(-1, -1, group_size).reshape(rows, cols)
    return (weight_fp4 * scales_exp * alpha).to(torch.bfloat16)


def compress_24(
    weight: torch.Tensor,   # (out, in) bfloat16, zeros already applied by pruning
    mask: torch.Tensor,     # (out, in) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack 2:4-sparse weight into compressed form + placeholder metadata.

    Returns:
        compressed  (out, in//2)  same dtype as weight — the 2 non-zero values per 4
        meta        (out, in//16) torch.uint16 — placeholder metadata layout.
    """
    assert weight.shape[-1] % 4 == 0, "K must be divisible by 4"
    rows, cols = weight.shape
    # Ensure mask is on the same device as weight (caller may pass CPU mask).
    mask = mask.to(weight.device)

    w4 = weight.reshape(rows, cols // 4, 4)     # (out, groups, 4)
    m4 = mask.reshape(rows, cols // 4, 4)       # (out, groups, 4)
    if not torch.all(m4.sum(dim=-1) == 2):
        raise ValueError("2:4 mask must keep exactly 2 values per group of 4")

    # Keep positions in deterministic order so compressed values and metadata match.
    keep_idx = torch.argsort(m4.to(torch.int64), dim=-1, descending=True)[..., :2]
    keep_idx, _ = torch.sort(keep_idx, dim=-1)
    compressed = torch.gather(w4, -1, keep_idx).reshape(rows, cols // 2)

    # Encode the two kept positions as 2-bit fields, then pack 4 groups into uint16.
    # Use int32 for bitwise ops (uint16 doesn't support shifts on CPU).
    p0 = keep_idx[..., 0].to(torch.int32) & 0x3
    p1 = keep_idx[..., 1].to(torch.int32) & 0x3
    nibble = p0 | (p1 << 2)                      # (out, groups) 4-bit per group, int32

    assert (cols // 4) % 4 == 0, "groups per row must be multiple of 4"
    nibble4 = nibble.reshape(rows, -1, 4)        # (out, col//16, 4)
    meta = (
        nibble4[:, :, 0]
        | (nibble4[:, :, 1] << 4)
        | (nibble4[:, :, 2] << 8)
        | (nibble4[:, :, 3] << 12)
    ).to(torch.uint16)

    return compressed.contiguous(), meta.contiguous()


def required_k_multiple(method: str) -> int:
    if method == "nvfp4_semi_structured_sparse":
        return 64
    if "nvfp4" in method:
        return 16
    return 64


def k_aligned(cols: int, method: str) -> bool:
    """Check K-dimension alignment required by the target kernel."""
    return cols % required_k_multiple(method) == 0


# ── internal ──────────────────────────────────────────────────────────────────

def _quantize_to_fp4_int(x: torch.Tensor) -> torch.Tensor:
    """Map float values to e2m1 fp4 unsigned code (0-15), sign in bit 3."""
    sign_bit = (x < 0).to(torch.int32) << 3
    y = x.abs()
    mag = torch.zeros_like(y, dtype=torch.int32)
    mag[(y > 0.25) & (y < 0.75)]  = 1
    mag[(y >= 0.75) & (y <= 1.25)] = 2
    mag[(y > 1.25) & (y < 1.75)]  = 3
    mag[(y >= 1.75) & (y <= 2.5)]  = 4
    mag[(y > 2.5) & (y < 3.5)]    = 5
    mag[(y >= 3.5) & (y <= 5.0)]   = 6
    mag[y > 5.0]                   = 7
    return (mag | sign_bit).to(torch.uint8)
