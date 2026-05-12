// 2:4 structured sparse + NVFP4 GEMM for Blackwell (sm_120), matches cutlass_5090_my bench_04:
//   *sm120*bssptensorop*ue4m3xe2m1*ue4m3xe2m1*f32_void_f32*
//
// Layouts:
//   A_packed          : (M, K/2)  uint8  — NVFP4 packed nibbles
//   A_scales          : (M, K/16) uint8  — ue4m3 per-group scales (over dense K)
//   B_packed_compressed: (N, K/4) uint8  — NVFP4 packed nibbles, 2:4-compressed
//   B_scales          : (N, K/16) uint8  — ue4m3 per-group scales (over dense K)
//   B_meta            : (N, K/8)  uint16 — CUTLASS 2:4 metadata E
//   D                 : (M, N)    bf16

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

// For sparse block-scaled tensor op, use float_e2m1_t as the base type
// The ue4m3 scales are specified separately in the mainloop arguments
using ElementA           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB           = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC           = void;
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentD = 8;

using ArchTag = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassBlockScaledSparseTensorOp;

// Use CuTe Shape for CUTLASS 3.x (same as dense nvfp4_gemm.cu)
using TileShape    = cute::Shape<cute::_128, cute::_128, cute::_256>;  // K tile larger for sparse
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutD, AlignmentD,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using ElementMeta = typename Gemm::ElementE;

torch::Tensor sparse24_nvfp4_gemm(
    torch::Tensor a_packed,             // (M, K/2) uint8
    torch::Tensor a_scales,             // (M, K/16) uint8 (ue4m3)
    torch::Tensor b_packed_compressed,  // (N, K/4) uint8 (2:4 compressed NVFP4)
    torch::Tensor b_scales,             // (N, K/16) uint8 (ue4m3)
    torch::Tensor b_meta,               // (N, K/8) uint16
    double  alpha,
    int64_t m, int64_t n, int64_t k)
{
    TORCH_CHECK(a_packed.dtype()             == torch::kUInt8);
    TORCH_CHECK(b_packed_compressed.dtype()  == torch::kUInt8);
    TORCH_CHECK(a_scales.dtype()             == torch::kUInt8);
    TORCH_CHECK(b_scales.dtype()             == torch::kUInt8);
    TORCH_CHECK(b_meta.dtype()               == torch::kUInt16);
    TORCH_CHECK(k % 64 == 0, "K must be multiple of 64 for 2:4 sparse NVFP4");

    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(a_packed.device());
    auto d = torch::empty({m, n}, options);

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(int(m), int(k), 1));
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(int(n), int(k), 1));
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(int(m), int(n), 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {int(m), int(n), int(k), 1},
        {
            reinterpret_cast<uint8_t const*>(a_packed.data_ptr<uint8_t>()),
            stride_a,
            reinterpret_cast<uint8_t const*>(b_packed_compressed.data_ptr<uint8_t>()),
            stride_b,
            reinterpret_cast<cutlass::float_ue4m3_t const*>(a_scales.data_ptr<uint8_t>()),
            reinterpret_cast<cutlass::float_ue4m3_t const*>(b_scales.data_ptr<uint8_t>()),
            reinterpret_cast<ElementMeta const*>(b_meta.data_ptr<uint16_t>()),
        },
        {
            { static_cast<float>(alpha), 0.f },
            nullptr, {},
            d.data_ptr<cutlass::bfloat16_t>(), stride_d,
        },
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({int64_t(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(a_packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
                "sparse24_nvfp4_gemm: can_implement failed");
    TORCH_CHECK(gemm.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess,
                "sparse24_nvfp4_gemm: init failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
                "sparse24_nvfp4_gemm: run failed");

    return d;
}
