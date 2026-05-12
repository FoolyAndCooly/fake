// Dense NVFP4 GEMM for Blackwell (sm_120), matches cutlass_5090_my bench_01:
//   *sm120*bstensorop*ue4m3xe2m1*ue4m3xe2m1*f32_void_f32*cooperative*
//
// Layouts (row-major for A, column-major for B, row-major for D):
//   A_packed    : (M, K/2) uint8       — fp4 nibbles (low=even, high=odd)
//   A_scales    : (M, K/16) uint8      — ue4m3 (float8_e4m3fn) per-group scales
//   B_packed    : (N, K/2) uint8       — fp4 nibbles, column-major logically ⇒ stored (N, K/2)
//   B_scales    : (N, K/16) uint8      — ue4m3 per-group scales
//   alpha       : float scalar
//   D           : (M, N) bf16 output
//
// This is a minimal wrapper around CUTLASS's Blackwell block-scaled NVFP4 collective
// GEMM. If you need other tile/cluster configs, add more Gemm_* typedefs and pick
// at runtime by shape.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

// NVFP4 data types on CUTLASS side.
using ElementA           = cutlass::nv_float4_t<cutlass::float_ue4m3_t>;
using ElementB           = cutlass::nv_float4_t<cutlass::float_ue4m3_t>;
using ElementC           = void;
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentD = 8;

using ArchTag   = cutlass::arch::Sm120;
using OpClass   = cutlass::arch::OpClassBlockScaledTensorOp;

// Matches cooperative kernel in bench_01.
using TileShape    = cutlass::gemm::GemmShape<128, 128, 128>;
using ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;

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

// Kernel entry. Returns a bf16 (M, N) tensor.
torch::Tensor nvfp4_gemm(
    torch::Tensor a_packed,   // (M, K/2) uint8
    torch::Tensor a_scales,   // (M, K/16) uint8, interpreted as ue4m3
    torch::Tensor b_packed,   // (N, K/2) uint8
    torch::Tensor b_scales,   // (N, K/16) uint8
    double        alpha,
    int64_t       m,
    int64_t       n,
    int64_t       k)
{
    TORCH_CHECK(a_packed.is_cuda() && b_packed.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(a_packed.dtype() == torch::kUInt8);
    TORCH_CHECK(b_packed.dtype() == torch::kUInt8);
    TORCH_CHECK(a_scales.dtype() == torch::kUInt8);
    TORCH_CHECK(b_scales.dtype() == torch::kUInt8);
    TORCH_CHECK(k % 16 == 0, "K must be multiple of 16");

    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(a_packed.device());
    auto d = torch::empty({m, n}, options);

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(int(m), int(k), int(1)));
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(int(n), int(k), int(1)));
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(int(m), int(n), int(1)));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {int(m), int(n), int(k), 1},
        {
            reinterpret_cast<typename ElementA::PackedElement const*>(a_packed.data_ptr<uint8_t>()),
            stride_a,
            reinterpret_cast<typename ElementB::PackedElement const*>(b_packed.data_ptr<uint8_t>()),
            stride_b,
            reinterpret_cast<cutlass::float_ue4m3_t const*>(a_scales.data_ptr<uint8_t>()),
            reinterpret_cast<cutlass::float_ue4m3_t const*>(b_scales.data_ptr<uint8_t>()),
        },
        {
            { static_cast<float>(alpha), 0.f },
            nullptr, {},  // C = null, void
            d.data_ptr<cutlass::bfloat16_t>(), stride_d,
        },
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({int64_t(workspace_size)}, torch::TensorOptions().dtype(torch::kUInt8).device(a_packed.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = gemm.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass gemm cannot implement: ", int(status));

    status = gemm.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass gemm init failed: ", int(status));

    status = gemm.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "cutlass gemm run failed: ", int(status));

    return d;
}
