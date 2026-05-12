// 2:4 structured sparse BF16 GEMM, matches cutlass_5090_my bench_03:
//   cutlass_tensorop_s16832spgemm_bf16_64x128_64x6_tn_align8
//
// Layouts:
//   A                  : (M, K)    bf16, row-major
//   B_compressed       : (N, K/2)  bf16, col-major   (CUTLASS LayoutB = ColumnMajor)
//   B_meta             : (N, K/8)  uint16            (CUTLASS LayoutE, already reordered)
//   D                  : (M, N)    bf16
//
// The host compressor compress_2_4_bf16() takes a dense (N, K) bf16 weight
// (with 2:4 zeros already applied) and produces the (compressed, meta) pair
// in the exact layout the kernel expects, by mirroring the setup code from
// CUTLASS example 15 (examples/15_ampere_sparse_tensorop_gemm/).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_uncompress.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using ElementA            = cutlass::bfloat16_t;
using ElementB            = cutlass::bfloat16_t;
using ElementC            = cutlass::bfloat16_t;
using ElementAccumulator  = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 8;
constexpr int AlignmentB = 8;

using ArchTag    = cutlass::arch::Sm80;
using OpClass    = cutlass::arch::OpClassTensorOp;

// Mirrors cutlass_tensorop_s16832spgemm_bf16_64x128_64x6_tn_align8.
using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape<32, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int Stages = 6;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator, ElementAccumulator>;

using Gemm = cutlass::gemm::device::SparseGemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator, OpClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
    AlignmentA, AlignmentB
>;

using ElementMeta            = typename Gemm::ElementE;
using LayoutMeta             = typename Gemm::LayoutE;
using ReorderedLayoutMeta    = typename Gemm::LayoutE;

constexpr int kSparse                = Gemm::kSparse;
constexpr int kElementsPerElementE   = Gemm::kElementsPerElementE;
constexpr int kMetaSizeInBits        = Gemm::kMetaSizeInBits;

torch::Tensor sparse24_gemm_bf16(
    torch::Tensor a,               // (M, K) bf16, row-major
    torch::Tensor b_compressed,    // (N, K/2) bf16, col-major
    torch::Tensor b_meta,          // (N, K/8) uint16, reordered
    int64_t m, int64_t n, int64_t k)
{
    TORCH_CHECK(a.dtype() == torch::kBFloat16);
    TORCH_CHECK(b_compressed.dtype() == torch::kBFloat16);
    TORCH_CHECK(b_meta.dtype() == torch::kUInt16);
    TORCH_CHECK(k % (kSparse * 16) == 0,
                "K must be multiple of 32 for s16832spgemm (kSparse=2, instruction K=16)");

    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(a.device());
    auto d = torch::empty({m, n}, options);

    cutlass::gemm::GemmCoord problem_size{(int)m, (int)n, (int)k};

    // CUTLASS SparseGemm::Arguments order: problem_size, ref_A, ref_B, ref_C, ref_D, ref_E (metadata), epilogue, split_k
    typename Gemm::Arguments args{
        problem_size,
        {reinterpret_cast<ElementA const*>(a.data_ptr<at::BFloat16>()), (int)k},
        {reinterpret_cast<ElementB const*>(b_compressed.data_ptr<at::BFloat16>()), (int)(k / kSparse)},
        {reinterpret_cast<ElementC*>(d.data_ptr<at::BFloat16>()), (int)n},
        {reinterpret_cast<ElementC*>(d.data_ptr<at::BFloat16>()), (int)n},
        {reinterpret_cast<ElementMeta const*>(b_meta.data_ptr<uint16_t>()),
         (int)(k / kSparse / kElementsPerElementE)},
        {1.0f, 0.0f},
        1,
    };

    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty({int64_t(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess,
                "sparse24_gemm_bf16: can_implement failed");
    TORCH_CHECK(gemm.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess,
                "sparse24_gemm_bf16: init failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
                "sparse24_gemm_bf16: run failed");

    return d;
}

// Host-side compressor: takes a dense (N, K) bf16 weight in column-major layout
// (CUTLASS LayoutB), with 2:4 zeros already applied, and produces:
//   compressed : (N, K/2) bf16, col-major — the 2 non-zero values per 4
//   meta       : (N, K/8) uint16          — reordered metadata E
//
// Setup mirrors example 15 (examples/15_ampere_sparse_tensorop_gemm/) so the
// metadata layout matches what the Sparse kernel expects.
std::tuple<torch::Tensor, torch::Tensor> compress_2_4_bf16(torch::Tensor dense_weight)
{
    TORCH_CHECK(dense_weight.dtype() == torch::kBFloat16);
    TORCH_CHECK(dense_weight.is_cuda());
    TORCH_CHECK(dense_weight.dim() == 2);
    int n = dense_weight.size(0);
    int k = dense_weight.size(1);
    TORCH_CHECK(k % (kSparse * kElementsPerElementE) == 0,
                "K must be divisible by kSparse * kElementsPerElementE");

    // Move dense → host for compressor (CUTLASS host-side utility).
    auto dense_cpu = dense_weight.detach().to(torch::kCPU).contiguous();

    cutlass::HostTensor<ElementB, LayoutB> dense_h({n, k});
    cutlass::HostTensor<ElementB, LayoutB> compressed_h({n, k / kSparse});
    cutlass::HostTensor<ElementMeta, LayoutMeta> meta_h({n, k / kSparse / kElementsPerElementE});
    cutlass::HostTensor<ElementMeta, ReorderedLayoutMeta> meta_reordered_h(
        {n, k / kSparse / kElementsPerElementE});

    // copy torch tensor → CUTLASS host tensor (LayoutB is ColumnMajor, but the
    // numerical content is what matters; we lay it out element-by-element).
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            dense_h.at({row, col}) =
                reinterpret_cast<ElementB const*>(dense_cpu.data_ptr<at::BFloat16>())[row * k + col];
        }
    }

    // Step 1: compress dense → (compressed, meta) using CUTLASS sparse helper.
    // Note: CUTLASS does not ship a single-call "compress_2_to_4" for arbitrary
    // dense input; instead, example 15 first fills random metadata then uses
    // uncompress to fill the dense. We do the inverse: derive metadata from the
    // 2:4 zero pattern, gather non-zeros, then reorder metadata.
    for (int row = 0; row < n; ++row) {
        for (int g = 0; g < k / 4; ++g) {
            ElementB v0 = dense_h.at({row, g * 4 + 0});
            ElementB v1 = dense_h.at({row, g * 4 + 1});
            ElementB v2 = dense_h.at({row, g * 4 + 2});
            ElementB v3 = dense_h.at({row, g * 4 + 3});
            bool nz0 = (float(v0) != 0.0f);
            bool nz1 = (float(v1) != 0.0f);
            bool nz2 = (float(v2) != 0.0f);
            bool nz3 = (float(v3) != 0.0f);
            // pick the 2 keep positions in ascending index order
            int keep0 = -1, keep1 = -1;
            int bits[4] = { nz0, nz1, nz2, nz3 };
            for (int i = 0; i < 4; ++i) {
                if (bits[i]) {
                    if (keep0 < 0) keep0 = i;
                    else if (keep1 < 0) keep1 = i;
                }
            }
            if (keep0 < 0) { keep0 = 0; keep1 = 1; }
            else if (keep1 < 0) { keep1 = (keep0 == 3) ? 2 : keep0 + 1; }

            ElementB picked[2];
            picked[0] = dense_h.at({row, g * 4 + keep0});
            picked[1] = dense_h.at({row, g * 4 + keep1});
            compressed_h.at({row, g * 2 + 0}) = picked[0];
            compressed_h.at({row, g * 2 + 1}) = picked[1];

            // Encode 2-bit positions into ElementE (uint16 holds 4 groups = 16 bits).
            uint16_t enc4 = uint16_t(keep0 & 0x3) | (uint16_t(keep1 & 0x3) << 2);
            // We're filling the un-reordered meta one nibble at a time; pack
            // 4 groups per uint16 slot.
            int e_col = g / 4;
            int e_sub = g % 4;
            uint16_t cur = uint16_t(meta_h.at({row, e_col}));
            cur |= (enc4 << (4 * e_sub));
            meta_h.at({row, e_col}) = ElementMeta(cur);
        }
    }

    // Step 2: reorder metadata into the layout consumed by the SparseGemm kernel.
    // reorder_meta expects problem_size as {M, N, K} where M=rows, N=cols of metadata tensor
    cutlass::reorder_meta<InstructionShape::kK, kSparse, kMetaSizeInBits>(
        meta_reordered_h.host_ref(),
        meta_h.host_ref(),
        {n, k / kSparse / kElementsPerElementE, 1}
    );

    // Step 3: copy results back to GPU tensors.
    auto bf16_opt = torch::TensorOptions().dtype(torch::kBFloat16).device(dense_weight.device());
    auto u16_opt  = torch::TensorOptions().dtype(torch::kUInt16).device(dense_weight.device());
    auto compressed = torch::empty({n, k / kSparse}, bf16_opt);
    auto meta_out   = torch::empty({n, k / kSparse / kElementsPerElementE}, u16_opt);

    cudaMemcpy(compressed.data_ptr<at::BFloat16>(),
               compressed_h.host_data(),
               sizeof(ElementB) * n * (k / kSparse),
               cudaMemcpyHostToDevice);
    cudaMemcpy(meta_out.data_ptr<uint16_t>(),
               meta_reordered_h.host_data(),
               sizeof(ElementMeta) * n * (k / kSparse / kElementsPerElementE),
               cudaMemcpyHostToDevice);

    return std::make_tuple(compressed, meta_out);
}
