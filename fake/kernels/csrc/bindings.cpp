#include <torch/extension.h>

torch::Tensor nvfp4_gemm(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor b_packed,
    torch::Tensor b_scales,
    double        alpha,
    int64_t       m,
    int64_t       n,
    int64_t       k);

torch::Tensor sparse24_gemm_bf16(
    torch::Tensor a,
    torch::Tensor b_compressed,
    torch::Tensor b_meta,
    int64_t m, int64_t n, int64_t k);

std::tuple<torch::Tensor, torch::Tensor> compress_2_4_bf16(
    torch::Tensor dense_weight);

torch::Tensor sparse24_nvfp4_gemm(
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    torch::Tensor b_packed_compressed,
    torch::Tensor b_scales,
    torch::Tensor b_meta,
    double  alpha,
    int64_t m, int64_t n, int64_t k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("nvfp4_gemm", &nvfp4_gemm,
            "Dense NVFP4 GEMM (sm_120 bstensorop ue4m3xe2m1)",
            py::arg("a_packed"), py::arg("a_scales"),
            py::arg("b_packed"), py::arg("b_scales"),
            py::arg("alpha"),
            py::arg("m"), py::arg("n"), py::arg("k"));

    mod.def("sparse24_gemm_bf16", &sparse24_gemm_bf16,
            "2:4 sparse bf16 GEMM (s16832spgemm)",
            py::arg("a"), py::arg("b_compressed"), py::arg("b_meta"),
            py::arg("m"), py::arg("n"), py::arg("k"));

    mod.def("compress_2_4_bf16", &compress_2_4_bf16,
            "Produce CUTLASS (compressed, meta) pair from dense bf16 weight with 2:4 zeros",
            py::arg("dense_weight"));

    mod.def("sparse24_nvfp4_gemm", &sparse24_nvfp4_gemm,
            "2:4 sparse NVFP4 GEMM (sm_120 bssptensorop ue4m3xe2m1)",
            py::arg("a_packed"), py::arg("a_scales"),
            py::arg("b_packed_compressed"), py::arg("b_scales"), py::arg("b_meta"),
            py::arg("alpha"),
            py::arg("m"), py::arg("n"), py::arg("k"));
}
