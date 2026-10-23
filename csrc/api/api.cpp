#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void register_sparse_prefill(pybind11::module_& m);
void register_sparse_decode(pybind11::module_& m);
void register_dense_fwd(pybind11::module_& m);
void register_dense_bwd(pybind11::module_& m);
void register_dense_decode(pybind11::module_& m);
void register_fused_norm_rope_attn_rope_cast_fwd(pybind11::module_& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    register_sparse_prefill(m);
    register_sparse_decode(m);
#ifdef FLASHMLA_ENABLE_SM100
    register_dense_fwd(m);
    register_dense_bwd(m);
#endif
    register_dense_decode(m);
#ifdef FLASHMLA_ENABLE_SM100
    register_fused_norm_rope_attn_rope_cast_fwd(m);
#endif
}
