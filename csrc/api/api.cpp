#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_fwd.h"

namespace {

std::optional<at::Tensor> optional_tensor(const pybind11::object &value) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return value.cast<at::Tensor>();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def(
        "sparse_decode_fwd",
        [](const at::Tensor &q,
           const at::Tensor &kv,
           const at::Tensor &indices,
           const pybind11::object &topk_length,
           const pybind11::object &attn_sink,
           const pybind11::object &tile_scheduler_metadata,
           const pybind11::object &num_splits,
           const pybind11::object &extra_kv,
           const pybind11::object &extra_indices,
           const pybind11::object &extra_topk_length,
           int d_v,
           float sm_scale) {
            auto topk_length_opt = optional_tensor(topk_length);
            auto attn_sink_opt = optional_tensor(attn_sink);
            auto tile_scheduler_metadata_opt = optional_tensor(tile_scheduler_metadata);
            auto num_splits_opt = optional_tensor(num_splits);
            auto extra_kv_opt = optional_tensor(extra_kv);
            auto extra_indices_opt = optional_tensor(extra_indices);
            auto extra_topk_length_opt = optional_tensor(extra_topk_length);
            return sparse_attn_decode_interface(
                q, kv, indices, topk_length_opt, attn_sink_opt,
                tile_scheduler_metadata_opt, num_splits_opt, extra_kv_opt,
                extra_indices_opt, extra_topk_length_opt, d_v, sm_scale);
        });
    m.def(
        "remnant_sparse_decode_fwd",
        [](const at::Tensor &q,
           const at::Tensor &kv,
           const at::Tensor &indices,
           const pybind11::object &topk_length,
           const pybind11::object &attn_sink,
           const pybind11::object &tile_scheduler_metadata,
           const pybind11::object &num_splits,
           const pybind11::object &extra_indices,
           const pybind11::object &extra_topk_length,
           const at::Tensor &remnant_values,
           const at::Tensor &remnant_bitmaps,
           const at::Tensor &remnant_scales,
           const at::Tensor &remnant_raw_indices,
           const at::Tensor &remnant_freqs,
           int d_v,
           float sm_scale) {
            auto topk_length_opt = optional_tensor(topk_length);
            auto attn_sink_opt = optional_tensor(attn_sink);
            auto tile_scheduler_metadata_opt = optional_tensor(tile_scheduler_metadata);
            auto num_splits_opt = optional_tensor(num_splits);
            auto extra_indices_opt = optional_tensor(extra_indices);
            auto extra_topk_length_opt = optional_tensor(extra_topk_length);
            return sparse_attn_decode_remnant_interface(
                q, kv, indices, topk_length_opt, attn_sink_opt,
                tile_scheduler_metadata_opt, num_splits_opt, extra_indices_opt,
                extra_topk_length_opt, remnant_values, remnant_bitmaps,
                remnant_scales, remnant_raw_indices, remnant_freqs, d_v, sm_scale);
        });
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
