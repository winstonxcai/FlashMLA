#include <torch/extension.h>
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
        [](const pybind11::args &args) {
            if (args.size() != 12) {
                throw pybind11::type_error("sparse_decode_fwd expects 12 positional arguments");
            }
            auto q = args[0].cast<at::Tensor>();
            auto kv = args[1].cast<at::Tensor>();
            auto indices = args[2].cast<at::Tensor>();
            auto topk_length_opt = optional_tensor(args[3]);
            auto attn_sink_opt = optional_tensor(args[4]);
            auto tile_scheduler_metadata_opt = optional_tensor(args[5]);
            auto num_splits_opt = optional_tensor(args[6]);
            auto extra_kv_opt = optional_tensor(args[7]);
            auto extra_indices_opt = optional_tensor(args[8]);
            auto extra_topk_length_opt = optional_tensor(args[9]);
            return sparse_attn_decode_interface(
                q, kv, indices, topk_length_opt, attn_sink_opt,
                tile_scheduler_metadata_opt, num_splits_opt, extra_kv_opt,
                extra_indices_opt, extra_topk_length_opt,
                args[10].cast<int>(), args[11].cast<float>());
        });
    m.def(
        "remnant_sparse_decode_fwd",
        [](const pybind11::args &args) {
            if (args.size() != 16) {
                throw pybind11::type_error("remnant_sparse_decode_fwd expects 16 positional arguments");
            }
            auto q = args[0].cast<at::Tensor>();
            auto kv = args[1].cast<at::Tensor>();
            auto indices = args[2].cast<at::Tensor>();
            auto topk_length_opt = optional_tensor(args[3]);
            auto attn_sink_opt = optional_tensor(args[4]);
            auto tile_scheduler_metadata_opt = optional_tensor(args[5]);
            auto num_splits_opt = optional_tensor(args[6]);
            auto extra_indices_opt = optional_tensor(args[7]);
            auto extra_topk_length_opt = optional_tensor(args[8]);
            auto remnant_values = args[9].cast<at::Tensor>();
            auto remnant_bitmaps = args[10].cast<at::Tensor>();
            auto remnant_scales = args[11].cast<at::Tensor>();
            auto remnant_raw_indices = args[12].cast<at::Tensor>();
            auto remnant_freqs = args[13].cast<at::Tensor>();
            return sparse_attn_decode_remnant_interface(
                q, kv, indices, topk_length_opt, attn_sink_opt,
                tile_scheduler_metadata_opt, num_splits_opt, extra_indices_opt,
                extra_topk_length_opt, remnant_values, remnant_bitmaps,
                remnant_scales, remnant_raw_indices, remnant_freqs,
                args[14].cast<int>(), args[15].cast<float>());
        });
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}
