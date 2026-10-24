// Author: Winston Cai. Packed dispatch on the v0.5.22 V4 API.
#include "common.h"

#include "kernels/params.h"

#include "kernels/sm90/decode/sparse/splitkv_mla.h"
#ifdef FLASHMLA_ENABLE_SM100
#include "kernels/sm100/decode/sparse/head64/kernel.h"
#include "kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h"
#endif
#include "kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "kernels/smxx/decode/combine/combine.h"

template<bool ENABLE_SPLIT_KV>
static constexpr SparseAttnFwdMode get_decode_fwd_mode() {
    if constexpr (ENABLE_SPLIT_KV) {
        return SparseAttnFwdMode::DecodeWithSplitKV;
    } else {
        return SparseAttnFwdMode::Decode;
    }
}

// Feature set of sparse decoding kernels
enum class DecodeFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    V32_KVCACHE_FORMAT,
    V32_NO_ROPE_KVCACHE_FORMAT,
    V4_KVCACHE_FORMAT,
    V41_KVCACHE_FORMAT,
    V41_FP4_KVCACHE_FORMAT,

    ATTN_SINK,
    TOPK_LENGTH,
    EXTRA_KVCACHE,
    EXTRA_TOPK_LENGTH
};

struct DecodeImplMeta {
    int num_sm_parts;
    int fixed_overhead_num_blocks;
    int block_size_topk;
};

class DecodeImplBase : public ImplBase<
    SparseAttnDecodeParams,
    DecodeFeatures
> {
public:
    virtual DecodeImplMeta get_meta(int h_q, int s_q) = 0;
};

class Decode_Sm90_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::V32_NO_ROPE_KVCACHE_FORMAT,
        DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q / (h_q/64), 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            DISPATCH_NUM_HEADS(params.h_q, NUM_HEADS, [&]() {
                sm90::decode::sparse::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE, NUM_HEADS>(params);
            });
        });
    }
};

template<int FORMAT>
class Decode_Sm90_FP4_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64, DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512, DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::V41_FP4_KVCACHE_FORMAT, DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH, DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )
public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        return {std::max(Arch().num_sms / s_q / (h_q/64), 1), 5, 64};
    }
protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &) override {
        DISPATCH_NUM_HEADS(params.h_q, NUM_HEADS, [&]() {
            sm90::decode::sparse::run_flash_splitkv_mla_fp8_sparse_kernel<
                ModelType::V4, NUM_HEADS, false, FORMAT>(params);
        });
    }
};

class Decode_Sm90_Remnant_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q / (h_q / 64), 1),
            1,  // Packed decode has a smaller per-request setup estimate.
            h_q == 64 ? 32 : 64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_NUM_HEADS(params.h_q, NUM_HEADS, [&]() {
            sm90::decode::sparse::run_flash_splitkv_mla_fp8_sparse_kernel<
                ModelType::V4, NUM_HEADS, true>(params);
        });
    }
};

#ifdef FLASHMLA_ENABLE_SM100
class Decode_Sm100_Head64_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::V32_NO_ROPE_KVCACHE_FORMAT,
        DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::V41_KVCACHE_FORMAT,
        DecodeFeatures::V41_FP4_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )
    using SupportedKVFormats = KVFormatPairs<KVFormatPair<ModelType::V32>, KVFormatPair<ModelType::V32_NO_ROPE>, KVFormatPair<ModelType::V4>,
                                             KVFormatPair<ModelType::V41>, KVFormatPair<ModelType::V41, ModelType::V41_FP4>>;

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q, 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        dispatch_kv_formats(SupportedKVFormats{}, params.model_type, params.extra_model_type, [&]<ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>() {
            DISPATCH_BOOLEAN_FLAG(params.enable_split_kv, ENABLE_SPLIT_KV, ([&]() {
                TORCH_CHECK(params.h_q == 64, "Unsupported h_q: ", params.h_q);
                using sm100::decode::sparse::head64::Config;
                sm100::decode::sparse::head64::run_flash_splitkv_mla_fp8_sparse_kernel<Config{MODEL_TYPE, EXTRA_MODEL_TYPE, ENABLE_SPLIT_KV}>(params);
            }));
        });
    }
};


// An implementation that calls the head64 kernel twice to process head128
// Necessary for running V3.2 shape (i.e. h = 128, d_qk = 576) on SM100f
class Decode_Sm100_Head64x2_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::V32_NO_ROPE_KVCACHE_FORMAT,
        DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )
    using SupportedKVFormats = KVFormatPairs<KVFormatPair<ModelType::V32>, KVFormatPair<ModelType::V32_NO_ROPE>, KVFormatPair<ModelType::V4>>;

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q, 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        dispatch_kv_formats(SupportedKVFormats{}, params.model_type, params.extra_model_type, [&]<ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>() {
            DISPATCH_BOOLEAN_FLAG(params.enable_split_kv, ENABLE_SPLIT_KV, ([&]() {
                for (int start_head_idx = 0; start_head_idx < 128; start_head_idx += 64) {
                    SparseAttnDecodeParams cur_params = params;
                    cur_params.q += start_head_idx * params.stride_q_h_q;
                    if (cur_params.attn_sink) {
                        cur_params.attn_sink += start_head_idx;
                    }
                    cur_params.lse += start_head_idx;
                    cur_params.out += start_head_idx * params.stride_o_h_q;
                    if (cur_params.enable_split_kv) {
                        cur_params.lse_accum += start_head_idx;
                        cur_params.o_accum += start_head_idx * params.stride_o_accum_h_q;
                    }
                    cur_params.h_q = 64;
                    using sm100::decode::sparse::head64::Config;
                    sm100::decode::sparse::head64::run_flash_splitkv_mla_fp8_sparse_kernel<Config{MODEL_TYPE, EXTRA_MODEL_TYPE, ENABLE_SPLIT_KV}>(cur_params);
                }
            }));
        });
    }
};


class Decode_Sm100_Head128_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::V4_KVCACHE_FORMAT,
        DecodeFeatures::V41_KVCACHE_FORMAT,
        DecodeFeatures::V41_FP4_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )
    using SupportedKVFormats = KVFormatPairs<KVFormatPair<ModelType::V4>, KVFormatPair<ModelType::V41>,
                                             KVFormatPair<ModelType::V41, ModelType::V41_FP4>>;

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q / 2, 1),
            3,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        SparseAttnDecodeParams hotfixed_params = params;
        if (params.s_q == 1 && params.b > 1) {
            // For this kernel, we require `params.stride_q_b % params.stride_q_s_q == 0`, since we "squeeze" the batch size dimention and the sequence length q dimension together during tensormap creation
            hotfixed_params.stride_q_s_q = hotfixed_params.stride_q_b;
        }
        dispatch_kv_formats(SupportedKVFormats{}, params.model_type, params.extra_model_type, [&]<ModelType MODEL_TYPE, ModelType EXTRA_MODEL_TYPE>() {
            DISPATCH_BOOLEAN_FLAG(params.enable_split_kv, ENABLE_SPLIT_KV, ([&]() {
                sm100::prefill::sparse_fwd_for_small_topk::head128::run_sparse_fwd_for_small_topk_phase1_kernel<get_decode_fwd_mode<ENABLE_SPLIT_KV>(), 512, MODEL_TYPE, EXTRA_MODEL_TYPE>(hotfixed_params);
            }));
        });
    }
};
#endif


template<bool REMNANT, int EXTRA_FP4 = 0>
static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface_impl(
    const at::Tensor &q,   // [b, s_q, h_q, d_qk]
    const at::Tensor &kv,   // [num_blocks, page_block_size, h_k, d_qk]
    const at::Tensor &indices,    // [b, s_q, topk]
    const std::optional<at::Tensor> &topk_length,   // [b, s_q]
    const std::optional<at::Tensor> &attn_sink, // [h_q]
    std::optional<at::Tensor> &tile_scheduler_metadata,   // num_sm_parts x (DecodingSchedMetaSize/4)
    std::optional<at::Tensor> &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_v,
    float sm_scale,
    // Names the format of `kv` ("V32", "V32_NO_ROPE", "V4", "V41"). Only needed to disambiguate V3.2-no-RoPE from V4.1,
    // which have the same d_qk and bytes per token; when omitted the format is detected from the shape and 528 B per
    // token means V4.1, as in upstream FlashMLA.
    const std::optional<std::string> &kv_format,
    const std::optional<at::Tensor> &remnant_values,
    const std::optional<at::Tensor> &remnant_bitmaps,
    const std::optional<at::Tensor> &remnant_scales,
    const std::optional<at::Tensor> &remnant_raw_indices,
    const std::optional<at::Tensor> &remnant_freqs
#ifdef FLASHMLA_ENABLE_READY_WAIT_CYCLES
    , std::optional<at::Tensor> *ready_wait_cycles_out = nullptr
#endif
) {
    using bf16 = cutlass::bfloat16_t;

    // Check the architecture
    Arch arch = Arch();

    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);

    int b = q.size(0);
    int s_q = q.size(1);
    int h_q = q.size(2);
    int d_qk = q.size(3);
    int num_blocks = kv.size(0);
    int page_block_size = kv.size(1);
    int h_kv = kv.size(2);
    int topk = indices.size(2);

    bool have_topk_length = topk_length.has_value();
    bool have_extra_kcache = extra_kv.has_value();
    bool have_remnant = remnant_values.has_value();
    bool have_extra_topk_length = extra_topk_length.has_value();
    bool have_attn_sink = attn_sink.has_value();

    int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
    int remnant_num_pages = 0, remnant_page_size = 0;
    if constexpr (REMNANT) {
        TORCH_CHECK(d_qk == 512, "Remnant C4 decode requires MODEL1 head_size_k == 512");
        TORCH_CHECK(!have_extra_kcache, "Remnant C4 decode does not accept a dense extra_kv tensor");
        TORCH_CHECK(remnant_values.has_value() && remnant_bitmaps.has_value() && remnant_scales.has_value(), "Remnant C4 decode requires values, bitmaps, and scales");
        TORCH_CHECK(remnant_raw_indices.has_value() && remnant_freqs.has_value(), "Remnant C4 decode requires raw indices and RoPE frequencies");
        TORCH_CHECK(extra_indices.has_value() && extra_topk_length.has_value(), "Remnant C4 decode requires physical indices and topk lengths");
    } else {
        TORCH_CHECK(!have_remnant, "Remnant buffers require the dedicated Remnant decode entry point");
    }

    if (have_extra_kcache) {
        KU_CHECK_NDIM(extra_kv, 4);
        extra_num_blocks = extra_kv->size(0);
        extra_page_block_size = extra_kv->size(1);
    }
    if (extra_indices.has_value()) {
        extra_topk = extra_indices->size(-1);
    }
    if constexpr (REMNANT) {
        KU_CHECK_NDIM(remnant_values, 3);
        KU_CHECK_NDIM(remnant_bitmaps, 3);
        KU_CHECK_NDIM(remnant_scales, 3);
        KU_CHECK_NDIM(remnant_raw_indices, 3);
        KU_CHECK_NDIM(remnant_freqs, 3);
        remnant_num_pages = remnant_values->size(0);
        remnant_page_size = remnant_values->size(1);
        TORCH_CHECK(remnant_values->size(2) == 256, "Remnant values must have 256 codes per row");
        TORCH_CHECK(remnant_bitmaps->size(0) == remnant_num_pages && remnant_bitmaps->size(1) == remnant_page_size && remnant_bitmaps->size(2) == 8, "Remnant bitmap shape must be [pages, page_size, 8]");
        TORCH_CHECK(remnant_scales->size(0) == remnant_num_pages && remnant_scales->size(1) == remnant_page_size && remnant_scales->size(2) == 8, "Remnant scale shape must be [pages, page_size, 8]");
        TORCH_CHECK(remnant_freqs->size(2) == 2, "Remnant frequencies must have real/imag pairs");
    }

    // Split-KV only pays off when a request has enough work. The sm90 kernel always splits.
    bool enable_split_kv = arch.is_sm90a() || !(topk + extra_topk <= 640);

    // metadata sanity check
    TORCH_CHECK(b > 0);
    TORCH_CHECK(s_q > 0);
    TORCH_CHECK(h_q > 0);
    TORCH_CHECK(h_kv == 1, "Currently only MQA (i.e. h_kv == 1) is supported for sparse decoding");
    TORCH_CHECK(d_qk == 576 || d_qk == 512, "Only head_size_k == 576 or 512 is supported for sparse decoding");
    TORCH_CHECK(d_v == 512, "Only head_size_v == 512 is supported for sparse decoding");
    TORCH_CHECK(topk > 0);

    if (have_extra_kcache) {
        TORCH_CHECK(extra_indices.has_value(), "extra_indices_in_kvcache must be provided when extra_kcache is provided for sparse attention");
    } else if constexpr (!REMNANT) {
        TORCH_CHECK(!extra_indices.has_value(), "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
        TORCH_CHECK(!extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_k_cache is not provided");
    }

    // Check device
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);
    KU_CHECK_DEVICE(remnant_values);
    KU_CHECK_DEVICE(remnant_bitmaps);
    KU_CHECK_DEVICE(remnant_scales);
    KU_CHECK_DEVICE(remnant_raw_indices);
    KU_CHECK_DEVICE(remnant_freqs);

    // Check data type
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kUInt8, "key must have dtype fp8_e4m3fn, int8 or uint8");
    if (extra_kv.has_value()) {
        TORCH_CHECK(extra_kv->dtype() == torch::kFloat8_e4m3fn || extra_kv->dtype() == torch::kInt8 || extra_kv->dtype() == torch::kUInt8, "extra k cache must have dtype fp8_e4m3fn, int8 or uint8");
    }
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);
    KU_CHECK_DTYPE(remnant_values, torch::kUInt8);
    KU_CHECK_DTYPE(remnant_bitmaps, torch::kUInt64);
    KU_CHECK_DTYPE(remnant_scales, torch::kUInt8);
    KU_CHECK_DTYPE(remnant_raw_indices, torch::kInt32);
    KU_CHECK_DTYPE(remnant_freqs, torch::kFloat32);
    
    // Check layout
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);

    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);

    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);
    KU_CHECK_CONTIGUOUS(remnant_values);
    KU_CHECK_CONTIGUOUS(remnant_bitmaps);
    KU_CHECK_CONTIGUOUS(remnant_scales);
    KU_CHECK_LAST_DIM_CONTIGUOUS(remnant_raw_indices);
    KU_CHECK_CONTIGUOUS(remnant_freqs);
    
    // Check shape
    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    // The formats of `kv` and `extra_kv`
    ModelType model_type, extra_model_type;
    const std::optional<ModelType> kv_format_hint =
        kv_format.has_value() ? std::make_optional(parse_kv_cache_format(*kv_format)) : std::nullopt;
    if (d_qk == 576 && d_v == 512) {
        model_type = extra_model_type = ModelType::V32;
    } else if (d_qk == 512 && d_v == 512) {
        model_type = kv_format_hint.value_or(detect_kv_cache_format_for_headdim_512(kv.size(3), kv_format_hint));
        extra_model_type = EXTRA_FP4 != 0 ? ModelType::V41_FP4 : (have_extra_kcache ? detect_kv_cache_format_for_headdim_512(extra_kv->size(3), model_type) : model_type);
    } else {
        TORCH_CHECK(false, "Unsupported head sizes for is_fp8_kvcache == True");
    }
    if (kv_format_hint.has_value()) {
        TORCH_CHECK(model_type == *kv_format_hint, "kv_format says ", get_dynamic_enum_name(*kv_format_hint),
            " but q/kv have d_qk ", d_qk, " and ", kv.size(3), " bytes per token");
    }
    TORCH_CHECK(model_type != ModelType::V41_FP4, "The fp4 KV cache is only supported as extra_kv");
    TORCH_CHECK((EXTRA_FP4 != 0 && model_type == ModelType::V4 && have_extra_kcache) || is_valid_kv_format_pair(model_type, extra_model_type), "invalid kv format pair, ", get_dynamic_enum_name(model_type), " and ", get_dynamic_enum_name(extra_model_type));
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, kv_cache_bytes_per_token(model_type));
    KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, (EXTRA_FP4 == 2 ? 224 : kv_cache_bytes_per_token(extra_model_type)));
    TORCH_CHECK(kv.stride(1) == kv_cache_bytes_per_token(model_type), "The whole block must be contiguous when is_fp8_cache is True for kv cache");
    if (have_extra_kcache) {
        TORCH_CHECK(extra_kv->stride(1) == (EXTRA_FP4 == 2 ? 224 : kv_cache_bytes_per_token(extra_model_type)), "The whole block must be contiguous when is_fp8_cache is True for extra kv cache");
    }
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
    KU_CHECK_SHAPE(extra_topk_length, b);
    if constexpr (REMNANT) {
        KU_CHECK_SHAPE(remnant_raw_indices, b, s_q, extra_topk);
        TORCH_CHECK(remnant_values->stride(2) == 1 && remnant_bitmaps->stride(2) == 1 && remnant_scales->stride(2) == 1, "Remnant records must be row-contiguous");
    }

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts);
    at::Tensor lse = torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat));

    std::vector<DecodeFeatures> features;
    if (h_q == 64) {
        features.push_back(DecodeFeatures::HEAD_64);
    } else if (h_q == 128) {
        features.push_back(DecodeFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        features.push_back(DecodeFeatures::HEAD_DIM_576);
    } else if (d_qk == 512) {
        features.push_back(DecodeFeatures::HEAD_DIM_512);
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (have_attn_sink) {
        features.push_back(DecodeFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        features.push_back(DecodeFeatures::TOPK_LENGTH);
    }
    if (have_extra_kcache) {
        features.push_back(DecodeFeatures::EXTRA_KVCACHE);
    }
    if (have_extra_topk_length) {
        features.push_back(DecodeFeatures::EXTRA_TOPK_LENGTH);
    }
    for (ModelType mt : {model_type, extra_model_type}) {
        if (mt == ModelType::V32) {
            features.push_back(DecodeFeatures::V32_KVCACHE_FORMAT);
        } else if (mt == ModelType::V32_NO_ROPE) {
            features.push_back(DecodeFeatures::V32_NO_ROPE_KVCACHE_FORMAT);
        } else if (mt == ModelType::V4) {
            features.push_back(DecodeFeatures::V4_KVCACHE_FORMAT);
        } else if (mt == ModelType::V41) {
            features.push_back(DecodeFeatures::V41_KVCACHE_FORMAT);
        } else if (mt == ModelType::V41_FP4) {
            features.push_back(DecodeFeatures::V41_FP4_KVCACHE_FORMAT);
        } else {
            TORCH_CHECK(false, "Unsupported model type: ", (int)mt);
        }
    }

    DecodeImplBase* impl;
    if constexpr (EXTRA_FP4 != 0) {
        TORCH_CHECK(arch.is_sm90a(), "V4.1 FP4 Direct requires SM90a");
        TORCH_CHECK(model_type == ModelType::V4 && have_extra_kcache, "FP4 Direct requires V4 SWA and a global FP4 cache");
        TORCH_CHECK(topk % 64 == 0 && extra_topk % 64 == 0, "FP4 Direct index widths must be padded to 64");
        TORCH_CHECK(extra_page_block_size > 0 && extra_page_block_size % 64 == 0, "FP4 page size must be a positive multiple of 64");
        TORCH_CHECK(extra_kv->stride(0) >= extra_page_block_size*(EXTRA_FP4 == 2 ? 224 : 288) && extra_kv->stride(0) % 256 == 0, "FP4 page stride must contain its SoA payload/scales and align to 256 bytes");
        impl = new Decode_Sm90_FP4_Impl<EXTRA_FP4>();
    } else if constexpr (REMNANT) {
        TORCH_CHECK(arch.is_sm90a(), "Remnant direct decode currently requires SM90a");
        impl = new Decode_Sm90_Remnant_Impl();
    }
#ifdef FLASHMLA_ENABLE_SM100
    else if (arch.is_sm100f()) {
        if (h_q == 64) {
            impl = new Decode_Sm100_Head64_Impl();
        } else if (h_q == 128) {
            if (d_qk == 576 || model_type == ModelType::V32_NO_ROPE) {
                impl = new Decode_Sm100_Head64x2_Impl();
            } else if (d_qk == 512) {
                impl = new Decode_Sm100_Head128_Impl();
            } else {
                TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
            }
        } else {
            TORCH_CHECK(false, "Unsupported h_q: ", h_q);
        }
    }
#endif
    else if (arch.is_sm90a()) {
        impl = new Decode_Sm90_Impl();
    } else {
        TORCH_CHECK(false, "Unsupported architecture for sparse decode fwd");
    }

    DecodeImplMeta impl_meta = impl->get_meta(h_q, s_q);

#ifdef FLASHMLA_ENABLE_READY_WAIT_CYCLES
    if (ready_wait_cycles_out != nullptr) {
        TORCH_CHECK(!REMNANT, "ready-wait cycle capture is native-only");
        TORCH_CHECK(arch.is_sm90a() && d_qk == 512 && d_v == 512 && h_q == 128,
                    "ready-wait cycle capture requires native SM90 H128 D512");
        const int ready_wait_blocks = (topk + 63) / 64 + (extra_topk + 63) / 64;
        *ready_wait_cycles_out = torch::full(
            {impl_meta.num_sm_parts, s_q, h_q / 64, ready_wait_blocks, 3},
            -1, opts.dtype(at::kLong));
    }
#endif

    SparseAttnDecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
        num_blocks, page_block_size, topk,
        model_type, extra_model_type,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        (float*)lse.data_ptr(),
        (bf16*)out.data_ptr(),

        extra_num_blocks, extra_page_block_size, extra_topk,
        ku::get_optional_tensor_ptr<bf16>(extra_kv),
        ku::get_optional_tensor_ptr<int>(extra_indices),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)), int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),

        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
        extra_indices.has_value() ? int64_stride_to_int(extra_indices->stride(0)) : 0,
        extra_indices.has_value() ? int64_stride_to_int(extra_indices->stride(1)) : 0,
        at::cuda::getCurrentCUDAStream().stream(),

        enable_split_kv,
    };

    if constexpr (REMNANT) {
        TORCH_CHECK(model_type == ModelType::V4, "Packed Direct requires V4 Native SWA bytes");
        params.remnant_values = remnant_values->data_ptr<uint8_t>();
        params.remnant_bitmaps = remnant_bitmaps->data_ptr<uint64_t>();
        params.remnant_scales = remnant_scales->data_ptr<uint8_t>();
        params.remnant_raw_indices = remnant_raw_indices->data_ptr<int>();
        params.remnant_freqs = remnant_freqs->data_ptr<float>();
        params.remnant_num_pages = remnant_num_pages;
        params.remnant_page_size = remnant_page_size;
        params.stride_remnant_values_page = int64_stride_to_int(remnant_values->stride(0));
        params.stride_remnant_values_row = int64_stride_to_int(remnant_values->stride(1));
        params.stride_remnant_bitmaps_page = int64_stride_to_int(remnant_bitmaps->stride(0));
        params.stride_remnant_bitmaps_row = int64_stride_to_int(remnant_bitmaps->stride(1));
        params.stride_remnant_scales_page = int64_stride_to_int(remnant_scales->stride(0));
        params.stride_remnant_scales_row = int64_stride_to_int(remnant_scales->stride(1));
        params.stride_remnant_raw_indices_b = int64_stride_to_int(remnant_raw_indices->stride(0));
        params.stride_remnant_raw_indices_s_q = int64_stride_to_int(remnant_raw_indices->stride(1));
    }

    at::Tensor o_accum, lse_accum;
    if (enable_split_kv) {
        // Get MLA metadata if necessary
        if (!tile_scheduler_metadata.has_value()) {
            tile_scheduler_metadata = torch::empty({impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4}, opts.dtype(torch::kInt32));
            num_splits = torch::empty({b+1}, opts.dtype(torch::kInt32));
            KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
            KU_CHECK_CONTIGUOUS(num_splits);

            GetDecodeSchedMetaParams get_sched_meta_params = {
                b, s_q,
                impl_meta.block_size_topk,
                impl_meta.fixed_overhead_num_blocks,
                topk,
                extra_topk,
                ku::get_optional_tensor_ptr<int>(topk_length),
                ku::get_optional_tensor_ptr<int>(extra_topk_length),
                nullptr,
                (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
                num_splits->data_ptr<int>(),
                impl_meta.num_sm_parts,
                at::cuda::getCurrentCUDAStream().stream()
            };
            smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
        }
        KU_CHECK_DEVICE(tile_scheduler_metadata);
        KU_CHECK_DEVICE(num_splits);
        KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
        KU_CHECK_DTYPE(num_splits, torch::kInt32);
        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
        KU_CHECK_CONTIGUOUS(num_splits);
        KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4);
        KU_CHECK_SHAPE(num_splits, b+1);
        // Stick the metadata pointers to `params`
        params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
        params.num_splits_ptr = num_splits->data_ptr<int>();
        params.num_sm_parts = impl_meta.num_sm_parts;
        // Allocate intermediate buffers for split-KV
        const int total_num_splits = b + params.num_sm_parts;
        lse_accum = torch::empty({total_num_splits, s_q, h_q}, opts.dtype(at::kFloat));
        o_accum = torch::empty({total_num_splits, s_q, h_q, d_v}, opts.dtype(at::kFloat));
        KU_CHECK_CONTIGUOUS(lse_accum);
        KU_CHECK_CONTIGUOUS(o_accum);
        params.lse_accum = lse_accum.data_ptr<float>();
        params.o_accum = o_accum.data_ptr<float>();
        params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
        params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
        params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
        params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
        params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));
    }

    impl->run(params, features);
    if (enable_split_kv) {
        CombineParams combine_params = {
            b, s_q, h_q, d_v,

            params.lse,
            params.out,
            params.stride_lse_b, params.stride_lse_s_q,
            params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q,

            params.lse_accum,
            params.o_accum,
            params.stride_lse_accum_split, params.stride_lse_accum_s_q,
            params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,

            params.tile_scheduler_metadata_ptr,
            params.num_splits_ptr,
            params.num_sm_parts,

            ku::get_optional_tensor_ptr<float>(attn_sink),
            at::cuda::getCurrentCUDAStream().stream()
        };
        if constexpr (REMNANT) {
            const int tile = impl_meta.block_size_topk;
            const int max_request_tiles = (std::max(topk, 1) + tile - 1) / tile
                + (extra_topk + tile - 1) / tile;
            combine_params.num_sm_parts = std::min(params.num_sm_parts, max_request_tiles);
        }
        smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);
    }

    delete impl;

    return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface(
    const at::Tensor &q, const at::Tensor &kv, const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    std::optional<at::Tensor> &tile_scheduler_metadata,
    std::optional<at::Tensor> &num_splits,
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_v, float sm_scale, const std::optional<std::string> &kv_format
) {
    return sparse_attn_decode_interface_impl<false>(
        q, kv, indices, topk_length, attn_sink, tile_scheduler_metadata,
        num_splits, extra_kv, extra_indices, extra_topk_length, d_v, sm_scale,
        kv_format, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_remnant_interface(
    const at::Tensor &q, const at::Tensor &kv, const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    const at::Tensor &remnant_values, const at::Tensor &remnant_bitmaps,
    const at::Tensor &remnant_scales, const at::Tensor &remnant_raw_indices,
    const at::Tensor &remnant_freqs, int d_v, float sm_scale
) {
    return sparse_attn_decode_interface_impl<true>(
        q, kv, indices, topk_length, attn_sink, tile_scheduler_metadata,
        num_splits, std::nullopt, extra_indices, extra_topk_length, d_v, sm_scale,
        std::string("V4"), remnant_values, remnant_bitmaps, remnant_scales,
        remnant_raw_indices, remnant_freqs);
}

// Dedicated opt-in API: never changes stock format inference or FP8 Remnant.
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_fp4_interface(
    const at::Tensor &q, const at::Tensor &kv, const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    const at::Tensor &extra_kv, const at::Tensor &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length, bool packed, float sm_scale
) {
    std::optional<at::Tensor> metadata, splits;
    if (packed) return sparse_attn_decode_interface_impl<false, 2>(
        q, kv, indices, topk_length, attn_sink, metadata, splits,
        extra_kv, extra_indices, extra_topk_length, 512, sm_scale,
        std::string("V4"), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    return sparse_attn_decode_interface_impl<false, 1>(
        q, kv, indices, topk_length, attn_sink, metadata, splits,
        extra_kv, extra_indices, extra_topk_length, 512, sm_scale,
        std::string("V4"), std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

#ifndef FLASH_MLA_LIBTORCH_ONLY
void register_sparse_decode(pybind11::module_& m) {
    m.def("fp4_sparse_decode_fwd", &sparse_attn_decode_fp4_interface);
    m.def("remnant_sparse_decode_fwd", &sparse_attn_decode_remnant_interface);
    m.def("sparse_decode_fwd",
        &sparse_attn_decode_interface,
        "Run Sparse Attention Decode Forward",
        pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("indices"),
        pybind11::arg("topk_length"), pybind11::arg("attn_sink"),
        pybind11::arg("tile_scheduler_metadata"), pybind11::arg("num_splits"),
        pybind11::arg("extra_kv"), pybind11::arg("extra_indices"), pybind11::arg("extra_topk_length"),
        pybind11::arg("d_v"), pybind11::arg("sm_scale"),
        pybind11::arg("kv_format") = pybind11::none());
}
#endif
