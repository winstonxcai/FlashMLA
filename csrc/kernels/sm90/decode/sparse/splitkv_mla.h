// Authored by Winston Cai.

#pragma once

#include "kernels/kv_cache_format.h"
#include "kernels/params.h"

namespace sm90::decode::sparse {

template <ModelType MODEL_TYPE, int NUM_HEADS, bool REMNANT = false,
          FP4DecodeMode FP4_MODE = FP4DecodeMode::Stock>
void run_flash_splitkv_mla_fp8_sparse_kernel(
    const SparseAttnDecodeParams &params);
}
