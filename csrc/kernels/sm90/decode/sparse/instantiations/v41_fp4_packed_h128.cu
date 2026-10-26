// Author: Winston Cai. SM90 post-RoPE FP4 Direct.
#include "../splitkv_mla.cuh"
namespace sm90::decode::sparse {
template void
run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V4, 128, false, 2>(
    const SparseAttnDecodeParams &params);
}
