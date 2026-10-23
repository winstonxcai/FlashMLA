#define E009_H64_REGISTER_PREFIX 1
#define E009_H64_BITMAP_TRANSPOSE 1
#define E009_H64_BITCAST_SCALE 1
#define E009_H64_BRANCHLESS_WINDOW 1
// Authored by Winston Cai.

#include "../splitkv_mla.cuh"

namespace sm90::decode::sparse {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V4, 64, true>(const SparseAttnDecodeParams &params);

}
