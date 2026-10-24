// Author: Winston Cai. Model-free FP4 prefill specialization.
#include "../phase1.h"
#include "../phase1.cuh"

namespace sm90::prefill::sparse_fwd {
template void run_fwd_phase1_kernel<512, true, true>(const SparseAttnFwdParams& params);
}
