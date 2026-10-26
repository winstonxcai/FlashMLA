// Author: Winston Cai. Separate opt-in FP4 prefill instantiations.
#pragma once

#include "kernels/params.h"

namespace sm90::prefill::sparse_fwd {

template <int D_QK, bool HAVE_TOPK_LENGTH, bool FP4_PREFILL = false>
void run_fwd_phase1_kernel(const SparseAttnFwdParams &params);
}
