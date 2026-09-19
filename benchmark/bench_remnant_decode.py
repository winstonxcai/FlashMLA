"""Model-free Native versus adapter versus direct Remnant decode benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import flash_mla

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from remnant_fixture import make_case, materialize_native  # noqa: E402


def _native(case, meta):
    return flash_mla.flash_mla_with_kvcache(
        case.q, case.swa_cache, None, None, 512, meta, None,
        case.sm_scale, False, True, case.swa_indices, case.sink,
        case.native_cache, case.native_indices, case.swa_length, case.extra_length,
    )


def _adapter(case, meta):
    materialize_native(case)
    return _native(case, meta)


def _direct(case, meta):
    return flash_mla.flash_mla_with_remnant_kvcache(
        case.q, case.swa_cache, case.swa_indices, case.packed_buffers,
        case.raw_indices, case.freqs, meta, topk_length=case.swa_length,
        extra_topk_length=case.extra_length, attn_sink=case.sink,
        sm_scale=case.sm_scale, extra_indices_in_kvcache=case.packed_indices,
    )


def _measure(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _kernel_time(fn) -> float:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    names = ("flash_fwd_splitkv_mla_fp8_sparse_kernel", "flash_fwd_mla_combine_kernel")
    return sum(
        event.device_time_total
        for event in prof.key_averages()
        if any(name in event.key for name in names)
    ) / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="8,16")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-regression-percent", type=float, default=2.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an H100")
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("Direct Remnant decode currently targets SM90")

    print("heads,batch,topk,native_total_ms,adapter_total_ms,direct_total_ms,"
          "native_kernel_ms,adapter_kernel_ms,direct_kernel_ms,direct_vs_native_pct,status")
    misses = []
    for heads in (64, 128):
        for batch in (int(value) for value in args.batches.split(",")):
            case = make_case(heads, 512, batch=batch)
            meta = flash_mla.get_mla_metadata()[0]
            native_total = _measure(lambda: _native(case, meta), args.warmup, args.repeats)
            adapter_total = _measure(lambda: _adapter(case, meta), args.warmup, args.repeats)
            direct_total = _measure(lambda: _direct(case, meta), args.warmup, args.repeats)
            native_kernel = _kernel_time(lambda: _native(case, meta))
            adapter_kernel = _kernel_time(lambda: _adapter(case, meta))
            direct_kernel = _kernel_time(lambda: _direct(case, meta))
            delta = 100.0 * (direct_total / native_total - 1.0)
            status = "PASS" if delta <= args.max_regression_percent else "MISS"
            if status == "MISS":
                misses.append((heads, batch, delta))
            print(
                f"{heads},{batch},512,"
                f"{native_total:.4f},{adapter_total:.4f},{direct_total:.4f},"
                f"{native_kernel:.4f},{adapter_kernel:.4f},{direct_kernel:.4f},{delta:.3f},{status}"
            )
    if misses:
        details = ", ".join(f"H{heads}/B{batch}={delta:.2f}%" for heads, batch, delta in misses)
        raise RuntimeError(
            f"direct decode exceeds the {args.max_regression_percent:.2f}% target: {details}"
        )


if __name__ == "__main__":
    main()
