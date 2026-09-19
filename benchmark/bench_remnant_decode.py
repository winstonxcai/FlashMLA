"""Model-free steady-state Native, adapter, and direct Remnant decode timing."""

from __future__ import annotations

import argparse
import random
import statistics
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


def _capture(fn, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(max(3, warmup)):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _measure_graph(graph: torch.cuda.CUDAGraph, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _measure_eager(fn, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95 + 0.5))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="8,16")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--max-regression-percent", type=float, default=2.0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark requires an H100")
    if args.repeats <= 0 or args.rounds <= 0 or args.warmup < 0:
        raise ValueError("repeats/rounds must be positive and warmup nonnegative")

    torch.manual_seed(20260919)
    print("heads,batch,topk,native_decode_ms,direct_decode_ms,adapter_total_ms,median_regression_pct,p95_regression_pct,status")
    misses = []
    for heads in (64, 128):
        for batch in (int(value) for value in args.batches.split(",")):
            case = make_case(heads, 512, batch=batch, unique_selections=True)
            native_meta = flash_mla.get_mla_metadata()[0]
            adapter_meta = flash_mla.get_mla_metadata()[0]
            direct_meta = flash_mla.get_mla_metadata()[0]
            native = lambda: _native(case, native_meta)
            adapter = lambda: _adapter(case, adapter_meta)
            direct = lambda: _direct(case, direct_meta)
            native_graph = _capture(native, args.warmup)
            direct_graph = _capture(direct, args.warmup)
            for _ in range(args.warmup):
                adapter()
            torch.cuda.synchronize()

            samples = {"native": [], "direct": [], "adapter": []}
            rng = random.Random(heads * 1000 + batch)
            for _ in range(args.rounds):
                order = ["native", "direct", "adapter"]
                rng.shuffle(order)
                for name in order:
                    if name == "native":
                        samples[name].append(_measure_graph(native_graph, args.repeats))
                    elif name == "direct":
                        samples[name].append(_measure_graph(direct_graph, args.repeats))
                    else:
                        samples[name].append(_measure_eager(adapter, args.repeats))
            ratios = [100.0 * (d / n - 1.0) for n, d in zip(samples["native"], samples["direct"])]
            median_delta = statistics.median(ratios)
            upper = _p95(ratios)
            status = "PASS" if upper <= args.max_regression_percent else "MISS"
            if status == "MISS":
                misses.append((heads, batch, median_delta, upper))
            print(
                f"{heads},{batch},512,{statistics.median(samples['native']):.5f},"
                f"{statistics.median(samples['direct']):.5f},{statistics.median(samples['adapter']):.5f},"
                f"{median_delta:.3f},{upper:.3f},{status}"
            )
    if misses:
        details = ", ".join(
            f"H{h}/B{b}=median {m:.2f}%, p95 {p:.2f}%" for h, b, m, p in misses
        )
        raise RuntimeError(f"direct decode exceeds the {args.max_regression_percent:.2f}% p95 target: {details}")


if __name__ == "__main__":
    main()
