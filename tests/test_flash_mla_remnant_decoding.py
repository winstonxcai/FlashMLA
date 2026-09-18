"""Direct Remnant decode parity and CUDA graph coverage."""

import importlib.util

import pytest
import torch

try:
    import flash_mla
except ImportError:
    flash_mla = None

from remnant_fixture import make_case


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or flash_mla is None
    or importlib.util.find_spec("flash_mla.cuda") is None,
    reason="An H100 and the compiled FlashMLA extension are required",
)


def _run_native(case):
    meta = flash_mla.get_mla_metadata()[0]
    return flash_mla.flash_mla_with_kvcache(
        case.q,
        case.swa_cache,
        None,
        None,
        512,
        meta,
        None,
        case.sm_scale,
        False,
        True,
        case.swa_indices,
        case.sink,
        case.native_cache,
        case.native_indices,
        case.swa_length,
        case.extra_length,
    )


def _run_remnant(case):
    meta = flash_mla.get_mla_metadata()[0]
    return flash_mla.flash_mla_with_remnant_kvcache(
        case.q,
        case.swa_cache,
        case.swa_indices,
        case.packed_buffers,
        case.raw_indices,
        case.freqs,
        meta,
        topk_length=case.swa_length,
        extra_topk_length=case.extra_length,
        attn_sink=case.sink,
        sm_scale=case.sm_scale,
        extra_indices_in_kvcache=case.packed_indices,
    )


@pytest.mark.parametrize("num_heads", [64, 128])
@pytest.mark.parametrize("topk_length", [512, 317])
def test_direct_decode_matches_native_adapter(num_heads: int, topk_length: int):
    case = make_case(num_heads, topk_length)
    native_out, native_lse = _run_native(case)
    direct_out, direct_lse = _run_remnant(case)
    torch.testing.assert_close(direct_out, native_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(direct_lse, native_lse, atol=2.0e-2, rtol=2.0e-2)


def test_direct_decode_cuda_graph_replay():
    case = make_case(64, 512)
    meta = flash_mla.get_mla_metadata()[0]
    for _ in range(3):
        _run_remnant(case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = flash_mla.flash_mla_with_remnant_kvcache(
            case.q,
            case.swa_cache,
            case.swa_indices,
            case.packed_buffers,
            case.raw_indices,
            case.freqs,
            meta,
            topk_length=case.swa_length,
            extra_topk_length=case.extra_length,
            attn_sink=case.sink,
            sm_scale=case.sm_scale,
            extra_indices_in_kvcache=case.packed_indices,
        )
    graph.replay()
    torch.cuda.synchronize()
    first = tuple(value.clone() for value in captured)
    graph.replay()
    torch.cuda.synchronize()
    second = tuple(value.clone() for value in captured)
    torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
    torch.testing.assert_close(first[1], second[1], atol=0, rtol=0)
