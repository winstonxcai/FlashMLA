"""Direct Remnant decode parity and CUDA graph coverage."""

import importlib.util

import pytest
import torch

try:
    import flash_mla
except ImportError:
    flash_mla = None

from remnant_fixture import make_case, materialize_native


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
@pytest.mark.parametrize("batch", [8, 16])
@pytest.mark.parametrize("topk_length", [512, 317])
def test_direct_decode_matches_native_adapter(num_heads: int, batch: int, topk_length: int):
    case = make_case(num_heads, topk_length, batch=batch)
    # The public [page, token, 1, 584] view has a synthetic row width; the
    # decoder addresses the physical 576-byte data rows plus the page scale
    # tail. Inspect the backing page bytes for the native-code check.
    native_code_bytes = case.native_storage[:, : 64 * 576].view(-1, 64, 576)[..., :448]
    packed_code_bytes = case.packed_buffers[0]
    # E4M3FN reserves both sign variants of exponent=15, mantissa=7 for NaN.
    native_nan_codes = ((native_code_bytes & 0x7F) == 0x7F).sum().item()
    packed_nan_codes = ((packed_code_bytes & 0x7F) == 0x7F).sum().item()
    assert native_nan_codes == 0, f"Native cache contains {native_nan_codes} NaN FP8 codes"
    assert packed_nan_codes == 0, f"Packed cache contains {packed_nan_codes} NaN FP8 codes"
    native_out, native_lse = _run_native(case)
    direct_out, direct_lse = _run_remnant(case)
    assert torch.isfinite(native_out).all(), "Native reference output is non-finite"
    assert torch.isfinite(native_lse).all(), "Native reference LSE is non-finite"
    assert torch.isfinite(direct_out).all(), "Direct Remnant output is non-finite"
    assert torch.isfinite(direct_lse).all(), "Direct Remnant LSE is non-finite"
    torch.testing.assert_close(direct_out, native_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(direct_lse, native_lse, atol=2.0e-2, rtol=2.0e-2)


@pytest.mark.parametrize("num_heads", [64, 128])
def test_direct_decode_mixed_lengths_and_poisoned_padding(num_heads: int):
    case = make_case(num_heads, 512, batch=8)
    lengths = torch.tensor([0, 1, 63, 64, 65, 317, 511, 512], device="cuda", dtype=torch.int32)
    case.extra_length.copy_(lengths)
    # Native was materialized before poisoning. Both paths must ignore every
    # selection at or beyond the per-request length.
    for request, length in enumerate(lengths.tolist()):
        case.packed_indices[request, :, length:] = torch.iinfo(torch.int32).max
        case.raw_indices[request, :, length:] = -1
    native_out, native_lse = _run_native(case)
    direct_out, direct_lse = _run_remnant(case)
    torch.testing.assert_close(direct_out, native_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(direct_lse, native_lse, atol=2.0e-2, rtol=2.0e-2)


@pytest.mark.parametrize("num_heads", [64, 128])
@pytest.mark.parametrize("batch", [8, 16])
def test_direct_decode_cuda_graph_replay(num_heads: int, batch: int):
    case = make_case(num_heads, 512, batch=batch)
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
    native_first = _run_native(case)
    torch.testing.assert_close(first[0], native_first[0], atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(first[1], native_first[1], atol=2.0e-2, rtol=2.0e-2)

    # Inputs must remain live after capture.  This catches a graph that only
    # proves replay of an unchanged buffer rather than replay correctness.
    case.q.normal_()
    case.packed_buffers[2].add_(1)
    case.packed_indices.copy_(case.packed_indices.roll(1, dims=-1))
    case.raw_indices.copy_(case.raw_indices.roll(1, dims=-1))
    case.extra_length.sub_(13)
    materialize_native(case)
    graph.replay()
    torch.cuda.synchronize()
    second = tuple(value.clone() for value in captured)
    native_second = _run_native(case)
    torch.testing.assert_close(second[0], native_second[0], atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(second[1], native_second[1], atol=2.0e-2, rtol=2.0e-2)
    assert not torch.equal(first[0], second[0])
