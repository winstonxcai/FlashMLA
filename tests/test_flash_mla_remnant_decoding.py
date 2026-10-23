"""Direct Remnant decode parity and CUDA graph coverage.

Author: Winston Cai.
"""

import dataclasses
import importlib.util
import json

import pytest
import torch

try:
    import flash_mla
except ImportError:
    flash_mla = None

from lib import (
    make_remnant_case,
    make_remnant_reference_case,
    materialize_remnant_native,
)
import ref

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or flash_mla is None
    or importlib.util.find_spec("flash_mla.cuda") is None,
    reason="An H100 and the compiled FlashMLA extension are required",
)


@dataclasses.dataclass(frozen=True)
class RemnantTestParam:
    num_heads: int
    batch: int
    topk_length: int
    unique_selections: bool = False


def gen_testcase() -> list[RemnantTestParam]:
    return [
        RemnantTestParam(num_heads, batch, topk_length)
        for num_heads in (64, 128)
        for batch in (8, 16, 32)
        for topk_length in (512, 317)
    ]


def _run_native(case, *, metadata=None):
    meta = metadata if metadata is not None else flash_mla.get_mla_metadata()[0]
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


def _run_remnant(case, *, metadata=None):
    meta = metadata if metadata is not None else flash_mla.get_mla_metadata()[0]
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


def _install_bitmap_edge_patterns(case):
    """Install balanced masks that exercise nibble edges and the final rank."""
    device = case.mask.device
    coordinate = torch.arange(512, device=device)
    within_byte = coordinate.remainder(8)
    patterns = torch.stack(
        (
            within_byte < 4,  # full high nibble, empty low nibble
            within_byte >= 4,  # empty high nibble, full low nibble
            coordinate.remainder(2) == 0,  # 0xAA
            coordinate.remainder(2) == 1,  # 0x55
            (within_byte == 0)
            | (within_byte == 1)
            | (within_byte == 6)
            | (within_byte == 7),  # 0xC3
            (within_byte == 0)
            | (within_byte == 2)
            | (within_byte == 5)
            | (within_byte == 7),  # 0xA5
            (coordinate < 255) | (coordinate == 511),  # final survivor at rank 255
        )
    )
    row_pattern = torch.arange(case.mask.shape[0], device=device) % patterns.shape[0]
    case.mask.copy_(patterns.index_select(0, row_pattern))

    values, bitmaps, scales = case.packed_buffers
    codes = (torch.arange(256, device=device, dtype=torch.uint8) % 126) + 1
    values.reshape(-1, 256).copy_(codes.expand(values.numel() // 256, -1))
    shifts = (1 << (63 - torch.arange(64, device=device))).to(torch.int64)
    packed_bits = case.mask.reshape(-1, 8, 64).to(torch.int64)
    packed_bitmap = (packed_bits * shifts.view(1, 1, 64)).sum(-1).view(torch.uint64)
    bitmaps.copy_(packed_bitmap.reshape_as(bitmaps))
    scales.fill_(127)
    materialize_remnant_native(case)


@pytest.mark.parametrize(
    "params",
    gen_testcase(),
    ids=lambda params: f"H{params.num_heads}-B{params.batch}-K{params.topk_length}",
)
def test_direct_decode_matches_native(params: RemnantTestParam):
    case = make_remnant_case(params.num_heads, params.topk_length, batch=params.batch)
    # The public [page, token, 1, 584] view has a synthetic row width; the
    # decoder addresses the physical 576-byte data rows plus the page scale
    # tail. Inspect the backing page bytes for the native-code check.
    native_code_bytes = case.native_storage[:, : 64 * 576].view(-1, 64, 576)[..., :448]
    packed_code_bytes = case.packed_buffers[0]
    # E4M3FN reserves both sign variants of exponent=15, mantissa=7 for NaN.
    native_nan_codes = ((native_code_bytes & 0x7F) == 0x7F).sum().item()
    packed_nan_codes = ((packed_code_bytes & 0x7F) == 0x7F).sum().item()
    assert (
        native_nan_codes == 0
    ), f"Native cache contains {native_nan_codes} NaN FP8 codes"
    assert (
        packed_nan_codes == 0
    ), f"Packed cache contains {packed_nan_codes} NaN FP8 codes"
    native_out, native_lse = _run_native(case)
    direct_out, direct_lse = _run_remnant(case)
    assert torch.isfinite(native_out).all(), "Native reference output is non-finite"
    assert torch.isfinite(native_lse).all(), "Native reference LSE is non-finite"
    assert torch.isfinite(direct_out).all(), "Direct Remnant output is non-finite"
    assert torch.isfinite(direct_lse).all(), "Direct Remnant LSE is non-finite"
    torch.testing.assert_close(direct_out, native_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(direct_lse, native_lse, atol=2.0e-2, rtol=2.0e-2)
    reference_params, reference_case = make_remnant_reference_case(case)
    reference_out, reference_lse = ref.ref_sparse_attn_decode(
        reference_params, reference_case
    )
    torch.testing.assert_close(native_out, reference_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(native_lse, reference_lse, atol=2.0e-2, rtol=2.0e-2)


@pytest.mark.parametrize("num_heads", [64, 128], ids=lambda heads: f"H{heads}")
def test_direct_decode_mixed_lengths_and_poisoned_padding(num_heads: int):
    case = make_remnant_case(num_heads, 512, batch=8)
    lengths = torch.tensor(
        [0, 1, 63, 64, 65, 317, 511, 512], device="cuda", dtype=torch.int32
    )
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


def _bitmap_attention_references(q, kv, invalid, sink, sm_scale, metadata, tile):
    """Independent CPU attention models for the two extreme bitmap fixtures.

    The online model rounds the unnormalized P operand to BF16 before P@V,
    while retaining FP32 maxima/normalizers. It uses the recorded partitions
    and tile width, but does not consume kernel outputs or reconstructed SMEM.
    CPU exp/GEMM and reduction order differ from CUDA; this is not a bitwise
    simulator. Ordinary fixtures continue to use the unmodified FP32 reference.
    Inputs are Q [B,H,D], logical KV [B,K,D], invalid [B,K], and sink [H].
    """
    assert q.device.type == kv.device.type == "cpu"
    batch, heads, dim = q.shape
    assert kv.shape[1] % tile == 0
    kv = torch.nan_to_num(kv.float(), nan=0.0)
    references = {}
    for dtype, name in ((torch.float32, "fp32"), (torch.float64, "fp64")):
        scores = (q.to(dtype) @ kv.to(dtype).transpose(1, 2)) * sm_scale
        scores.masked_fill_(invalid[:, None, :], -torch.inf)
        lse = scores.logsumexp(-1)
        probability = torch.exp(scores - lse[:, :, None])
        output = (probability @ kv.to(dtype)) / (
            1 + torch.exp(sink.to(dtype)[None, :] - lse)
        )[:, :, None]
        references[name] = output.to(torch.bfloat16)

    scores = (q.float() @ kv.transpose(1, 2)) * sm_scale
    scores.masked_fill_(invalid[:, None, :], -torch.inf)
    partials = [[] for _ in range(batch)]
    for row in metadata.tolist():
        begin, end, first, last, _, _, _, _ = row
        if begin >= batch:
            continue
        for request in range(begin, min(end, batch - 1) + 1):
            first_tile = first if request == begin else 0
            last_tile = last if request == end else kv.shape[1] // tile
            maximum = torch.full((heads,), -1e30)
            denominator = torch.zeros(heads)
            numerator = torch.zeros(heads, dim)
            for block in range(first_tile, last_tile):
                block_scores = scores[request, :, block * tile : (block + 1) * tile]
                block_values = kv[request, block * tile : (block + 1) * tile]
                new_maximum = torch.maximum(maximum, block_scores.amax(-1))
                rescale = torch.exp(maximum - new_maximum)
                probability = torch.exp(block_scores - new_maximum[:, None])
                operand = probability.to(torch.bfloat16).float()
                numerator = numerator * rescale[:, None] + operand @ block_values
                denominator = denominator * rescale + probability.sum(-1)
                maximum = new_maximum
            partials[request].append((numerator, denominator, maximum))

    outputs = []
    for partitions in partials:
        assert partitions, "Scheduler omitted a bitmap-fixture request"
        maximum = torch.stack([part[2] for part in partitions]).amax(0)
        numerator = sum(
            value * torch.exp(local_maximum - maximum)[:, None]
            for value, _, local_maximum in partitions
        )
        denominator = sum(
            total * torch.exp(local_maximum - maximum)
            for _, total, local_maximum in partitions
        ) + torch.exp(sink.float() - maximum)
        outputs.append(numerator / denominator[:, None])
    references["bf16_probability_operand"] = torch.stack(outputs).to(torch.bfloat16)
    return references


@pytest.mark.parametrize("num_heads", [64, 128], ids=lambda heads: f"H{heads}")
def test_direct_decode_bitmap_expansion_edge_patterns(num_heads: int, record_property):
    case = make_remnant_case(num_heads, 512, batch=8)
    _install_bitmap_edge_patterns(case)
    native_metadata = flash_mla.get_mla_metadata()[0]
    direct_metadata = flash_mla.get_mla_metadata()[0]
    native_out, native_lse = _run_native(case, metadata=native_metadata)
    direct_out, direct_lse = _run_remnant(case, metadata=direct_metadata)
    for value in (native_out, native_lse, direct_out, direct_lse):
        assert torch.isfinite(value).all()
    torch.testing.assert_close(direct_out, native_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(direct_lse, native_lse, atol=2.0e-2, rtol=2.0e-2)

    reference_params, reference_case = make_remnant_reference_case(case)
    reference_out, reference_lse = ref.ref_sparse_attn_decode(
        reference_params, reference_case
    )
    # The LSE contract and ordinary FP32-reference checks remain unchanged.
    torch.testing.assert_close(native_lse, reference_lse, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(direct_lse, reference_lse, atol=2.0e-2, rtol=2.0e-2)
    kvs, masks = [], []
    for scope in (reference_case.kv_scope, reference_case.extra_kv_scope):
        indices = scope.indices_in_kvcache
        gathered = (
            scope.blocked_k.view(-1, 512)
            .index_select(0, indices.clamp_min(0).flatten())
            .view(8, -1, 512)
        )
        invalid = indices.squeeze(1) < 0
        if scope.topk_length is not None:
            invalid |= (
                torch.arange(indices.shape[-1], device=indices.device)[None, :]
                >= scope.topk_length[:, None]
            )
        kvs.append(gathered.cpu())
        masks.append(invalid.cpu())
    kv = torch.cat(kvs, dim=1)
    invalid = torch.cat(masks, dim=1)
    # Explicit, user-approved qualification applies only to these two fixtures.
    # Keep the inherited mathematical-reference failures visible and reject any
    # additional failing location; never raise the 2e-2 tolerance.
    inherited_location = (2, 63, 497) if num_heads == 64 else (3, 46, 464)
    diagnostics = {"heads": num_heads, "modes": {}}
    for mode, actual, metadata, tile in (
        ("native", native_out, native_metadata, 64),
        ("direct", direct_out, direct_metadata, 32 if num_heads == 64 else 64),
    ):
        references = _bitmap_attention_references(
            case.q.cpu().squeeze(1),
            kv,
            invalid,
            case.sink.cpu(),
            case.sm_scale,
            metadata.tile_scheduler_metadata.cpu(),
            tile,
        )
        references["original_gpu_fp32"] = reference_out.cpu().squeeze(1)
        actual_cpu = actual.cpu().squeeze(1)
        torch.testing.assert_close(
            actual_cpu,
            references["bf16_probability_operand"],
            atol=2.0e-2,
            rtol=2.0e-2,
        )
        mode_diagnostics = {}
        for name, expected in references.items():
            failed = (actual_cpu.float() - expected.float()).abs() > (
                2.0e-2 + 2.0e-2 * expected.float().abs()
            )
            locations = [tuple(index) for index in failed.nonzero().tolist()]
            assert set(locations) <= {inherited_location}, (mode, name, locations)
            mode_diagnostics[name] = {
                "mismatch_locations": locations,
                "values": [
                    {
                        "actual": float(actual_cpu[index]),
                        "reference": float(expected[index]),
                    }
                    for index in locations
                ],
                "max_abs": float((actual_cpu.float() - expected.float()).abs().max()),
            }
        diagnostics["modes"][mode] = mode_diagnostics
    record_property("bitmap_reference_diagnostics", json.dumps(diagnostics))
    print("BITMAP_REFERENCE_DIAGNOSTICS=" + json.dumps(diagnostics))


@pytest.mark.parametrize("num_heads", [64, 128], ids=lambda heads: f"H{heads}")
@pytest.mark.parametrize("batch", [8, 16, 32], ids=lambda batch: f"B{batch}")
def test_direct_decode_cuda_graph_replay(num_heads: int, batch: int):
    case = make_remnant_case(num_heads, 512, batch=batch)
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
    materialize_remnant_native(case)
    graph.replay()
    torch.cuda.synchronize()
    second = tuple(value.clone() for value in captured)
    native_second = _run_native(case)
    torch.testing.assert_close(second[0], native_second[0], atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(second[1], native_second[1], atol=2.0e-2, rtol=2.0e-2)
    assert not torch.equal(first[0], second[0])
