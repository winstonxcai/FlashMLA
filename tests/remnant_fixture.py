"""Small CUDA fixtures for direct Remnant FlashMLA tests and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
KEEP_K = 256
PAGE_SIZE = 64


@dataclass
class RemnantCase:
    q: torch.Tensor
    swa_cache: torch.Tensor
    swa_indices: torch.Tensor
    swa_length: torch.Tensor
    packed_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    packed_indices: torch.Tensor
    raw_indices: torch.Tensor
    extra_length: torch.Tensor
    freqs: torch.Tensor
    native_cache: torch.Tensor
    native_indices: torch.Tensor
    native_storage: torch.Tensor
    mask: torch.Tensor
    sink: torch.Tensor
    sm_scale: float


def _bitmap(mask: torch.Tensor) -> torch.Tensor:
    shifts = (1 << (63 - torch.arange(64, device=mask.device))).to(torch.int64)
    bits = mask.reshape(mask.shape[0], 8, 64).to(torch.int64)
    return (bits * shifts.view(1, 1, 64)).sum(-1).view(torch.uint64)


def _pack(latent: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    mask = torch.zeros_like(latent, dtype=torch.bool)
    keep = latent.abs().topk(KEEP_K, dim=-1, largest=True).indices
    mask.scatter_(1, keep, True)
    masked = latent.masked_fill(~mask, 0).to(torch.bfloat16).float()
    tiles = masked.reshape(-1, 8, 64)
    max_abs = tiles.abs().amax(-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(max_abs / 448.0)).to(torch.int32)
    scale = torch.exp2(exponent.float())
    quantized = (tiles / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    quantized_bytes = quantized.view(torch.uint8).reshape(-1, HEAD_DIM)
    # E4M3FN reserves the two exponent=15/mantissa=7 encodings as NaN.  Some
    # CUDA/PyTorch conversion paths round the finite endpoint to those byte
    # patterns, so explicitly map them to the signed finite maximum.
    nan_codes = (quantized_bytes & 0x7F) == 0x7F
    quantized_bytes = torch.where(
        nan_codes,
        (quantized_bytes & 0x80) | 0x7E,
        quantized_bytes,
    )
    columns = mask.nonzero(as_tuple=False)[:, 1].reshape(-1, KEEP_K)
    values = quantized_bytes.gather(1, columns).reshape(-1, PAGE_SIZE, KEEP_K)
    bitmaps = _bitmap(mask).reshape(-1, PAGE_SIZE, 8)
    scales = (exponent + 127).to(torch.uint8).reshape(-1, PAGE_SIZE, 8)
    return (values, bitmaps, scales), mask


def _decode_packed(
    buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    values, _, scales = buffers
    codes = values.reshape(-1, KEEP_K).view(torch.float8_e4m3fn).float()
    scale = torch.exp2(scales.reshape(-1, 8).to(torch.int32).float() - 127)
    ranks = mask.to(torch.int32).cumsum(-1) - 1
    full_codes = codes.gather(1, ranks.clamp_min(0))
    decoded = full_codes * scale.repeat_interleave(64, -1)
    decoded = torch.where(mask, decoded, torch.zeros_like(decoded))
    return decoded.to(torch.bfloat16)


def _make_freqs(max_position: int, device: torch.device) -> torch.Tensor:
    pair = torch.arange(32, device=device, dtype=torch.float32)
    pos = torch.arange(max_position, device=device, dtype=torch.float32)[:, None]
    angle = (pos + 1.0) * (pair + 1.0) * 0.0017
    table = torch.zeros((1, max_position * 128 + 32, 2), device=device)
    table[..., 0] = 1.0
    table[0, : max_position * 128].view(max_position, 128, 2)[:, :32, 0] = torch.cos(angle)
    table[0, : max_position * 128].view(max_position, 128, 2)[:, :32, 1] = torch.sin(angle)
    return table.contiguous()


def _rotate_tail(tail: torch.Tensor, raw: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    result = tail.float().clone()
    flat_freqs = freqs.reshape(-1, 2)
    for pair in range(32):
        index = raw.to(torch.long) * 128 + pair
        c = flat_freqs.index_select(0, index.reshape(-1))[:, 0].reshape(raw.shape)
        s = flat_freqs.index_select(0, index.reshape(-1))[:, 1].reshape(raw.shape)
        x0 = result[..., 2 * pair].clone()
        x1 = result[..., 2 * pair + 1].clone()
        result[..., 2 * pair] = x0 * c - x1 * s
        result[..., 2 * pair + 1] = x0 * s + x1 * c
    return result.to(torch.bfloat16)


def _native_from_packed(
    buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mask: torch.Tensor,
    physical: torch.Tensor,
    raw: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values, _, scales = buffers
    batch, _, selected_k = physical.shape
    rows = raw.numel()
    pages = (rows + PAGE_SIZE - 1) // PAGE_SIZE
    bytes_per_page = ((PAGE_SIZE * 584 + 575) // 576) * 576
    storage = torch.zeros((pages, bytes_per_page), dtype=torch.uint8, device=raw.device)
    flat_values = values.reshape(-1, KEEP_K)
    flat_scales = scales.reshape(-1, 8)
    flat_mask = mask.reshape(-1, HEAD_DIM)
    ranks = flat_mask.to(torch.int32).cumsum(-1) - 1
    decoded = _decode_packed(buffers, mask).reshape(-1, HEAD_DIM)
    for row in range(rows):
        page, offset = divmod(row, PAGE_SIZE)
        data_base = offset * 576
        scale_base = PAGE_SIZE * 576 + offset * 8
        source = int(physical.reshape(-1)[row].item())
        rotated_tail = _rotate_tail(
            decoded[source : source + 1, NOPE_DIM:],
            raw.reshape(-1)[row : row + 1].clamp_min(0),
            freqs,
        )[0]
        codes = torch.zeros(NOPE_DIM, dtype=torch.uint8, device=raw.device)
        kept = flat_mask[source, :NOPE_DIM]
        codes[kept] = flat_values[source].index_select(0, ranks[source, :NOPE_DIM][kept])
        storage[page, data_base : data_base + NOPE_DIM] = codes
        storage[page, scale_base : scale_base + 8] = flat_scales[source, :8]
        storage[page, data_base + NOPE_DIM : data_base + 576] = (
            rotated_tail.contiguous().view(torch.uint8)
        )
    native_cache = storage.as_strided(
        (pages, PAGE_SIZE, 1, 584),
        (bytes_per_page, 584, 584, 1),
    )
    native_indices = torch.arange(
        rows, dtype=torch.int32, device=raw.device
    ).view(batch, 1, selected_k)
    return native_cache, native_indices, storage


def materialize_native(case: RemnantCase) -> None:
    """Reconstruct the selected Packed rows into the reusable Native workspace."""
    values, _, scales = case.packed_buffers
    source = case.packed_indices.reshape(-1)
    raw = case.raw_indices.reshape(-1)
    lengths = case.extra_length.repeat_interleave(case.packed_indices.shape[-1])
    valid = torch.arange(source.numel(), device=source.device) % case.packed_indices.shape[-1] < lengths
    flat_values = values.reshape(-1, KEEP_K)
    flat_scales = scales.reshape(-1, 8)
    flat_mask = case.mask.reshape(-1, HEAD_DIM)
    ranks = flat_mask.to(torch.int32).cumsum(-1) - 1
    selected_values = flat_values.index_select(0, source)
    selected_mask = flat_mask.index_select(0, source)
    selected_ranks = ranks.index_select(0, source).clamp_min(0)
    codes = torch.gather(selected_values, 1, selected_ranks)
    codes = torch.where(selected_mask, codes, torch.zeros_like(codes))

    decoded = _decode_packed(case.packed_buffers, case.mask)
    tail = decoded.index_select(0, source)[:, NOPE_DIM:].float().reshape(-1, 32, 2)
    freq_rows = case.freqs.reshape(-1, 2).index_select(
        0, (raw.clamp_min(0).to(torch.long)[:, None] * 128 + torch.arange(32, device=raw.device)[None, :]).reshape(-1)
    ).reshape(-1, 32, 2)
    x0 = tail[..., 0].clone()
    x1 = tail[..., 1].clone()
    rotated = torch.stack((x0 * freq_rows[..., 0] - x1 * freq_rows[..., 1],
                           x0 * freq_rows[..., 1] + x1 * freq_rows[..., 0]), dim=-1)
    tail_bytes = rotated.reshape(-1, ROPE_DIM).to(torch.bfloat16).view(torch.uint8)
    codes = torch.cat((codes[:, :NOPE_DIM], tail_bytes), dim=1)
    codes = torch.where(valid[:, None], codes, torch.zeros_like(codes))
    selected_scales = flat_scales.index_select(0, source)
    selected_scales = torch.where(valid[:, None], selected_scales, torch.zeros_like(selected_scales))

    case.native_storage.zero_()
    for page in range(case.native_storage.shape[0]):
        begin = page * PAGE_SIZE
        end = begin + PAGE_SIZE
        page_codes = codes[begin:end]
        page_scales = selected_scales[begin:end]
        case.native_storage[page, : PAGE_SIZE * 576].view(PAGE_SIZE, 576).copy_(page_codes)
        case.native_storage[page, PAGE_SIZE * 576 : PAGE_SIZE * 576 + PAGE_SIZE * 8].copy_(page_scales.reshape(-1))


def make_case(
    num_heads: int,
    topk_length: int,
    batch: int = 1,
    device: torch.device | str = "cuda",
    unique_selections: bool = False,
) -> RemnantCase:
    device = torch.device(device)
    rows_per_request = 512 if unique_selections else PAGE_SIZE
    rows = max(PAGE_SIZE, batch * rows_per_request)
    latent = torch.randn((rows, HEAD_DIM), device=device)
    packed, mask = _pack(latent)
    packed_indices = torch.stack([
        batch_index * rows_per_request
        + torch.arange(512, device=device, dtype=torch.int32) % rows_per_request
        for batch_index in range(batch)
    ]).view(batch, 1, 512)
    if unique_selections:
        raw_indices = packed_indices * 3 + 17
    else:
        raw_indices = packed_indices + 3 + (
            torch.arange(512, device=device, dtype=torch.int32).view(1, 1, 512) % 11
        )
    if topk_length < 512:
        packed_indices = packed_indices.clone()
        raw_indices = raw_indices.clone()
        packed_indices[:, :, topk_length:] = 0
        raw_indices[:, :, topk_length:] = -1
    extra_length = torch.full((batch,), topk_length, dtype=torch.int32, device=device)
    freqs = _make_freqs(max(128, int(raw_indices.max().item()) + 2), device)
    native_cache, native_indices, native_storage = _native_from_packed(
        packed, mask, packed_indices, raw_indices, freqs
    )
    q = torch.randn((batch, 1, num_heads, HEAD_DIM), device=device, dtype=torch.bfloat16)
    bytes_per_page = ((PAGE_SIZE * 584 + 575) // 576) * 576
    swa_storage = torch.zeros(
        (batch, bytes_per_page), dtype=torch.uint8, device=device
    )
    swa_cache = swa_storage.as_strided(
        (batch, PAGE_SIZE, 1, 584), (bytes_per_page, 584, 584, 1)
    )
    swa_data = swa_storage[:, : PAGE_SIZE * 576].view(batch, PAGE_SIZE, 576)
    swa_data[..., :448].copy_(
        (torch.arange(batch * PAGE_SIZE * 448, device=device) % 113 + 1)
        .to(torch.uint8)
        .view(batch, PAGE_SIZE, 448)
    )
    swa_tail = torch.linspace(-0.25, 0.25, batch * PAGE_SIZE * 64, device=device)
    swa_data[..., 448:576].copy_(
        swa_tail.to(torch.bfloat16).view(torch.uint8).reshape(batch, PAGE_SIZE, 128)
    )
    swa_storage[:, PAGE_SIZE * 576 : PAGE_SIZE * 584].view(batch, PAGE_SIZE, 8).fill_(127)
    swa_indices = torch.arange(PAGE_SIZE, device=device, dtype=torch.int32).view(1, 1, PAGE_SIZE)
    swa_indices = (swa_indices + torch.arange(batch, device=device, dtype=torch.int32).view(batch, 1, 1) * PAGE_SIZE).contiguous()
    swa_length = torch.full((batch,), PAGE_SIZE, dtype=torch.int32, device=device)
    sink = torch.linspace(-0.1, 0.1, num_heads, device=device)
    return RemnantCase(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_length=swa_length,
        packed_buffers=packed,
        packed_indices=packed_indices.contiguous(),
        raw_indices=raw_indices.contiguous(),
        extra_length=extra_length,
        freqs=freqs,
        native_cache=native_cache,
        native_indices=native_indices.contiguous(),
        native_storage=native_storage,
        mask=mask,
        sink=sink,
        sm_scale=HEAD_DIM ** -0.5,
    )
