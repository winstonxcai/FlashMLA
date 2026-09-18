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
        x0 = result[..., 2 * pair]
        x1 = result[..., 2 * pair + 1]
        result[..., 2 * pair] = x0 * c - x1 * s
        result[..., 2 * pair + 1] = x0 * s + x1 * c
    return result.to(torch.bfloat16)


def _native_from_packed(
    buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mask: torch.Tensor,
    physical: torch.Tensor,
    raw: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values, bitmaps, scales = buffers
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
        storage[page, scale_base : scale_base + 7] = flat_scales[source, :7]
        storage[page, data_base + NOPE_DIM : data_base + 576] = (
            rotated_tail.contiguous().view(torch.uint8)
        )
    return storage[:, : PAGE_SIZE * 576].view(pages, PAGE_SIZE, 1, 576), torch.arange(
        rows, dtype=torch.int32, device=raw.device
    ).view(1, 1, rows)


def make_case(
    num_heads: int,
    topk_length: int,
    batch: int = 1,
    device: torch.device | str = "cuda",
) -> RemnantCase:
    device = torch.device(device)
    rows = PAGE_SIZE
    latent = torch.randn((rows, HEAD_DIM), device=device)
    packed, mask = _pack(latent)
    packed_indices = torch.arange(512, device=device, dtype=torch.int32).view(1, 1, 512) % rows
    raw_indices = packed_indices + 3
    if topk_length < 512:
        packed_indices[:, :, topk_length:] = 0
        raw_indices[:, :, topk_length:] = -1
    extra_length = torch.full((batch,), topk_length, dtype=torch.int32, device=device)
    freqs = _make_freqs(128, device)
    native_cache, native_indices = _native_from_packed(
        packed, mask, packed_indices.reshape(-1), raw_indices.reshape(-1), freqs
    )
    q = torch.randn((batch, 1, num_heads, HEAD_DIM), device=device, dtype=torch.bfloat16)
    swa_cache = torch.zeros((1, PAGE_SIZE, 1, 576), dtype=torch.uint8, device=device)
    swa_indices = torch.arange(PAGE_SIZE, device=device, dtype=torch.int32).view(1, 1, PAGE_SIZE)
    swa_indices = swa_indices.expand(batch, -1, -1).contiguous()
    swa_length = torch.full((batch,), PAGE_SIZE, dtype=torch.int32, device=device)
    sink = torch.linspace(-0.1, 0.1, num_heads, device=device)
    return RemnantCase(
        q=q,
        swa_cache=swa_cache,
        swa_indices=swa_indices,
        swa_length=swa_length,
        packed_buffers=packed,
        packed_indices=packed_indices.expand(batch, -1, -1).contiguous(),
        raw_indices=raw_indices.expand(batch, -1, -1).contiguous(),
        extra_length=extra_length,
        freqs=freqs,
        native_cache=native_cache,
        native_indices=native_indices.expand(batch, -1, -1).contiguous(),
        sink=sink,
        sm_scale=HEAD_DIM ** -0.5,
    )
