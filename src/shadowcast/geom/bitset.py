"""Packed-uint64 bitset primitives.

Visibility masks are bitsets because a team's visible region is the union over its
sources, and union over a bitset is a bitwise OR. A 512x512 team mask is 32 KB —
it fits in L2, so assembling one per tick costs a memcpy plus a few dozen shifted
ORs instead of touching a quarter-million bytes.

Every kernel here exists twice: a Numba version and a NumPy reference with an
identical signature. That is not redundancy. The reference is the test oracle, the
readable specification, and the fallback when `SHADOWCAST_NO_NUMBA=1` — and a
bit-for-bit differential test between them is the only thing that catches the
shift bugs described below.

Bit order is little throughout: bit `k` of a row lives at bit `k & 63` of word
`k >> 6`. That matches `1 << k`, and it matches what a TypeScript `DataView`
reader does, which matters because the frontend reads these masks.

Two hazards, both of which produce silently wrong output rather than a crash:

1. **`x >> 64` is undefined.** In LLVM a shift at or beyond the operand width is
   UB, so the carry term `src >> (64 - sh)` must be skipped when `sh == 0` rather
   than evaluated. This is the single most likely defect in the codebase and it
   only manifests for sources whose x-offset happens to be a multiple of 64.
2. **Numba promotes `uint64 >> int64` to float64**, following NumPy's rules. Every
   shift amount is therefore cast to `uint64` explicitly. A missed cast turns a
   bitmask into a float and the result is garbage without a type error.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "WORD_BITS",
    "words_for",
    "pack_rows",
    "unpack_rows",
    "bit_get",
    "bit_set",
    "or_row_into",
    "or_row_into_ref",
    "popcount",
    "mask_range",
]

WORD_BITS = 64
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_U1 = np.uint64(1)


def words_for(bits: int) -> int:
    """Words needed to hold `bits` bits."""
    return (int(bits) + WORD_BITS - 1) // WORD_BITS


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------
def pack_rows(mask: np.ndarray, row_words: int | None = None) -> np.ndarray:
    """Pack a 2-D boolean array to uint64, one packed row per input row.

    `row_words` pads each row out to a fixed stride. Padding is what buys the
    fast blit: a stride that is a multiple of 8 words keeps every row 64-byte
    aligned, and the padding bits are guaranteed zero so no consumer has to mask
    them off.
    """
    mask = np.ascontiguousarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    rows, width = mask.shape
    need = words_for(width)
    if row_words is None:
        row_words = need
    elif row_words < need:
        raise ValueError(f"row_words={row_words} cannot hold {width} bits (needs {need})")

    out = np.zeros((rows, row_words), dtype=np.uint64)
    # np.packbits with bitorder="little" gives bit k at bit (k & 7) of byte k >> 3,
    # which is exactly the little-endian uint64 layout once viewed as uint64.
    padded_bytes = row_words * 8
    packed = np.packbits(mask, axis=1, bitorder="little")
    buf = np.zeros((rows, padded_bytes), dtype=np.uint8)
    buf[:, : packed.shape[1]] = packed
    out[:] = buf.view(np.uint64)
    return out


def unpack_rows(packed: np.ndarray, width: int) -> np.ndarray:
    """Inverse of `pack_rows`, truncated back to `width` bits per row."""
    packed = np.ascontiguousarray(packed, dtype=np.uint64)
    if packed.ndim != 2:
        raise ValueError(f"expected 2-D packed rows, got shape {packed.shape}")
    as_bytes = packed.view(np.uint8)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")
    return bits[:, :width].astype(bool)


def bit_get(words: np.ndarray, k: int) -> bool:
    """Read bit `k` of a 1-D packed row."""
    return bool((words[k >> 6] >> np.uint64(k & 63)) & _U1)


def bit_set(words: np.ndarray, k: int) -> None:
    """Set bit `k` of a 1-D packed row, in place."""
    words[k >> 6] |= _U1 << np.uint64(k & 63)


def mask_range(lo: int, hi: int) -> np.uint64:
    """Word mask with bits [lo, hi) set. Requires 0 <= lo < hi <= 64.

    Both shifts are in range for those bounds: `<< lo` with lo <= 63, and
    `>> (64 - hi)` with hi >= 1 so the shift is at most 63. hi == 64 gives a
    shift of 0, which is fine — it is only shifts *at or beyond* the width that
    are undefined.
    """
    if not (0 <= lo < hi <= 64):
        raise ValueError(f"mask_range({lo}, {hi}) out of bounds")
    return np.uint64((_MASK64 << np.uint64(lo)) & (_MASK64 >> np.uint64(64 - hi)))


@njit(cache=True, inline="always")
def _mask_range_jit(lo: int, hi: int) -> np.uint64:
    m = np.uint64(0xFFFFFFFFFFFFFFFF)
    return (m << np.uint64(lo)) & (m >> np.uint64(64 - hi))


# ---------------------------------------------------------------------------
# The blit
# ---------------------------------------------------------------------------
@njit(cache=True)
def or_row_into(
    dst: np.ndarray,  # uint64[dst_words], modified in place
    src: np.ndarray,  # uint64[src_words]
    x0: int,  # destination bit index of src bit 0; may be negative
    src_width: int,  # meaningful bits in src
    dst_width: int,  # meaningful bits in dst
) -> None:
    """OR a packed bit row into a packed destination row at bit offset `x0`.

    `x0` may be negative and `x0 + src_width` may exceed `dst_width`; the overhang
    is clipped. That clipping is not an edge case to handle later — every source
    within RMAX_CELLS of a map border overhangs, which is a large fraction of the
    map perimeter.

    Bits are clipped in *source* space first, by masking the source word down to
    the bits whose destination lands inside [0, dst_width). Only then are the
    destination word indices computed, so the bounds checks below are sufficient
    and no out-of-range write is possible. Numba does not bounds-check by default
    and negative indices wrap, so those guards are load-bearing.
    """
    n_src_words = src.shape[0]
    n_dst_words = dst.shape[0]

    for sw in range(n_src_words):
        base = sw * 64  # index, within src, of this word's bit 0

        # Bits of this source word that are both meaningful and land in range.
        lo = 0
        if -x0 - base > lo:
            lo = -x0 - base
        hi = 64
        if src_width - base < hi:
            hi = src_width - base
        if dst_width - x0 - base < hi:
            hi = dst_width - x0 - base
        if hi <= lo:
            continue

        v = src[sw]
        if lo > 0 or hi < 64:
            v &= _mask_range_jit(lo, hi)
        if v == np.uint64(0):
            continue

        # Destination of src bit `base`. Floor division, because x0 may be
        # negative and Python-style floor is what the bit layout needs.
        dbit = x0 + base
        dw = dbit >> 6 if dbit >= 0 else -((-dbit + 63) >> 6)
        sh = dbit - dw * 64  # always in [0, 64)

        if 0 <= dw < n_dst_words:
            dst[dw] |= v << np.uint64(sh)
        # The carry into the next word. Skipped entirely when sh == 0: `v >> 64`
        # is undefined, and there is nothing to carry anyway.
        if sh != 0:
            dw1 = dw + 1
            if 0 <= dw1 < n_dst_words:
                dst[dw1] |= v >> np.uint64(64 - sh)


def or_row_into_ref(
    dst: np.ndarray,
    src: np.ndarray,
    x0: int,
    src_width: int,
    dst_width: int,
) -> None:
    """NumPy reference for `or_row_into`. Obviously correct, deliberately slow.

    Unpacks, slices, ORs, repacks. This is the oracle the jitted kernel is
    differentially tested against, and the readable statement of what the shifts
    above are meant to accomplish.
    """
    src_bits = np.unpackbits(
        np.ascontiguousarray(src, dtype=np.uint64).view(np.uint8), bitorder="little"
    )[:src_width].astype(bool)
    dst_bits = np.unpackbits(
        np.ascontiguousarray(dst, dtype=np.uint64).view(np.uint8), bitorder="little"
    )[:dst_width].astype(bool)

    # Overlap of [x0, x0 + src_width) with [0, dst_width), in source coordinates.
    s_lo = max(0, -x0)
    s_hi = min(src_width, dst_width - x0)
    if s_hi > s_lo:
        d_lo = x0 + s_lo
        d_hi = x0 + s_hi
        dst_bits[d_lo:d_hi] |= src_bits[s_lo:s_hi]

    repacked = pack_rows(dst_bits[None, :], row_words=dst.shape[0])[0]
    dst[:] = repacked


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def popcount(words: np.ndarray) -> int:
    """Total set bits. Assumes padding bits are zero, which `pack_rows` guarantees."""
    return int(np.bitwise_count(np.ascontiguousarray(words, dtype=np.uint64)).sum())


@njit(cache=True, inline="always")
def _popcount_word(v: np.uint64) -> int:
    # SWAR popcount. Numba has no np.bitwise_count intrinsic, and calling out to
    # Python per word would dominate the mask-assembly cost it exists to serve.
    v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((v * np.uint64(0x0101010101010101)) >> np.uint64(56)) & 0xFF


@njit(cache=True)
def popcount_jit(words: np.ndarray) -> int:
    total = 0
    for i in range(words.shape[0]):
        total += _popcount_word(words[i])
    return total
