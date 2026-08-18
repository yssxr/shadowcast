"""Differential tests for the packed-bitset kernels.

The blit is the one place in the engine where a bug produces plausible output
rather than an error: a mask that is wrong by one word still looks like a
visibility mask, still unions, still renders. So it is tested exhaustively
against a NumPy reference, with the shift offsets that trigger the known hazards
enumerated explicitly rather than left to random sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadowcast.geom import bitset as bs


def test_words_for():
    assert bs.words_for(0) == 0
    assert bs.words_for(1) == 1
    assert bs.words_for(64) == 1
    assert bs.words_for(65) == 2
    assert bs.words_for(107) == 2


@pytest.mark.parametrize("width", [1, 7, 63, 64, 65, 107, 128, 129, 512])
def test_pack_unpack_round_trip(width):
    rng = np.random.default_rng(width)
    mask = rng.random((13, width)) < 0.4
    packed = bs.pack_rows(mask)
    assert packed.dtype == np.uint64
    np.testing.assert_array_equal(bs.unpack_rows(packed, width), mask)


@pytest.mark.parametrize("width", [107, 512])
def test_pack_rows_zeroes_padding(width):
    """Padding bits must be zero, popcount and the blit both rely on it."""
    mask = np.ones((4, width), dtype=bool)
    packed = bs.pack_rows(mask, row_words=16)
    assert bs.popcount(packed) == 4 * width


def test_pack_rows_rejects_too_narrow_stride():
    with pytest.raises(ValueError, match="cannot hold"):
        bs.pack_rows(np.zeros((2, 107), dtype=bool), row_words=1)


def test_bit_get_set():
    words = np.zeros(3, dtype=np.uint64)
    for k in (0, 1, 63, 64, 65, 127, 128, 191):
        bs.bit_set(words, k)
    for k in range(192):
        assert bs.bit_get(words, k) == (k in (0, 1, 63, 64, 65, 127, 128, 191))


def test_mask_range():
    assert bs.mask_range(0, 64) == np.uint64(0xFFFFFFFFFFFFFFFF)
    assert bs.mask_range(0, 1) == np.uint64(1)
    assert bs.mask_range(63, 64) == np.uint64(1) << np.uint64(63)
    assert bs.mask_range(8, 16) == np.uint64(0xFF00)
    for bad in [(0, 0), (5, 5), (-1, 4), (0, 65), (10, 9)]:
        with pytest.raises(ValueError, match="out of bounds"):
            bs.mask_range(*bad)


# ---------------------------------------------------------------------------
# The blit
# ---------------------------------------------------------------------------
SRC_WIDTH = 107  # FOV_WINDOW. The width that actually gets blitted
DST_WIDTH = 512  # GRID


def _blit_case(x0: int, seed: int, src_width: int = SRC_WIDTH, dst_width: int = DST_WIDTH):
    rng = np.random.default_rng(seed)
    src_bits = rng.random(src_width) < 0.35
    dst_bits = rng.random(dst_width) < 0.10

    src = bs.pack_rows(src_bits[None, :])[0]
    a = bs.pack_rows(dst_bits[None, :])[0]
    b = a.copy()

    bs.or_row_into(a, src, x0, src_width, dst_width)
    bs.or_row_into_ref(b, src, x0, src_width, dst_width)
    return a, b, src_bits, dst_bits


# Every multiple of 64 is a shift==0 case, which is the `v >> 64` undefined-behaviour
# trap. Negative offsets exercise the floor-division path. Offsets past dst_width
# exercise total clipping.
_CRITICAL_OFFSETS = [
    -SRC_WIDTH,
    -SRC_WIDTH + 1,
    -107,
    -65,
    -64,
    -63,
    -10,
    -1,
    0,
    1,
    63,
    64,
    65,
    127,
    128,
    129,
    191,
    192,
    255,
    256,
    319,
    320,
    383,
    384,
    447,
    448,
    DST_WIDTH - SRC_WIDTH,
    DST_WIDTH - 64,
    DST_WIDTH - 1,
    DST_WIDTH,
    DST_WIDTH + 5,
]


@pytest.mark.parametrize("x0", _CRITICAL_OFFSETS)
def test_blit_matches_reference_at_critical_offsets(x0):
    a, b, _, _ = _blit_case(x0, seed=abs(x0) + 1)
    np.testing.assert_array_equal(a, b, err_msg=f"blit != reference at x0={x0}")


def test_blit_matches_reference_at_every_offset():
    """Exhaustive over the whole valid range of x0, not sampled.

    The window is 107 bits and the grid 512, so there are only ~726 distinct
    offsets. Checking all of them is cheap and removes any argument about whether
    a random sweep happened to hit the aligned cases.
    """
    for x0 in range(-SRC_WIDTH - 2, DST_WIDTH + 3):
        a, b, _, _ = _blit_case(x0, seed=(x0 * 2654435761) & 0xFFFF)
        np.testing.assert_array_equal(a, b, err_msg=f"blit != reference at x0={x0}")


def test_blit_shift_zero_carries_nothing_wrongly():
    """A directed test for the `sh == 0` path.

    With x0 a multiple of 64 and the source's top bits set, a naive
    `dst[dw + 1] |= v >> (64 - sh)` would evaluate `v >> 64`. On x86 that yields
    `v` unshifted, which would scribble the source's low bits into the following
    word. A corruption of exactly 64 cells, 64 cells away from the source.
    """
    src_bits = np.zeros(SRC_WIDTH, dtype=bool)
    src_bits[[0, 1, 63]] = True
    src = bs.pack_rows(src_bits[None, :])[0]

    for x0 in (0, 64, 128, 192, 256, 320, 384, 448):
        got = np.zeros(bs.words_for(DST_WIDTH), dtype=np.uint64)
        bs.or_row_into(got, src, x0, SRC_WIDTH, DST_WIDTH)
        bits = bs.unpack_rows(got[None, :], DST_WIDTH)[0]
        expected = np.zeros(DST_WIDTH, dtype=bool)
        for k in (0, 1, 63):
            if 0 <= x0 + k < DST_WIDTH:
                expected[x0 + k] = True
        np.testing.assert_array_equal(bits, expected, err_msg=f"x0={x0}")


def test_blit_never_writes_outside_the_row():
    """Fully-clipped blits must be no-ops, not wraparound writes.

    Numba does not bounds-check and negative indices wrap, so a missing guard
    here would write to the far end of the row, visibility appearing on the
    opposite side of the map.
    """
    src = bs.pack_rows(np.ones((1, SRC_WIDTH), dtype=bool))[0]
    for x0 in (-SRC_WIDTH, -SRC_WIDTH - 1, -1000, DST_WIDTH, DST_WIDTH + 1000):
        dst = np.zeros(bs.words_for(DST_WIDTH), dtype=np.uint64)
        bs.or_row_into(dst, src, x0, SRC_WIDTH, DST_WIDTH)
        assert bs.popcount(dst) == 0, f"x0={x0} wrote {bs.popcount(dst)} bits"


def test_blit_is_a_union_not_an_assignment():
    """OR semantics: pre-existing bits survive. Team masks depend on this."""
    dst_bits = np.zeros(DST_WIDTH, dtype=bool)
    dst_bits[500] = True
    dst = bs.pack_rows(dst_bits[None, :])[0]
    src = bs.pack_rows(np.ones((1, SRC_WIDTH), dtype=bool))[0]
    bs.or_row_into(dst, src, 0, SRC_WIDTH, DST_WIDTH)
    out = bs.unpack_rows(dst[None, :], DST_WIDTH)[0]
    assert out[500]
    assert out[:SRC_WIDTH].all()
    assert not out[SRC_WIDTH:500].any()


def test_blit_with_padded_destination_stride():
    """A destination padded beyond dst_width must leave the padding untouched."""
    src = bs.pack_rows(np.ones((1, SRC_WIDTH), dtype=bool))[0]
    dst = np.zeros(16, dtype=np.uint64)  # 1024 bits of storage, 512 meaningful
    bs.or_row_into(dst, src, DST_WIDTH - 10, SRC_WIDTH, DST_WIDTH)
    assert bs.popcount(dst) == 10
    bits = bs.unpack_rows(dst[None, :], 1024)[0]
    assert bits[DST_WIDTH - 10 : DST_WIDTH].all()
    assert not bits[DST_WIDTH:].any()


# ---------------------------------------------------------------------------
# popcount
# ---------------------------------------------------------------------------
def test_popcount_agrees_with_jit():
    rng = np.random.default_rng(11)
    for n in (1, 2, 8, 216):
        words = rng.integers(0, 2**64, size=n, dtype=np.uint64)
        assert bs.popcount(words) == bs.popcount_jit(words)


def test_popcount_known_values():
    assert bs.popcount(np.array([0], dtype=np.uint64)) == 0
    assert bs.popcount(np.array([np.uint64(0xFFFFFFFFFFFFFFFF)], dtype=np.uint64)) == 64
    assert bs.popcount_jit(np.array([np.uint64(0xFFFFFFFFFFFFFFFF)], dtype=np.uint64)) == 64
