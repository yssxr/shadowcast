"""Union many vision sources into one per-team visibility mask.

A team's visible region is the union over its champions, wards, turrets and minions,
so the hot loop is a bitwise OR. The global mask is `uint64[512, 8]`, 32 KB, which
fits in L2, so assembling one costs a memcpy plus a few dozen shifted ORs rather
than touching a quarter-million bytes.

Each source contributes its table row, ANDed with the disc for its sight radius, at
a window origin that may be negative or overhang the far edge. Border overhang is
not an edge case to bolt on later: every source within 53 cells of a map boundary
overhangs, which is most of the map perimeter.

The kernel is paired with a NumPy reference of identical signature and checked
bit-for-bit against it. Two hazards here produce wrong masks rather than errors:
`x >> 64` is undefined in LLVM so the carry term is skipped when the shift is zero,
and Numba does not bounds-check while negative indices wrap, so the destination word
guards are load-bearing, without them a source near the left edge would light cells
on the right side of the map.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from shadowcast.fov.table import MISS, FovTable

__all__ = [
    "mask_bit",
    "mask_popcount",
    "mask_to_bool",
    "new_mask",
    "or_live_window",
    "union_sources",
    "union_sources_ref",
]


def new_mask(grid: int) -> np.ndarray:
    """A zeroed team mask: one packed row of `grid` bits per grid row."""
    return np.zeros((grid, (grid + 63) // 64), dtype=np.uint64)


@njit(cache=True, inline="always")
def _mask_range(lo: int, hi: int) -> np.uint64:
    m = np.uint64(0xFFFFFFFFFFFFFFFF)
    return (m << np.uint64(lo)) & (m >> np.uint64(64 - hi))


@njit(cache=True)
def union_sources(
    rows: np.ndarray,  # uint64[n_rows, row_words]
    discs: np.ndarray,  # uint64[n_radii, window, src_words]
    row_ids: np.ndarray,  # int32[n_src]
    origin_i: np.ndarray,  # int32[n_src]  window origin, grid x of window column 0
    origin_j: np.ndarray,  # int32[n_src]  window origin, grid y of window row 0
    radius_idx: np.ndarray,  # int32[n_src]
    window: int,
    src_words: int,
    out: np.ndarray,  # uint64[grid, dst_words], OR-ed in place
) -> None:
    grid = out.shape[0]
    dst_words = out.shape[1]
    n_src = row_ids.shape[0]

    for s in range(n_src):
        row = row_ids[s]
        if row < 0:
            continue  # table miss; the caller handles it via or_live_window
        x0 = origin_i[s]
        y0 = origin_j[s]
        ri = radius_idx[s]

        for wj in range(window):
            gj = y0 + wj
            if gj < 0 or gj >= grid:
                continue
            base = wj * src_words

            for sw in range(src_words):
                v = rows[row, base + sw] & discs[ri, wj, sw]
                if v == np.uint64(0):
                    continue

                bit0 = sw * 64  # window-space index of this word's bit 0

                # Clip in SOURCE space first, so the destination guards below are
                # sufficient and no out-of-range write is possible.
                lo = 0
                if -x0 - bit0 > lo:
                    lo = -x0 - bit0
                hi = 64
                if window - bit0 < hi:
                    hi = window - bit0
                if grid - x0 - bit0 < hi:
                    hi = grid - x0 - bit0
                if hi <= lo:
                    continue
                if lo > 0 or hi < 64:
                    v &= _mask_range(lo, hi)
                    if v == np.uint64(0):
                        continue

                dbit = x0 + bit0
                dw = dbit >> 6 if dbit >= 0 else -((-dbit + 63) >> 6)
                sh = dbit - dw * 64  # always in [0, 64)

                if 0 <= dw < dst_words:
                    out[gj, dw] |= v << np.uint64(sh)
                # Skipped when sh == 0: `v >> 64` is undefined and there is nothing
                # to carry. This is the likeliest defect in the whole engine, and it
                # only manifests for sources whose x-origin is a multiple of 64.
                if sh != 0:
                    dw1 = dw + 1
                    if 0 <= dw1 < dst_words:
                        out[gj, dw1] |= v >> np.uint64(64 - sh)


def union_sources_ref(
    rows: np.ndarray,
    discs: np.ndarray,
    row_ids: np.ndarray,
    origin_i: np.ndarray,
    origin_j: np.ndarray,
    radius_idx: np.ndarray,
    window: int,
    src_words: int,
    out: np.ndarray,
) -> None:
    """NumPy reference: unpack, slice, OR. Obviously correct, deliberately slow."""
    from shadowcast.geom.bitset import pack_rows, unpack_rows

    grid = out.shape[0]
    acc = unpack_rows(out, grid)

    for s in range(len(row_ids)):
        row = int(row_ids[s])
        if row < 0:
            continue
        packed = np.asarray(rows[row])[: window * src_words].reshape(window, src_words)
        win = unpack_rows(packed, window) & unpack_rows(
            np.asarray(discs[int(radius_idx[s])]), window
        )
        x0, y0 = int(origin_i[s]), int(origin_j[s])

        sj_lo, sj_hi = max(0, -y0), min(window, grid - y0)
        si_lo, si_hi = max(0, -x0), min(window, grid - x0)
        if sj_hi <= sj_lo or si_hi <= si_lo:
            continue
        acc[y0 + sj_lo : y0 + sj_hi, x0 + si_lo : x0 + si_hi] |= win[sj_lo:sj_hi, si_lo:si_hi]

    out[:] = pack_rows(acc, row_words=out.shape[1])


def or_live_window(out: np.ndarray, window_bool: np.ndarray, x0: int, y0: int) -> None:
    """OR a live-computed boolean window into a packed mask.

    The fallback path for table misses, sources in walls, sources whose continuous
    position sits in a brush their cell does not, runtime occluders. Rare enough that
    a NumPy slice is fine.
    """
    from shadowcast.geom.bitset import pack_rows, unpack_rows

    grid = out.shape[0]
    window = window_bool.shape[0]
    acc = unpack_rows(out, grid)
    sj_lo, sj_hi = max(0, -y0), min(window, grid - y0)
    si_lo, si_hi = max(0, -x0), min(window, grid - x0)
    if sj_hi > sj_lo and si_hi > si_lo:
        acc[y0 + sj_lo : y0 + sj_hi, x0 + si_lo : x0 + si_hi] |= window_bool[
            sj_lo:sj_hi, si_lo:si_hi
        ]
    out[:] = pack_rows(acc, row_words=out.shape[1])


def mask_bit(mask: np.ndarray, i: int, j: int) -> bool:
    """Read one cell from a packed mask, without unpacking the whole row."""
    return bool((mask[j, i >> 6] >> np.uint64(i & 63)) & np.uint64(1))


def mask_to_bool(mask: np.ndarray, grid: int) -> np.ndarray:
    from shadowcast.geom.bitset import unpack_rows

    return unpack_rows(mask, grid)


def mask_popcount(mask: np.ndarray) -> int:
    return int(np.bitwise_count(mask).sum())


# ---------------------------------------------------------------------------
# Convenience assembly
# ---------------------------------------------------------------------------
def assemble(
    table: FovTable,
    terrain,
    sources: list[tuple[int, int, float, int]],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Union a list of `(i, j, radius_units, src_brush)` sources into a team mask.

    Dispatches per source: a table hit goes through the bit-blit, a miss is computed
    live and OR-ed in. `src_brush` is the brush from the source's continuous
    position, so passing a value that disagrees with the snapped cell deliberately
    forces the live path rather than silently using the wrong occluder set.
    """
    from shadowcast.fov.shadowcast import fov_bool

    grid = table.grid
    if out is None:
        out = new_mask(grid)

    hit_rows, hit_i, hit_j, hit_r = [], [], [], []
    for i, j, radius, src_brush in sources:
        cell = j * grid + i
        cell_brush = int(terrain.brush_id[j, i]) if 0 <= i < grid and 0 <= j < grid else -1
        ri = table.radius_index(radius)
        row = table.lookup(cell, src_brush, cell_brush) if ri >= 0 else int(MISS)
        if row >= 0:
            hit_rows.append(row)
            hit_i.append(i - table.half)
            hit_j.append(j - table.half)
            hit_r.append(ri)
        else:
            win = fov_bool(terrain, i, j, radius, half=table.half, src_brush=src_brush)
            or_live_window(out, win, i - table.half, j - table.half)

    if hit_rows:
        union_sources(
            np.asarray(table.rows),
            table.discs,
            np.array(hit_rows, dtype=np.int32),
            np.array(hit_i, dtype=np.int32),
            np.array(hit_j, dtype=np.int32),
            np.array(hit_r, dtype=np.int32),
            table.window,
            table.src_words,
            out,
        )
    return out
