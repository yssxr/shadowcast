"""Brute-force field of view by per-target ray marching.

The oracle that `shadowcast.py` is checked against. It is deliberately a different
*class* of algorithm — march a ray from the source's centre to each candidate
cell's centre and ask whether anything opaque lies between — rather than a second
octant sweep. A reimplementation of the same idea would share the same mistakes;
this shares almost nothing, so a disagreement carries information.

It is far too slow for production (O(cells × ray length) per source instead of
O(cells)), which is the entire reason recursive shadowcasting exists. Its job is to
be obviously correct.

The two algorithms are not expected to agree perfectly, and forcing them to would
be a mistake. Shadowcasting decides visibility per *cell* from accumulated slope
intervals; ray marching decides it per *centre-to-centre segment*. They differ on
cells whose centre is just inside or just outside a shadow boundary, so tests
compare them outside a one-cell band around shadow edges and report where the
disagreements land. If disagreements are NOT concentrated at boundaries, that is a
real bug rather than quantisation, which makes the distribution more useful than
the headline rate.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from shadowcast.constants import RMAX_CELLS
from shadowcast.terrain.terrain import Terrain

__all__ = ["boundary_band", "fov_reference", "fov_reference_into"]

#: Ray sampling step, in cells. 0.1 cell is 2.9 world units — far finer than the
#: 50-unit resolution of the source terrain, so the march cannot tunnel through a
#: one-cell wall, which is the only failure mode that would matter.
_STEP = 0.1


@njit(cache=True, inline="always")
def _blocks(
    blocks_vision: np.ndarray,
    brush_id: np.ndarray,
    j: int,
    i: int,
    src_brush: int,
) -> bool:
    if blocks_vision[j, i]:
        return True
    b = brush_id[j, i]
    return b >= 0 and b != src_brush


@njit(cache=True)
def fov_reference_into(
    out: np.ndarray,  # bool[window, window]
    blocks_vision: np.ndarray,
    brush_id: np.ndarray,
    si: int,
    sj: int,
    src_brush: int,
    r_cells_sq: float,
    half: int,
) -> None:
    grid_h, grid_w = blocks_vision.shape
    out[half, half] = True

    for dj in range(-half, half + 1):
        for di in range(-half, half + 1):
            if di * di + dj * dj > r_cells_sq:
                continue
            gi = si + di
            gj = sj + dj
            if not (0 <= gi < grid_w and 0 <= gj < grid_h):
                continue
            # Foreign brush is never visible, however clear the line to it.
            b = brush_id[gj, gi]
            if b >= 0 and b != src_brush:
                continue
            if di == 0 and dj == 0:
                continue

            length = np.sqrt(float(di * di + dj * dj))
            steps = int(length / _STEP) + 1
            blocked = False
            for s in range(1, steps):
                t = (s * _STEP) / length
                if t >= 1.0:
                    break
                px = di * t
                py = dj * t
                ci = int(np.floor(px + 0.5))
                cj = int(np.floor(py + 0.5))
                # Neither endpoint occludes: the source cannot block itself, and a
                # wall face is visible from outside.
                if (ci == 0 and cj == 0) or (ci == di and cj == dj):
                    continue
                qi = si + ci
                qj = sj + cj
                if not (0 <= qi < grid_w and 0 <= qj < grid_h):
                    blocked = True
                    break
                if _blocks(blocks_vision, brush_id, qj, qi, src_brush):
                    blocked = True
                    break
            if not blocked:
                out[dj + half, di + half] = True


def fov_reference(
    terrain: Terrain,
    i: int,
    j: int,
    radius_units: float,
    half: int = RMAX_CELLS,
    src_brush: int | None = None,
) -> np.ndarray:
    from shadowcast.geom.grid import radius_cells_sq

    window = 2 * half + 1
    out = np.zeros((window, window), dtype=bool)
    if src_brush is None:
        src_brush = int(terrain.brush_id[j, i])
    fov_reference_into(
        out,
        terrain.blocks_vision,
        terrain.brush_id,
        int(i),
        int(j),
        int(src_brush),
        radius_cells_sq(radius_units),
        int(half),
    )
    return out


def boundary_band(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Cells within `width` of a visibility boundary in `mask`.

    Used to exclude shadow edges when comparing two algorithms, since that is
    exactly where a per-cell decision and a per-segment decision legitimately
    differ. Dilating the boundary rather than the mask keeps the exclusion
    symmetric: a cell adjacent to a disagreement is excluded whichever side of the
    edge it sits on.
    """
    m = mask.astype(bool)
    edge = np.zeros_like(m)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[1:, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]
    edge[:, 1:] |= m[:, :-1] != m[:, 1:]
    for _ in range(width - 1):
        grown = edge.copy()
        grown[:-1, :] |= edge[1:, :]
        grown[1:, :] |= edge[:-1, :]
        grown[:, :-1] |= edge[:, 1:]
        grown[:, 1:] |= edge[:, :-1]
        edge = grown
    return edge
