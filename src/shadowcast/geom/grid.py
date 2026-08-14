"""World coordinates <-> grid cells, and the radius predicate.

The radius predicate is the load-bearing part of this module. The FOV table
stores visibility at RMAX and serves every smaller radius by intersecting with a
circular mask, which is only exact if the mask and the FOV's own radius test are
the *same* test. So both call `in_radius_sq`, and there is no second
implementation anywhere in the package.

Cells are square. That is why the grid spans a square region (the larger navgrid
axis) rather than the navgrid's slightly non-square extent: a non-square cell
turns a circular sight radius into an ellipse in cell space, and every disc mask
would be subtly wrong in a way that is invisible until it is compared against
ground truth.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from shadowcast.constants import (
    FOV_WINDOW,
    GRID,
    GRID_CELL_SIZE,
    RMAX_CELLS,
    RMAX_UNITS,
    WORLD_MIN_X,
    WORLD_MIN_Z,
    WORLD_SPAN,
)

__all__ = [
    "cell_to_world",
    "disc_mask",
    "disc_mask_cells",
    "flat_index",
    "in_bounds",
    "in_radius_sq",
    "radius_cells_sq",
    "ring_offsets",
    "unflatten",
    "world_to_cell",
    "world_to_cell_array",
]


# ---------------------------------------------------------------------------
# The radius predicate
# ---------------------------------------------------------------------------
@njit(cache=True, inline="always")
def in_radius_sq(dx: int, dy: int, r_cells_sq: float) -> bool:
    """Is the cell offset (dx, dy) within a radius whose square is r_cells_sq?

    Distances are centre-to-centre, matching the game: "a unit becomes visible
    when its centre enters the observer's radius". Both operands are in cell
    units, so callers convert a world radius once via `radius_cells_sq`.

    The comparison is `<=`. That tie-break is arbitrary but it must be identical
    here and in every disc mask, or `fov(Rmax) & disc(r)` stops being exactly
    `fov(r)` for cells sitting precisely on the boundary.
    """
    return dx * dx + dy * dy <= r_cells_sq


def radius_cells_sq(radius_units: float) -> float:
    """Square of a world-unit radius, expressed in cells.

    Deliberately *not* rounded to whole cells. Rounding here would quantise every
    sight radius to 28.8 u steps for no benefit: the disc masks are precomputed
    per distinct radius (there are six in the game), so an exact float costs
    nothing and keeps radii faithful to the wiki values.
    """
    r = radius_units / GRID_CELL_SIZE
    return r * r


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _to_cell_scalar(x: float, z: float) -> tuple[int, int]:
    i = int((x - WORLD_MIN_X) / GRID_CELL_SIZE)
    j = int((z - WORLD_MIN_Z) / GRID_CELL_SIZE)
    return i, j


def world_to_cell(x: float, z: float) -> tuple[int, int]:
    """World position -> (i, j) cell, unclamped.

    Returns out-of-range indices for out-of-bounds input rather than clamping,
    because clamping would silently map a position 3000 units off the map onto a
    real map edge. Callers that want clamping ask for it; callers that want to
    notice bad data check `in_bounds`.
    """
    return _to_cell_scalar(float(x), float(z))


def world_to_cell_array(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised `world_to_cell`. Returns int32 arrays, unclamped."""
    i = ((np.asarray(x, dtype=np.float64) - WORLD_MIN_X) / GRID_CELL_SIZE).astype(np.int32)
    j = ((np.asarray(z, dtype=np.float64) - WORLD_MIN_Z) / GRID_CELL_SIZE).astype(np.int32)
    return i, j


def cell_to_world(i: int | np.ndarray, j: int | np.ndarray):
    """Cell -> world position of the cell *centre*.

    The centre, not the corner, because `in_radius_sq` measures centre to centre.
    Using the corner here would introduce a half-cell bias into every distance.
    """
    x = WORLD_MIN_X + (np.asarray(i, dtype=np.float64) + 0.5) * GRID_CELL_SIZE
    z = WORLD_MIN_Z + (np.asarray(j, dtype=np.float64) + 0.5) * GRID_CELL_SIZE
    if np.isscalar(i) and np.isscalar(j):
        return float(x), float(z)
    return x, z


@njit(cache=True, inline="always")
def flat_index(i: int, j: int) -> int:
    """Row-major flat index, j selecting the row. Matches the display convention."""
    return j * GRID + i


def unflatten(k: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `flat_index`."""
    arr = np.asarray(k, dtype=np.int64)
    return np.mod(arr, GRID).astype(np.int32), np.floor_divide(arr, GRID).astype(np.int32)


@njit(cache=True, inline="always")
def in_bounds(i: int, j: int) -> bool:
    return 0 <= i < GRID and 0 <= j < GRID


# ---------------------------------------------------------------------------
# Disc masks
# ---------------------------------------------------------------------------
def disc_mask(radius_units: float, window: int = FOV_WINDOW) -> np.ndarray:
    """Boolean circular mask over an FOV window, centred on the source cell.

    Built from `in_radius_sq` so it agrees with the FOV scan bit for bit. This is
    the mask that makes one max-radius table serve every radius:

        fov(r) == fov(RMAX) & disc(r)

    which holds exactly because shadowcasting decides a cell using only shadow
    intervals cast by strictly *nearer* occluders. An occluder outside disc(r)
    can therefore never affect a cell inside it, so the two runs make identical
    decisions everywhere inside disc(r).
    """
    if radius_units > RMAX_UNITS:
        raise ValueError(
            f"radius {radius_units} exceeds RMAX_UNITS={RMAX_UNITS}; such a source must "
            "take the live-compute path rather than the table"
        )
    half = window // 2
    r_sq = radius_cells_sq(radius_units)
    dy, dx = np.mgrid[-half : half + 1, -half : half + 1]
    return (dx * dx + dy * dy) <= r_sq


def disc_mask_cells(radius_units: float) -> int:
    """How many cells a disc of this radius contains. Useful for size budgeting."""
    return int(disc_mask(radius_units).sum())


def ring_offsets(radius_cells: int = RMAX_CELLS) -> np.ndarray:
    """(dy, dx) offsets sorted by increasing distance from the origin.

    Used by the reachability sampler and by tests that want to walk outward from
    a source in a deterministic order.
    """
    dy, dx = np.mgrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    dy, dx = dy.ravel(), dx.ravel()
    d2 = dy * dy + dx * dx
    keep = d2 <= radius_cells * radius_cells
    dy, dx, d2 = dy[keep], dx[keep], d2[keep]
    order = np.argsort(d2, kind="stable")
    return np.stack([dy[order], dx[order]], axis=1).astype(np.int32)


def describe() -> dict[str, float | int | str]:
    """The grid's own parameters, for artifact headers and `shadowcast doctor`."""
    return {
        "grid": GRID,
        "cell_size_units": GRID_CELL_SIZE,
        "world_min_x": WORLD_MIN_X,
        "world_min_z": WORLD_MIN_Z,
        "world_span": WORLD_SPAN,
        "rmax_units": RMAX_UNITS,
        "rmax_cells": RMAX_CELLS,
        "fov_window": FOV_WINDOW,
    }
