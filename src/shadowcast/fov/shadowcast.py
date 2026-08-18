"""Recursive shadowcasting over the terrain grid, with brush as a conditional occluder.

The algorithm is Björn Bergström's recursive shadowcasting: split the plane into
eight octants around the source, scan each octant outward row by row, and track the
slope interval that is still unshadowed. When a row hits an occluder, recurse into
the narrowed interval beyond it.

Two properties of this implementation are load-bearing and must not be
"improved":

**Radius separability.** `fov(r) == fov(RMAX) & disc(r)` exactly, which is what
lets one precomputed table serve every sight radius in the game and turns a
naively 8.6 TB all-pairs problem into 286 MB. It holds because a cell's visibility
is decided using only shadow intervals cast by *strictly nearer* occluders, so an
occluder outside `disc(r)` cannot influence any cell inside it. Two specific
changes break it, both verified empirically:

- A **wall-lighting post-pass**, marking occluder faces visible after the scan
  finishes, breaks 68% of cases, because a wall's visibility then depends on a
  *farther* cell's visibility. Occluders are therefore lit inside the scan, using
  the same slope and radius tests as everything else.
- **Flood-revealing the source's whole brush** breaks 1.2%, exactly when the brush
  extends past `r`, which is the long mid and river brushes. Brush the source
  stands in is transparent *and the radius still applies*.

Anti-aliased or fractional visibility would also break it, since intersecting with
a binary disc is lossy at the boundary. Visibility here is a bit, not a coverage
fraction.

**Permissiveness is left alone.** Shadowcasting tracks a slope *interval* and lights
any cell whose extremities overlap it, so it over-reports relative to a
centre-to-centre visibility test. MEASURED against the ray-march reference over
3,064,927 cells of deliberately adversarial geometry: 4,286 cells lit here but not
there, and **zero** the other way. The disagreement is strictly one-directional,
and every instance lies within two cells of a shadow boundary (4,241 at distance
one, 45 at distance two, none beyond).

That one-directionality is asserted as a test rather than merely observed, because
it is an algebraic property where the agreement *rate* is only a summary statistic:
a transposed octant, an off-by-one in a slope, or an inverted brush comparison
would all produce restrictive disagreements immediately, while any of them could
hide inside a 99.9% rate.

Classic shadowcasting is also about 3% asymmetric at wall corners. A sees B while
B does not see A. Both properties are of the algorithm, not of this code, and
"fixing" either would be optimising for the wrong target: we are trying to match
Riot's engine, which can only be settled against the fog oracle. The residual is
absorbed by the belief filter's soft detection probability rather than papered over
here.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from shadowcast.constants import RMAX_CELLS
from shadowcast.terrain.terrain import Terrain

__all__ = ["SCRATCH_FRAMES", "StackOverflowInFOV", "fov_bool", "fov_into", "new_scratch"]

#: Pending-frame capacity for the explicit scan stack.
#:
#: Numba supports recursion poorly, so the recursive scan is flattened onto a
#: stack the caller supplies (allocating per call would dominate the cost of
#: 165k table rows). The bound is not obvious analytically. A pathological
#: checkerboard could in principle push O(radius^2) frames, so the capacity is
#: generous and `fov_into` returns the high-water mark. A test measures the real
#: maximum across every walkable cell of Summoner's Rift and asserts the headroom,
#: which is a stronger guarantee than an argument about worst cases.
SCRATCH_FRAMES = 8192

_OCTANTS = np.array(
    [
        # (xx, xy, yx, yy) transforms scan-space (dx, dy) into grid offsets.
        [1, 0, 0, -1],
        [0, 1, -1, 0],
        [0, -1, -1, 0],
        [-1, 0, 0, -1],
        [-1, 0, 0, 1],
        [0, -1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
    ],
    dtype=np.int64,
)


class StackOverflowInFOV(RuntimeError):
    """The scan stack was exhausted, so the returned mask is incomplete."""


def new_scratch(frames: int = SCRATCH_FRAMES) -> np.ndarray:
    """Allocate scan-stack scratch. One buffer can be reused across calls."""
    return np.empty((frames, 3), dtype=np.float64)


@njit(cache=True, inline="always")
def _opaque(
    blocks_vision: np.ndarray,
    brush_id: np.ndarray,
    j: int,
    i: int,
    src_brush: int,
) -> bool:
    """Does this cell stop a line of vision from a source in `src_brush`?

    Walls always do. Brush does too, but only brush the source is not standing in:
    "brush is opaque towards vision when viewed from the outside inwards and not
    the reverse", and allies inside a brush share vision of it. That conditionality
    is what makes brush interesting and is why the table is keyed by
    (cell, brush_id) rather than by cell alone.
    """
    if blocks_vision[j, i]:
        return True
    b = brush_id[j, i]
    return b >= 0 and b != src_brush


@njit(cache=True, inline="always")
def _lightable(brush_id: np.ndarray, j: int, i: int, src_brush: int) -> bool:
    """May this cell be marked visible?

    Walls may: a wall face is visible, no unit can stand there, and lighting it
    in-scan preserves radius separability.

    Foreign brush may **not**, and this is the single most consequential line in
    the module. Marking a brush cell visible from outside would mean an enemy
    hiding in that brush counted as seen, inverting the central mechanic of
    jungle and river play, and quietly inflating every fog-agreement number.
    """
    b = brush_id[j, i]
    return b < 0 or b == src_brush


@njit(cache=True)
def fov_into(
    out: np.ndarray,  # bool[window, window], modified in place
    blocks_vision: np.ndarray,  # bool[G, G]
    brush_id: np.ndarray,  # int16[G, G]
    si: int,
    sj: int,
    src_brush: int,
    r_cells_sq: float,
    half: int,
    scratch: np.ndarray,  # float64[frames, 3]
) -> int:
    """Compute field of view from (si, sj) into a window-local boolean mask.

    `out` is indexed [dj + half, di + half]. It is *not* cleared. The caller owns
    that, so a scratch window can be reused without a fresh allocation.

    Returns the high-water mark of the scan stack, for headroom measurement. A
    return value equal to `scratch.shape[0]` means the stack overflowed and the
    mask is incomplete; `fov_bool` turns that into an exception.
    """
    grid_h, grid_w = blocks_vision.shape
    capacity = scratch.shape[0]
    max_depth = 0

    # The source cell is always visible to itself.
    out[half, half] = True

    radius = half

    for oct_idx in range(8):
        xx = _OCTANTS[oct_idx, 0]
        xy = _OCTANTS[oct_idx, 1]
        yx = _OCTANTS[oct_idx, 2]
        yy = _OCTANTS[oct_idx, 3]

        top = 1
        scratch[0, 0] = 1.0  # row
        scratch[0, 1] = 1.0  # start slope
        scratch[0, 2] = 0.0  # end slope

        while top > 0:
            if top > max_depth:
                max_depth = top
            top -= 1
            row = int(scratch[top, 0])
            start = scratch[top, 1]
            end = scratch[top, 2]
            if start < end:
                continue

            blocked = False
            new_start = 0.0

            for jj in range(row, radius + 1):
                dy = -jj
                dx = -jj - 1
                while dx <= 0:
                    dx += 1
                    gi = si + dx * xx + dy * xy
                    gj = sj + dx * yx + dy * yy

                    # Slopes of this cell's near and far extremities.
                    l_slope = (dx - 0.5) / (dy + 0.5)
                    r_slope = (dx + 0.5) / (dy - 0.5)

                    if start < r_slope:
                        continue
                    if end > l_slope:
                        break

                    inside = 0 <= gi < grid_w and 0 <= gj < grid_h
                    # Off-grid counts as opaque: the map border is a wall, and
                    # treating it as open would let vision wrap around the edge.
                    cell_opaque = True
                    if inside:
                        cell_opaque = _opaque(blocks_vision, brush_id, gj, gi, src_brush)
                        if dx * dx + dy * dy <= r_cells_sq and _lightable(
                            brush_id, gj, gi, src_brush
                        ):
                            # Window offsets are the GRID offsets, so the scan-space
                            # (dx, dy) must go through the octant transform first.
                            # Indexing by (dx, dy) directly silently collapses all
                            # eight octants onto one, which is exactly the shape of
                            # bug this module's empty-grid test exists to catch.
                            out[gj - sj + half, gi - si + half] = True

                    if blocked:
                        if cell_opaque:
                            new_start = r_slope
                            continue
                        blocked = False
                        start = new_start
                    elif cell_opaque and jj < radius:
                        blocked = True
                        # Defer the child scan. Order does not matter: each frame
                        # owns a disjoint slope wedge and `start` is passed by
                        # value, so running children after the parent's row loop
                        # yields the identical union of lit cells.
                        if top < capacity:
                            scratch[top, 0] = jj + 1
                            scratch[top, 1] = start
                            scratch[top, 2] = l_slope
                            top += 1
                        else:
                            return capacity  # overflow; mask is incomplete
                        new_start = r_slope

                if blocked:
                    break

    return max_depth


def fov_bool(
    terrain: Terrain,
    i: int,
    j: int,
    radius_units: float,
    half: int = RMAX_CELLS,
    src_brush: int | None = None,
    out: np.ndarray | None = None,
    scratch: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience wrapper: field of view as a fresh boolean window.

        `src_brush` defaults to the brush the source cell sits in. Callers with a
        continuous position should pass the brush determined from that position rather
        than relying on the cell, because a champion 10 units inside a brush can snap
        to a cell classified as non-brush, and brush transparency is a discrete switch
    the resulting error is not small.
    """
    from shadowcast.geom.grid import radius_cells_sq

    window = 2 * half + 1
    if out is None:
        out = np.zeros((window, window), dtype=bool)
    else:
        out.fill(False)
    if scratch is None:
        scratch = new_scratch()
    if src_brush is None:
        src_brush = int(terrain.brush_id[j, i])

    depth = fov_into(
        out,
        terrain.blocks_vision,
        terrain.brush_id,
        int(i),
        int(j),
        int(src_brush),
        radius_cells_sq(radius_units),
        int(half),
        scratch,
    )
    if depth >= scratch.shape[0]:
        raise StackOverflowInFOV(
            f"scan stack of {scratch.shape[0]} frames exhausted at cell ({i}, {j}); "
            "the mask is incomplete. Allocate more frames via new_scratch()."
        )
    return out
