"""Pathfinding and distance fields on the walkable grid.

Two things, both needed downstream for different reasons.

`astar` gives point-to-point paths. The synthetic match generator uses it so its
champions move on navmesh-legal routes — which matters because the belief filter
later constrains particles to the navmesh, and a ground truth that walked through
walls would make the filter look wrong when it was right.

`geodesic_field` gives distance-from-a-set over the whole map in one sweep. That is
what the belief filter's reachability set needs: "every cell an enemy could have
reached in the 12 seconds since we last saw them" is a geodesic ball, not a
Euclidean one, and the difference is most of the map when the last sighting was
next to a wall.

Distances are 8-connected with octile weights, held as scaled integers (70 for a
step, 99 for a diagonal — 99/70 = 1.41428, within 0.01% of sqrt 2). Integer weights
let the field use a bucket queue instead of a heap, and they make the metric exactly
reproducible rather than dependent on float summation order.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "DIAG_COST",
    "STEP_COST",
    "UNREACHABLE",
    "astar",
    "chord_walkable",
    "diagonal_ok",
    "field_to_units",
    "geodesic_field",
    "nearest_walkable",
    "simplify_path",
]

STEP_COST = 70
DIAG_COST = 99
#: Sentinel for cells no path reaches. Chosen so `field < budget` is safe without a
#: separate reachability mask — nothing can be closer than a real distance.
UNREACHABLE = np.int32(2**30)

# (dj, di, cost) — 8-connected.
_NEIGHBOURS = np.array(
    [
        (-1, 0, STEP_COST),
        (1, 0, STEP_COST),
        (0, -1, STEP_COST),
        (0, 1, STEP_COST),
        (-1, -1, DIAG_COST),
        (-1, 1, DIAG_COST),
        (1, -1, DIAG_COST),
        (1, 1, DIAG_COST),
    ],
    dtype=np.int32,
)


@njit(cache=True, inline="always")
def diagonal_ok(walkable: np.ndarray, j: int, i: int, dj: int, di: int) -> bool:
    """A diagonal move requires BOTH of its orthogonal neighbours to be open.

    This is the strict rule, and the permissive variant (either neighbour suffices) was
    tried first and abandoned for two reasons.

    Geometrically it is wrong: the straight line between the two cell centres passes
    through one of the orthogonal cells, so if that cell is a wall the move clips
    terrain. `chord_walkable` traverses exactly and rejects such a chord, and the
    disagreement showed up as 14 synthetic ground-truth positions per match sitting
    inside walls — routes a navmesh-constrained belief filter could never explain.

    Physically it is also wrong for League: a one-cell gap here is 28.8 world units,
    well under a champion's ~65-unit collision radius, so squeezing through is not
    something the game permits either. The strict rule sealing such gaps is the more
    faithful behaviour, not a limitation.
    """
    if dj == 0 or di == 0:
        return True
    return walkable[j + dj, i] and walkable[j, i + di]


@njit(cache=True, inline="always")
def _heap_push(heap_k, heap_f, size, k, f):
    heap_k[size] = k
    heap_f[size] = f
    c = size
    while c > 0:
        par = (c - 1) // 2
        if heap_f[par] <= heap_f[c]:
            break
        heap_k[par], heap_k[c] = heap_k[c], heap_k[par]
        heap_f[par], heap_f[c] = heap_f[c], heap_f[par]
        c = par
    return size + 1


@njit(cache=True, inline="always")
def _heap_pop(heap_k, heap_f, size):
    k = heap_k[0]
    size -= 1
    heap_k[0] = heap_k[size]
    heap_f[0] = heap_f[size]
    p = 0
    while True:
        left = 2 * p + 1
        right = left + 1
        m = p
        if left < size and heap_f[left] < heap_f[m]:
            m = left
        if right < size and heap_f[right] < heap_f[m]:
            m = right
        if m == p:
            break
        heap_k[p], heap_k[m] = heap_k[m], heap_k[p]
        heap_f[p], heap_f[m] = heap_f[m], heap_f[p]
        p = m
    return k, size


@njit(cache=True)
def geodesic_field(walkable: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Scaled-integer geodesic distance from the nearest of `seeds` (flat cells).

    Plain Dijkstra with a binary heap. A bucket queue would be asymptotically better
    given the two small integer weights, but it needs either 100 buckets sized for
    the worst-case frontier (~105 MB at this grid) or a linked-list arena, and this
    runs in tens of milliseconds — the wrong thing to optimise before a caller exists
    that cares.
    """
    h, w = walkable.shape
    n = h * w
    dist = np.full(n, UNREACHABLE, dtype=np.int32)
    done = np.zeros(n, dtype=np.bool_)

    cap = 8 * n  # each cell can be improved at most once per incoming edge
    heap_k = np.empty(cap, dtype=np.int32)
    heap_f = np.empty(cap, dtype=np.int32)
    size = 0

    for s in range(seeds.shape[0]):
        k = seeds[s]
        if walkable[k // w, k % w] and dist[k] != 0:
            dist[k] = 0
            size = _heap_push(heap_k, heap_f, size, k, 0)

    while size > 0:
        k, size = _heap_pop(heap_k, heap_f, size)
        if done[k]:
            continue
        done[k] = True
        d = dist[k]
        j = k // w
        i = k % w

        for t in range(_NEIGHBOURS.shape[0]):
            dj = _NEIGHBOURS[t, 0]
            di = _NEIGHBOURS[t, 1]
            nj = j + dj
            ni = i + di
            if nj < 0 or nj >= h or ni < 0 or ni >= w:
                continue
            if not walkable[nj, ni]:
                continue
            if not diagonal_ok(walkable, j, i, dj, di):
                continue
            nk = nj * w + ni
            if done[nk]:
                continue
            nd = d + _NEIGHBOURS[t, 2]
            if nd < dist[nk] and size < cap:
                dist[nk] = nd
                size = _heap_push(heap_k, heap_f, size, nk, nd)

    return dist


@njit(cache=True, inline="always")
def _octile(j0: int, i0: int, j1: int, i1: int) -> int:
    dj = abs(j1 - j0)
    di = abs(i1 - i0)
    lo = dj if dj < di else di
    hi = di if dj < di else dj
    return lo * DIAG_COST + (hi - lo) * STEP_COST


@njit(cache=True)
def _astar(walkable: np.ndarray, start: int, goal: int) -> np.ndarray:
    """A* with a manual binary heap. Returns flat cells start..goal, or empty."""
    h, w = walkable.shape
    n = h * w
    g = np.full(n, UNREACHABLE, dtype=np.int32)
    came = np.full(n, -1, dtype=np.int32)
    closed = np.zeros(n, dtype=np.bool_)

    cap = 8 * n
    heap_k = np.empty(cap, dtype=np.int32)
    heap_f = np.empty(cap, dtype=np.int32)

    gj, gi = goal // w, goal % w
    g[start] = 0
    size = _heap_push(heap_k, heap_f, 0, start, _octile(start // w, start % w, gj, gi))

    while size > 0:
        k, size = _heap_pop(heap_k, heap_f, size)
        if closed[k]:
            continue
        closed[k] = True
        if k == goal:
            break

        j = k // w
        i = k % w
        for t in range(_NEIGHBOURS.shape[0]):
            dj = _NEIGHBOURS[t, 0]
            di = _NEIGHBOURS[t, 1]
            nj = j + dj
            ni = i + di
            if nj < 0 or nj >= h or ni < 0 or ni >= w:
                continue
            if not walkable[nj, ni]:
                continue
            if not diagonal_ok(walkable, j, i, dj, di):
                continue
            nk = nj * w + ni
            if closed[nk]:
                continue
            ng = g[k] + _NEIGHBOURS[t, 2]
            if ng < g[nk] and size < cap:
                g[nk] = ng
                came[nk] = k
                size = _heap_push(heap_k, heap_f, size, nk, ng + _octile(nj, ni, gj, gi))

    if g[goal] >= UNREACHABLE:
        return np.empty(0, dtype=np.int32)

    length = 0
    k = goal
    while k != -1:
        length += 1
        k = came[k]
    out = np.empty(length, dtype=np.int32)
    k = goal
    for q in range(length - 1, -1, -1):
        out[q] = k
        k = came[k]
    return out


def astar(walkable: np.ndarray, start: int, goal: int) -> np.ndarray:
    """Shortest 8-connected walkable path between two flat cells.

    Returns cells from `start` to `goal` inclusive, or an empty array if no path
    exists — which callers must handle rather than assume, since a mis-snapped
    endpoint inside a wall is the usual cause.
    """
    walkable = np.ascontiguousarray(walkable, dtype=np.bool_)
    h, w = walkable.shape
    for name, k in (("start", start), ("goal", goal)):
        if not (0 <= k < h * w):
            raise ValueError(f"{name} cell {k} out of range for a {h}x{w} grid")
        if not walkable[k // w, k % w]:
            raise ValueError(f"{name} cell {k} is not walkable")
    return _astar(walkable, int(start), int(goal))


@njit(cache=True)
def chord_walkable(walkable: np.ndarray, j0: int, i0: int, j1: int, i1: int) -> bool:
    """Is the straight line between two cell centres entirely on walkable ground?

    Exact voxel traversal (Amanatides & Woo), not point sampling: it visits every cell
    the segment passes through, including cells it merely clips at a corner.

    That exactness is the point. A sampled version at 0.25-cell steps looks adequate
    and is not — a segment can cut the corner of a wall cell without any sample
    landing inside it, which put roughly 15 synthetic ground-truth positions per match
    inside terrain even though every chord had supposedly been verified. Those are
    exactly the positions a navmesh-constrained belief filter could never explain.
    """
    h, w = walkable.shape
    if not (0 <= i0 < w and 0 <= j0 < h and 0 <= i1 < w and 0 <= j1 < h):
        return False
    if not walkable[j0, i0] or not walkable[j1, i1]:
        return False

    # Cell centres in cell-unit space, matching `geom.grid.world_to_cell`, where the
    # cell containing a point is its truncation.
    x = i0 + 0.5
    y = j0 + 0.5
    dx = (i1 + 0.5) - x
    dy = (j1 + 0.5) - y

    ci = i0
    cj = j0
    step_i = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_j = 1 if dy > 0 else (-1 if dy < 0 else 0)

    # Parametric distance to the next cell boundary on each axis, and between them.
    big = 1.0e30
    t_max_i = big
    t_delta_i = big
    if step_i != 0:
        next_i = ci + (1 if step_i > 0 else 0)
        t_max_i = (next_i - x) / dx
        t_delta_i = abs(1.0 / dx)
    t_max_j = big
    t_delta_j = big
    if step_j != 0:
        next_j = cj + (1 if step_j > 0 else 0)
        t_max_j = (next_j - y) / dy
        t_delta_j = abs(1.0 / dy)

    # Bounded by the Manhattan cell distance, which is how many boundary crossings a
    # straight segment can make.
    max_steps = abs(i1 - i0) + abs(j1 - j0) + 2
    for _ in range(max_steps):
        if ci == i1 and cj == j1:
            return True
        if t_max_i < t_max_j:
            ci += step_i
            t_max_i += t_delta_i
        else:
            cj += step_j
            t_max_j += t_delta_j
        if ci < 0 or ci >= w or cj < 0 or cj >= h:
            return False
        if not walkable[cj, ci]:
            return False
    return True


def simplify_path(walkable: np.ndarray, cells: np.ndarray, max_points: int = 8) -> np.ndarray:
    """Reduce a dense path to at most `max_points` cells, keeping every chord walkable.

    Greedy: from the current vertex, reach as far ahead as a legal straight chord
    allows, then repeat. This is what a movement order actually looks like — a client
    sends a few waypoints, not a cell-by-cell route — and the walkability guarantee is
    what keeps the shortcut honest.

    If the greedy pass still needs more than `max_points`, the result is **truncated**
    — the first `max_points` vertices are kept and the destination is dropped.

    Subsampling to fit the budget was the first attempt and it is wrong: evenly
    spaced vertices from a legal chain create new chords that were never checked, so
    the returned path cuts wall corners while appearing to have been verified. A real
    movement order is also finite; a client sends what fits and the unit is ordered
    again on arrival. Truncation matches that and keeps every chord legal.
    """
    walkable = np.ascontiguousarray(walkable, dtype=np.bool_)
    w = walkable.shape[1]
    if cells.size <= 2:
        return cells.copy()

    js, is_ = np.divmod(cells, w)
    keep = [0]
    cursor = 0
    n = len(cells)
    while cursor < n - 1:
        # Falling back to the immediate successor is safe because the path came from
        # A* with the strict diagonal rule, so every adjacent step is a legal chord.
        # Under the permissive rule it was NOT safe: an adjacent diagonal step could
        # clip a wall corner, and this loop would then keep an unverified chord.
        best = cursor + 1
        # Binary search would assume monotone visibility, which corners violate.
        for cand in range(n - 1, cursor, -1):
            if chord_walkable(walkable, js[cursor], is_[cursor], js[cand], is_[cand]):
                best = cand
                break
        keep.append(best)
        cursor = best
    out = cells[np.array(keep, dtype=np.int64)]
    return out[:max_points] if out.size > max_points else out


def field_to_units(field: np.ndarray, cell_size: float) -> np.ndarray:
    """Convert a scaled-integer distance field to world units, NaN where unreachable."""
    out = field.astype(np.float64) * (cell_size / STEP_COST)
    out[field >= UNREACHABLE] = np.nan
    return out


def nearest_walkable(walkable: np.ndarray, j: int, i: int, max_radius: int = 24) -> tuple[int, int]:
    """Nearest walkable cell to (j, i), searched outward in rings.

    Used to snap hand-authored landmark coordinates onto legal ground. Raises rather
    than returning the input, because a landmark that cannot be snapped is a data
    error and silently leaving it in a wall would put a vision source inside terrain.
    """
    h, w = walkable.shape
    if 0 <= j < h and 0 <= i < w and walkable[j, i]:
        return int(j), int(i)
    for r in range(1, max_radius + 1):
        best = None
        best_d = None
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if max(abs(dj), abs(di)) != r:
                    continue
                nj, ni = j + dj, i + di
                if 0 <= nj < h and 0 <= ni < w and walkable[nj, ni]:
                    d = dj * dj + di * di
                    if best_d is None or d < best_d:
                        best_d = d
                        best = (nj, ni)
        if best is not None:
            return int(best[0]), int(best[1])
    raise ValueError(f"no walkable cell within {max_radius} of ({j}, {i})")
