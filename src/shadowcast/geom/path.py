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
    "STEP_COST",
    "DIAG_COST",
    "UNREACHABLE",
    "astar",
    "geodesic_field",
    "field_to_units",
    "nearest_walkable",
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
def _diagonal_ok(walkable: np.ndarray, j: int, i: int, dj: int, di: int) -> bool:
    """Forbid cutting a diagonal between two walls.

    Without this a unit slips through the join of two wall corners, which on
    Summoner's Rift means walking through the point where two jungle walls meet.
    Every such shortcut would then appear in the ground truth as a legal route the
    belief filter's navmesh-constrained motion could never reproduce.
    """
    if dj == 0 or di == 0:
        return True
    return walkable[j + dj, i] or walkable[j, i + di]


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
            if not _diagonal_ok(walkable, j, i, dj, di):
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
            if not _diagonal_ok(walkable, j, i, dj, di):
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
