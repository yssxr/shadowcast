"""Where an enemy could have walked to by now.

When a champion disappears, the set of places they might be is a **geodesic** ball
around the last sighting, not a Euclidean one, and on Summoner's Rift the difference is
most of the map. A Euclidean disc drawn around a champion who vanished into the tri-brush
includes the whole enemy jungle across a wall they would need eight seconds to walk
around. Filters that grow circles are wrong in a way that is invisible until you overlay
the terrain.

This module answers the question once per sighting and then answers it in constant time
per particle, which is what makes it affordable: the field is computed by one Dijkstra
sweep, the cells are sorted by distance, and "sample a cell within radius r" becomes a
`searchsorted` plus a random index.

**The lattice is coarse on purpose.** 128² at 115 u/cell, not the 512² the rest of the
engine uses. A reachability set is a claim about which *region* an enemy could be in, and
115 u is already finer than that question deserves — while Dijkstra over 16k cells is
some sixty times cheaper than over 262k, which matters because the geodesic baseline
wants a fresh field at every sighting and there are hundreds of those per match.
A coarse cell counts as walkable if *any* fine cell inside it is, because the opposite
rule would close one-cell corridors that the map genuinely has and make parts of the
jungle unreachable — a reachability set that is too small breaks calibration outright,
while one that is too large only costs sharpness.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from shadowcast import constants as C
from shadowcast.geom.path import STEP_COST, UNREACHABLE, geodesic_field
from shadowcast.terrain.terrain import Terrain

__all__ = ["ReachabilityIndex"]

#: How many fields to keep. Each is 64 KB, so this is 4 MB — small next to the cost of
#: recomputing one, and a match revisits sighting locations constantly (lanes, camps,
#: the fountain), so the hit rate is high.
CACHE_ENTRIES = 64


class ReachabilityIndex:
    """Geodesic balls on a coarse lattice, cached per seed cell."""

    def __init__(self, terrain: Terrain, lattice: int = C.REACH_LATTICE) -> None:
        if C.GRID % lattice:
            raise ValueError(f"lattice {lattice} must divide the engine grid {C.GRID}")
        self.terrain = terrain
        self.lattice = lattice
        self.block = C.GRID // lattice
        self.cell_size = C.WORLD_SPAN / lattice

        fine = terrain.walkable.reshape(lattice, self.block, lattice, self.block)
        self.walkable = fine.any(axis=(1, 3))
        self.n_coarse = lattice * lattice

        # Fine cells grouped by their coarse parent, as a CSR so that picking a random
        # fine cell inside a coarse one is two array reads.
        jj, ii = np.nonzero(terrain.walkable)
        parent = (jj // self.block) * lattice + (ii // self.block)
        order = np.argsort(parent, kind="stable")
        self._fine_cells = (jj[order] * C.GRID + ii[order]).astype(np.int32)
        counts = np.bincount(parent, minlength=self.n_coarse)
        self._fine_offsets = np.zeros(self.n_coarse + 1, dtype=np.int64)
        np.cumsum(counts, out=self._fine_offsets[1:])

        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    # -- lattice conversion -------------------------------------------
    def coarse_of_fine(self, cell: int) -> int:
        j, i = divmod(int(cell), C.GRID)
        return (j // self.block) * self.lattice + (i // self.block)

    def _units(self, scaled: np.ndarray) -> np.ndarray:
        return scaled.astype(np.float64) * (self.cell_size / STEP_COST)

    # -- the field ----------------------------------------------------
    def field(self, coarse_seed: int) -> tuple[np.ndarray, np.ndarray]:
        """`(cells sorted by distance, their distances in world units)`.

        Unreachable cells are dropped rather than kept with a sentinel, so a
        `searchsorted` on the distances never has to reason about the sentinel's
        magnitude.
        """
        cached = self._cache.get(coarse_seed)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(coarse_seed)
            return cached
        self.misses += 1

        seed = np.array([coarse_seed], dtype=np.int32)
        if not self.walkable.reshape(-1)[coarse_seed]:
            seed = self._nearest_walkable_coarse(coarse_seed)
        dist = geodesic_field(self.walkable, seed)
        reachable = np.nonzero(dist != UNREACHABLE)[0]
        order = np.argsort(dist[reachable], kind="stable")
        cells = reachable[order].astype(np.int32)
        units = self._units(dist[cells])

        self._cache[coarse_seed] = (cells, units)
        if len(self._cache) > CACHE_ENTRIES:
            self._cache.popitem(last=False)
        return cells, units

    def _nearest_walkable_coarse(self, coarse: int) -> np.ndarray:
        """A seed for a sighting that snapped into a wall.

        Reconstructed positions land in non-walkable cells occasionally — dashes over
        walls, and the 28.8 u quantisation near a wall face. Refusing to produce a field
        would leave the filter with no prior at all, which is strictly worse than
        seeding from the walkable cell next door.
        """
        j0, i0 = divmod(coarse, self.lattice)
        flat = self.walkable.reshape(-1)
        for radius in range(1, self.lattice):
            found = []
            for dj in range(-radius, radius + 1):
                for di in range(-radius, radius + 1):
                    if max(abs(dj), abs(di)) != radius:
                        continue
                    j, i = j0 + dj, i0 + di
                    if not (0 <= j < self.lattice and 0 <= i < self.lattice):
                        continue
                    if flat[j * self.lattice + i]:
                        found.append(j * self.lattice + i)
            if found:
                return np.array(found, dtype=np.int32)
        raise ValueError("no walkable cell anywhere on the reachability lattice")

    # -- sampling -----------------------------------------------------
    def sample(
        self,
        seed_cell: int,
        radius_units: float,
        n: int,
        rnd: np.ndarray,
        blocked: np.ndarray | None = None,
    ) -> np.ndarray:
        """`n` fine cells drawn uniformly from the geodesic ball, minus `blocked`.

        `rnd` is pre-drawn uniform noise of shape `(n, 2)`: one draw to pick the coarse
        cell, one to pick a fine cell inside it. Randomness is passed in rather than
        generated so that the NumPy and Numba paths, and successive runs, are
        bit-comparable — the barrier test compares outputs exactly.

        `blocked` is the observer's current visibility as a flat boolean over fine
        cells. Excluding it *is* the negative information, applied here as a hard
        constraint rather than a weight because a reinitialised particle has no history
        to contradict: we know the enemy is not somewhere we are looking right now.
        """
        if rnd.shape != (n, 2):
            raise ValueError(f"rnd must be ({n}, 2) pre-drawn uniforms, got {rnd.shape}")
        cells, units = self.field(self.coarse_of_fine(seed_cell))
        k = int(np.searchsorted(units, max(radius_units, self.cell_size), side="right"))
        k = max(k, 1)

        pick = (rnd[:, 0] * k).astype(np.int64)
        np.clip(pick, 0, k - 1, out=pick)
        coarse = cells[pick]

        lo = self._fine_offsets[coarse]
        hi = self._fine_offsets[coarse + 1]
        span = np.maximum(hi - lo, 1)
        idx = lo + (rnd[:, 1] * span).astype(np.int64)
        np.clip(idx, 0, self._fine_cells.size - 1, out=idx)
        out = self._fine_cells[idx]

        if blocked is not None and k > 1:
            # One rejection pass, not a loop. A loop would be unbounded when the enemy
            # is genuinely cornered inside the visible region, and the count of
            # survivors is more useful than a guarantee: it says how much of the
            # reachable set the observer has actually eliminated.
            bad = blocked[out]
            if bad.any():
                allowed = cells[:k]
                keep = ~blocked[self._fine_cells[self._fine_offsets[allowed]]]
                if keep.any():
                    pool = allowed[keep]
                    repick = (rnd[bad, 0] * pool.size).astype(np.int64)
                    np.clip(repick, 0, pool.size - 1, out=repick)
                    chosen = pool[repick]
                    out[bad] = self._fine_cells[self._fine_offsets[chosen]]
        return out.astype(np.int32)

    def describe(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        return {
            "lattice": self.lattice,
            "cell_size": round(self.cell_size, 2),
            "walkable_cells": int(self.walkable.sum()),
            "field_cache_hit_rate": round(self.hits / total, 4) if total else 0.0,
            "dijkstra_runs": self.misses,
        }
