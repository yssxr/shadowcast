"""Classifying *how* a belief is wrong, not just how much.

A calibration number says the 90% region contains the truth 43% of the time. It does not
say why, and the two possibilities need opposite fixes:

**Collapse**: the truth is outside the particle cloud's support entirely. The filter
killed the correct hypothesis and cannot recover it. That is a defect in the machinery:
weights, resampling, or an over-aggressive negative update.

**Drift**: the truth is inside the support, or near it, but the cloud's mass has moved
somewhere else. The machinery is fine and the *motion model* believes champions go
somewhere they do not. That is a modelling error, and no amount of filter work fixes it.

The distinction is worth a module because getting it backwards costs weeks. It also has a
particular trap attached: drift can always be reduced by teaching the motion model more
about the scenario it is being scored against, and on synthetic data that is fitting to
the generator, which measures nothing. So this reports the classification and leaves the
conclusion to whoever is reading, rather than optimising against it.

Run it on real data and the same output means something.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast.config import FilterSpec
from shadowcast.l3_infer.metrics import LatticeIndex
from shadowcast.l3_infer.pf import BeliefFilter
from shadowcast.l3_infer.policy import NO_CELL, Observation, PublicInfo, TruthTable
from shadowcast.terrain.terrain import Terrain

__all__ = ["DARKNESS_BANDS", "BeliefDiagnostic", "diagnose_belief"]

#: How long the enemy has been unseen, in seconds. The interesting structure is which
#: band the belief fails in, not the average across all of them.
DARKNESS_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 5.0),
    (5.0, 15.0),
    (15.0, 30.0),
    (30.0, 60.0),
    (60.0, float("inf")),
)


@dataclass(frozen=True, slots=True)
class BeliefDiagnostic:
    """Where the truth sits relative to the cloud, broken down by darkness."""

    scored: int
    #: Fraction of scored moments where a particle shares the truth's lattice bin.
    in_support: float
    #: When it does, how the truth's bin ranks by mass among occupied bins. 0 is the peak.
    median_rank: float
    #: Distance to the closest particle, small means the cloud covers the right ground.
    nearest_percentiles: dict[int, float]
    #: Distance to the cloud's centre of mass, large with a small nearest means DRIFT.
    centroid_percentiles: dict[int, float]
    #: `(label, n, median nearest, median centroid)` per darkness band.
    by_darkness: list[tuple[str, int, float, float]] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Which defect this looks like, stated in one word plus the reason."""
        near = self.nearest_percentiles.get(50, 0.0)
        centre = self.centroid_percentiles.get(50, 0.0)
        if self.in_support > 0.4 and centre > 4 * max(near, 1.0):
            return "drift: the cloud covers the right ground and puts its mass elsewhere"
        if self.in_support < 0.2:
            return "collapse: the truth is usually outside the cloud entirely"
        return "mixed, neither drift nor collapse dominates"

    def describe(self) -> dict[str, Any]:
        return {
            "scored": self.scored,
            "truth_in_support": round(self.in_support, 4),
            "median_density_rank": round(self.median_rank, 3),
            "nearest_particle_p50": round(self.nearest_percentiles.get(50, 0.0)),
            "nearest_particle_p90": round(self.nearest_percentiles.get(90, 0.0)),
            "centroid_error_p50": round(self.centroid_percentiles.get(50, 0.0)),
            "centroid_error_p90": round(self.centroid_percentiles.get(90, 0.0)),
            "verdict": self.verdict,
        }


def diagnose_belief(
    spec: FilterSpec,
    terrain: Terrain,
    obs: Observation,
    public: PublicInfo,
    truth: TruthTable,
    masks: Iterator[tuple[int, np.ndarray, np.ndarray]],
    lattice: LatticeIndex | None = None,
    stride: int = 8,
) -> BeliefDiagnostic:
    """Run one filter and measure where the truth falls relative to its cloud."""
    lattice = lattice or LatticeIndex(terrain)
    grid = terrain.grid
    filt = BeliefFilter(spec, terrain)

    in_support = 0
    total = 0
    ranks: list[float] = []
    nearest: list[float] = []
    centroid: list[float] = []
    darkness: list[float] = []
    last_seen = np.zeros((C.N_TEAMS, C.N_ENEMIES))

    for belief in filt.run(obs, public, masks):
        t = belief.tick / C.TICK_HZ
        for o in range(C.N_TEAMS):
            for e in range(C.N_ENEMIES):
                if belief.seen[o, e]:
                    last_seen[o, e] = t
                    continue
                if not belief.alive[o, e] or belief.tick % stride:
                    continue
                truth_cell = int(truth.cell[belief.tick, o, e])
                if truth_cell == NO_CELL:
                    continue

                cells = belief.cell[o, e]
                w = np.exp(belief.logw[o, e] - belief.logw[o, e].max())
                w /= w.sum()
                total += 1

                truth_bin = int(lattice.bin_of_cell[truth_cell])
                bins = lattice.bin_of_cell[cells]
                if (bins == truth_bin).any():
                    in_support += 1
                    mass = np.bincount(bins, weights=w, minlength=lattice.lattice**2)
                    occupied = np.sort(mass[mass > 0])[::-1]
                    ranks.append(
                        float(np.searchsorted(-occupied, -mass[truth_bin]) / occupied.size)
                    )

                tj, ti = divmod(truth_cell, grid)
                pj, pi = np.divmod(cells, grid)
                d = np.hypot(pj - tj, pi - ti) * C.GRID_CELL_SIZE
                nearest.append(float(d.min()))
                centroid.append(
                    float(np.hypot((pj * w).sum() - tj, (pi * w).sum() - ti) * C.GRID_CELL_SIZE)
                )
                darkness.append(t - last_seen[o, e])

    near = np.asarray(nearest)
    cent = np.asarray(centroid)
    dark = np.asarray(darkness)

    bands: list[tuple[str, int, float, float]] = []
    for lo, hi in DARKNESS_BANDS:
        hit = (dark >= lo) & (dark < hi)
        if hit.sum() < 20:
            continue
        label = f"{lo:.0f}-{hi:.0f}s" if np.isfinite(hi) else f"{lo:.0f}s+"
        bands.append(
            (label, int(hit.sum()), float(np.median(near[hit])), float(np.median(cent[hit])))
        )

    pct = (50, 75, 90, 99)
    return BeliefDiagnostic(
        scored=total,
        in_support=in_support / max(total, 1),
        median_rank=float(np.median(ranks)) if ranks else float("nan"),
        nearest_percentiles={q: float(np.percentile(near, q)) for q in pct} if near.size else {},
        centroid_percentiles={q: float(np.percentile(cent, q)) for q in pct} if cent.size else {},
        by_darkness=bands,
    )
