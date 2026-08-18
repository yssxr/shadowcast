"""Scoring a belief, and the units that make the score mean something.

This is where truth is finally allowed in, and it is deliberately a separate module from
the filter for that reason, `pf.run` yields beliefs, `evaluate` zips them against a
`TruthTable`, and the two never share a scope.

Four numbers, and the reason there are four is that no one of them is sufficient:

**Entropy** is what the design puts on screen, but it is a measurement against a chosen
lattice and a chosen particle budget, so it is comparable only within this project.

**Credible-region area** is the same information in units a reader can check against the
map, and it does not move when the lattice or the particle count changes. It is the
primary number in the write-up for exactly that reason.

**Negative log-likelihood** of the truth is what ranks models, because it is the only one
of the four that punishes a confident wrong answer. Entropy rewards it. It is computed
against the same frozen lattice as the entropy rather than against a kernel density
estimate, which was the earlier plan. Two reasons: a KDE needs a bandwidth, and a
bandwidth is a knob that can be turned until the preferred model wins; and a 300-unit
kernel smooths probability straight through walls, handing mass back to exactly the
off-navmesh region the terrain-aware models spent their effort excluding. Sharing one
reference measure across all four metrics also means entropy, area, likelihood and
calibration cannot quietly disagree about what space they are defined on.

**Calibration** is what stops NLL being gamed by a filter that is sharp and wrong. A
model with lower entropy and broken calibration is overconfident garbage, and there is no
way to tell from entropy alone, so a reliability curve is reported next to every score,
and a model that beats another on NLL while sitting off-diagonal has not won anything.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast.config import FilterSpec
from shadowcast.l3_infer.pf import TickBelief
from shadowcast.l3_infer.policy import NO_CELL, TruthTable
from shadowcast.terrain.terrain import Terrain

__all__ = [
    "CALIBRATION_LEVELS",
    "BeliefScore",
    "LatticeIndex",
    "belief_summary",
    "evaluate",
]

#: Credible levels the reliability curve is reported at. A perfectly calibrated filter's
#: truth falls inside its `q` region exactly `q` of the time, for every one of these.
CALIBRATION_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

#: Area unit. One "ku" is a thousand game units, so Summoner's Rift, 14,759 units on a
#: side: is 217.8 ku². Named rather than called km² because a game unit is not a metre
#: and quietly implying it is would make every area figure a small lie.
UNITS_PER_KU = 1000.0


class LatticeIndex:
    """The frozen entropy lattice: fine cells to coarse bins, walkable bins only.

    Entropy in bits is defined only against a reference measure, so this object *is* part
    of the metric's definition rather than an implementation detail. It is hashed into
    every artifact header, and a number computed against a different lattice is not
    comparable to one computed against this one and must never be averaged with it.
    """

    def __init__(
        self,
        terrain: Terrain,
        lattice: int = C.ENTROPY_LATTICE,
        smoothing: float = C.SCORING_SMOOTHING,
    ) -> None:
        if C.GRID % lattice:
            raise ValueError(f"entropy lattice {lattice} must divide the grid {C.GRID}")
        self.lattice = lattice
        self.block = C.GRID // lattice
        self.cell_size = C.WORLD_SPAN / lattice
        self.bin_area_ku2 = (self.cell_size / UNITS_PER_KU) ** 2

        coarse = terrain.walkable.reshape(lattice, self.block, lattice, self.block)
        self.walkable = coarse.any(axis=(1, 3)).reshape(-1)
        self.n_bins = int(self.walkable.sum())
        self.smoothing = smoothing

        # Fine cell -> lattice bin, dense, so binning a particle set is one gather
        # instead of an arithmetic chain per particle. Bins are kept in lattice order
        # rather than compacted, because smoothing needs the 2-D neighbourhood.
        jj, ii = np.divmod(np.arange(C.GRID * C.GRID), C.GRID)
        self.bin_of_cell = ((jj // self.block) * lattice + (ii // self.block)).astype(np.int32)

    @property
    def max_bits(self) -> float:
        return float(np.log2(self.n_bins))

    def histogram(self, cell: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Smoothed probability over the lattice, zero outside walkable bins.

        The smoothing is part of the metric, not a tidy-up: a particle set cannot
        resolve a distribution below one particle per bin, so an unsmoothed histogram
        assigns exactly zero to any bin that happens to hold none, and then every
        credible region at every level excludes a truth that lands there. That is a
        statement about the sample rather than about the belief.

        Mass that lands off the navmesh is kept and smoothed rather than dropped. The
        `disc` and `constant_velocity` baselines deliberately ignore terrain, and
        renormalising their off-navmesh mass away would quietly turn them into better
        models than they are.
        """
        h = np.bincount(self.bin_of_cell[cell], weights=weight, minlength=self.lattice**2)
        if self.smoothing > 0.0:
            g = h.reshape(self.lattice, self.lattice)
            # Zero-padded rather than wrapped: `np.roll` would carry mass off the west
            # edge of the map onto the east one, which is a shortcut that happens to be
            # nearly harmless here (the border bins are walls) and would stop being so
            # the moment the lattice or the map changed.
            pad = np.zeros((self.lattice + 2, self.lattice + 2))
            pad[1:-1, 1:-1] = g
            acc = np.zeros_like(g)
            for dj in (0, 1, 2):
                for di in (0, 1, 2):
                    if dj == 1 and di == 1:
                        continue
                    acc += pad[dj : dj + self.lattice, di : di + self.lattice]
            h = ((1.0 - self.smoothing) * g + (self.smoothing / 8.0) * acc).reshape(-1)
        h = np.where(self.walkable, h, 0.0)
        total = h.sum()
        if total <= 0:
            return self.walkable.astype(np.float64) / self.n_bins
        return h / total


def _normalised_weights(logw: np.ndarray) -> np.ndarray:
    m = logw.max()
    if not np.isfinite(m):
        return np.full(logw.shape, 1.0 / logw.size)
    w = np.exp(logw - m)
    s = w.sum()
    return w / s if s > 0 else np.full(logw.shape, 1.0 / logw.size)


def _entropy_bits(p: np.ndarray, ess: float, max_bits: float) -> float:
    """Plug-in entropy with the Miller-Madow correction, capped at the lattice ceiling.

    The correction matters more than it looks. Plug-in entropy of a finite sample is
    biased *downward* by roughly `(K-1) / (2N ln 2)` bits, where K is the number of
    occupied bins. And K grows as the belief spreads, so the bias is a function of
    precisely the quantity being measured. Uncorrected, a diffuse belief is understated
    more than a concentrated one, which flatters the model's apparent sharpness in the
    exact regime the project cares about.

    The cap is not cosmetic. Miller-Madow is asymptotic in particles per bin, and near a
    uniform belief there is barely one of those, measured, the correction added 0.63
    bits to a 9.35-bit estimate on an 890-bin lattice whose maximum is 9.80. Entropy
    over K bins cannot exceed log2 K by any amount, so a correction that pushes past it
    has left its validity range and the ceiling is the right answer.
    """
    nz = p[p > 0]
    h = float(-(nz * np.log2(nz)).sum())
    if ess > 0:
        h += (nz.size - 1) / (2.0 * ess * np.log(2.0))
    return min(h, max_bits)


def _credible(p: np.ndarray, mass: float) -> tuple[int, float]:
    """`(bins in the highest-density region, cumulative mass at its edge)`."""
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    k = int(np.searchsorted(cum, mass) + 1)
    k = min(k, p.size)
    return k, float(cum[k - 1])


def _pit(p: np.ndarray, truth_bin: int) -> float:
    """Smallest credible level whose region contains the truth.

    Under a correctly calibrated filter this is Uniform(0, 1), so the empirical CDF of
    these values against the diagonal *is* the reliability curve, with no binning choices
    to argue about. A filter that is systematically overconfident piles these up near 1.
    """
    pt = p[truth_bin]
    return float(p[p > pt].sum() + pt)


def belief_summary(
    lattice: LatticeIndex,
    cell: np.ndarray,
    weights: np.ndarray,
    spec: FilterSpec,
) -> tuple[float, float]:
    """`(entropy in bits, 90% credible-region area in ku²)` for one belief.

    The pair the artifact ships per tick. Exported here rather than recomputed in
    `l4_export` so the number on the site and the number in the validation report come
    from the same three lines. A display that quietly used a different lattice or
    skipped the Miller-Madow correction would disagree with the report and nobody would
    know which was right.
    """
    p = lattice.histogram(cell, weights)
    ess = 1.0 / float((weights * weights).sum()) if weights.size else 0.0
    bins, _ = _credible(p, spec.credible_mass)
    return _entropy_bits(p, ess, lattice.max_bits), bins * lattice.bin_area_ku2


@dataclass(frozen=True, slots=True)
class BeliefScore:
    """One model's performance over one match."""

    name: str
    spec: FilterSpec
    #: Mean over scored ticks (alive, unseen, truth known).
    nll: float
    entropy_bits: float
    credible_area_ku2: float
    credible_area_map_fraction: float
    #: Empirical coverage at each of `CALIBRATION_LEVELS`.
    coverage: dict[float, float]
    calibration_error: float
    scored_ticks: int
    darkness_strict: float
    darkness_naive: float
    depletion_events: int
    stats: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "nll": round(self.nll, 4),
            "entropy_bits": round(self.entropy_bits, 4),
            "area_ku2": round(self.credible_area_ku2, 3),
            "area_map_fraction": round(self.credible_area_map_fraction, 4),
            "ece": round(self.calibration_error, 4),
            "coverage": {k: round(v, 4) for k, v in self.coverage.items()},
            "scored_ticks": self.scored_ticks,
            "darkness_strict": round(self.darkness_strict, 4),
            "darkness_naive": round(self.darkness_naive, 4),
            "depletion_events": self.depletion_events,
        }


def evaluate(
    name: str,
    spec: FilterSpec,
    beliefs: Iterable[TickBelief],
    truth: TruthTable,
    lattice: LatticeIndex,
    stride: int = 1,
    depletion_events: int = 0,
) -> BeliefScore:
    """Score a belief stream against the truth.

    **Only ticks where the enemy is alive and unseen are scored.** A seen enemy's belief
    is a point mass at their exact cell by construction, so including those ticks would
    average in a perfect score for every moment the question was not being asked. And
    since visibility runs 25-40%, that alone would move every model a third of the way
    toward looking good. It would also flatter whichever model happens to see more, which
    is none of them: visibility is identical across the ablation.
    """
    n_scored = 0
    nll_sum = 0.0
    ent_sum = 0.0
    area_sum = 0.0
    pits: list[float] = []
    dark_strict = 0
    dark_naive = 0
    alive_ticks = 0
    total_ticks = 0
    bin_area = lattice.bin_area_ku2
    map_area = lattice.n_bins * bin_area

    for belief in beliefs:
        tick = belief.tick
        for o in range(C.N_TEAMS):
            for e in range(C.N_ENEMIES):
                total_ticks += 1
                alive = bool(belief.alive[o, e])
                seen = bool(belief.seen[o, e])
                if not seen:
                    dark_naive += 1
                if alive:
                    alive_ticks += 1
                    if not seen:
                        dark_strict += 1
                if seen or not alive or tick % stride:
                    continue
                tcell = int(truth.cell[tick, o, e])
                if tcell == NO_CELL:
                    continue

                w = _normalised_weights(belief.logw[o, e])
                p = lattice.histogram(belief.cell[o, e], w)
                ess = 1.0 / float((w * w).sum())

                ent_sum += _entropy_bits(p, ess, lattice.max_bits)
                k, _ = _credible(p, spec.credible_mass)
                area_sum += k * bin_area
                tbin = int(lattice.bin_of_cell[tcell])
                pits.append(_pit(p, tbin))
                # Density per ku^2, floored at half a particle's worth of mass. A bin
                # with no particles in it has plug-in probability exactly zero, and one
                # such tick would send the mean NLL to infinity and decide the whole
                # comparison on a single sample. Half of 1/P is the smallest mass the
                # estimator could have resolved, so the floor says "below what we can
                # measure" rather than inventing a number.
                dens = max(p[tbin], 0.5 / spec.particles) / bin_area
                nll_sum += -np.log(dens)
                n_scored += 1

    pit = np.asarray(pits)
    coverage = {
        q: float((pit <= q).mean()) if pit.size else float("nan") for q in CALIBRATION_LEVELS
    }
    ece = (
        float(np.mean([abs(coverage[q] - q) for q in CALIBRATION_LEVELS]))
        if pit.size
        else float("nan")
    )
    denom = max(n_scored, 1)
    return BeliefScore(
        name=name,
        spec=spec,
        nll=nll_sum / denom,
        entropy_bits=ent_sum / denom,
        credible_area_ku2=area_sum / denom,
        credible_area_map_fraction=(area_sum / denom) / map_area,
        coverage=coverage,
        calibration_error=ece,
        scored_ticks=n_scored,
        darkness_strict=dark_strict / max(alive_ticks, 1),
        darkness_naive=dark_naive / max(total_ticks, 1),
        depletion_events=depletion_events,
        stats={
            "lattice_bins": lattice.n_bins,
            "lattice_max_bits": round(lattice.max_bits, 3),
            "map_area_ku2": round(map_area, 2),
        },
    )
