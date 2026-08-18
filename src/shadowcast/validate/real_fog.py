"""Why real fog agreement is what it is, decomposed into causes.

`fog_oracle.validate_fog` says *how much* the reconstruction disagrees with the corpus.
On synthetic data that is enough, because the remaining error is small and its causes are
known. On real data it is not: a single 68% tells you nothing about whether to go and fix
trajectories, model more vision sources, or re-examine the geometry.

So this splits the disagreement along the one axis that separates those causes, **how
stale the position being used is**, measured as time since a *labelled* anchor, a
`CastSpellAns` or `BasicAttackPos` that states a champion's coordinates outright.

Two conditionings, and the difference between them is the point:

**On the target's own staleness**, we learn whether we are looking in the right place. Its
signature is the "no source in range" column: with a fresh anchor the position is nearly
exact, so a visible champion with no modelled source nearby means a source is genuinely
missing from the model, not misplaced. That column is a floor on how much of the gap
better trajectories can ever close.

**On the nearest observer's staleness**, we learn whether the sources are in the right
place. A champion is visible because *someone else* can see it, so an observer whose
position is fifteen seconds stale takes its vision with it. MEASURED: agreement falls from
76.8% to 51.0% across that range, which is what moved the project's attention from
modelling more entity types to reconstructing better trajectories.

The inversion check is the sharpest single number here. If visible champions sit *further*
from the nearest source than hidden ones, the position estimate is not merely noisy. It
is anti-informative, and no amount of source modelling will help until it is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast import sr
from shadowcast.fov.union import mask_to_bool
from shadowcast.geom.grid import world_to_cell
from shadowcast.l1_events.schema import UNKNOWN, MatchEvents
from shadowcast.l2_reconstruct.vision import VisionStream
from shadowcast.validate.fog_oracle import _oracle_timeline

__all__ = ["ANCHOR_BANDS", "AnchorBand", "RealFogReport", "decompose_fog"]

#: Bands of "seconds since this champion's last labelled anchor". Chosen so the first is
#: effectively exact position, the last is effectively no position information, and the
#: middle shows which way the trend runs.
ANCHOR_BANDS: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, np.inf))

#: Bands of "seconds since the *nearest enemy* champion's last anchor".
OBSERVER_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),
    (1.0, 5.0),
    (5.0, 15.0),
    (15.0, np.inf),
)


@dataclass(frozen=True, slots=True)
class AnchorBand:
    """One row of the decomposition."""

    lo: float
    hi: float
    n: int
    agree: int
    visible_source_distance: float
    hidden_source_distance: float
    visible_without_source: float

    @property
    def label(self) -> str:
        if not np.isfinite(self.hi):
            return f"over {self.lo:g} s"
        return f"{self.lo:g}-{self.hi:g} s"

    @property
    def rate(self) -> float:
        return self.agree / self.n if self.n else float("nan")

    @property
    def informative(self) -> bool:
        """Is a visible champion actually closer to a source than a hidden one?

        When this goes False the position estimate has stopped carrying information about
        visibility, which is a different failure from being merely imprecise.
        """
        return self.visible_source_distance < self.hidden_source_distance


@dataclass(frozen=True, slots=True)
class RealFogReport:
    by_target_age: list[AnchorBand]
    by_observer_age: list[AnchorBand]
    stats: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        def rows(bands: list[AnchorBand]) -> list[dict[str, Any]]:
            return [
                {
                    "band": b.label,
                    "n": b.n,
                    "agreement": round(b.rate, 4),
                    "visible_to_source_u": round(b.visible_source_distance, 0),
                    "hidden_to_source_u": round(b.hidden_source_distance, 0),
                    "visible_without_source": round(b.visible_without_source, 4),
                    "informative": b.informative,
                }
                for b in bands
            ]

        return {
            "by_target_age": rows(self.by_target_age),
            "by_observer_age": rows(self.by_observer_age),
            **self.stats,
        }


def _anchor_age(events: MatchEvents, n_ticks: int, n_slots: int) -> np.ndarray:
    """Seconds since each champion's last labelled anchor, per tick. `inf` before the first."""
    tick_t = np.arange(n_ticks) * C.TICK_DT
    out = np.full((n_ticks, n_slots), np.inf)
    for slot in range(n_slots):
        ts = np.sort(events.anchors["t"][events.anchors["slot"] == slot])
        if ts.size == 0:
            continue
        j = np.searchsorted(ts, tick_t, side="right") - 1
        seen = j >= 0
        out[seen, slot] = tick_t[seen] - ts[np.maximum(j, 0)][seen]
    return out


def _team_sources(
    events: MatchEvents, pos: np.ndarray, valid: np.ndarray, front: dict[str, np.ndarray], tick: int
) -> dict[int, np.ndarray]:
    """Every modelled vision source position, per team, at one tick.

    Positions only: this asks "was anything of ours near enough to have seen it", which
    is a question about modelling coverage rather than about geometry, so occlusion is
    deliberately not applied. A source behind a wall still counts here; if the answer is
    still "nothing was near", no field-of-view fix can help.
    """
    t = tick * C.TICK_DT
    team = events.heroes["team"].astype(np.int64)
    out: dict[int, np.ndarray] = {}
    for observer in (C.TEAM_ORDER, C.TEAM_CHAOS):
        pts: list[np.ndarray] = []
        for slot in range(events.n_heroes):
            if int(team[slot]) == observer and valid[tick, slot]:
                pts.append(pos[tick, slot])
        for ward in events.wards:
            if int(ward["team"]) == observer and ward["t0"] <= t <= ward["t1"]:
                pts.append(np.array([ward["x"], ward["z"]]))
        for site in events.turret_sites:
            if int(site["team"]) == observer and np.isfinite(site["x"]):
                pts.append(np.array([site["x"], site["z"]]))
        for wave in events.minion_waves:
            if int(wave["team"]) != observer:
                continue
            lane = str(wave["lane"])
            p = sr.minion_clump_position(
                lane,
                observer,
                float(wave["t0"]),
                t,
                float(wave["t1"]),
                front_s=float(front[lane][tick]),
            )
            if p is not None:
                pts.append(np.asarray(p, dtype=float))
        out[observer] = np.stack(pts) if pts else np.zeros((0, 2))
    return out


def decompose_fog(
    events: MatchEvents, attribution, terrain, table, stride: int = 4
) -> RealFogReport:
    """Split the fog disagreement by how stale the positions involved are."""
    n_ticks, n_slots = attribution.pos.shape[:2]
    pos, valid = attribution.pos, attribution.valid
    team = events.heroes["team"].astype(np.int64)
    oracle = _oracle_timeline(events, n_ticks, C.TICK_DT)
    age = _anchor_age(events, n_ticks, n_slots)

    stream = VisionStream(events, attribution, terrain, table)
    front = stream._front

    # (band index, conditioning) -> accumulators
    acc = {
        key: [
            {"n": 0, "agree": 0, "vis": [], "hid": [], "vis_n": 0, "nosrc": 0}
            for _ in (OBSERVER_BANDS if key == "observer" else ANCHOR_BANDS)
        ]
        for key in ("target", "observer")
    }
    compared = 0

    for tick, mask_order, mask_chaos in stream.masks():
        if tick % stride:
            continue
        bools = (mask_to_bool(mask_order, terrain.grid), mask_to_bool(mask_chaos, terrain.grid))
        sources = _team_sources(events, pos, valid, front, tick)

        for slot in range(n_slots):
            own = int(team[slot])
            if own == UNKNOWN or not valid[tick, slot]:
                continue
            x, z = pos[tick, slot]
            i, j = world_to_cell(float(x), float(z))
            if not (0 <= i < terrain.grid and 0 <= j < terrain.grid):
                continue

            observer = 1 - own
            ours = bool(bools[observer][j, i])
            theirs = bool(oracle[tick, slot])
            compared += 1

            points = sources[observer]
            if points.size:
                gaps = np.hypot(points[:, 0] - x, points[:, 1] - z)
                nearest = float(gaps.min())
            else:
                nearest = float("inf")

            # Staleness of the nearest enemy champion specifically: the sources that move.
            best_age, best_gap = float("inf"), float("inf")
            for other in range(n_slots):
                if int(team[other]) != observer or not valid[tick, other]:
                    continue
                gap = float(np.hypot(pos[tick, other, 0] - x, pos[tick, other, 1] - z))
                if gap < best_gap:
                    best_gap, best_age = gap, float(age[tick, other])

            for key, value, bands in (
                ("target", float(age[tick, slot]), ANCHOR_BANDS),
                ("observer", best_age, OBSERVER_BANDS),
            ):
                for b, (lo, hi) in enumerate(bands):
                    if not (lo <= value < hi):
                        continue
                    cell = acc[key][b]
                    cell["n"] += 1
                    cell["agree"] += int(ours == theirs)
                    if theirs:
                        cell["vis"].append(nearest)
                        cell["vis_n"] += 1
                        cell["nosrc"] += int(nearest > C.SIGHT_CHAMPION)
                    else:
                        cell["hid"].append(nearest)
                    break

    def finish(key: str, bands: tuple[tuple[float, float], ...]) -> list[AnchorBand]:
        out = []
        for b, (lo, hi) in enumerate(bands):
            cell = acc[key][b]
            out.append(
                AnchorBand(
                    lo=lo,
                    hi=hi,
                    n=cell["n"],
                    agree=cell["agree"],
                    visible_source_distance=float(np.median(cell["vis"]))
                    if cell["vis"]
                    else np.nan,
                    hidden_source_distance=float(np.median(cell["hid"])) if cell["hid"] else np.nan,
                    visible_without_source=cell["nosrc"] / cell["vis_n"]
                    if cell["vis_n"]
                    else np.nan,
                )
            )
        return out

    return RealFogReport(
        by_target_age=finish("target", ANCHOR_BANDS),
        by_observer_age=finish("observer", OBSERVER_BANDS),
        stats={
            "compared": compared,
            "stride": stride,
            "minion_waves": int(events.minion_waves.size),
            "minion_contacts": int(events.minion_contacts.size),
            "order_attribution_rate": round(events.order_attribution_rate, 4),
        },
    )
