"""Measuring reconstructed visibility against the dataset's own fog events.

This is the project's central validation, and the reason it can claim to be measured
rather than merely plausible. The corpus records when each champion entered and left fog;
we reconstruct, independently, which cells each team could see at each moment. Comparing
the two tests the whole stack at once — champion trajectories, ward lifetimes, turret
positions, minion modelling and the field-of-view geometry.

**Direction of error matters more than the headline rate**, so the two are always reported
separately:

- A **false positive** is our mask claiming vision the game did not grant. These inflate
  every information metric: an enemy we wrongly believe was seen produces too little
  darkness, too little entropy, and an understated information advantage.
- A **false negative** is the game granting vision we failed to reconstruct. These come
  from a missing source — an unmodelled ward, a destroyed-turret assumption, a minion wave
  in the wrong place — and understate what a team knew.

They have different causes and different consequences, and a single agreement percentage
hides both.

**Where errors land matters too.** A brush-adjacent disagreement is expected: brush is a
conditional occluder and the grid quantises its boundary, so a champion a few units either
side of an entrance is genuinely borderline. A disagreement in the middle of open lane is
not expected and means something is wrong. Reporting the breakdown is what makes the
number diagnostic rather than merely a score.

**Transition timing** is measured separately from state agreement. Getting the visibility
state right 99% of the time while being half a second late on every transition would look
excellent by state agreement alone and would still ruin any metric that integrates over a
ward's lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast.geom.grid import world_to_cell
from shadowcast.l1_events.schema import UNKNOWN, MatchEvents
from shadowcast.l2_reconstruct.vision import VisionStream

__all__ = ["REGIONS", "FogAgreement", "validate_fog"]

#: Regions to break the agreement down by. `brush_adjacent` is the one expected to be
#: worst, so it is separated rather than averaged into the rest.
REGIONS = ("lane", "jungle", "river", "brush_adjacent", "base")

#: How close to brush a cell must be to count as brush-adjacent.
_BRUSH_ADJACENT_CELLS = 2
#: A cell within this distance of a lane centre line is "lane".
_LANE_RADIUS = 1200.0
#: A cell within this distance of a fountain is "base".
_BASE_RADIUS = 2600.0


@dataclass(frozen=True, slots=True)
class FogAgreement:
    """Reconstructed visibility measured against the oracle's fog events."""

    compared: int
    agree: int
    false_positive: int
    false_negative: int
    by_region: dict[str, tuple[int, int]]  # region -> (compared, agree)
    transition_errors: np.ndarray  # seconds, signed: ours minus the oracle's
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.agree / self.compared if self.compared else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.false_positive / self.compared if self.compared else 0.0

    @property
    def false_negative_rate(self) -> float:
        return self.false_negative / self.compared if self.compared else 0.0

    def region_rates(self) -> dict[str, float]:
        return {
            region: (agree / compared if compared else float("nan"))
            for region, (compared, agree) in self.by_region.items()
        }

    def timing(self) -> dict[str, float]:
        e = self.transition_errors[np.isfinite(self.transition_errors)]
        if e.size == 0:
            return {}
        return {
            "median_s": float(np.median(e)),
            "abs_median_s": float(np.median(np.abs(e))),
            "abs_p98_s": float(np.percentile(np.abs(e), 98)),
            "within_150ms": float((np.abs(e) <= 0.150).mean()),
            "n": int(e.size),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "compared": self.compared,
            "agreement": round(self.rate, 5),
            "false_positive_rate": round(self.false_positive_rate, 5),
            "false_negative_rate": round(self.false_negative_rate, 5),
            "by_region": {k: round(v, 5) for k, v in self.region_rates().items()},
            "timing": {k: round(v, 4) for k, v in self.timing().items()},
            **self.stats,
        }


def _region_map(terrain) -> np.ndarray:
    """Label every cell with a region, for the agreement breakdown."""
    from shadowcast import sr

    grid = terrain.grid
    labels = np.full((grid, grid), REGIONS.index("jungle"), dtype=np.int8)

    js, is_ = np.mgrid[0:grid, 0:grid]
    xs = C.WORLD_MIN_X + (is_ + 0.5) * C.GRID_CELL_SIZE
    zs = C.WORLD_MIN_Z + (js + 0.5) * C.GRID_CELL_SIZE

    # Lanes, then river (the anti-diagonal band that is not lane), then bases.
    lane_d = np.full((grid, grid), np.inf)
    for pts in sr.LANES.values():
        ref = np.stack([sr.lerp_polyline(pts, s) for s in np.linspace(0, 1, 300)])
        for point in ref:
            lane_d = np.minimum(lane_d, np.hypot(xs - point[0], zs - point[1]))
    labels[lane_d <= _LANE_RADIUS] = REGIONS.index("lane")

    mid_ref = np.stack([sr.lerp_polyline(sr.LANES["mid"], s) for s in np.linspace(0, 1, 300)])
    mid_d = np.full((grid, grid), np.inf)
    for point in mid_ref:
        mid_d = np.minimum(mid_d, np.hypot(xs - point[0], zs - point[1]))
    river = (mid_d > _LANE_RADIUS) & (mid_d <= 3200.0) & (labels != REGIONS.index("lane"))
    labels[river] = REGIONS.index("river")

    for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
        f = sr.FOUNTAINS[team]
        labels[np.hypot(xs - f[0], zs - f[1]) <= _BASE_RADIUS] = REGIONS.index("base")

    # Brush-adjacent last, so it wins: it is the category we most want isolated.
    brush = terrain.brush
    near = brush.copy()
    for _ in range(_BRUSH_ADJACENT_CELLS):
        grown = near.copy()
        grown[:-1, :] |= near[1:, :]
        grown[1:, :] |= near[:-1, :]
        grown[:, :-1] |= near[:, 1:]
        grown[:, 1:] |= near[:, :-1]
        near = grown
    labels[near] = REGIONS.index("brush_adjacent")
    return labels


def _oracle_timeline(events: MatchEvents, n_ticks: int, dt: float) -> np.ndarray:
    """Replay the fog events into a per-tick visibility array.

    The events are transitions, so this is the state they describe. The implicit initial
    state is "not visible", matching how the transitions are emitted.
    """
    n_slots = max(1, events.n_heroes)
    out = np.zeros((n_ticks, n_slots), dtype=bool)
    for slot in range(n_slots):
        rows = events.fog[events.fog["slot"] == slot]
        if rows.size == 0:
            continue
        rows = rows[np.argsort(rows["t"], kind="stable")]
        state = False
        cursor = 0
        for tick in range(n_ticks):
            t = tick * dt
            while cursor < rows.size and rows[cursor]["t"] <= t + 1e-9:
                state = bool(rows[cursor]["visible"])
                cursor += 1
            out[tick, slot] = state
    return out


def _transition_ticks(timeline: np.ndarray, slot: int) -> np.ndarray:
    col = timeline[:, slot]
    return np.flatnonzero(np.diff(col.astype(np.int8)) != 0) + 1


def validate_fog(
    events: MatchEvents,
    attribution,
    terrain,
    table,
    tick_hz: int = C.TICK_HZ,
    stride: int = 1,
) -> FogAgreement:
    """Compare reconstructed masks against the oracle's fog timeline.

    `stride` samples ticks rather than checking all of them, which is useful when running
    over many matches. It does not change the measurement, only its precision.
    """
    stream = VisionStream(events, attribution, terrain, table, tick_hz=tick_hz)
    dt = 1.0 / tick_hz
    n_ticks = attribution.pos.shape[0]
    n_slots = max(1, events.n_heroes)
    team = events.heroes["team"].astype(np.int64)

    oracle = _oracle_timeline(events, n_ticks, dt)
    regions = _region_map(terrain)
    ours = np.zeros((n_ticks, n_slots), dtype=bool)
    ours_known = np.zeros((n_ticks, n_slots), dtype=bool)

    compared = agree = fp = fn = 0
    region_compared = dict.fromkeys(REGIONS, 0)
    region_agree = dict.fromkeys(REGIONS, 0)
    unknown_position = 0

    from shadowcast.fov.union import mask_to_bool

    for tick, mask_order, mask_chaos in stream.masks():
        masks = (mask_order, mask_chaos)
        bools = None
        for slot in range(n_slots):
            own = int(team[slot])
            if own == UNKNOWN:
                continue
            observer = 1 - own
            if not attribution.valid[tick, slot]:
                # No position claim, so nothing to compare. Counted, not hidden: a large
                # number here means the comparison covered less than it appears to.
                unknown_position += 1
                continue
            x, z = attribution.pos[tick, slot]
            i, j = world_to_cell(float(x), float(z))
            if not (0 <= i < terrain.grid and 0 <= j < terrain.grid):
                unknown_position += 1
                continue
            if bools is None:
                bools = (
                    mask_to_bool(masks[0], terrain.grid),
                    mask_to_bool(masks[1], terrain.grid),
                )
            seen = bool(bools[observer][j, i])
            ours[tick, slot] = seen
            ours_known[tick, slot] = True

            if tick % stride:
                continue
            truth = bool(oracle[tick, slot])
            compared += 1
            region = REGIONS[int(regions[j, i])]
            region_compared[region] += 1
            if seen == truth:
                agree += 1
                region_agree[region] += 1
            elif seen:
                fp += 1
            else:
                fn += 1

    # Transition timing: for each oracle transition, the nearest of ours for that champion.
    errors: list[float] = []
    for slot in range(n_slots):
        if int(team[slot]) == UNKNOWN:
            continue
        theirs = _transition_ticks(oracle, slot)
        mine = _transition_ticks(ours & ours_known, slot)
        if theirs.size == 0 or mine.size == 0:
            continue
        for tk in theirs:
            nearest = mine[int(np.argmin(np.abs(mine - tk)))]
            errors.append(float((nearest - tk) * dt))

    counts = stream.counts()
    return FogAgreement(
        compared=compared,
        agree=agree,
        false_positive=fp,
        false_negative=fn,
        by_region={r: (region_compared[r], region_agree[r]) for r in REGIONS},
        transition_errors=np.array(errors),
        stats={
            "ticks": n_ticks,
            "no_position_claim": unknown_position,
            "oracle_transitions": int(
                sum(_transition_ticks(oracle, s).size for s in range(n_slots))
            ),
            "our_transitions": int(
                sum(_transition_ticks(ours & ours_known, s).size for s in range(n_slots))
            ),
            "sources": {
                "turrets": counts.turrets,
                "wards": counts.wards,
                "minion_waves": counts.extra.get("minion_waves", 0),
                "live_fov_fallbacks": counts.live_fallbacks,
            },
        },
    )
