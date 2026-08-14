"""Attributing anonymous movement orders to champions, and reconstructing trajectories.

The corpus's defining defect: a movement order carries a polyline and a timestamp and
no entity id whatsoever. Ten champions issue thousands of orders per match into one
undifferentiated stream, and every position-dependent thing this project computes needs
to know who moved.

**The anchors are what make it possible.** `CastSpellAns` and `BasicAttackPos` pair a
net_id with a position, giving one labelled observation per champion roughly every
0.75 s — 546 to 1,085 per champion per match in the real corpus. Interpolating between a
champion's own anchors gives a position estimate that is accurate to a measured 10-unit
median, and crucially it depends on *no* prior assignment. So the design is three passes
with no feedback loop:

    A. Build an anchor skeleton per champion — position from labelled observations only.
    B. Assign each order to the champion whose skeleton position best matches the
       order's first waypoint, jointly across simultaneous orders.
    C. Integrate the assigned orders into a full-rate trajectory, resetting at each
       anchor.

An earlier version used the *integrated* position as the assignment cost, which creates
a feedback loop: one wrong assignment moves a champion onto another's path, and every
subsequent assignment for both is then made from a corrupted estimate. The skeleton
breaks that loop, because an error in pass B cannot influence pass B's other decisions.

**There is an irreducible ambiguity, and understanding it changes what to report.** For
the first ten seconds of a match five champions per team stand on one fountain: measured
nearest-same-team separation is *zero* at t=2 s and still 6 units at t=10 s. No
position-based method can tell their orders apart there, because there is no information
with which to do it.

But that ambiguity is almost entirely **harmless**, and measuring it that way is the
honest framing. Against ground truth, a misattributed order's true and assigned owners
sit a median of **zero units** apart — the assignment is arbitrary exactly when it makes
no difference to any position. Raw attribution accuracy is about 97%; genuinely harmful
misattributions, where the two champions were more than 300 units apart, are 8 of 5,470
orders (0.15%) on a clean stream and **zero** on the fully adversarial one.

`order_margin` exposes that distinction without needing the truth: it is the cost gap to
the runner-up champion, and it separates the two cases cleanly. Orders with a margin of
at least 100 units are 92% of the total and 99.7% accurate; at 300 units they are 85% of
the total and 99.9-100% accurate. Downstream consumers should discount low-margin
attributions rather than treat every one as equally certain.

Three residuals come out of this and they measure different things:

- The **order residual** is the gap between the skeleton position and the assigned
  order's first waypoint. It is the assignment cost, so it is not independent evidence —
  and its magnitude (~20 units median) is set by the skeleton's interpolation error, not
  by how well anything was reconstructed. It should not be read as an accuracy figure.
- The **order margin** is the confidence signal described above.
- The **anchor residual** is the gap between the integrated trajectory and a labelled
  observation, measured *before* the estimate is reset to it. Nothing about it feeds the
  assignment, so it is the honest error figure — and on a clean stream with an exact
  frame offset it is 0.0003 units, which is float32 noise.

Two ideas that seemed obviously right and measurably were not, both kept as ablatable
knobs rather than as claims: a **direction term** in the cost (see
`AttributionSpec.direction_weight`) and **iterating** assignment against the integrated
trajectory (see `AttributionSpec.iterations`). Each is argued for at its definition
along with the numbers that rejected it.

Trajectory accuracy is bounded below by the frame calibration, visibly so: with a
2.5-unit offset error the median residual is 3.54 units, which is exactly sqrt(2) times
2.5. Not a bug in either component — it is the frame's half-cell resolution limit
propagating, and it means no trajectory claim can be tighter than the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast.l1_events.schema import UNKNOWN, MatchEvents

__all__ = ["Attribution", "AttributionSpec", "attribute", "with_owners"]

_KIND_SPEED, _KIND_ANCHOR, _KIND_ORDER, _KIND_DEATH = 0, 1, 2, 3


@dataclass(frozen=True, slots=True)
class AttributionSpec:
    """Gating, cost weighting and integration parameters."""

    tick_hz: int = C.TICK_HZ
    #: Assignment/integration rounds.
    #:
    #: **Default one, because more does not help.** Round 0 assigns from the anchor
    #: skeleton; later rounds reassign from the previous round's integrated trajectory,
    #: which between anchors is more accurate because it follows the real order
    #: polylines. Measured: accuracy is flat from one round to four (97.0%, 96.6%,
    #: 97.0%, 96.9% on a clean stream) while the 99th-percentile trajectory error grows
    #: from 419 to 691 units, because reassigning from an estimate that depends on the
    #: previous assignment reintroduces exactly the feedback the skeleton exists to
    #: avoid. Kept as a knob so the ablation stays reproducible.
    iterations: int = 1
    #: Cost gate on the skeleton-to-order-start distance. Has to cover the genuine
    #: disagreement between a champion's position and the first waypoint it publishes
    #: (server-side smoothing) plus the skeleton's own interpolation error, whose
    #: measured 90th percentile is about 100 units.
    base_gate: float = 400.0
    #: Extra gate allowed per second of gap in the skeleton, since interpolation across
    #: a long silence is much weaker than across a typical 0.75-second one.
    gap_gate_rate: float = 200.0
    max_gate: float = 3000.0
    #: Cost added for an order heading the opposite way to the champion's motion.
    #:
    #: **Default zero, because it measurably hurts.** The reasoning for adding it was
    #: sound — while five champions stand on one fountain, where they are *going* is the
    #: only thing that separates them — but the measurement disagreed: at 600 it cost
    #: 6.7 points of accuracy on the adversarial stream (96.9% -> 90.2%) and took the
    #: worst trajectory error from 855 to 8,371 units. The heading estimate is
    #: interpolated between anchors and therefore noisy, and a champion's instantaneous
    #: heading routinely disagrees with its next order's opening direction because it
    #: stops and turns. A noisy term applied with confidence is worse than no term.
    #: Kept as a knob so the ablation is reproducible rather than a claim in a comment.
    direction_weight: float = 0.0
    #: Below this speed a champion has no meaningful heading and the direction term is
    #: skipped rather than applied to noise.
    direction_min_speed: float = 40.0
    #: How far outside its anchor range a champion's skeleton may be extrapolated.
    extrapolate_seconds: float = 2.0
    #: An anchor further than this from the integrated position is counted as evidence
    #: that a recent assignment was wrong. It still resets the estimate.
    suspect_anchor_error: float = 400.0
    default_speed: float = C.MOVE_SPEED_DEFAULT


@dataclass(frozen=True, slots=True)
class Attribution:
    """Order ownership plus the trajectory that produced it."""

    owner: np.ndarray  # i1[n_orders], UNKNOWN where unassigned
    pos: np.ndarray  # f8[n_ticks, n_slots, 2]
    valid: np.ndarray  # bool[n_ticks, n_slots]
    speed: np.ndarray  # f8[n_ticks, n_slots]
    order_residual: np.ndarray  # f8[n_orders], nan where unassigned
    order_margin: np.ndarray  # f8[n_orders], cost gap to the runner-up champion
    anchor_residual: np.ndarray  # f8[n_anchors], nan where no prior estimate
    stats: dict[str, Any] = field(default_factory=dict)

    def confident(self, min_margin: float = 100.0) -> np.ndarray:
        """Orders whose attribution was not a near-tie.

        Measured against ground truth, a misattributed order's true and assigned owners
        sit a median of ZERO units apart — the assignment is arbitrary precisely when it
        does not matter. This is the truth-free way to spot those.
        """
        return (self.owner != UNKNOWN) & (self.order_margin >= min_margin)

    @property
    def attributed_fraction(self) -> float:
        return float((self.owner != UNKNOWN).mean()) if self.owner.size else 0.0

    def residual_percentiles(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, arr in (("order", self.order_residual), ("anchor", self.anchor_residual)):
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            out[f"{name}_median"] = float(np.median(finite))
            out[f"{name}_p99"] = float(np.percentile(finite, 99))
            out[f"{name}_max"] = float(finite.max())
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "orders": int(self.owner.size),
            "attributed": round(self.attributed_fraction, 4),
            **{k: round(v, 3) for k, v in self.residual_percentiles().items()},
            **self.stats,
        }


# ---------------------------------------------------------------------------
# Pass A: the anchor skeleton
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Skeleton:
    """Per-champion position from labelled observations alone.

    Assignment-independent by construction, which is the whole point: an attribution
    mistake cannot corrupt the basis on which later attributions are made.
    """

    t: list[np.ndarray]
    x: list[np.ndarray]
    z: list[np.ndarray]

    def positions_at(self, times: np.ndarray, slot: int) -> tuple[np.ndarray, np.ndarray]:
        at = self.t[slot]
        if at.size == 0:
            nan = np.full(times.shape, np.nan)
            return nan, nan
        return np.interp(times, at, self.x[slot]), np.interp(times, at, self.z[slot])

    def gap_at(self, times: np.ndarray, slot: int) -> np.ndarray:
        """Length of the anchor interval bracketing each time, or inf outside the range."""
        at = self.t[slot]
        if at.size < 2:
            return np.full(times.shape, np.inf)
        idx = np.clip(np.searchsorted(at, times, side="right"), 1, at.size - 1)
        return at[idx] - at[idx - 1]

    def out_of_range(self, times: np.ndarray, slot: int, slack: float) -> np.ndarray:
        at = self.t[slot]
        if at.size == 0:
            return np.ones(times.shape, dtype=bool)
        return (times < at[0] - slack) | (times > at[-1] + slack)

    def heading_at(self, times: np.ndarray, slot: int, dt: float) -> np.ndarray:
        """Unit direction of motion, from the skeleton. NaN where undetermined."""
        ax, az = self.positions_at(times + dt, slot)
        bx, bz = self.positions_at(times - dt, slot)
        vx, vz = ax - bx, az - bz
        norm = np.hypot(vx, vz)
        speed = norm / (2 * dt)
        out = np.stack(
            [vx / np.where(norm > 0, norm, np.nan), vz / np.where(norm > 0, norm, np.nan)], axis=1
        )
        out[speed < 1e-9] = np.nan
        return out, speed


def _build_skeleton(events: MatchEvents, n_slots: int) -> _Skeleton:
    t, x, z = [], [], []
    for slot in range(n_slots):
        a = events.anchors[events.anchors["slot"] == slot]
        a = a[np.argsort(a["t"], kind="stable")]
        # Duplicate timestamps break np.interp's monotonicity requirement.
        keep = np.concatenate([[True], np.diff(a["t"]) > 0]) if a.size else np.zeros(0, bool)
        a = a[keep]
        t.append(a["t"].astype(np.float64))
        x.append(a["x"].astype(np.float64))
        z.append(a["z"].astype(np.float64))
    return _Skeleton(t, x, z)


@dataclass(frozen=True, slots=True)
class _TrajectoryEstimate:
    """Position source backed by a previously integrated trajectory.

    Used by iteration rounds after the first. Between anchors it is far more accurate
    than linear interpolation — it follows the actual order polylines — at the cost of
    depending on the previous round's assignments, which is why the number of rounds is
    bounded rather than iterated to convergence.
    """

    pos: np.ndarray  # f8[n_ticks, n_slots, 2]
    valid: np.ndarray  # bool[n_ticks, n_slots]
    dt: float

    def _frames(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = np.clip(times / self.dt, 0.0, self.pos.shape[0] - 1.0)
        return np.floor(f).astype(np.int64), f - np.floor(f)

    def positions_at(self, times: np.ndarray, slot: int) -> tuple[np.ndarray, np.ndarray]:
        k, frac = self._frames(times)
        k1 = np.minimum(k + 1, self.pos.shape[0] - 1)
        ok = self.valid[k, slot] & self.valid[k1, slot]
        x = self.pos[k, slot, 0] * (1 - frac) + self.pos[k1, slot, 0] * frac
        z = self.pos[k, slot, 1] * (1 - frac) + self.pos[k1, slot, 1] * frac
        return np.where(ok, x, np.nan), np.where(ok, z, np.nan)

    def gap_at(self, times: np.ndarray, _slot: int) -> np.ndarray:
        return np.zeros(times.shape)

    def out_of_range(self, times: np.ndarray, slot: int, _slack: float) -> np.ndarray:
        k, _ = self._frames(times)
        return ~self.valid[k, slot]

    def heading_at(self, times: np.ndarray, slot: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
        ax, az = self.positions_at(times + dt, slot)
        bx, bz = self.positions_at(times - dt, slot)
        vx, vz = ax - bx, az - bz
        norm = np.hypot(vx, vz)
        speed = norm / (2 * dt)
        safe = np.where(norm > 0, norm, np.nan)
        out = np.stack([vx / safe, vz / safe], axis=1)
        out[~np.isfinite(speed) | (speed < 1e-9)] = np.nan
        return out, np.nan_to_num(speed)


# ---------------------------------------------------------------------------
# Pass C: integration
# ---------------------------------------------------------------------------
class _Track:
    """One champion's integrated position, following its most recent order."""

    __slots__ = ("cum", "known", "poly", "pos", "prog", "speed", "total")

    def __init__(self, speed: float) -> None:
        self.poly = np.zeros((1, 2))
        self.cum = np.zeros(1)
        self.total = 0.0
        self.prog = 0.0
        self.speed = speed
        self.pos = np.zeros(2)
        self.known = False

    def set_order(self, poly: np.ndarray) -> None:
        self.poly = poly
        seg = np.hypot(*np.diff(poly, axis=0).T) if poly.shape[0] > 1 else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.total = float(self.cum[-1])
        self.prog = 0.0
        self.pos = poly[0].copy()
        self.known = True

    def fix_at(self, x: float, z: float) -> None:
        """Reset to an observed position, discarding the current order.

        The order is dropped rather than re-based: an observation that disagrees with
        the integrated path means the path was wrong, and continuing along it from a
        corrected start would keep heading the wrong way.
        """
        self.pos = np.array([x, z])
        self.poly = self.pos[None, :].copy()
        self.cum = np.zeros(1)
        self.total = 0.0
        self.prog = 0.0
        self.known = True

    def advance(self, dt: float) -> None:
        if not self.known or self.total <= 0.0:
            return
        self.prog = min(self.total, self.prog + self.speed * dt)
        k = min(
            max(int(np.searchsorted(self.cum, self.prog, side="right")) - 1, 0),
            self.poly.shape[0] - 2,
        )
        span = self.cum[k + 1] - self.cum[k]
        f = (self.prog - self.cum[k]) / span if span > 0 else 0.0
        self.pos = self.poly[k] + (self.poly[k + 1] - self.poly[k]) * f


def _merged_timeline(events: MatchEvents, hp_zero: np.ndarray) -> np.ndarray:
    parts = []
    for kind, arr in (
        (_KIND_SPEED, events.speed),
        (_KIND_ANCHOR, events.anchors),
        (_KIND_ORDER, events.orders),
        (_KIND_DEATH, hp_zero),
    ):
        if arr.size == 0:
            continue
        block = np.empty(arr.size, dtype=[("t", "f8"), ("kind", "i1"), ("idx", "i8")])
        block["t"] = arr["t"]
        block["kind"] = kind
        block["idx"] = np.arange(arr.size)
        parts.append(block)
    if not parts:
        return np.empty(0, dtype=[("t", "f8"), ("kind", "i1"), ("idx", "i8")])
    merged = np.concatenate(parts)
    merged.sort(order=["t", "kind", "idx"], kind="stable")
    return merged


# ---------------------------------------------------------------------------
# Pass B: assignment
# ---------------------------------------------------------------------------
def _assign_orders(
    events: MatchEvents,
    skeleton: _Skeleton,
    n_slots: int,
    spec: AttributionSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Assign every order to a champion using the skeleton only."""
    n_orders = events.orders.size
    owner = np.full(n_orders, UNKNOWN, dtype=np.int8)
    residual = np.full(n_orders, np.nan)
    margin = np.full(n_orders, np.nan)
    if n_orders == 0:
        return owner, residual, margin, {}

    times = events.orders["t"].astype(np.float64)
    starts = np.empty((n_orders, 2))
    dirs = np.full((n_orders, 2), np.nan)
    for n in range(n_orders):
        poly = events.order_polyline(n)
        starts[n] = poly[0]
        if poly.shape[0] > 1:
            d = poly[1] - poly[0]
            norm = float(np.hypot(*d))
            if norm > 1e-9:
                dirs[n] = d / norm

    # Skeleton position, gap and heading for every (order, slot) pair, vectorised.
    hdt = 0.5  # half-window for the heading estimate, in seconds
    sx = np.empty((n_orders, n_slots))
    sz = np.empty((n_orders, n_slots))
    gap = np.empty((n_orders, n_slots))
    oor = np.empty((n_orders, n_slots), dtype=bool)
    head = np.empty((n_orders, n_slots, 2))
    hspeed = np.empty((n_orders, n_slots))
    for slot in range(n_slots):
        sx[:, slot], sz[:, slot] = skeleton.positions_at(times, slot)
        gap[:, slot] = skeleton.gap_at(times, slot)
        oor[:, slot] = skeleton.out_of_range(times, slot, spec.extrapolate_seconds)
        h, s = skeleton.heading_at(times, slot, hdt)
        head[:, slot] = h
        hspeed[:, slot] = s

    dist = np.hypot(sx - starts[:, 0:1], sz - starts[:, 1:2])
    # Direction penalty: 0 for aligned, 2 * weight for opposed. Skipped where either
    # heading is undetermined or the champion is barely moving.
    dot = head[:, :, 0] * dirs[:, None, 0] + head[:, :, 1] * dirs[:, None, 1]
    usable = np.isfinite(dot) & (hspeed >= spec.direction_min_speed)
    penalty = np.where(usable, spec.direction_weight * (1.0 - dot), 0.0)
    cost = dist + penalty
    gate = np.minimum(spec.max_gate, spec.base_gate + spec.gap_gate_rate * np.minimum(gap, 30.0))
    cost[oor | ~np.isfinite(cost) | (dist > gate)] = np.inf

    # Group simultaneous orders and assign jointly: several champions may issue at one
    # timestamp, and the globally cheapest pairing is what stops two co-located
    # champions both claiming the nearer order.
    # Ambiguity margin: how much better the chosen champion is than the runner-up.
    #
    # This is the honest, truth-free confidence signal for an attribution. A margin near
    # zero means two champions were equally good candidates — which, measured against
    # ground truth, is exactly when a "wrong" assignment is harmless, because the two
    # occupy the same point. Downstream consumers can discount positions derived from
    # low-margin orders instead of treating every attribution as equally certain.
    sorted_cost = np.sort(cost, axis=1)
    if n_slots > 1:
        best = sorted_cost[:, 0]
        runner = sorted_cost[:, 1]
        # Both infinite means no candidate passed the gate, so there is no margin to
        # report; subtracting would give nan-from-inf and a spurious warning.
        margin = np.full(n_orders, np.nan)
        ok = np.isfinite(best)
        np.subtract(runner, best, out=margin, where=ok)
    else:
        margin = np.full(n_orders, np.inf)

    stats = {"groups": 0, "ambiguous_groups": 0}
    start = 0
    while start < n_orders:
        end = start
        while end < n_orders and times[end] == times[start]:
            end += 1
        block = cost[start:end]
        stats["groups"] += 1
        pairs = [
            (float(block[i, s]), i, s)
            for i in range(block.shape[0])
            for s in range(n_slots)
            if np.isfinite(block[i, s])
        ]
        pairs.sort()
        if len(pairs) > (end - start):
            stats["ambiguous_groups"] += 1
        used_o: set[int] = set()
        used_s: set[int] = set()
        for _c, i, s in pairs:
            if i in used_o or s in used_s:
                continue
            used_o.add(i)
            used_s.add(s)
            owner[start + i] = s
            residual[start + i] = dist[start + i, s]
        start = end

    return owner, residual, margin, stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _integrate(
    events: MatchEvents,
    owner: np.ndarray,
    hp_zero: np.ndarray,
    n_slots: int,
    n_ticks: int,
    dt: float,
    spec: AttributionSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Walk the timeline, following assigned orders and resetting at every anchor."""
    tracks = [_Track(spec.default_speed) for _ in range(n_slots)]
    anchor_residual = np.full(events.anchors.size, np.nan)
    pos = np.full((n_ticks, n_slots, 2), np.nan)
    valid = np.zeros((n_ticks, n_slots), dtype=bool)
    speed_out = np.full((n_ticks, n_slots), spec.default_speed)

    timeline = _merged_timeline(events, hp_zero)
    suspect_anchors = 0
    deaths_seen = 0
    clock = 0.0
    next_tick = 0

    def record_until(t_limit: float) -> None:
        nonlocal clock, next_tick
        while next_tick < n_ticks and next_tick * dt <= t_limit + 1e-12:
            target = next_tick * dt
            if target > clock:
                for tr in tracks:
                    tr.advance(target - clock)
                clock = target
            for slot, tr in enumerate(tracks):
                if tr.known:
                    pos[next_tick, slot] = tr.pos
                    valid[next_tick, slot] = True
                speed_out[next_tick, slot] = tr.speed
            next_tick += 1

    cursor = 0
    n_events = timeline.size
    while cursor < n_events:
        t = float(timeline[cursor]["t"])
        record_until(t)
        if t > clock:
            for tr in tracks:
                tr.advance(t - clock)
            clock = t

        end_idx = cursor
        while end_idx < n_events and timeline[end_idx]["t"] == timeline[cursor]["t"]:
            end_idx += 1
        group = timeline[cursor:end_idx]

        for row in group[group["kind"] == _KIND_SPEED]:
            rec = events.speed[int(row["idx"])]
            value = float(rec["value"])
            if C.MOVE_SPEED_MIN <= value <= C.MOVE_SPEED_MAX:
                tracks[int(rec["slot"])].speed = value

        for row in group[group["kind"] == _KIND_ANCHOR]:
            ai = int(row["idx"])
            rec = events.anchors[ai]
            tr = tracks[int(rec["slot"])]
            if tr.known:
                err = float(np.hypot(tr.pos[0] - rec["x"], tr.pos[1] - rec["z"]))
                anchor_residual[ai] = err
                if err > spec.suspect_anchor_error:
                    suspect_anchors += 1
            tr.fix_at(float(rec["x"]), float(rec["z"]))

        for row in group[group["kind"] == _KIND_ORDER]:
            oi = int(row["idx"])
            slot = int(owner[oi])
            if slot != UNKNOWN:
                tracks[slot].set_order(events.order_polyline(oi))

        for row in group[group["kind"] == _KIND_DEATH]:
            # A dead champion is not where its last order was heading, and the stream
            # carries no respawn timer. Marking the estimate unknown makes the next
            # anchor re-establish it, which is self-healing given how dense anchors are.
            tracks[int(hp_zero[int(row["idx"])]["slot"])].known = False
            deaths_seen += 1

        cursor = end_idx

    record_until((n_ticks - 1) * dt)
    return pos, valid, speed_out, anchor_residual, suspect_anchors, deaths_seen


def attribute(events: MatchEvents, spec: AttributionSpec | None = None) -> Attribution:
    """Assign every movement order to a champion and reconstruct trajectories.

    Alternates assignment and integration for `spec.iterations` rounds. The first round
    assigns from the anchor skeleton, which cannot propagate an error into its own later
    decisions; subsequent rounds reassign from the previous round's integrated
    trajectory, which is much more accurate between anchors because it follows the real
    order polylines rather than interpolating across them.
    """
    spec = spec or AttributionSpec()
    n_slots = max(1, events.n_heroes)
    dt = 1.0 / spec.tick_hz
    n_ticks = round(events.duration * spec.tick_hz) + 1
    hp_zero = events.hp[events.hp["value"] == 0.0] if events.hp.size else events.hp

    estimate: Any = _build_skeleton(events, n_slots)
    owner = np.full(events.orders.size, UNKNOWN, dtype=np.int8)
    order_residual = np.full(events.orders.size, np.nan)
    order_margin = np.full(events.orders.size, np.nan)
    assign_stats: dict[str, int] = {}
    pos = valid = speed_out = anchor_residual = None
    suspect_anchors = deaths_seen = 0
    changed_per_round: list[int] = []

    for round_index in range(max(1, spec.iterations)):
        previous = owner.copy()
        owner, order_residual, order_margin, assign_stats = _assign_orders(
            events, estimate, n_slots, spec
        )
        changed_per_round.append(int((owner != previous).sum()))
        pos, valid, speed_out, anchor_residual, suspect_anchors, deaths_seen = _integrate(
            events, owner, hp_zero, n_slots, n_ticks, dt, spec
        )
        if round_index + 1 < max(1, spec.iterations):
            estimate = _TrajectoryEstimate(pos=pos, valid=valid, dt=dt)

    stats = {
        "unattributed": int((owner == UNKNOWN).sum()),
        "suspect_anchors": suspect_anchors,
        "deaths_seen": deaths_seen,
        "valid_tick_fraction": round(float(valid.mean()), 4),
        "iterations": max(1, spec.iterations),
        "reassigned_per_round": changed_per_round,
        "orders_per_slot": np.bincount(
            owner[owner != UNKNOWN].astype(np.int64), minlength=n_slots
        ).tolist(),
        "confident_fraction": round(
            float((order_margin[owner != UNKNOWN] >= 100.0).mean())
            if (owner != UNKNOWN).any()
            else 0.0,
            4,
        ),
        **assign_stats,
    }
    return Attribution(
        owner=owner,
        pos=pos,
        valid=valid,
        speed=speed_out,
        order_residual=order_residual,
        order_margin=order_margin,
        anchor_residual=anchor_residual,
        stats=stats,
    )


def with_owners(events: MatchEvents, attribution: Attribution) -> MatchEvents:
    """Return a copy of `events` with order ownership filled in."""
    import dataclasses

    orders = events.orders.copy()
    orders["owner"] = attribution.owner
    return dataclasses.replace(events, orders=orders)
