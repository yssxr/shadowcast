"""Inferring each champion's role from where it went and what it did.

The corpus contains no role field, and unlike team there is no entity that states one
outright, so this is the softest inference in the pipeline, and it is the one whose
accuracy should be quoted most carefully.

Two kinds of evidence, and mixing them is what makes it work:

**Where a champion spends the laning phase.** A top laner is near the top lane, a jungler
is near none of the three lanes. That separates top, mid and jungle cleanly, and puts two
champions in the bottom lane.

**Who places the wards.** Distinguishing the bottom laner from the support by position
alone is unreliable. They stand together by design, but ward ownership is *directly
observed*: `SpawnMinion` gives a ward's position and `targetable_on_client` gives the
owning champion's net_id. Supports ward far more than carries do, so this is a real
measurement rather than a behavioural guess, and it is what decides the pair.

**The one-of-each constraint does the rest.** Every team fields exactly one of each role,
so the five champions are assigned jointly by exhausting all 120 permutations and taking
the best total score. That stops two champions both being called the jungler and lets a
strong signal for one role settle a weak one for another.

Roles are only inferred once teams are known, because the constraint is per team.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np

from shadowcast import constants as C
from shadowcast import sr

if TYPE_CHECKING:
    from shadowcast.l1_events.resolve.attribute import Attribution

from shadowcast.l1_events.schema import MatchEvents

__all__ = ["ROLES", "RoleResolution", "resolve_roles"]

ROLES: tuple[str, ...] = ("top", "jungle", "mid", "bot", "support")

#: The laning phase, in seconds. Before this champions are still walking out of base;
#: after it, roles stop predicting position because everyone roams and groups.
_LANE_WINDOW = (90.0, 600.0)
#: A champion within this distance of a lane's centre line counts as being in that lane.
_LANE_RADIUS = 1400.0
#: Weight on ward share when scoring the support role, in the same units as the
#: occupancy fractions it is added to.
_WARD_WEIGHT = 1.5


@dataclass(frozen=True, slots=True)
class RoleResolution:
    role: np.ndarray  # U8[n_slots]
    lane_time: np.ndarray  # f8[n_slots, 3] fractions for top/mid/bot
    jungle_time: np.ndarray  # f8[n_slots]
    ward_share: np.ndarray  # f8[n_slots] share of own team's wards
    score: np.ndarray  # f8[n_slots] score of the chosen assignment
    stats: dict[str, object]

    @property
    def resolved(self) -> bool:
        return bool((self.role != "").all())


def _lane_reference() -> dict[str, np.ndarray]:
    """Densely resampled lane centre lines, for nearest-lane tests."""
    out: dict[str, np.ndarray] = {}
    for lane, pts in sr.LANES.items():
        samples = np.linspace(0.0, 1.0, 400)
        out[lane] = np.stack([sr.lerp_polyline(pts, s) for s in samples])
    return out


def _occupancy(
    pos: np.ndarray, valid: np.ndarray, tick_hz: int, n_slots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of laning-phase time each champion spends in each lane, and off-lane."""
    lanes = _lane_reference()
    lane_names = ("top", "mid", "bot")
    lo = int(_LANE_WINDOW[0] * tick_hz)
    hi = min(int(_LANE_WINDOW[1] * tick_hz), pos.shape[0])

    lane_time = np.zeros((n_slots, 3))
    jungle_time = np.zeros(n_slots)
    for slot in range(n_slots):
        ok = valid[lo:hi, slot]
        if not ok.any():
            continue
        p = pos[lo:hi, slot][ok]
        dists = np.empty((3, p.shape[0]))
        for k, lane in enumerate(lane_names):
            ref = lanes[lane]
            # Nearest sample on each lane's centre line, per tick.
            d = np.hypot(p[:, 0][:, None] - ref[None, :, 0], p[:, 1][:, None] - ref[None, :, 1])
            dists[k] = d.min(axis=1)
        nearest = dists.argmin(axis=0)
        close = dists.min(axis=0) <= _LANE_RADIUS
        for k in range(3):
            lane_time[slot, k] = float(((nearest == k) & close).mean())
        jungle_time[slot] = float((~close).mean())
    return lane_time, jungle_time


def _ward_share(events: MatchEvents, team: np.ndarray, n_slots: int) -> np.ndarray:
    """Each champion's share of its own team's wards.

    Directly observed rather than inferred: the ward's `targetable_on_client` names the
    owner. This is what separates the support from the bottom laner, who are otherwise
    standing in the same place on purpose.
    """
    share = np.zeros(n_slots)
    if events.wards.size == 0:
        return share
    counts = np.zeros(n_slots)
    for ward in events.wards:
        owner = int(ward["owner_slot"])
        if 0 <= owner < n_slots:
            counts[owner] += 1
    for t in (C.TEAM_ORDER, C.TEAM_CHAOS):
        members = np.flatnonzero(team == t)
        total = counts[members].sum()
        if total > 0:
            share[members] = counts[members] / total
    return share


def _role_scores(lane_time, jungle_time, ward_share, slot) -> dict[str, float]:
    """Score for placing this champion in each role. Higher is better."""
    top, mid, bot = lane_time[slot]
    return {
        "top": top,
        "mid": mid,
        "bot": bot - _WARD_WEIGHT * ward_share[slot],
        "support": bot + _WARD_WEIGHT * ward_share[slot],
        "jungle": jungle_time[slot],
    }


def resolve_roles(
    events: MatchEvents,
    pos: np.ndarray,
    valid: np.ndarray,
    tick_hz: int = C.TICK_HZ,
) -> RoleResolution:
    """Infer roles from reconstructed trajectories and observed ward ownership."""
    n_slots = max(1, events.n_heroes)
    team = events.heroes["team"].astype(np.int64)
    role = np.full(n_slots, "", dtype="U8")
    score = np.zeros(n_slots)

    lane_time, jungle_time = _occupancy(pos, valid, tick_hz, n_slots)
    ward_share = _ward_share(events, team, n_slots)

    unresolved_teams = 0
    for t in (C.TEAM_ORDER, C.TEAM_CHAOS):
        members = np.flatnonzero(team == t)
        if members.size != len(ROLES):
            unresolved_teams += 1
            continue
        table = [_role_scores(lane_time, jungle_time, ward_share, s) for s in members]
        # Exhaustive over 120 permutations: one of each role per team is a hard fact, and
        # enforcing it jointly lets a confident signal for one role settle a weak one for
        # another instead of two champions both being called the jungler.
        best_total, best_perm = -np.inf, tuple(ROLES)
        for perm in permutations(ROLES):
            total = sum(table[k][perm[k]] for k in range(len(ROLES)))
            if total > best_total:
                best_total, best_perm = total, perm
        for k, slot in enumerate(members):
            role[slot] = best_perm[k]
            score[slot] = table[k][best_perm[k]]

    return RoleResolution(
        role=role,
        lane_time=lane_time,
        jungle_time=jungle_time,
        ward_share=ward_share,
        score=score,
        stats={
            "teams_without_five_members": unresolved_teams,
            "resolved": int((role != "").sum()),
            "mean_score": round(float(score.mean()), 3),
        },
    )


def with_roles(events: MatchEvents, resolution: RoleResolution) -> MatchEvents:
    heroes = events.heroes.copy()
    heroes["role"] = resolution.role
    return dataclasses.replace(events, heroes=heroes)


def resolve_all(
    events: MatchEvents,
    attribution: Attribution,
    tick_hz: int = C.TICK_HZ,
) -> tuple[MatchEvents, dict[str, object]]:
    """Run every resolver in dependency order and return the filled-in events.

    Teams first, because roles are constrained per team and fog observer teams are
    derived from champion teams. Deaths are independent of both.

    Takes the whole `Attribution` rather than its `pos`/`valid` arrays because order
    ownership belongs on the events too. Passing the two arrays was how ownership came to
    be silently dropped: the attributor computed it, nothing wrote it back, and
    `describe()` reported `orders_attributed: False` on every match including ones where
    91% of ticks had a recovered position.
    """
    from shadowcast.l1_events.resolve.attribute import with_owners
    from shadowcast.l1_events.resolve.deaths import resolve_deaths, with_deaths
    from shadowcast.l1_events.resolve.teams import resolve_teams, with_teams

    pos, valid = attribution.pos, attribution.valid
    events = with_owners(events, attribution)

    team_res = resolve_teams(events)
    events = with_teams(events, team_res)

    death_res = resolve_deaths(events)
    events = with_deaths(events, death_res)

    role_res = resolve_roles(events, pos, valid, tick_hz)
    events = with_roles(events, role_res)

    return events, {
        "teams": {
            "method": team_res.method,
            "min_margin": round(team_res.min_margin, 1),
            **team_res.stats,
        },
        "deaths": death_res.stats,
        "roles": role_res.stats,
        "orders": attribution.describe(),
    }
