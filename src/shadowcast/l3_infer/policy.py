"""The information barrier: what a belief filter is allowed to know.

This module exists because the central claim of the project — that reconstructing
*negative* information produces better position estimates than diffusion alone — is
worth exactly nothing if the filter can see the answer. A leak here does not crash, does
not look wrong, and produces numbers that are better than the honest ones. It would be
found, if at all, by a reader rather than by us.

So the barrier is structural rather than aspirational. Truth enters through exactly one
function, `observe`, and leaves it as an `Observation`: a boolean per (observer, enemy,
tick) and a cell index that is meaningful **only where that boolean is true**. Everything
else a filter is entitled to know is a `PublicInfo`, assembled from facts that were on
every player's screen — the clock, the kill feed, respawn timers, who plays which role.
`pf.step` takes those two and a pre-drawn randomness array. It holds no reference to the
trajectory table, and there is no path from a filter to a position it did not observe.

`tests/test_policy.py` then checks the barrier the only way it can be checked: it moves
every unobserved enemy 2,000 units and asserts the filter's output is bit-identical.
A structural argument is what makes that test cheap; the test is what makes the
structural argument true.

One asymmetry worth naming. Deaths, respawn timers and levels *are* public — the game
puts them on the scoreboard — so a filter that ignores them is not being principled, it
is being under-informed, and it would understate what a real player knows. Being wrong
in the flattering direction and being wrong in the modest direction are both wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast.fov.union import mask_bit
from shadowcast.geom.grid import flat_index, world_to_cell
from shadowcast.l1_events.resolve.attribute import Attribution
from shadowcast.l1_events.schema import UNKNOWN, MatchEvents
from shadowcast.l2_reconstruct.vision import VisionStream

__all__ = ["Observation", "PublicInfo", "TruthTable", "observe"]

#: Cell sentinel for "no position", used wherever `seen` is false so that reading the
#: cell without checking the flag fails loudly rather than silently returning cell 0.
NO_CELL = -1


@dataclass(frozen=True, slots=True)
class PublicInfo:
    """Everything a filter may condition on that is not an observation.

    Indexed `[observer_team, enemy_index]` throughout, where `enemy_index` runs 0..4 over
    the members of the opposing team. `enemy_slot` maps that back to a hero row; it is
    public because champion select is public.
    """

    #: i1[2, 5] — hero slot of each enemy, from the observer's point of view.
    enemy_slot: np.ndarray
    #: U8[2, 5] — resolved role. Public: everyone can see who walked to the jungle.
    enemy_role: np.ndarray
    #: bool[n_ticks, 2, 5] — alive, from the kill feed. A dead enemy's position is known.
    alive: np.ndarray
    #: f8[n_ticks] — match clock in seconds.
    clock: np.ndarray
    tick_hz: int = C.TICK_HZ

    @property
    def n_ticks(self) -> int:
        return int(self.alive.shape[0])


@dataclass(frozen=True, slots=True)
class Observation:
    """The one channel through which truth reaches a filter.

    `cell` is `NO_CELL` wherever `seen` is false. That is deliberate: a filter that
    forgets to check `seen` gets an out-of-range index rather than a plausible position,
    which turns a silent leak into an exception.
    """

    #: bool[n_ticks, 2, 5]
    seen: np.ndarray
    #: i4[n_ticks, 2, 5] — flat cell index, valid only where `seen`.
    cell: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_ticks(self) -> int:
        return int(self.seen.shape[0])

    def visible_fraction(self) -> float:
        return float(self.seen.mean()) if self.seen.size else 0.0


@dataclass(frozen=True, slots=True)
class TruthTable:
    """Where the enemies actually were. **For scoring only.**

    Kept in a separate type from `Observation` for the same reason a loaded firearm is
    kept in a separate room from the ammunition: so that passing the wrong one to
    `pf.step` is a type error a reader can see, not a subtle mistake in an index.

    Nothing in `l3_infer` outside `metrics` and the validators may import this.
    """

    #: i4[n_ticks, 2, 5] — flat cell index, `NO_CELL` where the trajectory has no estimate.
    cell: np.ndarray
    #: f8[n_ticks, 2, 5, 2] — world position, NaN where unknown.
    pos: np.ndarray
    #: bool[n_ticks, 2, 5] — whether the trajectory had an estimate at all.
    valid: np.ndarray


def _enemy_index(events: MatchEvents) -> np.ndarray:
    """`[observer_team, enemy_index] -> hero slot`.

    Sorted by slot so the mapping is deterministic across runs, which matters because a
    seeded filter's output is compared bit-for-bit in the barrier test.
    """
    out = np.full((C.N_TEAMS, C.N_ENEMIES), UNKNOWN, dtype=np.int8)
    for observer in (C.TEAM_ORDER, C.TEAM_CHAOS):
        enemies = np.sort(events.heroes["slot"][events.heroes["team"] == 1 - observer])
        if enemies.size != C.N_ENEMIES:
            raise ValueError(
                f"team {1 - observer} has {enemies.size} members, not {C.N_ENEMIES}; "
                "teams must be resolved before belief can be computed"
            )
        out[observer] = enemies
    return out


def _alive_table(events: MatchEvents, enemy_slot: np.ndarray, n_ticks: int, dt: float):
    """Alive/dead per tick from the kill feed.

    Deaths are inferred rather than recorded (the corpus has no death packet), so this
    inherits that uncertainty — but it is inherited *publicly*: a filter using the kill
    feed is using what the scoreboard showed, and if our kill feed is wrong then so is
    the belief, in the same direction a mistaken player's would be.
    """
    alive = np.ones((n_ticks, C.N_TEAMS, C.N_ENEMIES), dtype=bool)
    if not events.deaths.size:
        return alive
    for death in events.deaths:
        victim = int(death["victim"])
        t0 = float(death["t"])
        t1 = float(death["respawn_t"])
        if not np.isfinite(t1) or t1 <= t0:
            t1 = t0 + C.RESPAWN_FALLBACK_SECONDS
        lo = max(0, int(np.ceil(t0 / dt)))
        hi = min(n_ticks, int(np.ceil(t1 / dt)))
        for observer in (C.TEAM_ORDER, C.TEAM_CHAOS):
            hit = np.nonzero(enemy_slot[observer] == victim)[0]
            if hit.size:
                alive[lo:hi, observer, int(hit[0])] = False
    return alive


def observe(
    events: MatchEvents,
    attribution: Attribution,
    vision: VisionStream,
    tick_hz: int = C.TICK_HZ,
) -> tuple[Observation, PublicInfo, TruthTable]:
    """Run the vision stream and gate the trajectories through it.

    **This is the only function in the project that reads positions and writes something
    a filter consumes.** It returns the truth alongside, because the validators need it —
    but as a distinct object, so that handing it to a filter requires writing the wrong
    type rather than forgetting a flag.
    """
    dt = 1.0 / tick_hz
    enemy_slot = _enemy_index(events)
    n_ticks = int(attribution.pos.shape[0])
    grid = vision.terrain.grid

    seen = np.zeros((n_ticks, C.N_TEAMS, C.N_ENEMIES), dtype=bool)
    obs_cell = np.full((n_ticks, C.N_TEAMS, C.N_ENEMIES), NO_CELL, dtype=np.int32)
    truth_cell = np.full((n_ticks, C.N_TEAMS, C.N_ENEMIES), NO_CELL, dtype=np.int32)
    truth_pos = np.full((n_ticks, C.N_TEAMS, C.N_ENEMIES, 2), np.nan)
    truth_valid = np.zeros((n_ticks, C.N_TEAMS, C.N_ENEMIES), dtype=bool)

    for tick, mask_order, mask_chaos in vision.masks():
        if tick >= n_ticks:
            break
        masks = (mask_order, mask_chaos)
        for observer in (C.TEAM_ORDER, C.TEAM_CHAOS):
            for e in range(C.N_ENEMIES):
                slot = int(enemy_slot[observer, e])
                if slot == UNKNOWN or not attribution.valid[tick, slot]:
                    continue
                x, z = attribution.pos[tick, slot]
                i, j = world_to_cell(float(x), float(z))
                if not (0 <= i < grid and 0 <= j < grid):
                    continue
                cell = flat_index(i, j)
                truth_cell[tick, observer, e] = cell
                truth_pos[tick, observer, e] = (x, z)
                truth_valid[tick, observer, e] = True
                if mask_bit(masks[observer], i, j):
                    seen[tick, observer, e] = True
                    obs_cell[tick, observer, e] = cell

    public = PublicInfo(
        enemy_slot=enemy_slot,
        enemy_role=np.array(
            [[events.heroes["role"][s] for s in enemy_slot[o]] for o in range(C.N_TEAMS)],
            dtype="U8",
        ),
        alive=_alive_table(events, enemy_slot, n_ticks, dt),
        clock=np.arange(n_ticks) * dt,
        tick_hz=tick_hz,
    )
    observation = Observation(
        seen=seen,
        cell=obs_cell,
        stats={
            "visible_fraction": round(float(seen.mean()), 4) if seen.size else 0.0,
            "trajectory_coverage": round(float(truth_valid.mean()), 4) if truth_valid.size else 0.0,
        },
    )
    truth = TruthTable(cell=truth_cell, pos=truth_pos, valid=truth_valid)
    return observation, public, truth
