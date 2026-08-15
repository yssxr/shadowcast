"""The belief filter: ten particle clouds, stepped together.

Two observing teams times five enemies is ten distributions, each over "where is that
champion right now, given everything this team could have seen". They share one state
array and one code path; a baseline is a `FilterSpec`, not a separate implementation,
which is what makes the ablation table trustworthy — `behavioural` and `full` differ in
one enum value, so a gap between them cannot be an artefact of one having been written
more carefully.

**The filter never sees the answer.** `run` takes an `Observation`, a `PublicInfo` and
the observer's own visibility masks. It does not take a `TruthTable`, and there is no
path from here to one. Scoring happens outside, in `metrics`, by zipping this generator
against the truth — so the code that knows where the enemy is and the code that guesses
never meet. See `policy.py` for why that is enforced structurally rather than by care.

**Randomness is drawn from a seeded generator per tick and passed down**, never generated
inside a kernel. Numba's RNG is a different stream from NumPy's, so a kernel that draws
its own noise can never be compared against a NumPy reference, and two runs of the same
seed would not be bit-identical — which the barrier test requires.

A note on what happens when an enemy is dead. Deaths are public, so the belief collapses:
a dead champion's position is known, and a filter that kept spreading probability over
the map during a 30-second respawn would report enormous uncertainty at exactly the
moment everyone on the server knows precisely where five of the ten champions are. The
`darkness` metric treats those ticks separately for the same reason — getting this
backwards makes a team look informationally dominant precisely when it is winning fights.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast import sr
from shadowcast.config import FilterSpec
from shadowcast.geom.grid import cell_to_world, flat_index, world_to_cell
from shadowcast.l3_infer import motion, observation
from shadowcast.l3_infer.policy import NO_CELL, Observation, PublicInfo
from shadowcast.l3_infer.reachability import ReachabilityIndex
from shadowcast.terrain.terrain import Terrain

__all__ = ["BeliefFilter", "BeliefState", "TickBelief"]


@dataclass(slots=True)
class BeliefState:
    """`(2, 5, P)` particles, plus the little that has to persist between ticks."""

    cell: np.ndarray  # i4[2, 5, P]
    logw: np.ndarray  # f8[2, 5, P]
    heading: np.ndarray  # i1[2, 5, P]
    goal: np.ndarray  # i4[2, 5, P] — each particle's current destination
    vel: np.ndarray  # f8[2, 5, 2] — cells/tick at the last sighting, for `constant_velocity`
    last_cell: np.ndarray  # i4[2, 5] — last observed cell, NO_CELL before the first sighting
    last_tick: np.ndarray  # i4[2, 5]
    was_alive: np.ndarray  # bool[2, 5]
    depletions: np.ndarray  # i4[2, 5]
    resamples: np.ndarray  # i4[2, 5]


@dataclass(frozen=True, slots=True)
class TickBelief:
    """One tick of belief, as views into the filter's own buffers.

    The arrays are **not copied**. A consumer that stores one gets the next tick's
    contents, which is the same bargain `VisionStream.masks` makes and for the same
    reason — copying `(2, 5, 400)` particles for 7,201 ticks is 115 MB of garbage to
    produce a number that is consumed immediately.
    """

    tick: int
    cell: np.ndarray  # i4[2, 5, P]
    logw: np.ndarray  # f8[2, 5, P]
    seen: np.ndarray  # bool[2, 5]
    alive: np.ndarray  # bool[2, 5]


class BeliefFilter:
    """One filter configuration, run over one match."""

    def __init__(
        self,
        spec: FilterSpec,
        terrain: Terrain,
        reach: ReachabilityIndex | None = None,
    ) -> None:
        self.spec = spec
        self.terrain = terrain
        self.grid = terrain.grid
        self.reach = reach if reach is not None else ReachabilityIndex(terrain)
        self.n_sub = spec.sub_steps
        self.p = spec.particles

        self._walkable_cells = np.flatnonzero(terrain.walkable.reshape(-1)).astype(np.int32)
        self._walkable_flat = terrain.walkable.reshape(-1)
        self._targets: dict[str, np.ndarray] = {}

    # -- the behavioural prior ------------------------------------------
    def _targets_for(self, role: str, team: int) -> np.ndarray:
        """Cached per (role, team) — the team decides which fountain they recall to."""
        key = f"{role}/{team}"
        if key not in self._targets:
            self._targets[key] = motion.role_targets(role, self.terrain, team)
        return self._targets[key]

    # -- setup ---------------------------------------------------------
    def initial_state(self) -> BeliefState:
        """Everyone starts on their own fountain, which is not an assumption.

        A match begins with all ten champions standing in their base, visible to their
        own team and known by position to the other — it is the one moment of the game
        with no information asymmetry at all. Starting from a uniform prior over the map
        would be modelling ignorance nobody has.
        """
        cell = np.empty((C.N_TEAMS, C.N_ENEMIES, self.p), dtype=np.int32)
        for observer in range(C.N_TEAMS):
            fountain = sr.FOUNTAINS[1 - observer]
            i, j = world_to_cell(float(fountain[0]), float(fountain[1]))
            cell[observer, :, :] = flat_index(i, j)
        return BeliefState(
            cell=cell,
            logw=np.zeros((C.N_TEAMS, C.N_ENEMIES, self.p)),
            heading=np.full((C.N_TEAMS, C.N_ENEMIES, self.p), motion.STAY, dtype=np.int8),
            goal=cell.copy(),
            vel=np.zeros((C.N_TEAMS, C.N_ENEMIES, 2)),
            last_cell=cell[:, :, 0].copy(),
            last_tick=np.zeros((C.N_TEAMS, C.N_ENEMIES), dtype=np.int32),
            was_alive=np.ones((C.N_TEAMS, C.N_ENEMIES), dtype=bool),
            depletions=np.zeros((C.N_TEAMS, C.N_ENEMIES), dtype=np.int32),
            resamples=np.zeros((C.N_TEAMS, C.N_ENEMIES), dtype=np.int32),
        )

    # -- motion --------------------------------------------------------
    def _move(
        self,
        state: BeliefState,
        o: int,
        e: int,
        tick: int,
        role: str,
        enemy_team: int,
        rng: np.random.Generator,
    ) -> None:
        spec = self.spec
        cell = state.cell[o, e]
        elapsed = max(tick - int(state.last_tick[o, e]), 0) / C.TICK_HZ
        radius = spec.v_max * elapsed

        if spec.motion == "uniform":
            idx = rng.integers(0, self._walkable_cells.size, size=self.p)
            cell[:] = self._walkable_cells[idx]
            state.logw[o, e].fill(0.0)
            return

        if spec.motion == "disc":
            # Deliberately NOT clamped to the navmesh. This is the naive overlay every
            # other tool draws, and letting it put probability mass inside walls is the
            # honest representation of it -- `geodisc` is the ablation that adds terrain,
            # and the gap between them is the measurement.
            seed = int(state.last_cell[o, e])
            if seed == NO_CELL:
                seed = int(cell[0])
            cx, cz = cell_to_world(seed % self.grid, seed // self.grid)
            theta = rng.uniform(0.0, 2.0 * np.pi, self.p)
            r = radius * np.sqrt(rng.uniform(0.0, 1.0, self.p))
            x = np.clip(cx + r * np.cos(theta), C.WORLD_MIN_X, C.WORLD_MIN_X + C.WORLD_SPAN - 1)
            z = np.clip(cz + r * np.sin(theta), C.WORLD_MIN_Z, C.WORLD_MIN_Z + C.WORLD_SPAN - 1)
            i = ((x - C.WORLD_MIN_X) / C.GRID_CELL_SIZE).astype(np.int32)
            j = ((z - C.WORLD_MIN_Z) / C.GRID_CELL_SIZE).astype(np.int32)
            np.clip(i, 0, self.grid - 1, out=i)
            np.clip(j, 0, self.grid - 1, out=j)
            cell[:] = j * self.grid + i
            state.logw[o, e].fill(0.0)
            return

        if spec.motion == "geodisc":
            seed = int(state.last_cell[o, e])
            if seed == NO_CELL:
                seed = int(cell[0])
            cell[:] = self.reach.sample(
                seed, radius, self.p, rng.uniform(size=(self.p, 2)), blocked=None
            )
            state.logw[o, e].fill(0.0)
            return

        if spec.motion == "constant_velocity":
            # No terrain clamp, by construction: B2 exists to show what extrapolation
            # without a navmesh is worth, and clamping it would be quietly making the
            # baseline better than the thing it stands for.
            v = state.vel[o, e]
            spread = 0.35 * (1.0 + elapsed)
            j = cell // self.grid
            i = cell - j * self.grid
            i = i + v[0] + rng.normal(0.0, spread, self.p)
            j = j + v[1] + rng.normal(0.0, spread, self.p)
            ii = np.clip(i.astype(np.int32), 0, self.grid - 1)
            jj = np.clip(j.astype(np.int32), 0, self.grid - 1)
            cell[:] = jj * self.grid + ii
            return

        beta = spec.effective_goal_beta
        if beta > 0.0:
            motion.refresh_goals(
                cell,
                state.goal[o, e],
                self._targets_for(role, enemy_team),
                self.grid,
                spec.goal_arrive_cells,
                rng,
            )
        motion.propose_cells(
            cell,
            state.heading[o, e],
            self.terrain.walkable,
            self.n_sub,
            spec.p_stay,
            spec.persistence,
            state.goal[o, e],
            beta,
            rng.uniform(size=(self.p, self.n_sub)),
            cell,
            state.heading[o, e],
        )

    # -- the tick ------------------------------------------------------
    def run(
        self,
        obs: Observation,
        public: PublicInfo,
        masks: Iterator[tuple[int, np.ndarray, np.ndarray]],
        rng: np.random.Generator | None = None,
    ) -> Iterator[TickBelief]:
        """Step the ten filters over the match, yielding the belief at every tick.

        `masks` is the observer's own visibility — its own information, not the enemy's,
        which is why it is allowed through the barrier. It arrives as the same generator
        `VisionStream.masks` produces, consumed once.
        """
        rng = rng if rng is not None else np.random.default_rng(self.spec.seed)
        state = self.initial_state()
        pd = np.empty(self.p, dtype=np.float64)
        idx = np.empty(self.p, dtype=np.int32)
        spec = self.spec
        ess_resample = spec.ess_resample * self.p
        ess_depletion = spec.ess_depletion * self.p

        for tick, mask_order, mask_chaos in masks:
            if tick >= obs.n_ticks:
                break
            team_masks = (mask_order, mask_chaos)
            for o in range(C.N_TEAMS):
                blocked = None  # built lazily, and only when a depletion needs it
                for e in range(C.N_ENEMIES):
                    alive = bool(public.alive[tick, o, e])
                    if not alive:
                        # Public knowledge: they are dead and their position is not in
                        # question. Freeze rather than diffuse.
                        state.cell[o, e].fill(max(int(state.last_cell[o, e]), 0))
                        state.logw[o, e].fill(0.0)
                        state.was_alive[o, e] = False
                        continue
                    if not state.was_alive[o, e]:
                        # Just respawned: the fountain is public too.
                        fountain = sr.FOUNTAINS[1 - o]
                        fi, fj = world_to_cell(float(fountain[0]), float(fountain[1]))
                        state.cell[o, e].fill(flat_index(fi, fj))
                        state.logw[o, e].fill(0.0)
                        state.heading[o, e].fill(motion.STAY)
                        state.last_cell[o, e] = flat_index(fi, fj)
                        state.last_tick[o, e] = tick
                    state.was_alive[o, e] = True

                    role = str(public.enemy_role[o, e]) or "unknown"
                    self._move(state, o, e, tick, role, 1 - o, rng)

                    if obs.seen[tick, o, e] and spec.obs != "none":
                        cur = int(obs.cell[tick, o, e])
                        prev = int(state.last_cell[o, e])
                        gap = tick - int(state.last_tick[o, e])
                        if prev != NO_CELL and 0 < gap <= C.TICK_HZ:
                            state.vel[o, e] = (
                                (cur % self.grid - prev % self.grid) / gap,
                                (cur // self.grid - prev // self.grid) / gap,
                            )
                        observation.collapse_to_cell(state.cell[o, e], state.logw[o, e], cur)
                        state.heading[o, e].fill(motion.STAY)
                        state.last_cell[o, e] = cur
                        state.last_tick[o, e] = tick
                        continue

                    if spec.obs == "positive_and_negative":
                        observation.detection_field(
                            state.cell[o, e],
                            team_masks[o],
                            self.grid,
                            spec.pd_edge_ring_cells,
                            spec.pd_interior,
                            spec.pd_edge,
                            pd,
                        )
                        observation.negative_update(state.logw[o, e], pd)

                        ess = observation.effective_sample_size(state.logw[o, e])
                        if ess < ess_depletion:
                            if blocked is None:
                                blocked = self._blocked_flat(team_masks[o])
                            seed = int(state.last_cell[o, e])
                            if seed == NO_CELL:
                                seed = int(state.cell[o, e, 0])
                            elapsed = max(tick - int(state.last_tick[o, e]), 1) / C.TICK_HZ
                            state.cell[o, e] = self.reach.sample(
                                seed,
                                spec.v_max * elapsed,
                                self.p,
                                rng.uniform(size=(self.p, 2)),
                                blocked=blocked,
                            )
                            state.logw[o, e].fill(0.0)
                            state.heading[o, e].fill(motion.STAY)
                            state.depletions[o, e] += 1
                        elif ess < ess_resample:
                            observation.systematic_resample(
                                state.logw[o, e], float(rng.uniform()), idx
                            )
                            state.cell[o, e] = state.cell[o, e][idx]
                            state.heading[o, e] = state.heading[o, e][idx]
                            state.logw[o, e].fill(0.0)
                            state.resamples[o, e] += 1

            yield TickBelief(
                tick=tick,
                cell=state.cell,
                logw=state.logw,
                seen=obs.seen[tick],
                alive=public.alive[tick],
            )
        self.state = state

    def _blocked_flat(self, mask: np.ndarray) -> np.ndarray:
        from shadowcast.fov.union import mask_to_bool

        return np.ascontiguousarray(mask_to_bool(mask, self.grid).reshape(-1))

    def describe(self) -> dict[str, Any]:
        state = getattr(self, "state", None)
        out: dict[str, Any] = {
            "motion": self.spec.motion,
            "obs": self.spec.obs,
            "particles": self.p,
            "sub_steps": self.n_sub,
            "config_hash": self.spec.content_hash,
        }
        if state is not None:
            out["depletion_events"] = int(state.depletions.sum())
            out["resample_events"] = int(state.resamples.sum())
        out.update(self.reach.describe())
        return out
