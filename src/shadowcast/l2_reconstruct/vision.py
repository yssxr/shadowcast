"""Assembling per-team visibility masks over a match.

Everything upstream exists to make this possible: resolved teams say whose vision is
whose, reconstructed trajectories say where the champions are, ward lifetimes say which
wards are alive, and the field-of-view table says what each source can see.

**The masks are streamed, never materialised.** A 512² mask for two teams at 8 Hz over a
fifteen-minute match is 472 MB, and nothing needs all of it at once — the belief filter
and the artifact exporter both consume it tick by tick. So this yields masks and moves on,
and the peak memory is two 32 KB buffers.

**Sources are layered by how often they change**, because recomputing every source every
tick would be wasteful in a way that compounds over 14,400 team-ticks:

    static   turrets              computed once
    semi     static | wards       recomputed only when a ward appears or expires
    dynamic  champions, minion waves, reveal-on-attack   every tick

A team mask is then a 32 KB copy of its semi-static layer plus a union of about eighteen
dynamic sources — six to twelve microseconds of work.

**Reveal-on-attack is gated, and the gate was measured rather than reasoned about.** The
game rule is that attacking from your team's fog reveals a 400-unit disc around the
attacker for 4.5 seconds (at patch 12.22 — it became 300 units and 2 seconds in V13.22).
Applying it unconditionally looks safe, on the argument that a reveal centred on an
already-visible champion lies inside vision the observer had anyway. It is not: see
`_reveal_sources`, where the number that killed that argument is recorded.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast import sr
from shadowcast.fov.table import MISS, FovTable
from shadowcast.fov.union import assemble, mask_bit, new_mask, union_sources
from shadowcast.geom.grid import world_to_cell
from shadowcast.l1_events.resolve.attribute import Attribution
from shadowcast.l1_events.schema import ANCHOR_ATTACK, UNKNOWN, MatchEvents
from shadowcast.l2_reconstruct.front import estimate_front
from shadowcast.terrain.terrain import Terrain

__all__ = ["UNKNOWN_TARGET_REVEALS", "SourceCounts", "VisionStream"]

#: Whether an attack on a target we cannot resolve grants a reveal.
#:
#: The rule needs an ENEMY target. Champions, turrets and wards resolve; minions and
#: neutral monsters do not, and the two fall on opposite sides of the rule — an enemy
#: minion counts, a jungle camp does not. Since farming is most of what champions attack,
#: this choice is not marginal.
#:
#: Left FALSE: claiming vision the game did not grant is the worse error, because it
#: understates darkness and every downstream information metric is built on darkness.
#: The measured cost is in `docs/validation.md`.
UNKNOWN_TARGET_REVEALS = False


@dataclass(frozen=True, slots=True)
class SourceCounts:
    """How many of each kind of source contributed, for reporting and sanity checks."""

    turrets: int
    wards: int
    champion_ticks: int
    minion_ticks: int
    reveal_ticks: int
    live_fallbacks: int
    extra: dict[str, Any] = field(default_factory=dict)


class VisionStream:
    """Per-team visibility masks, tick by tick.

    Requires teams to be resolved — a mask is per team, so without them there is nothing
    to assemble. Raises rather than silently producing one combined mask, because a
    combined mask would look plausible and make every information-asymmetry metric
    meaningless.
    """

    def __init__(
        self,
        events: MatchEvents,
        attribution: Attribution,
        terrain: Terrain,
        table: FovTable,
        tick_hz: int = C.TICK_HZ,
    ) -> None:
        if not events.teams_resolved:
            raise ValueError(
                "teams must be resolved before vision can be assembled: a mask is "
                "per team, and a combined mask would make every asymmetry metric "
                "meaningless while still looking plausible"
            )
        self.events = events
        self.attribution = attribution
        self.terrain = terrain
        self.table = table
        self.tick_hz = tick_hz
        self.dt = 1.0 / tick_hz
        self.n_ticks = attribution.pos.shape[0]
        self.grid = terrain.grid
        self.team = events.heroes["team"].astype(np.int64)

        self._live_fallbacks = 0
        self._front = estimate_front(events, attribution.pos, attribution.valid)
        self._enemy_targets = self._build_target_teams()
        self._static = self._build_static()
        self._ward_boundaries = self._ward_boundary_ticks()
        self._counts = {"champion": 0, "minion": 0, "reveal": 0}

    def _build_target_teams(self) -> dict[int, int]:
        """`net_id -> owning team`, for every unit an attack can name as its target.

        The fog-attack reveal is conditioned on this. The rule is that a champion is
        revealed "when attacking an ENEMY (including wards) from their team's fog of
        war" — so the target's team decides whether a reveal happens at all, and an
        attack that names no target reveals nobody.

        Getting this wrong was not subtle. Applying the reveal to every attack anchor
        meant every champion revealed themselves roughly once a second wherever they
        stood, including in their own fountain at 0:00 before either team had left base —
        so both teams lit each other's spawn from the first tick of the match.

        Champions, turrets and wards are resolvable here. Minions are not: they are
        modelled as wave clumps rather than tracked as entities, so an attack on one
        lands in `UNKNOWN_TARGET_REVEALS` below.
        """
        out: dict[int, int] = {}
        for hero in self.events.heroes:
            out[int(hero["net_id"])] = int(hero["team"])
        for site in self.events.turret_sites:
            if int(site["team"]) != UNKNOWN:
                out[int(site["net_id"])] = int(site["team"])
        for ward in self.events.wards:
            team = int(ward["team"])
            if team == UNKNOWN:
                owner = int(ward["owner_slot"])
                if 0 <= owner < self.team.size:
                    team = int(self.team[owner])
            if team != UNKNOWN:
                out[int(ward["net_id"])] = team
        return out

    # -- layers -------------------------------------------------------
    def _build_static(self) -> list[np.ndarray]:
        """Turret vision. Structures do not move, so this is computed once.

        Turret destruction is not modelled in v1: the corpus has no building-death
        packet (grep found zero `BuildingDie`, `TurretDie` or `ObjectDie` occurrences),
        so a destroyed turret would keep granting vision. That inflates late-game vision
        for whichever team is losing structures, and it is a stated limitation rather
        than a hidden one.
        """
        masks = [new_mask(self.grid) for _ in range(2)]
        for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
            sources = []
            for site in self.events.turret_sites:
                if int(site["team"]) != team:
                    continue
                if not (np.isfinite(site["x"]) and np.isfinite(site["z"])):
                    continue  # no recovered position, so no vision claim
                i, j = world_to_cell(float(site["x"]), float(site["z"]))
                if not (0 <= i < self.grid and 0 <= j < self.grid):
                    continue
                sources.append((i, j, C.SIGHT_TURRET, int(self.terrain.brush_id[j, i])))
            if sources:
                assemble(self.table, self.terrain, sources, out=masks[team])
        return masks

    def _ward_boundary_ticks(self) -> set[int]:
        """Ticks at which the semi-static layer must be rebuilt.

        Both the floor and the ceiling of every boundary are included, and that is not
        belt-and-braces. Rounding to the nearest tick was the first version, and when a
        ward's placement time rounded DOWN the rebuild happened at a tick where the ward
        did not yet exist — so it was excluded, and since no later rebuild was scheduled
        it contributed no vision for its entire lifetime. Roughly half of all wards, and
        it surfaced only as an elevated false-negative rate in the fog agreement.
        """
        out = {0}
        for ward in self.events.wards:
            for t in (ward["t0"], ward["t1"]):
                if not np.isfinite(t):
                    continue
                exact = float(t) / self.dt
                for tick in (int(np.floor(exact)), int(np.ceil(exact))):
                    if 0 <= tick < self.n_ticks:
                        out.add(tick)
        return out

    def _semi_at(self, tick: int, out: list[np.ndarray]) -> None:
        """Static plus every ward alive at this tick."""
        t = tick * self.dt
        for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
            out[team][:] = self._static[team]
        for ward in self.events.wards:
            owner = int(ward["owner_slot"])
            team = int(ward["team"])
            if team == UNKNOWN and 0 <= owner < self.team.size:
                team = int(self.team[owner])
            if team not in (C.TEAM_ORDER, C.TEAM_CHAOS):
                continue
            if not (ward["t0"] <= t <= ward["t1"]):
                continue
            i, j = world_to_cell(float(ward["x"]), float(ward["z"]))
            if not (0 <= i < self.grid and 0 <= j < self.grid):
                continue
            assemble(
                self.table,
                self.terrain,
                [(i, j, float(ward["sight"]), int(self.terrain.brush_id[j, i]))],
                out=out[team],
            )

    # -- dynamic sources ----------------------------------------------
    def _champion_sources(self, tick: int, team: int) -> list[tuple[int, int, float, int]]:
        out: list[tuple[int, int, float, int]] = []
        for slot in range(self.team.size):
            if int(self.team[slot]) != team or not self.attribution.valid[tick, slot]:
                continue
            x, z = self.attribution.pos[tick, slot]
            i, j = world_to_cell(float(x), float(z))
            if not (0 <= i < self.grid and 0 <= j < self.grid):
                continue
            out.append((i, j, C.SIGHT_CHAMPION, int(self.terrain.brush_id[j, i])))
        return out

    def _minion_sources(self, tick: int, team: int) -> list[tuple[int, int, float, int]]:
        t = tick * self.dt
        out: list[tuple[int, int, float, int]] = []
        for wave in self.events.minion_waves:
            if int(wave["team"]) != team:
                continue
            lane = str(wave["lane"])
            p = sr.minion_clump_position(
                lane,
                int(wave["team"]),
                float(wave["t0"]),
                t,
                float(wave["t1"]),
                front_s=float(self._front[lane][tick]),
            )
            if p is None:
                continue
            i, j = world_to_cell(float(p[0]), float(p[1]))
            if not (0 <= i < self.grid and 0 <= j < self.grid):
                continue
            out.append((i, j, C.SIGHT_MINION, int(self.terrain.brush_id[j, i])))
        return out

    def _reveal_sources(self, tick: int, team: int) -> list[tuple[int, int, float, int]]:
        """Discs revealed by enemies who attacked *while in this team's fog*.

        The gate is essential and applying it unconditionally was measurably wrong. An
        earlier version added a reveal for every enemy attack, reasoning that a reveal
        centred on an already-visible champion lies inside vision the observer had anyway
        and so changes nothing. That is false over time: a champion that attacks while
        visible and then walks into fog would keep being revealed for 4.5 seconds it never
        earned. With attacks arriving every ~1.5 seconds the effect was not subtle — fog
        agreement fell from 98.8% to 43.4%, with a 56.6% false-positive rate.

        So the condition is checked against BASE visibility recorded earlier in this same
        streaming pass: whether the attacker was in fog at the moment it attacked, before
        any reveals. Attacks are always in the past by the time they matter, so the
        history needed is already available, and a reveal can never trigger another.

        **And the attack has to have hit an enemy.** The wiki's wording is "when attacking
        an enemy (including wards) from their team's fog of war", so an attack on a
        neutral monster or on nothing at all reveals no one. Ignoring the target was worth
        488 spurious reveals in the first four seconds of a match — enough for both teams
        to see each other's fountain before anybody had moved.
        """
        t = tick * self.dt
        out: list[tuple[int, int, float, int]] = []
        lo = t - C.FOG_ATTACK_REVEAL_DURATION
        window = self._attacks[(self._attacks["t"] > lo) & (self._attacks["t"] <= t)]
        for row in window:
            slot = int(row["slot"])
            if not (0 <= slot < self.team.size) or int(self.team[slot]) == team:
                continue  # our own attacks reveal nothing to us
            attack_tick = round(float(row["t"]) / self.dt)
            if not (0 <= attack_tick < self.n_ticks):
                continue
            if self._base_visible[attack_tick, team, slot]:
                continue  # it was not attacking from fog, so nothing was revealed
            if not self._attacked_an_enemy(int(row["target"]), int(self.team[slot])):
                continue  # the rule needs an enemy target; a camp or a miss reveals no one
            i, j = world_to_cell(float(row["x"]), float(row["z"]))
            if not (0 <= i < self.grid and 0 <= j < self.grid):
                continue
            out.append((i, j, C.FOG_ATTACK_REVEAL_RADIUS, int(self.terrain.brush_id[j, i])))
        return out

    def _attacked_an_enemy(self, target: int, attacker_team: int) -> bool:
        """Whether this attack's target was an enemy of the attacker."""
        if target == 0:
            return False  # no target at all
        known = self._enemy_targets.get(target)
        if known is None:
            # A minion, a neutral monster, or a unit we never saw created. Minions are
            # enemies and neutrals are not, and the stream does not let us tell them
            # apart here — see `UNKNOWN_TARGET_REVEALS` for which way that is resolved
            # and what the choice was measured to cost.
            return UNKNOWN_TARGET_REVEALS
        return known != attacker_team

    def _record_base_visibility(self, tick: int, live: list[np.ndarray]) -> None:
        """Note which champions the base mask sees, before reveals are added.

        A single-bit read per champion, so this costs nothing next to the mask assembly
        it follows — and it is what makes the reveal rule well-defined.
        """
        for slot in range(self.team.size):
            if not self.attribution.valid[tick, slot]:
                continue
            observer = 1 - int(self.team[slot])
            x, z = self.attribution.pos[tick, slot]
            i, j = world_to_cell(float(x), float(z))
            if 0 <= i < self.grid and 0 <= j < self.grid:
                self._base_visible[tick, observer, slot] = mask_bit(live[observer], i, j)

    # -- streaming ----------------------------------------------------
    def masks(self, copy: bool = False) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield `(tick, order_mask, chaos_mask)` for every tick.

        The buffers are **reused** unless `copy=True`. That is the whole point — holding
        every mask would be 472 MB — but it means a consumer that stashes a yielded array
        gets the last tick's contents, so anything needing to keep one must ask.
        """
        self._attacks = self.events.anchors[self.events.anchors["kind"] == ANCHOR_ATTACK]
        semi = [new_mask(self.grid) for _ in range(2)]
        live = [new_mask(self.grid) for _ in range(2)]
        rows = np.asarray(self.table.rows)
        self._counts = {"champion": 0, "minion": 0, "reveal": 0}
        self._live_fallbacks = 0
        self._base_visible = np.zeros((self.n_ticks, 2, self.team.size), dtype=bool)

        for tick in range(self.n_ticks):
            if tick in self._ward_boundaries:
                self._semi_at(tick, semi)

            # Base layer first: everything except reveals, whose own condition depends on
            # base visibility and would otherwise be circular.
            for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
                live[team][:] = semi[team]
                champs = self._champion_sources(tick, team)
                minions = self._minion_sources(tick, team)
                self._counts["champion"] += len(champs)
                self._counts["minion"] += len(minions)
                self._union(rows, champs + minions, live[team])

            self._record_base_visibility(tick, live)

            for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
                reveals = self._reveal_sources(tick, team)
                self._counts["reveal"] += len(reveals)
                if reveals:
                    self._union(rows, reveals, live[team])
            if copy:
                yield tick, live[0].copy(), live[1].copy()
            else:
                yield tick, live[0], live[1]

    def _union(self, rows, sources, out) -> None:
        """Union a source list into a mask, using the table where it applies."""
        hit_rows, hit_i, hit_j, hit_r = [], [], [], []
        for i, j, radius, brush in sources:
            cell = j * self.grid + i
            cell_brush = int(self.terrain.brush_id[j, i])
            ri = self.table.radius_index(radius)
            row = self.table.lookup(cell, brush, cell_brush) if ri >= 0 else int(MISS)
            if row >= 0:
                hit_rows.append(row)
                hit_i.append(i - self.table.half)
                hit_j.append(j - self.table.half)
                hit_r.append(ri)
            else:
                self._live_fallbacks += 1
                assemble(self.table, self.terrain, [(i, j, radius, brush)], out=out)
        if hit_rows:
            union_sources(
                rows,
                self.table.discs,
                np.array(hit_rows, dtype=np.int32),
                np.array(hit_i, dtype=np.int32),
                np.array(hit_j, dtype=np.int32),
                np.array(hit_r, dtype=np.int32),
                self.table.window,
                self.table.src_words,
                out,
            )

    # -- reporting ----------------------------------------------------
    def counts(self) -> SourceCounts:
        return SourceCounts(
            turrets=int(
                np.isfinite(self.events.turret_sites["x"]).sum()
                if self.events.turret_sites.size
                else 0
            ),
            wards=int(self.events.wards.size),
            champion_ticks=self._counts["champion"],
            minion_ticks=self._counts["minion"],
            reveal_ticks=self._counts["reveal"],
            live_fallbacks=self._live_fallbacks,
            extra={"minion_waves": int(self.events.minion_waves.size)},
        )
