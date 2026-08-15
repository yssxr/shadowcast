"""Reconnaissance on a real shard: does the fog oracle actually hold?

Everything downstream of L2 rests on one claim, taken from the plan:

    A team always sees its own members, so a fog event naming champion C can only come
    from C's OPPONENTS' view — which makes the observer team derivable per event and
    gives a ground-truth visibility oracle for both sides.

That is the project's central asset and it had never been tested against real packets.
This module tests it, on any shard, and prints what it found rather than asserting it.

The test that settles it is the last one. Fog semantics cannot be read off the packet
names — `EnterFog` could plausibly mean "entered the fog" or "entered view from the fog"
— and the counts do not settle it either, because both are heavily duplicated. What does
settle it is geometry: if a fog event is about the opposing team's vision, then a champion
marked visible should be *close to an enemy* and one marked hidden should be far from
enemies, while its distance to its own allies should barely change. That is a prediction
no other interpretation makes. Camera-based interest culling, for instance, predicts that
a hidden champion is far from everyone.

MEASURED on `12_22/batch_001.jsonl.gz`, first match: visible champions sit a median 885
units from the nearest enemy — inside the 1,350-unit champion sight radius — and 2,748
units when hidden, while the ally distance moves only from 2,858 to 2,638. The oracle
holds.
"""

from __future__ import annotations

import collections
import gzip
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FogReport", "ShardMatch", "inspect_fog", "read_matches"]

#: Champion sight radius at patch 12.22, for interpreting the distance test.
CHAMPION_SIGHT = 1350.0

#: A hero-to-hero damage pair below this count is treated as noise rather than evidence
#: of enmity — reflected damage, shared effects and the odd mis-parse all produce ones.
MIN_DAMAGE_FOR_ENMITY = 3


@dataclass(slots=True)
class ShardMatch:
    """One match's packets, grouped into the handful of streams the recon needs."""

    heroes: dict[int, str]
    duration: float
    fog: dict[int, list[tuple[float, str]]]
    anchors: dict[int, list[tuple[float, float, float]]]
    damage: collections.Counter
    n_packets: int
    kinds: collections.Counter


def read_matches(path: Path | str, limit: int = 1) -> Iterator[ShardMatch]:
    """Parse the first `limit` matches from a gzipped JSONL shard.

    Fifteen lines of `gzip` and `json`, deliberately. The official `…-gym` loader treats
    `WaypointGroup`'s dict key as a net_id when it is the list length, so its position
    tracking is wrong throughout — reading the file directly is both simpler and the only
    version that is correct.
    """
    with gzip.open(path, "rt") as fh:
        for n, line in enumerate(fh):
            if n >= limit:
                return
            yield _group(json.loads(line)["events"])


def _group(events: list[dict[str, Any]]) -> ShardMatch:
    heroes: dict[int, str] = {}
    fog: dict[int, list[tuple[float, str]]] = collections.defaultdict(list)
    anchors: dict[int, list[tuple[float, float, float]]] = collections.defaultdict(list)
    damage: collections.Counter = collections.Counter()
    kinds: collections.Counter = collections.Counter()
    duration = 0.0

    for event in events:
        kind = next(iter(event))
        payload = event[kind]
        kinds[kind] += 1
        duration = max(duration, float(payload.get("time", 0.0)))

        if kind == "CreateHero":
            heroes[payload["net_id"]] = payload.get("champion", "")
        elif kind in ("EnterFog", "LeaveFog"):
            fog[payload["net_id"]].append((float(payload["time"]), kind))
        elif kind == "UnitApplyDamage":
            damage[(payload.get("source_net_id"), payload.get("target_net_id"))] += 1
        elif kind in ("BasicAttackPos", "CastSpellAns"):
            net_id = payload.get("source_net_id") or payload.get("caster_net_id")
            where = payload.get("source_position")
            if net_id is not None and where:
                anchors[net_id].append((float(payload["time"]), where["x"], where["z"]))

    return ShardMatch(
        heroes=heroes,
        duration=duration,
        fog={k: sorted(v) for k, v in fog.items() if k in heroes},
        anchors={k: sorted(v) for k, v in anchors.items() if k in heroes},
        damage=damage,
        n_packets=len(events),
        kinds=kinds,
    )


def teams_from_damage(match: ShardMatch) -> dict[int, int]:
    """Recover the team split by maximising cross-team damage over every balanced split.

    Champions damage enemies and not allies, so the true split is the one that puts
    almost all hero-to-hero damage across the cut. With ten champions there are only
    `C(10,5)/2 = 126` balanced splits, so this is solved exactly by enumeration rather
    than approximated.

    **Two-colouring the damage graph was the obvious method and it is wrong twice over.**
    It colours each connected component from zero independently, so "colour 0" means a
    different team in each component and a disconnected graph comes out 6/4 with nothing
    raising. And it assumes the graph is bipartite, which real matches are not: in one of
    twelve, a champion traded 75 hits with a genuine enemy and 3 with a teammate — enough
    to close an odd cycle and make any colouring arbitrary. A maximum cut is indifferent
    to a few stray edges; a two-colouring is decided by them.

    Recovered without turret names, spawn sides or any position data, which makes it a
    genuinely independent check on the resolver — if the two disagree, one is wrong and it
    is worth knowing which.
    """
    ids = sorted(match.heroes)
    n = len(ids)
    index = {net_id: i for i, net_id in enumerate(ids)}

    weight = np.zeros((n, n))
    for (source, target), count in match.damage.items():
        if source in index and target in index and source != target:
            weight[index[source], index[target]] += count
            weight[index[target], index[source]] += count

    if n % 2 or not weight.any():
        return {}

    best_cut = -1.0
    best: dict[int, int] = {}
    # Champion 0 is pinned to team 0 so each split is considered once rather than twice;
    # the other four members of its team are chosen from the remaining nine.
    for combo in itertools.combinations(range(1, n), n // 2 - 1):
        first = np.zeros(n, dtype=bool)
        first[0] = True
        first[list(combo)] = True
        cut = float(weight[first][:, ~first].sum())
        if cut > best_cut:
            best_cut = cut
            best = {ids[i]: (0 if first[i] else 1) for i in range(n)}

    total = float(weight.sum()) / 2
    # Recorded so a caller can see how clean the separation was. Real matches put 97-100%
    # of hero damage across the cut; anything much lower means the split is not trustworthy.
    if best:
        best["_cut_fraction"] = best_cut / total if total else 0.0  # type: ignore[assignment]
    return best


def transitions(rows: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Deduped, run-collapsed fog transitions for one unit.

    Two separate reductions and both are needed. `LeaveFog` is 65-70% of all packets and
    maknee documents 20+ repeats, so the first pass drops exact `(time, kind)` duplicates;
    the second collapses consecutive same-kind events, which is what leaves an alternating
    sequence. Skipping either makes the raw 5.5:1 hero EnterFog:LeaveFog ratio look like a
    semantic asymmetry rather than the duplication artefact it is.
    """
    out: list[tuple[float, str]] = []
    for time, kind in rows:
        if out and out[-1] == (time, kind):
            continue
        if out and out[-1][1] == kind:
            continue
        out.append((time, kind))
    return out


@dataclass(slots=True)
class FogReport:
    """What the recon found. Every field is a measurement, not an assumption."""

    n_heroes: int
    duration: float
    raw_ratio: float
    alternates: bool
    n_transitions: int
    position_packets_while_visible: float
    teams: dict[int, int]
    bipartite: bool
    #: Share of hero-to-hero damage that crosses the recovered split. 1.0 is perfect.
    cut_fraction: float
    visible_ally_distance: float
    visible_enemy_distance: float
    hidden_ally_distance: float
    hidden_enemy_distance: float
    samples: int
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def enemy_ratio(self) -> float:
        """How much further from an enemy a hidden champion is than a visible one."""
        return self.hidden_enemy_distance / max(self.visible_enemy_distance, 1.0)

    @property
    def ally_ratio(self) -> float:
        """The same for allies. Should be near 1: fog is not about your own team."""
        return self.hidden_ally_distance / max(self.visible_ally_distance, 1.0)

    @property
    def oracle_holds(self) -> bool:
        """Whether the fog stream behaves like the OPPOSING team's vision.

        The discriminating prediction is a CONTRAST, not an absolute distance: hiding
        moves a champion much further from enemies than it does from allies. Interest
        culling around a camera moves both together, and a stream carrying only one
        team's view would not toggle that team's own members at all.

        Stated as a ratio on purpose. An earlier version asserted that a visible champion
        sits within the 1,350-unit sight radius, which is true on average and fails on any
        match where the position track — interpolated between sparse labelled anchors — is
        a few hundred units loose. That tested the reconstruction, not the claim.
        """
        return (
            self.alternates and self.enemy_ratio > 1.5 and self.enemy_ratio > 1.3 * self.ally_ratio
        )

    def describe(self) -> dict[str, Any]:
        return {
            "heroes": self.n_heroes,
            "duration_s": round(self.duration),
            "raw_enter_leave_ratio": round(self.raw_ratio, 2),
            "transitions_alternate": self.alternates,
            "transitions": self.n_transitions,
            "position_packets_while_visible": round(self.position_packets_while_visible, 3),
            "teams_balanced": self.bipartite,
            "damage_across_the_split": round(self.cut_fraction, 4),
            "visible_nearest_enemy": round(self.visible_enemy_distance),
            "hidden_nearest_enemy": round(self.hidden_enemy_distance),
            "visible_nearest_ally": round(self.visible_ally_distance),
            "hidden_nearest_ally": round(self.hidden_ally_distance),
            "enemy_distance_ratio": round(self.enemy_ratio, 2),
            "ally_distance_ratio": round(self.ally_ratio, 2),
            "oracle_holds": self.oracle_holds,
        }


def inspect_fog(match: ShardMatch, grid_seconds: float = 2.0, stale: float = 6.0) -> FogReport:
    """Run every R1 check against one match."""
    heroes = match.heroes
    colour = teams_from_damage(match)
    cut_fraction = float(colour.pop("_cut_fraction", 0.0))

    raw = collections.Counter()
    for rows in match.fog.values():
        for _, kind in rows:
            raw[kind] += 1
    raw_ratio = raw["EnterFog"] / max(raw["LeaveFog"], 1)

    alternates = True
    n_transitions = 0
    per_hero: dict[int, list[tuple[float, str]]] = {}
    for net_id, rows in match.fog.items():
        seq = transitions(rows)
        per_hero[net_id] = seq
        n_transitions += len(seq)
        alternates &= all(a[1] != b[1] for a, b in itertools.pairwise(seq))

    # A packet carrying a unit's coordinates can only reach a client that can see it, so
    # the polarity of the two names is decided by where those packets land.
    inside = total = 0
    for net_id, rows in match.anchors.items():
        seq = per_hero.get(net_id)
        if not seq:
            continue
        times = [t for t, _ in seq]
        kinds = [k for _, k in seq]
        for time, _, _ in rows:
            index = np.searchsorted(times, time, side="right") - 1
            if index < 0:
                continue
            total += 1
            inside += kinds[index] == "LeaveFog"

    grid = np.arange(0.0, match.duration, grid_seconds)
    track: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    visible: dict[int, np.ndarray] = {}
    for net_id in heroes:
        rows = match.anchors.get(net_id, [])
        seq = per_hero.get(net_id)
        if len(rows) < 5 or not seq:
            continue
        at = np.array([r[0] for r in rows])
        track[net_id] = (
            np.interp(grid, at, np.array([r[1] for r in rows])),
            np.interp(grid, at, np.array([r[2] for r in rows])),
            np.abs(grid[:, None] - at[None, :]).min(axis=1),
        )
        times = np.array([t for t, _ in seq])
        kinds = [k for _, k in seq]
        index = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(kinds) - 1)
        visible[net_id] = np.array([kinds[i] == "LeaveFog" for i in index])

    buckets: dict[bool, list[tuple[float, float]]] = {True: [], False: []}
    ids = list(track)
    for g in range(len(grid)):
        # Only where the interpolation is anchored recently enough to be worth trusting.
        present = [n for n in ids if track[n][2][g] < stale]
        if len(present) < 6:
            continue
        for net_id in present:
            x, z = track[net_id][0][g], track[net_id][1][g]
            allies = [m for m in present if m != net_id and colour.get(m) == colour.get(net_id)]
            enemies = [m for m in present if colour.get(m) != colour.get(net_id)]
            if not allies or not enemies:
                continue
            buckets[bool(visible[net_id][g])].append(
                (
                    min(np.hypot(track[m][0][g] - x, track[m][1][g] - z) for m in allies),
                    min(np.hypot(track[m][0][g] - x, track[m][1][g] - z) for m in enemies),
                )
            )

    seen = np.array(buckets[True]) if buckets[True] else np.zeros((1, 2))
    hidden = np.array(buckets[False]) if buckets[False] else np.zeros((1, 2))
    sizes = collections.Counter(colour.values())

    return FogReport(
        n_heroes=len(heroes),
        duration=match.duration,
        raw_ratio=raw_ratio,
        alternates=alternates,
        n_transitions=n_transitions,
        position_packets_while_visible=inside / max(total, 1),
        teams=colour,
        bipartite=len(colour) == len(heroes) and set(sizes.values()) == {len(heroes) // 2},
        cut_fraction=cut_fraction,
        visible_ally_distance=float(np.median(seen[:, 0])),
        visible_enemy_distance=float(np.median(seen[:, 1])),
        hidden_ally_distance=float(np.median(hidden[:, 0])),
        hidden_enemy_distance=float(np.median(hidden[:, 1])),
        samples=len(seen) + len(hidden),
        stats={"packets": match.n_packets, "kinds": dict(match.kinds.most_common(8))},
    )
