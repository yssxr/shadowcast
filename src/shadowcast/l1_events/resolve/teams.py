"""Resolving which team each champion is on.

`CreateHero` carries a summoner name and a champion name and nothing else — no team, no
side, no position. So team membership has to be inferred, and it is inferred from
turrets, which are the only entities that state their side outright: the internal name
encodes `T1` for ORDER and `T2` for CHAOS.

Turrets have no coordinates either, but their positions are recoverable from their attack
packets (see `l1_events.normalise`), so a turret is the one thing in the stream that has
both a known team and a known place. Champions start the match at their own fountain, so
whichever team's structures a champion is standing among at its first observation is its
team.

**The 5/5 constraint does the real work.** Nearest-shrine alone is a decent signal but it
can misfire — a champion that leaves the fountain before its first anchor, or a match
where a shrine turret never fires and so has no recovered position. Every Summoner's
Rift match has exactly five champions per side, which is a hard structural fact, so
instead of thresholding a distance the champions are *ranked* by how ORDER-leaning they
are and split down the middle. That converts a per-champion judgement into a global one
and makes a single ambiguous champion harmless.

## Why the damage graph is tried first

All of that scored 100% on synthetic matches and is **wrong on 2-4 champions in 7 of 8
real ones**. The synthetic scenario holds champions at their fountain until their first
anchor, which the validation report already flagged as unusually clean; real champions
leave base immediately, so "whose structures were you standing among at your first
observation" is a much weaker signal than it looked.

Damage is not. Champions damage enemies and not allies, so the true split is the one that
puts essentially all hero-to-hero damage *across* it — and with ten champions there are
only `C(10,5)/2 = 126` balanced splits, so the maximum cut is found exactly by
enumeration rather than approximated. MEASURED on eight real matches: the recovered split
carries 100.0% of hero damage in seven and 99.4% in the eighth.

It needs no positions, no turret names and no trajectory quality, which makes it both
more accurate and independent of every layer it feeds. Turret proximity remains as the
fallback for a match with too little combat to separate the teams — the first minute of
a truncated stream, say — and the two disagreeing is worth logging, because one of them
is wrong and the cut fraction says which.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass

import numpy as np

from shadowcast import constants as C
from shadowcast.l1_events.schema import UNKNOWN, MatchEvents

__all__ = ["TeamResolution", "resolve_teams", "teams_from_damage"]

#: Hero-to-hero damage below this is treated as noise rather than evidence of enmity.
#: Reflected damage and shared effects produce ones and twos; a real matchup produces
#: dozens. Measured across eight real matches, no legitimate pair sits below three.
_MIN_DAMAGE = 3

#: Below this share of hero damage across the recovered split, the damage graph has not
#: separated the teams and the geometric fallback is used instead. Real matches sit at
#: 99.4-100%, so anything under 90% means there was barely any combat to learn from.
_MIN_CUT = 0.90

#: How many of a champion's earliest anchors to average over. More than one because a
#: single anchor can be an outlier; few enough that the champion is still near its base.
_EARLY_ANCHORS = 5


@dataclass(frozen=True, slots=True)
class TeamResolution:
    team: np.ndarray  # i1[n_slots]
    #: Signed evidence per champion: distance to CHAOS structures minus distance to
    #: ORDER structures at its first observation. Positive means ORDER-leaning. The
    #: magnitude is the margin, so a champion near zero was a coin flip.
    lean: np.ndarray
    method: str
    stats: dict[str, object]

    @property
    def resolved(self) -> bool:
        return bool((self.team != UNKNOWN).all())

    @property
    def min_margin(self) -> float:
        """Smallest absolute lean. Near zero means at least one champion was ambiguous."""
        finite = np.abs(self.lean[np.isfinite(self.lean)])
        return float(finite.min()) if finite.size else 0.0


def _team_reference_points(events: MatchEvents) -> dict[int, np.ndarray]:
    """Positions to measure champions against, per team.

    Shrine turrets sit on the fountain and are therefore the ideal reference, but one may
    have no recovered position if it never attacked. Falling back to every turret of that
    team keeps the signal available; it is weaker because turrets line the whole map, but
    a champion at its own fountain is still nearer its own side's structures overall.
    """
    out: dict[int, np.ndarray] = {}
    sites = events.turret_sites
    if sites.size == 0:
        return out
    usable = sites[np.isfinite(sites["x"]) & np.isfinite(sites["z"])]
    for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
        of_team = usable[usable["team"] == team]
        shrines = of_team[
            [any(tok in str(n) for tok in C.TURRET_SHRINE_TOKENS) for n in of_team["name"]]
        ]
        chosen = shrines if shrines.size else of_team
        if chosen.size:
            out[team] = np.stack([chosen["x"], chosen["z"]], axis=1)
    return out


def _first_positions(events: MatchEvents, n_slots: int) -> np.ndarray:
    """Each champion's earliest observed position, as the median of its first anchors."""
    out = np.full((n_slots, 2), np.nan)
    for slot in range(n_slots):
        a = events.anchors[events.anchors["slot"] == slot]
        if a.size == 0:
            continue
        a = a[np.argsort(a["t"], kind="stable")][:_EARLY_ANCHORS]
        out[slot] = [np.median(a["x"]), np.median(a["z"])]
    return out


def teams_from_damage(events: MatchEvents) -> tuple[np.ndarray, float]:
    """`(team per slot, share of hero damage across the split)`, by exact maximum cut.

    Returns `UNKNOWN` everywhere and a zero fraction when there is not enough combat to
    decide, which the caller treats as "fall back to geometry" rather than as an answer.
    """
    n_slots = max(1, events.n_heroes)
    team = np.full(n_slots, UNKNOWN, dtype=np.int8)
    if n_slots % 2 or events.damage.size == 0:
        return team, 0.0

    # `DAMAGE_EVENT` is already resolved to slots, with UNKNOWN for anything that is not
    # a champion — so minion and turret damage drops out here without a filter.
    weight = np.zeros((n_slots, n_slots))
    source = events.damage["source"].astype(np.int64)
    target = events.damage["target"].astype(np.int64)
    hit = (source >= 0) & (target >= 0) & (source != target)
    np.add.at(weight, (source[hit], target[hit]), 1)
    weight += weight.T
    weight[weight < _MIN_DAMAGE] = 0.0
    total = float(weight.sum()) / 2
    if total <= 0:
        return team, 0.0

    # Slot 0 is pinned to ORDER so each split is considered once rather than twice; the
    # other four members of its side are chosen from the remaining nine.
    best_cut = -1.0
    best: np.ndarray | None = None
    ties = 0
    for combo in itertools.combinations(range(1, n_slots), n_slots // 2 - 1):
        side = np.zeros(n_slots, dtype=bool)
        side[0] = True
        side[list(combo)] = True
        cut = float(weight[side][:, ~side].sum())
        if cut > best_cut:
            best_cut, best, ties = cut, side.copy(), 1
        elif cut == best_cut:
            ties += 1

    # **A unique maximum is the whole point.** A sparse damage graph can put 100% of its
    # edges across many different splits at once — the synthetic generator emits about
    # thirty-five hero-to-hero damage events where a real twelve-minute match has
    # eighteen thousand, and dozens of partitions tie there. The cut fraction cannot see
    # that: it is 100% for every one of them. So a tie means the graph does not determine
    # the teams and the geometric method has to decide instead.
    if best is None or ties > 1:
        return np.full(n_slots, UNKNOWN, dtype=np.int8), 0.0
    team[best] = C.TEAM_ORDER
    team[~best] = C.TEAM_CHAOS
    return team, best_cut / total


def resolve_teams(events: MatchEvents) -> TeamResolution:
    """Infer champion teams, preferring the damage graph over geometry.

    The damage split is exact when there is combat to learn from; turret proximity is the
    fallback. See the module docstring for why the order is that way round — the
    geometric method scored 100% on synthetic data and is wrong on 2-4 champions in 7 of
    8 real matches.
    """
    n_slots = max(1, events.n_heroes)
    refs = _team_reference_points(events)
    starts = _first_positions(events, n_slots)
    lean = np.full(n_slots, np.nan)

    if len(refs) == 2:
        for slot in range(n_slots):
            if not np.isfinite(starts[slot]).all():
                continue
            d = {}
            for team, pts in refs.items():
                d[team] = float(np.hypot(*(pts - starts[slot]).T).min())
            lean[slot] = d[C.TEAM_CHAOS] - d[C.TEAM_ORDER]

    damage_team, cut = teams_from_damage(events)
    if cut >= _MIN_CUT:
        # Orient the damage split so ORDER is the side whose champions lean ORDER by the
        # geometric signal. The cut recovers the partition; only geometry knows which
        # half is which, and the sign of the mean lean is a far more robust way to ask
        # than any single champion's position.
        order_side = damage_team == C.TEAM_ORDER
        finite = np.isfinite(lean)
        if finite.any() and np.nanmean(lean[order_side & finite]) < np.nanmean(
            lean[~order_side & finite]
        ):
            damage_team = np.where(damage_team == C.TEAM_ORDER, C.TEAM_CHAOS, C.TEAM_ORDER).astype(
                np.int8
            )
        counts = {int(t): int((damage_team == t).sum()) for t in (C.TEAM_ORDER, C.TEAM_CHAOS)}
        return TeamResolution(
            team=damage_team,
            lean=lean,
            method="damage_max_cut",
            stats={
                "method": "damage_max_cut",
                "damage_across_the_split": round(cut, 4),
                "counts": counts,
                "balanced": set(counts.values()) == {n_slots // 2},
                "geometric_min_margin": round(
                    float(np.nanmin(np.abs(lean))) if np.isfinite(lean).any() else 0.0, 1
                ),
            },
        )

    team = np.full(n_slots, UNKNOWN, dtype=np.int8)
    method = "unresolved"
    if np.isfinite(lean).all() and n_slots % 2 == 0:
        # Rank and split. A champion whose own evidence is weak is carried by the
        # constraint rather than decided by a threshold it happens to fall near.
        order = np.argsort(-lean, kind="stable")
        half = n_slots // 2
        team[order[:half]] = C.TEAM_ORDER
        team[order[half:]] = C.TEAM_CHAOS
        method = "turret_proximity_with_5v5_constraint"
    elif np.isfinite(lean).any():
        # Partial evidence: decide only the champions that have any, by sign.
        known = np.isfinite(lean)
        team[known] = np.where(lean[known] > 0, C.TEAM_ORDER, C.TEAM_CHAOS)
        method = "turret_proximity_partial"

    counts = {int(t): int((team == t).sum()) for t in (C.TEAM_ORDER, C.TEAM_CHAOS, UNKNOWN)}
    return TeamResolution(
        team=team,
        lean=lean,
        method=method,
        stats={
            "reference_teams": len(refs),
            "counts": counts,
            "balanced": counts[C.TEAM_ORDER] == counts[C.TEAM_CHAOS],
        },
    )


def with_teams(events: MatchEvents, resolution: TeamResolution) -> MatchEvents:
    """Write teams into the hero table, and derive every team field that follows from it.

    The fog derivation is the point: a team never loses sight of its own members, so a
    fog event naming champion C can only be from C's opponents. That is what recovers a
    per-team visibility oracle from packets that carry no observer field at all.

    Ward teams are derived here too, from the owner recorded in `targetable_on_client`.
    They were previously left as `UNKNOWN` and patched locally wherever they were needed,
    which worked for exactly as long as there was one consumer: the vision layer filled
    them in and nothing else looked. The artifact then shipped ten wards all labelled
    team -1, and the frontend's ward-yield metric indexed off the end of an array with
    it. Deriving once, here, is the difference between a fact and a convention every
    consumer has to know about.
    """
    heroes = events.heroes.copy()
    heroes["team"] = resolution.team
    team_by_slot = resolution.team

    fog = events.fog.copy()
    if fog.size:
        slots = fog["slot"]
        valid = (slots >= 0) & (slots < team_by_slot.size)
        observer = np.full(fog.size, UNKNOWN, dtype=np.int8)
        subject_team = team_by_slot[np.clip(slots, 0, team_by_slot.size - 1)]
        known = valid & (subject_team != UNKNOWN)
        observer[known] = 1 - subject_team[known]
        fog["observer_team"] = observer

    wards = events.wards.copy()
    if wards.size:
        owners = wards["owner_slot"]
        valid = (owners >= 0) & (owners < team_by_slot.size)
        owner_team = team_by_slot[np.clip(owners, 0, team_by_slot.size - 1)]
        wards["team"] = np.where(valid & (owner_team != UNKNOWN), owner_team, UNKNOWN)

    return dataclasses.replace(events, heroes=heroes, fog=fog, wards=wards)
