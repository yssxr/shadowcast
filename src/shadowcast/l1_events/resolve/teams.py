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
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from shadowcast import constants as C
from shadowcast.l1_events.schema import UNKNOWN, MatchEvents

__all__ = ["TeamResolution", "resolve_teams"]

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


def resolve_teams(events: MatchEvents) -> TeamResolution:
    """Infer champion teams from turret positions and the 5/5 constraint."""
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
    """Write teams into the hero table, and derive each fog event's observer team.

    The derivation is the point: a team never loses sight of its own members, so a fog
    event naming champion C can only be from C's opponents. That is what recovers a
    per-team visibility oracle from packets that carry no observer field at all.
    """
    heroes = events.heroes.copy()
    heroes["team"] = resolution.team

    fog = events.fog.copy()
    if fog.size:
        team_by_slot = resolution.team
        slots = fog["slot"]
        valid = (slots >= 0) & (slots < team_by_slot.size)
        observer = np.full(fog.size, UNKNOWN, dtype=np.int8)
        subject_team = team_by_slot[np.clip(slots, 0, team_by_slot.size - 1)]
        known = valid & (subject_team != UNKNOWN)
        observer[known] = 1 - subject_team[known]
        fog["observer_team"] = observer

    return dataclasses.replace(events, heroes=heroes, fog=fog)
