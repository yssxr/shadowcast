"""Typed event tables: the L1 output, and the vocabulary every later layer speaks.

Two conventions matter here.

**Champions are addressed by `slot` (0..9), not by net_id.** Slots are assigned by
sorted net_id, so they are stable for a given match and let every downstream array be
a fixed `(n_ticks, 10, ...)` shape rather than a dictionary keyed on 32-bit ids.
`slot = -1` means "not a champion" or "not yet known".

**Everything the packets do not state is `-1` until something resolves it.** Team,
role, order ownership and observer team all start unknown, and the resolvers fill them.
That is deliberately visible in the types: a field that is `-1` has not been inferred
yet, so no consumer can mistake an unresolved value for a real one — which matters
because the corpus omits team, role, ownership and kills entirely, and every one of
them is a guess we make rather than a fact we read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shadowcast.config import StageHeader

__all__ = [
    "ANCHOR",
    "ANCHOR_ATTACK",
    "ANCHOR_CAST",
    "DAMAGE_EVENT",
    "DEATH",
    "FOG_EVENT",
    "HERO",
    "MINION_WAVE",
    "ORDER",
    "ORDER_XZ",
    "REPL_VALUE",
    "TURRET_SITE",
    "UNKNOWN",
    "WARD_EVENT",
    "FrameCalibration",
    "MatchEvents",
]

#: Sentinel for any field that has not been resolved yet.
UNKNOWN = -1

ANCHOR_CAST = 0
ANCHOR_ATTACK = 1

#: One row per champion. `team` and `role` are `UNKNOWN` until resolved.
HERO = np.dtype(
    [
        ("slot", "i1"),
        ("net_id", "u4"),
        ("name", "U24"),
        ("champion", "U24"),
        ("team", "i1"),
        ("role", "U8"),
    ]
)

#: A movement order. `owner` is the champion slot, `UNKNOWN` until attributed — which
#: is the project's first hard problem, because the packets carry no entity id.
ORDER = np.dtype([("t", "f8"), ("off", "i8"), ("n", "i4"), ("owner", "i1"), ("seq", "i8")])

#: Order polyline payload, converted to the **world** frame by calibration.
ORDER_XZ = np.dtype([("x", "f8"), ("z", "f8")])

#: A labelled position: a champion net_id paired with a coordinate. The only such
#: packets in the corpus, and therefore the anchors that make anonymous movement
#: orders attributable at all.
#: `target` is the attacked unit's net id, or 0 for a cast and for an attack that hit
#: nothing. It is here because the fog-attack reveal rule is conditioned on it: the wiki
#: says a champion is revealed "when attacking an ENEMY (including wards) from their
#: team's fog of war", so an attack with no enemy target reveals nobody. Dropping the
#: field made every attack anchor look like a reveal, and both teams lit each other's
#: fountain from the first second of the match.
ANCHOR = np.dtype(
    [
        ("t", "f8"),
        ("slot", "i1"),
        ("x", "f8"),
        ("z", "f8"),
        ("kind", "u1"),
        ("target", "u4"),
    ]
)

#: A visibility transition, deduped. `observer_team` is derived from the champion's
#: own team once that is known: a team never loses sight of its own members, so an
#: event about champion C can only come from C's opponents.
FOG_EVENT = np.dtype([("t", "f8"), ("slot", "i1"), ("observer_team", "i1"), ("visible", "u1")])

#: A turret, with the position recovered from its attack packets rather than read from
#: a create packet, which carries none.
TURRET_SITE = np.dtype(
    [("net_id", "u4"), ("name", "U40"), ("team", "i1"), ("x", "f8"), ("z", "f8"), ("n_obs", "i4")]
)

#: A ward's whole life. `t1_known` distinguishes an observed destruction from a
#: modelled expiry, which matters because ward information yield integrates over
#: lifetime and a modelled endpoint carries more uncertainty than an observed one.
WARD_EVENT = np.dtype(
    [
        ("net_id", "u4"),
        ("owner_slot", "i1"),
        ("team", "i1"),
        ("kind", "U10"),
        ("x", "f8"),
        ("z", "f8"),
        ("t0", "f8"),
        ("t1", "f8"),
        ("t1_known", "u1"),
        ("sight", "f8"),
    ]
)

#: A replicated scalar for a champion — movement speed, health.
REPL_VALUE = np.dtype([("t", "f8"), ("slot", "i1"), ("value", "f8")])

#: A minion wave, reduced to one clump. Lane and side come from the spawn position,
#: spawn time from the stream clock, and death from the NPC death packets — so no minion
#: tracking is needed, which is fortunate because none is possible: movement orders carry
#: no entity id and minions have none of the labelled position packets that make champion
#: attribution work.
MINION_WAVE = np.dtype(
    [
        ("net_id", "u4"),
        ("lane", "U4"),
        ("team", "i1"),
        ("t0", "f8"),
        ("t1", "f8"),
        ("t1_known", "u1"),
    ]
)

#: Champion-on-champion damage. The corpus has no killing-blow flag and no death
#: packet, so a kill is a health value reaching zero and its killer is whoever dealt
#: damage just before. This table is the only evidence for the second half of that.
DAMAGE_EVENT = np.dtype([("t", "f8"), ("source", "i1"), ("target", "i1"), ("amount", "f4")])

#: An inferred champion death. There is no death packet in the corpus, so both the
#: victim and the killer are inferences and both carry a confidence.
DEATH = np.dtype(
    [
        ("t", "f8"),
        ("victim", "i1"),
        ("killer", "i1"),
        ("respawn_t", "f8"),
        ("killer_confidence", "f4"),
    ]
)


@dataclass(frozen=True, slots=True)
class FrameCalibration:
    """The recovered map-centred to world-frame offset, with evidence for it.

    Waypoint coordinates arrive in a different frame from every other position, and
    the offset is close to 7500 but not equal to it. `walkable_fraction` is the score
    that was maximised; `plateau_width` is how wide the near-optimal region is, and a
    wide plateau means the offset is only weakly determined — worth knowing before
    trusting reconstructed positions to a few units.
    """

    offset: float
    walkable_fraction: float
    plateau_width: float
    baseline_fraction: float
    n_samples: int

    @property
    def well_determined(self) -> bool:
        """Is the optimum sharp enough, and good enough, to rely on?

        The thresholds are loose on purpose: the point is to catch a calibration that
        failed outright — a flat response, or a peak no better than chance — rather
        than to certify precision.
        """
        return self.walkable_fraction > 0.9 and self.plateau_width <= 120.0


@dataclass(frozen=True, slots=True)
class MatchEvents:
    """One match, normalised."""

    match_id: str
    duration: float
    header: StageHeader
    frame: FrameCalibration
    heroes: np.ndarray
    orders: np.ndarray
    order_xz: np.ndarray
    anchors: np.ndarray
    turret_sites: np.ndarray
    wards: np.ndarray
    fog: np.ndarray
    speed: np.ndarray
    hp: np.ndarray
    damage: np.ndarray
    deaths: np.ndarray
    minion_waves: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)

    # ---- lookups ------------------------------------------------------
    @property
    def n_heroes(self) -> int:
        return int(self.heroes.size)

    def slot_of(self, net_id: int) -> int:
        hit = self.heroes["net_id"] == net_id
        return int(self.heroes["slot"][hit][0]) if hit.any() else UNKNOWN

    def team_of(self, slot: int) -> int:
        return int(self.heroes["team"][slot])

    @property
    def deaths_resolved(self) -> bool:
        return bool(self.deaths.size > 0)

    @property
    def teams_resolved(self) -> bool:
        return bool((self.heroes["team"] != UNKNOWN).all())

    @property
    def roles_resolved(self) -> bool:
        return bool((self.heroes["role"] != "").all())

    @property
    def orders_attributed(self) -> bool:
        return bool((self.orders["owner"] != UNKNOWN).all())

    def order_polyline(self, index: int) -> np.ndarray:
        row = self.orders[index]
        seg = self.order_xz[row["off"] : row["off"] + row["n"]]
        return np.stack([seg["x"], seg["z"]], axis=1)

    def orders_of(self, slot: int) -> np.ndarray:
        return self.orders[self.orders["owner"] == slot]

    def anchors_of(self, slot: int) -> np.ndarray:
        return self.anchors[self.anchors["slot"] == slot]

    def describe(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "duration": round(self.duration, 1),
            "heroes": self.n_heroes,
            "orders": int(self.orders.size),
            "anchors": int(self.anchors.size),
            "anchors_per_hero": round(self.anchors.size / max(1, self.n_heroes), 1),
            "fog_events": int(self.fog.size),
            "wards": int(self.wards.size),
            "wards_with_observed_end": int(self.wards["t1_known"].sum()) if self.wards.size else 0,
            "turret_sites": int(self.turret_sites.size),
            "minion_waves": int(self.minion_waves.size),
            "speed_samples": int(self.speed.size),
            "frame_offset": round(self.frame.offset, 2),
            "frame_walkable_fraction": round(self.frame.walkable_fraction, 4),
            "teams_resolved": self.teams_resolved,
            "roles_resolved": self.roles_resolved,
            "orders_attributed": self.orders_attributed,
            **self.stats,
        }

    # ---- persistence --------------------------------------------------
    _ARRAYS = (
        "heroes",
        "orders",
        "order_xz",
        "anchors",
        "turret_sites",
        "wards",
        "fog",
        "speed",
        "hp",
        "damage",
        "deaths",
        "minion_waves",
    )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: getattr(self, name) for name in self._ARRAYS}
        meta = {
            "match_id": self.match_id,
            "duration": self.duration,
            "header": self.header.to_dict(),
            "frame": {
                "offset": self.frame.offset,
                "walkable_fraction": self.frame.walkable_fraction,
                "plateau_width": self.frame.plateau_width,
                "baseline_fraction": self.frame.baseline_fraction,
                "n_samples": self.frame.n_samples,
            },
            "stats": self.stats,
        }
        np.savez_compressed(
            path,
            meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
            **payload,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> MatchEvents:
        with np.load(path) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            arrays = {name: z[name] for name in cls._ARRAYS}
        return cls(
            match_id=meta["match_id"],
            duration=meta["duration"],
            header=StageHeader.from_dict(meta["header"]),
            frame=FrameCalibration(**meta["frame"]),
            stats=meta["stats"],
            **arrays,
        )
