"""The packet-source seam.

This module imports nothing else from the package. It defines what a decoded replay
looks like, and everything upstream of it — synthetic generator, HuggingFace reader —
implements the same protocol, so swapping one for the other is a new file rather than
a refactor.

**The dtypes mirror the real packets faithfully, including their defects.** That is
deliberate and it is the whole value of the seam. Every field the real stream lacks is
lacked here too, so the code that copes with the absence is exercised from day one
instead of being bolted on when real data arrives:

- `WAYPOINT` carries **no entity id**. In the real data the waypoint dict's key is the
  list length, not a net_id (verified: 100.0000% of 41,129 pairs). Movement orders are
  anonymous, and recovering which champion each belongs to is the project's first hard
  problem. A synthetic source that helpfully attached ids would leave that entire
  layer untested.
- `CREATE_HERO` has no team, no role and no position. All three are inferred.
- There is no death packet. `HeroDie` is declared in the published schema and fires
  zero times in 965,768 real packets; no hero net_id ever appears in the NPC death
  packets either. Champion deaths come from health replication.
- `SPAWN_MINION.t` may be garbage. Real `SpawnMinion.time` is denormal-float noise, so
  the field exists but consumers must take timing from the stream clock.
- Waypoint coordinates are in a **map-centred** frame while every other position is
  world-framed. The offset is near 7500 but not exactly, so it is calibrated rather
  than assumed.

Wards are not a packet type. They arrive as `SPAWN_MINION` rows whose `name` and
`skin_name` identify them, with the owner's hero net_id in `targetable_on_client` and
destruction signalled by a `WardCorpse` unit — so placement, owner and lifetime are
all directly observable, which is better than the project plan assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BASIC_ATTACK",
    "CAST_SPELL",
    "CREATE_HERO",
    "CREATE_NEUTRAL",
    "CREATE_TURRET",
    "DAMAGE",
    "FOG",
    "NPC_DIE",
    "PACKET_KINDS",
    "REPLICATION",
    "SPAWN_MINION",
    "USE_ITEM",
    "WAYPOINT",
    "WAYPOINT_XZ",
    "MatchMeta",
    "PacketBundle",
    "PacketSource",
]

# ---------------------------------------------------------------------------
# Packet dtypes
# ---------------------------------------------------------------------------
#: Every packet dtype carries `seq`: its index in the original interleaved event
#: stream. That field exists because splitting one stream into per-kind arrays throws
#: away the interleaving — and the interleaving is the only thing that makes
#: `SPAWN_MINION`'s corrupt timestamps recoverable. A ward's placement time comes from
#: the surrounding packets' clock, which requires knowing what "surrounding" means.
#: Ward lifetimes are the project's headline metric, so losing that would be expensive.

#: `CreateHero` — re-emitted roughly every 60 s as a keyframe resync, so consumers
#: must dedupe by net_id. No team, no role, no position.
CREATE_HERO = np.dtype(
    [("t", "f8"), ("net_id", "u4"), ("name", "U24"), ("champion", "U24"), ("seq", "i8")]
)

#: `WaypointGroup` / `WaypointGroupWithSpeed`. **No entity id, by design.**
#: `off`/`n` index into a shared `WAYPOINT_XZ` buffer (CSR form), which is the honest
#: normalisation of a variable-length polyline payload and vectorises cleanly.
WAYPOINT = np.dtype([("t", "f8"), ("off", "i8"), ("n", "i4"), ("with_speed", "u1"), ("seq", "i8")])

#: Waypoint polyline payload, in the **map-centred** frame.
WAYPOINT_XZ = np.dtype([("x", "f4"), ("z", "f4")])

#: `EnterFog` / `LeaveFog`, distinguished by `leaving`. No team or observer field —
#: the observing team is derived from the fact that a team always sees its own
#: members, so an event about champion C can only come from C's opponents.
#: Heavily duplicated in the real stream; dedupe on (t, net_id, leaving).
FOG = np.dtype([("t", "f8"), ("net_id", "u4"), ("leaving", "u1"), ("seq", "i8")])

#: `Replication`. One attribute per net_id per packet, so full state needs
#: accumulation. `name` is empty for the majority of real entries, leaving only the
#: index pair — hence both are carried.
REPLICATION = np.dtype(
    [
        ("t", "f8"),
        ("net_id", "u4"),
        ("primary", "u2"),
        ("secondary", "u2"),
        ("name", "U32"),
        ("value", "f8"),
        ("is_float", "u1"),
        ("seq", "i8"),
    ]
)

#: `CreateTurret`. No position — the name is the only identity, and it encodes team
#: (`T1` = ORDER, `T2` = CHAOS), which makes turrets the anchor for team resolution.
CREATE_TURRET = np.dtype(
    [("t", "f8"), ("net_id", "u4"), ("owner_net_id", "u4"), ("name", "U40"), ("seq", "i8")]
)

#: `SpawnMinion`. Also how wards arrive. `t` may be garbage; `targetable_on_client`
#: holds the owning hero's net_id for wards.
SPAWN_MINION = np.dtype(
    [
        ("t", "f8"),
        ("net_id", "u4"),
        ("x", "f4"),
        ("z", "f4"),
        ("name", "U28"),
        ("skin_name", "U32"),
        ("targetable_on_client", "u4"),
        ("t_valid", "u1"),
        ("seq", "i8"),
    ]
)

#: `CreateNeutral` — jungle camps, dragon, herald. `position1` is trustworthy.
CREATE_NEUTRAL = np.dtype(
    [
        ("t", "f8"),
        ("net_id", "u4"),
        ("x", "f4"),
        ("z", "f4"),
        ("name", "U32"),
        ("camp_id", "i4"),
        ("neutral_type", "i4"),
        ("seq", "i8"),
    ]
)

#: `CastSpellAns`. One of the two sources of *labelled* positions, which is what
#: makes anonymous movement orders attributable at all. `spell_name` is often empty
#: in real data, so `spell_hash` is the reliable identifier.
CAST_SPELL = np.dtype(
    [
        ("t", "f8"),
        ("caster_net_id", "u4"),
        ("spell_name", "U32"),
        ("spell_hash", "u4"),
        ("src_x", "f4"),
        ("src_z", "f4"),
        ("tgt_x", "f4"),
        ("tgt_z", "f4"),
        ("slot", "i2"),
        ("seq", "i8"),
    ]
)

#: `BasicAttackPos`. The other source of labelled positions, and the route to real
#: turret coordinates (a turret's location is the mode of its attack positions).
BASIC_ATTACK = np.dtype(
    [
        ("t", "f8"),
        ("source_net_id", "u4"),
        ("target_net_id", "u4"),
        ("src_x", "f4"),
        ("src_z", "f4"),
        ("tgt_x", "f4"),
        ("tgt_z", "f4"),
        ("seq", "i8"),
    ]
)

#: `UnitApplyDamage`. No damage type and no killing blow, so kill attribution is the
#: last damage source before a health-replication zero.
DAMAGE = np.dtype(
    [("t", "f8"), ("source_net_id", "u4"), ("target_net_id", "u4"), ("damage", "f4"), ("seq", "i8")]
)

#: `NPCDieMapView` / `NPCDieMapViewBroadcast`. Covers minions, camps and structures.
#: **Never champions** — verified across 45,851 real death packets.
NPC_DIE = np.dtype(
    [
        ("t", "f8"),
        ("killer_net_id", "u4"),
        ("killed_net_id", "u4"),
        ("broadcast", "u1"),
        ("seq", "i8"),
    ]
)

#: `UseItem`. No position, so a trinket use alone cannot locate a ward — the ward's
#: own `SpawnMinion` row does that.
USE_ITEM = np.dtype([("t", "f8"), ("net_id", "u4"), ("slot", "i2"), ("seq", "i8")])


#: Field name on `PacketBundle` -> dtype. Iterated by the conformance suite and by
#: anything that wants to treat the bundle generically.
PACKET_KINDS: dict[str, np.dtype] = {
    "heroes": CREATE_HERO,
    "waypoints": WAYPOINT,
    "waypoint_xz": WAYPOINT_XZ,
    "fog": FOG,
    "replication": REPLICATION,
    "turrets": CREATE_TURRET,
    "minions": SPAWN_MINION,
    "neutrals": CREATE_NEUTRAL,
    "casts": CAST_SPELL,
    "attacks": BASIC_ATTACK,
    "damage": DAMAGE,
    "deaths": NPC_DIE,
    "items": USE_ITEM,
}

#: Kinds whose `t` column must be non-decreasing. `minions` is excluded because real
#: `SpawnMinion.time` is denormal garbage.
TIME_ORDERED_KINDS: tuple[str, ...] = (
    "heroes",
    "waypoints",
    "fog",
    "replication",
    "turrets",
    "neutrals",
    "casts",
    "attacks",
    "damage",
    "deaths",
    "items",
)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MatchMeta:
    """What little is known about a match before normalisation.

    `match_id` is synthetic for real data: the organized corpus carries no match id,
    region, patch, rank, win/loss or duration anywhere, so an identifier is
    constructed from shard and line. Anything claiming otherwise is inferred and
    should say so.
    """

    match_id: str
    source: str
    duration: float
    n_packets: int
    patch: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PacketBundle:
    """One match's packets, as structured arrays."""

    meta: MatchMeta
    heroes: np.ndarray
    waypoints: np.ndarray
    waypoint_xz: np.ndarray
    fog: np.ndarray
    replication: np.ndarray
    turrets: np.ndarray
    minions: np.ndarray
    neutrals: np.ndarray
    casts: np.ndarray
    attacks: np.ndarray
    damage: np.ndarray
    deaths: np.ndarray
    items: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in PACKET_KINDS}

    def counts(self) -> dict[str, int]:
        return {name: int(arr.size) for name, arr in self.arrays().items()}

    def total_packets(self) -> int:
        """Packets, not rows: replication rows are per-attribute, waypoint_xz is payload."""
        return sum(int(getattr(self, name).size) for name in PACKET_KINDS if name != "waypoint_xz")

    def order_polyline(self, index: int) -> np.ndarray:
        """The (n, 2) polyline of waypoint order `index`, in its native frame."""
        row = self.waypoints[index]
        seg = self.waypoint_xz[row["off"] : row["off"] + row["n"]]
        return np.stack([seg["x"], seg["z"]], axis=1).astype(np.float64)


@runtime_checkable
class PacketSource(Protocol):
    """A provider of decoded replay packets.

    Two methods. Implementing them is the entire cost of adding a data source, and
    `packets.conformance.validate_source` runs the same invariant suite against every
    implementation — so "the real reader has a subtle bug" surfaces as a named
    assertion failure rather than as metrics that look slightly odd.
    """

    def match_ids(self) -> list[str]:
        """Available match identifiers, in a stable order."""
        ...

    def read(self, match_id: str) -> PacketBundle:
        """All of one match's packets."""
        ...


def empty_bundle(meta: MatchMeta) -> PacketBundle:
    """A bundle with every array empty. Useful for tests and for degenerate matches."""
    return PacketBundle(
        meta=meta,
        **{name: np.empty(0, dtype=dt) for name, dt in PACKET_KINDS.items()},
    )
