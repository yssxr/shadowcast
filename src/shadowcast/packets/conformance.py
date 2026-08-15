"""Invariant checks that every `PacketSource` must satisfy.

Run against the synthetic generator and, later, against the real HuggingFace reader.
The point of sharing one suite is that "the real reader has a subtle bug" becomes a
named violation on a named row rather than metrics that look slightly off three layers
downstream.

**It returns a report, it does not assert.** On synthetic data the first failure would
be enough, but on real data we want every violation at once: the corpus has known
defects, several of them undocumented, and discovering them one exception at a time
across a 2 GB match would be miserable. So violations are collected, classified as
errors or warnings, and returned alongside statistics worth eyeballing.

The error/warning split encodes what is a bug versus what is merely the corpus being
itself. Unknown entity ids are a warning — the real stream has pets and clones that
cast spells without ever being created. Out-of-order timestamps are a warning within a
tolerance, because arrival jitter is real. A waypoint payload that runs off the end of
its buffer is an error, because nothing legitimate produces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shadowcast import constants as C
from shadowcast.packets.source import (
    PACKET_KINDS,
    TIME_ORDERED_KINDS,
    PacketBundle,
)

__all__ = ["ConformanceReport", "validate_bundle", "validate_source"]

#: Tolerated out-of-order arrival. The real stream is chunk-structured and the
#: synthetic generator models arrival jitter, so a small inversion is expected;
#: anything larger means a sort was skipped or two streams were interleaved wrongly.
TIME_INVERSION_TOLERANCE = 0.25

#: Positions are checked against the map with generous slack rather than exactly.
#: Real `CastSpellAns.target_position` ranges to +-32,000 — far off-map — so tight
#: bounds would fire constantly on legitimate data.
POSITION_SLACK = 2000.0


@dataclass
class ConformanceReport:
    """Violations and statistics from one source or bundle."""

    match_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        head = f"{self.match_id}: {'OK' if self.ok else 'FAIL'}"
        if self.errors:
            head += f" ({len(self.errors)} errors)"
        if self.warnings:
            head += f" ({len(self.warnings)} warnings)"
        return head

    def render(self) -> str:
        lines = [self.summary()]
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  warning {w}")
        for k, v in self.stats.items():
            lines.append(f"  stat    {k} = {v}")
        return "\n".join(lines)


def validate_bundle(bundle: PacketBundle) -> ConformanceReport:
    """Check one match's packets against every invariant we can state."""
    rep = ConformanceReport(match_id=bundle.meta.match_id)
    arrays = bundle.arrays()

    _check_dtypes(bundle, arrays, rep)
    _check_time_order(arrays, rep)
    _check_waypoints(bundle, rep)
    _check_entities(bundle, rep)
    _check_fog(bundle, rep)
    _check_positions(bundle, rep)
    _check_turrets(bundle, rep)
    _collect_stats(bundle, rep)
    return rep


def validate_source(source, limit: int | None = None) -> list[ConformanceReport]:
    """Validate every match a source offers (or the first `limit` of them)."""
    ids = source.match_ids()
    if limit is not None:
        ids = ids[:limit]
    return [validate_bundle(source.read(mid)) for mid in ids]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _check_dtypes(bundle, arrays, rep) -> None:
    for name, expected in PACKET_KINDS.items():
        arr = arrays.get(name)
        if arr is None:
            rep.error(f"bundle is missing packet kind {name!r}")
            continue
        if arr.dtype != expected:
            rep.error(f"{name}: dtype {arr.dtype} != expected {expected}")

    # Structural, not incidental: if a waypoint row ever grows an entity id, the
    # attribution layer silently becomes dead code and every test that depends on
    # attribution being hard would start passing for the wrong reason.
    if any(f in bundle.waypoints.dtype.names for f in ("net_id", "entity", "entity_id")):
        rep.error(
            "waypoints carry an entity id. Real movement orders do not, so a source "
            "that supplies one is not modelling the corpus and would leave the "
            "attribution layer untested."
        )


def _check_time_order(arrays, rep) -> None:
    for name in TIME_ORDERED_KINDS:
        arr = arrays[name]
        if arr.size < 2:
            continue
        d = np.diff(arr["t"])
        worst = float(-d.min()) if d.min() < 0 else 0.0
        if worst > TIME_INVERSION_TOLERANCE:
            rep.error(f"{name}: timestamps out of order by up to {worst:.3f}s")
        elif worst > 0:
            rep.warn(f"{name}: timestamps out of order by up to {worst:.3f}s (within tolerance)")
        if not np.isfinite(arr["t"]).all():
            rep.error(f"{name}: non-finite timestamps")
        if (arr["t"] < 0).any():
            rep.error(f"{name}: negative timestamps")


def _check_waypoints(bundle, rep) -> None:
    wp, xz = bundle.waypoints, bundle.waypoint_xz
    if wp.size == 0:
        rep.warn("no waypoint orders at all")
        return
    if (wp["n"] < 1).any():
        rep.error("waypoint order with fewer than one point")
    ends = wp["off"].astype(np.int64) + wp["n"].astype(np.int64)
    if (wp["off"] < 0).any() or (ends > xz.size).any():
        rep.error("waypoint payload runs off the end of the coordinate buffer")
    if not np.isfinite(xz["x"]).all() or not np.isfinite(xz["z"]).all():
        rep.error("non-finite waypoint coordinates")

    # Real waypoints are map-centred, so they straddle zero. World-framed
    # coordinates here would mean the frame conversion was applied twice or not at
    # all, which shifts every reconstructed position by half a map.
    if xz.size:
        if xz["x"].min() >= 0 and xz["x"].max() > C.WORLD_SPAN * 0.75:
            rep.warn(
                "waypoint coordinates look world-framed, not map-centred "
                f"(x range {xz['x'].min():.0f}..{xz['x'].max():.0f})"
            )
        span = max(np.ptp(xz["x"]), np.ptp(xz["z"]))
        if span > C.WORLD_SPAN * 1.5:
            rep.error(f"waypoint coordinate span {span:.0f} exceeds the map by too much")


def _known_net_ids(bundle) -> set[int]:
    known: set[int] = set()
    for arr, fields in (
        (bundle.heroes, ("net_id",)),
        (bundle.turrets, ("net_id", "owner_net_id")),
        (bundle.minions, ("net_id",)),
        (bundle.neutrals, ("net_id",)),
        # Lane minions are created here and nowhere else — they never appear in
        # `SpawnMinion`. The barrack itself is deliberately NOT counted: the real stream
        # emits no create packet for one, so treating it as known would hide exactly the
        # kind of dangling reference this check exists to surface.
        (bundle.barracks, ("minion_net_id",)),
    ):
        for f in fields:
            if arr.size:
                known.update(int(v) for v in arr[f])
    return known


def _check_entities(bundle, rep) -> None:
    heroes = bundle.heroes
    if heroes.size == 0:
        rep.error("no CreateHero rows: nothing can be resolved without them")
        return

    # CreateHero is re-emitted as a keyframe resync, so distinctness is by net_id.
    uniq = np.unique(heroes["net_id"])
    if uniq.size != C.N_HEROES:
        rep.error(f"{uniq.size} distinct hero net_ids, expected {C.N_HEROES}")

    # Every keyframe for a net_id must agree, or the resync is not a resync.
    for nid in uniq:
        rows = heroes[heroes["net_id"] == nid]
        if len(np.unique(rows["champion"])) > 1 or len(np.unique(rows["name"])) > 1:
            rep.error(f"hero {nid} has inconsistent identity across keyframes")

    known = _known_net_ids(bundle)
    unknown: dict[str, int] = {}
    for name, fields in (
        ("fog", ("net_id",)),
        ("replication", ("net_id",)),
        ("casts", ("caster_net_id",)),
        ("attacks", ("source_net_id",)),
        ("damage", ("source_net_id", "target_net_id")),
        ("deaths", ("killer_net_id", "killed_net_id")),
        ("items", ("net_id",)),
    ):
        arr = getattr(bundle, name)
        if arr.size == 0:
            continue
        for f in fields:
            ids = np.unique(arr[f])
            missing = [int(v) for v in ids if int(v) not in known and int(v) != 0]
            if missing:
                unknown[f"{name}.{f}"] = len(missing)
    if unknown:
        # A warning, not an error: the real stream contains pets, clones and ward
        # corpses that act without a create packet the decoder kept.
        rep.warn(f"references to entities that were never created: {unknown}")
    rep.stats["unknown_entity_refs"] = sum(unknown.values())


def _check_fog(bundle, rep) -> None:
    fog = bundle.fog
    if fog.size == 0:
        rep.warn("no fog transitions: the visibility oracle will have nothing to check")
        return
    heroes = np.unique(bundle.heroes["net_id"]) if bundle.heroes.size else np.array([])
    hero_fog = np.isin(fog["net_id"], heroes)
    rep.stats["fog_rows"] = int(fog.size)
    rep.stats["fog_rows_about_heroes"] = int(hero_fog.sum())

    # Transitions must alternate per entity. A repeated enter or leave means either
    # duplicates were not deduped or the two event types were confused — and the real
    # corpus does duplicate heavily, so this check earns its keep there.
    for nid in np.unique(fog["net_id"]):
        seq = fog[fog["net_id"] == nid]
        order = np.argsort(seq["t"], kind="stable")
        leaving = seq["leaving"][order]
        if leaving.size > 1 and (np.diff(leaving.astype(np.int8)) == 0).any():
            rep.warn(f"entity {nid}: fog transitions do not alternate (needs deduping)")
            break


def _check_positions(bundle, rep) -> None:
    lo_x = C.WORLD_MIN_X - POSITION_SLACK
    hi_x = C.WORLD_MIN_X + C.WORLD_SPAN + POSITION_SLACK
    lo_z = C.WORLD_MIN_Z - POSITION_SLACK
    hi_z = C.WORLD_MIN_Z + C.WORLD_SPAN + POSITION_SLACK

    for name, fx, fz in (
        ("minions", "x", "z"),
        ("neutrals", "x", "z"),
        ("casts", "src_x", "src_z"),
        ("attacks", "src_x", "src_z"),
    ):
        arr = getattr(bundle, name)
        if arr.size == 0:
            continue
        x, z = arr[fx], arr[fz]
        out = ((x < lo_x) | (x > hi_x) | (z < lo_z) | (z > hi_z)).sum()
        if out:
            rep.warn(f"{name}: {int(out)} of {arr.size} source positions outside the map")


def _check_turrets(bundle, rep) -> None:
    turrets = bundle.turrets
    if turrets.size == 0:
        rep.error(
            "no CreateTurret rows. Turret names are the anchor for team resolution, "
            "since CreateHero carries no team, so without them teams are unrecoverable."
        )
        return
    names = np.unique(turrets["name"])
    unresolved = [
        n
        for n in names
        if not any(tok in str(n) for tok in C.TURRET_TEAM_TOKENS)
        and not any(tok in str(n) for tok in C.TURRET_SHRINE_TOKENS)
    ]
    if unresolved:
        rep.warn(f"{len(unresolved)} turret names carry no team token: {unresolved[:4]}")
    rep.stats["distinct_turrets"] = int(names.size)


def _collect_stats(bundle, rep) -> None:
    rep.stats.update(
        {
            "duration": round(float(bundle.meta.duration), 1),
            "packets": bundle.total_packets(),
            "orders": int(bundle.waypoints.size),
            "labelled_anchors": int(bundle.casts.size + bundle.attacks.size),
        }
    )
    if bundle.heroes.size:
        n = max(1, int(np.unique(bundle.heroes["net_id"]).size))
        rep.stats["anchors_per_champion"] = round((bundle.casts.size + bundle.attacks.size) / n, 1)
    wards = 0
    if bundle.minions.size:
        pairs = list(zip(bundle.minions["name"], bundle.minions["skin_name"]))
        wards = sum(1 for p in pairs if (str(p[0]), str(p[1])) in C.WARD_UNITS)
    rep.stats["ward_placements"] = wards
