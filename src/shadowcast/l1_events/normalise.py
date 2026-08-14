"""Packets to event tables: dedupe, calibrate, recover what the decoder mangled.

This layer does the work that makes the corpus usable, and every step of it exists
because of a specific measured defect rather than as generic tidying:

- **Fog is deduped.** `LeaveFog` is 65–70% of all packets in the real stream and
  maknee documents "20+ repeats sometimes" at an identical timestamp. Beyond exact
  duplicates, transitions that do not change state are dropped too, so the output is a
  clean alternating timeline.
- **The waypoint frame is calibrated, not assumed.** Waypoint coordinates are
  map-centred while every other position is world-framed. The offset is near 7500 and
  demonstrably not equal to it, so it is recovered by maximising the fraction of
  waypoints that land on walkable ground — a score that only peaks when the frame is
  right, and which needs no labels.
- **Turret positions are recovered from attacks.** `CreateTurret` carries a name but no
  coordinates. `BasicAttackPos` carries `source_net_id` and `source_position`, and a
  turret never moves, so the median of its attack positions is its location.
- **Ward timing comes from stream position.** `SpawnMinion.time` is denormal-float
  garbage, so a ward's placement time is read from the packets around it via `seq`.
- **Replication is keyed on the index pair as well as the name**, because the name is
  empty for the majority of real entries and a name-only reader would drop them.

Nothing here infers team, role, order ownership or deaths. Those are guesses rather
than readings, they live in `resolve/`, and until they run the corresponding fields
stay at `UNKNOWN` so no consumer can mistake one for the other.
"""

from __future__ import annotations

import numpy as np

from shadowcast import constants as C
from shadowcast.config import GridSpec, StageHeader, TerrainSpec
from shadowcast.geom.grid import world_to_cell_array
from shadowcast.l1_events.schema import (
    ANCHOR,
    ANCHOR_ATTACK,
    ANCHOR_CAST,
    DEATH,
    FOG_EVENT,
    HERO,
    ORDER,
    ORDER_XZ,
    REPL_VALUE,
    TURRET_SITE,
    UNKNOWN,
    WARD_EVENT,
    FrameCalibration,
    MatchEvents,
)
from shadowcast.packets.source import PacketBundle
from shadowcast.terrain.terrain import Terrain

__all__ = ["STAGE_VERSION", "calibrate_waypoint_frame", "normalise"]

STAGE = "normalise"
STAGE_VERSION = 1


# ---------------------------------------------------------------------------
# Frame calibration
# ---------------------------------------------------------------------------
def calibrate_waypoint_frame(
    waypoint_xz: np.ndarray,
    terrain: Terrain,
    guess: float = C.WAYPOINT_OFFSET_GUESS,
    search: float = C.WAYPOINT_OFFSET_SEARCH,
    step: float = 0.5,
    max_samples: int = 40_000,
    seed: int = 0,
) -> FrameCalibration:
    """Recover the map-centred to world offset by maximising walkable coverage.

    Champions walk on walkable ground, so the correct offset is the one that puts the
    most waypoints there. The score needs no labels, has a single sharp optimum, and is
    robust to the fraction of waypoints that legitimately sit in odd places.

    Reported alongside the offset: the plateau width (how wide the near-optimal region
    is, so a weakly-determined frame is visible) and the baseline score at the naive
    guess (so "calibration did nothing" is distinguishable from "calibration worked").

    **The method cannot resolve the offset finer than one cell.** Two offsets differing
    by less than a cell width assign every waypoint to the same cell and therefore
    score identically, so the plateau is about 28.8 units wide no matter how much data
    is thrown at it. Measured plateau width on synthetic matches is 28.0 units, which
    is that limit rather than a weakness in the signal. Half a cell of frame error is
    acceptable — it is below the resolution of the terrain the offset is being fitted
    against — but it does mean this is not the way to pin the frame to single units,
    and a claim to that precision would be unfounded.
    """
    if waypoint_xz.size == 0:
        return FrameCalibration(guess, 0.0, 0.0, 0.0, 0)

    rng = np.random.default_rng(seed)
    if waypoint_xz.size > max_samples:
        idx = rng.choice(waypoint_xz.size, size=max_samples, replace=False)
        sample = waypoint_xz[idx]
    else:
        sample = waypoint_xz
    x = sample["x"].astype(np.float64)
    z = sample["z"].astype(np.float64)
    grid = terrain.grid
    walkable = terrain.walkable

    offsets = np.arange(guess - search, guess + search + step, step)
    scores = np.empty(offsets.size)
    for n, off in enumerate(offsets):
        i, j = world_to_cell_array(x + off, z + off)
        inside = (i >= 0) & (i < grid) & (j >= 0) & (j < grid)
        if not inside.any():
            scores[n] = 0.0
            continue
        hit = np.zeros(i.shape, dtype=bool)
        hit[inside] = walkable[j[inside], i[inside]]
        scores[n] = hit.mean()

    best = int(np.argmax(scores))
    peak = float(scores[best])
    # Width of the region within 0.5% of the peak, in world units. A broad plateau
    # means many offsets explain the data about equally well.
    near = offsets[scores >= peak - 0.005]
    plateau = float(near.max() - near.min()) if near.size else 0.0
    baseline = float(scores[int(np.argmin(np.abs(offsets - guess)))])

    return FrameCalibration(
        offset=float(offsets[best]),
        walkable_fraction=peak,
        plateau_width=plateau,
        baseline_fraction=baseline,
        n_samples=int(sample.size),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hero_table(bundle: PacketBundle) -> np.ndarray:
    """Dedupe the keyframe-repeated CreateHero rows into one row per champion.

    Slots are assigned by sorted net_id so they are stable across runs, which matters
    because every downstream array is indexed by slot.
    """
    if bundle.heroes.size == 0:
        return np.empty(0, dtype=HERO)
    net_ids = np.unique(bundle.heroes["net_id"])
    out = np.empty(net_ids.size, dtype=HERO)
    for slot, nid in enumerate(net_ids):
        rows = bundle.heroes[bundle.heroes["net_id"] == nid]
        out[slot] = (slot, nid, str(rows[0]["name"]), str(rows[0]["champion"]), UNKNOWN, "")
    return out


def _slot_lookup(heroes: np.ndarray) -> dict[int, int]:
    return {int(r["net_id"]): int(r["slot"]) for r in heroes}


def _clock_from_stream(bundle: PacketBundle) -> tuple[np.ndarray, np.ndarray]:
    """A (seq, time) index over every packet whose timestamp is trustworthy.

    Used to date packets whose own `t` is unusable. `minions` is excluded because that
    is precisely the array being repaired.
    """
    seqs, times = [], []
    for name, arr in bundle.arrays().items():
        if name in ("waypoint_xz", "minions") or arr.size == 0:
            continue
        valid = arr["seq"] >= 0
        seqs.append(arr["seq"][valid])
        times.append(arr["t"][valid])
    if not seqs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    seq = np.concatenate(seqs)
    t = np.concatenate(times)
    order = np.argsort(seq, kind="stable")
    return seq[order], t[order]


def _time_at_seq(seq_index: np.ndarray, time_index: np.ndarray, seq: np.ndarray) -> np.ndarray:
    """Interpolate a timestamp for each `seq` from the surrounding trustworthy clock."""
    if seq_index.size == 0:
        return np.zeros(seq.shape, dtype=np.float64)
    return np.interp(seq.astype(np.float64), seq_index.astype(np.float64), time_index)


def _minion_times(bundle: PacketBundle) -> np.ndarray:
    """Timestamps for SpawnMinion rows, from their own `t` if usable or the stream if not."""
    if bundle.minions.size == 0:
        return np.empty(0, dtype=np.float64)
    own = bundle.minions["t"].astype(np.float64)
    usable = bundle.minions["t_valid"].astype(bool) & (own > C.MIN_VALID_PACKET_TIME)
    if usable.all():
        return own
    seq_index, time_index = _clock_from_stream(bundle)
    recovered = _time_at_seq(seq_index, time_index, bundle.minions["seq"])
    return np.where(usable, own, recovered)


def _dedupe_fog(bundle: PacketBundle, slots: dict[int, int]) -> np.ndarray:
    """Collapse the fog stream into an alternating per-champion timeline.

    Two stages, and both are needed. Exact duplicates go first — the real stream
    repeats a single transition up to twenty times at one timestamp. Then transitions
    that do not change state go, which catches repeats spread across nearby
    timestamps that an exact-match dedupe would keep.
    """
    fog = bundle.fog
    if fog.size == 0:
        return np.empty(0, dtype=FOG_EVENT)

    keep = np.array([int(n) in slots for n in fog["net_id"]], dtype=bool)
    fog = fog[keep]
    if fog.size == 0:
        return np.empty(0, dtype=FOG_EVENT)

    rows: list[tuple[float, int, int, int]] = []
    for nid in np.unique(fog["net_id"]):
        seq = fog[fog["net_id"] == nid]
        seq = seq[np.lexsort((seq["seq"], seq["t"]))]
        # Exact duplicates on (t, leaving).
        _, first = np.unique(
            np.stack([seq["t"], seq["leaving"].astype(np.float64)], axis=1),
            axis=0,
            return_index=True,
        )
        seq = seq[np.sort(first)]
        state = UNKNOWN
        for row in seq:
            visible = int(row["leaving"])
            if visible == state:
                continue
            state = visible
            rows.append((float(row["t"]), slots[int(nid)], UNKNOWN, visible))

    out = np.empty(len(rows), dtype=FOG_EVENT)
    for n, r in enumerate(rows):
        out[n] = r
    out.sort(order=["t", "slot"], kind="stable")
    return out


def _turret_sites(bundle: PacketBundle) -> np.ndarray:
    """Recover turret positions from their attack packets and teams from their names.

    `CreateTurret` has no coordinates, so this is the only route to turret vision — and
    turrets are also the anchor for resolving champion teams, since the name encodes
    T1/T2 and `CreateHero` encodes nothing.
    """
    turrets = bundle.turrets
    if turrets.size == 0:
        return np.empty(0, dtype=TURRET_SITE)

    net_ids = np.unique(turrets["net_id"])
    name_of = {int(r["net_id"]): str(r["name"]) for r in turrets}

    rows: list[tuple] = []
    attacks = bundle.attacks
    for nid in net_ids:
        name = name_of[int(nid)]
        team = UNKNOWN
        for token, value in C.TURRET_TEAM_TOKENS.items():
            if token in name:
                team = value
                break
        else:
            for token, value in C.TURRET_SHRINE_TOKENS.items():
                if token in name:
                    team = value
                    break

        mine = attacks[attacks["source_net_id"] == nid] if attacks.size else attacks[:0]
        if mine.size:
            # Median, not mean: a turret does not move, so any spread is noise or a
            # mis-attributed packet, and the median ignores both.
            x = float(np.median(mine["src_x"]))
            z = float(np.median(mine["src_z"]))
        else:
            x = z = np.nan
        rows.append((int(nid), name, team, x, z, int(mine.size)))

    out = np.empty(len(rows), dtype=TURRET_SITE)
    for n, r in enumerate(rows):
        out[n] = r
    return out


def _wards(bundle: PacketBundle, slots: dict[int, int], times: np.ndarray) -> np.ndarray:
    """Extract ward lifetimes from SpawnMinion rows.

    Placement, position and owner are all directly observed. Destruction is observed
    when a `WardCorpse` appears, and modelled from the ward kind's duration when it does
    not — which happens in the real stream, so `t1_known` records which case applied.
    """
    minions = bundle.minions
    if minions.size == 0:
        return np.empty(0, dtype=WARD_EVENT)

    order = np.argsort(times, kind="stable")
    placements: list[dict] = []
    corpses: list[tuple[float, float, float]] = []

    for k in order:
        row = minions[k]
        key = (str(row["name"]), str(row["skin_name"]))
        t = float(times[k])
        if key in C.WARD_UNITS:
            kind = C.WARD_UNITS[key]
            owner = slots.get(int(row["targetable_on_client"]), UNKNOWN)
            placements.append(
                {
                    "net_id": int(row["net_id"]),
                    "owner_slot": owner,
                    "kind": kind,
                    "x": float(row["x"]),
                    "z": float(row["z"]),
                    "t0": t,
                    "t1": np.nan,
                    "t1_known": 0,
                }
            )
        elif key == C.WARD_CORPSE_UNIT:
            corpses.append((t, float(row["x"]), float(row["z"])))

    # A corpse carries a different net_id from the ward it replaces, so it is matched
    # by position among wards still alive at that moment.
    for ct, cx, cz in corpses:
        best, best_d = None, 1e18
        for p in placements:
            if p["t1_known"] or p["t0"] > ct:
                continue
            d = (p["x"] - cx) ** 2 + (p["z"] - cz) ** 2
            if d < best_d:
                best_d, best = d, p
        if best is not None and best_d <= (2.0 * C.GRID_CELL_SIZE) ** 2:
            best["t1"] = ct
            best["t1_known"] = 1

    out = np.empty(len(placements), dtype=WARD_EVENT)
    for n, p in enumerate(placements):
        if not p["t1_known"]:
            # Modelled expiry. Totem duration scales with average champion level, which
            # is not resolved here, so the midpoint of the documented range is used and
            # `t1_known` stays 0 to mark the estimate.
            if p["kind"] == "totem":
                life = (C.WARD_TOTEM_DURATION_MIN + C.WARD_TOTEM_DURATION_MAX) / 2.0
            elif p["kind"] in ("control", "farsight"):
                life = np.inf
            else:
                life = C.WARD_TOTEM_DURATION_MAX
            p["t1"] = p["t0"] + life
        out[n] = (
            p["net_id"],
            p["owner_slot"],
            UNKNOWN,
            p["kind"],
            p["x"],
            p["z"],
            p["t0"],
            p["t1"],
            p["t1_known"],
            C.WARD_SIGHT_BY_KIND.get(p["kind"], C.SIGHT_WARD_TOTEM),
        )
    return out


def _anchors(bundle: PacketBundle, slots: dict[int, int]) -> np.ndarray:
    """Labelled champion positions, from spell casts and basic attacks."""
    rows: list[tuple[float, int, float, float, int]] = []
    for arr, id_field, kind in (
        (bundle.casts, "caster_net_id", ANCHOR_CAST),
        (bundle.attacks, "source_net_id", ANCHOR_ATTACK),
    ):
        if arr.size == 0:
            continue
        for row in arr:
            slot = slots.get(int(row[id_field]))
            if slot is None:
                continue  # a turret, a pet, or the ghost caster
            rows.append((float(row["t"]), slot, float(row["src_x"]), float(row["src_z"]), kind))

    out = np.empty(len(rows), dtype=ANCHOR)
    for n, r in enumerate(rows):
        out[n] = r
    out.sort(order=["t", "slot"], kind="stable")
    return out


def _replication(bundle: PacketBundle, slots: dict[int, int], want: str) -> np.ndarray:
    """Pull one replicated attribute for champions.

    Matched on the name OR the (primary, secondary) index pair, because the name is
    empty for the majority of real entries and keying on it alone silently discards
    them.
    """
    repl = bundle.replication
    if repl.size == 0:
        return np.empty(0, dtype=REPL_VALUE)

    named = repl["name"] == want
    pairs = [k for k, v in C.REPL_INDEX_NAMES.items() if v == want]
    by_index = np.zeros(repl.size, dtype=bool)
    for primary, secondary in pairs:
        by_index |= (repl["primary"] == primary) & (repl["secondary"] == secondary)
    # An entry with a name that disagrees with its index pair is trusted on the name.
    unnamed = repl["name"] == ""
    hit = named | (by_index & unnamed)

    rows: list[tuple[float, int, float]] = []
    for row in repl[hit]:
        slot = slots.get(int(row["net_id"]))
        if slot is not None:
            rows.append((float(row["t"]), slot, float(row["value"])))

    out = np.empty(len(rows), dtype=REPL_VALUE)
    for n, r in enumerate(rows):
        out[n] = r
    out.sort(order=["t", "slot"], kind="stable")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def normalise(
    bundle: PacketBundle,
    terrain: Terrain,
    grid_spec: GridSpec | None = None,
    frame: FrameCalibration | None = None,
) -> MatchEvents:
    """Turn one match's packets into event tables.

    `frame` may be supplied to reuse a calibration across matches from the same source,
    which is the sensible thing to do for a whole shard: the offset is a property of the
    decoder, not of the match.
    """
    grid = grid_spec if grid_spec is not None else GridSpec()
    heroes = _hero_table(bundle)
    slots = _slot_lookup(heroes)

    calibration = (
        frame if frame is not None else calibrate_waypoint_frame(bundle.waypoint_xz, terrain)
    )

    order_xz = np.empty(bundle.waypoint_xz.size, dtype=ORDER_XZ)
    order_xz["x"] = bundle.waypoint_xz["x"].astype(np.float64) + calibration.offset
    order_xz["z"] = bundle.waypoint_xz["z"].astype(np.float64) + calibration.offset

    orders = np.empty(bundle.waypoints.size, dtype=ORDER)
    for f in ("t", "off", "n", "seq"):
        orders[f] = bundle.waypoints[f]
    orders["owner"] = UNKNOWN
    orders.sort(order=["t", "seq"], kind="stable")

    times = _minion_times(bundle)
    events = MatchEvents(
        match_id=bundle.meta.match_id,
        duration=float(bundle.meta.duration),
        header=StageHeader(
            stage=STAGE,
            stage_version=STAGE_VERSION,
            config_hash=grid.content_hash,
            input_hash=bundle.meta.source,
            extra={"frame_offset": calibration.offset, "packets": bundle.total_packets()},
        ),
        frame=calibration,
        heroes=heroes,
        orders=orders,
        order_xz=order_xz,
        anchors=_anchors(bundle, slots),
        turret_sites=_turret_sites(bundle),
        wards=_wards(bundle, slots, times),
        fog=_dedupe_fog(bundle, slots),
        speed=_replication(bundle, slots, C.ATTR_MOVE_SPEED),
        hp=_replication(bundle, slots, C.ATTR_HP),
        deaths=np.empty(0, dtype=DEATH),
        stats={
            "fog_rows_in": int(bundle.fog.size),
            "fog_rows_out": 0,  # filled below, once the array exists
            "minion_times_recovered": int(
                (~(bundle.minions["t_valid"].astype(bool))).sum() if bundle.minions.size else 0
            ),
        },
    )
    events.stats["fog_rows_out"] = int(events.fog.size)
    events.stats["fog_dedupe_ratio"] = (
        round(bundle.fog.size / max(1, events.fog.size), 1) if bundle.fog.size else 0.0
    )
    return events


def terrain_spec_of(terrain: Terrain) -> TerrainSpec:
    return terrain.spec
