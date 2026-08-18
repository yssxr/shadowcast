"""The real packet source: decoded replay shards behind the same seam as the synthetic one.

This is the file the whole `packets/source.py` seam exists for. Everything upstream of
here was built and validated against a synthetic generator; swapping in real data is one
new class implementing two methods, and `conformance.validate_source` runs the identical
invariant suite against both. So "the real reader has a subtle bug" surfaces as a named
assertion rather than as metrics that look slightly odd.

Reading the shards is fifteen lines of `gzip` and `json`. The official
[`…-gym` loader](https://github.com/Maknee/league-of-legends-decoded-replay-packets-gym)
is not usable: `parse_waypoints` treats `WaypointGroup`'s dict key as a net_id when it is
the list length, confirmed here in 100.0000% of 16,602 real pairs, so all of its
position tracking is wrong, and `get_heroes_by_team` reads a `team` field that does not
exist anywhere in the data.

## What the real stream does that the synthetic one does not

Four shape differences, each absorbed here rather than leaked upward:

**`Replication` is a dict, not a row.** It arrives as
`net_id_to_replication_datas: {net_id: {primary_index, secondary_index, name, data}}`,
one entry per attribute per unit, so a single packet fans out to many rows. 38% carry a
non-empty name, which is why the index pair is the key and the name is only a label.

**`SpawnMinion.time` is denormal garbage**, `1.59e-39` in the first row of the first
match. A running clock from the last trustworthy packet dates them instead, and `t_valid`
records which ones needed it.

**`BarrackSpawnUnit.minion_type` is garbage too**, arriving as 2^64-1. Minions are
classified by `SpawnMinion.name` and `skin_name`, which are correct.

**Positions are nested** under `position1` / `source_position` as `{x, z}` rather than
being flat fields.

## The identifier

The corpus has no match id, region, patch, rank, win/loss or duration anywhere. So the id
is constructed from the shard and line, `12_22/batch_001:7`: and it is honest about
being synthetic. The mockup's `EUW1_6412887731` is a string with nothing behind it.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shadowcast.packets.source import (
    PACKET_KINDS,
    MatchMeta,
    PacketBundle,
)

__all__ = ["ReplaySource", "read_shard_line"]

#: A timestamp below this is denormal garbage rather than a time near zero. Real
#: `SpawnMinion.time` values come through as ~1e-39; the smallest legitimate time is 0.0
#: and the next is a 30 Hz tick at 0.033.
MIN_VALID_TIME = 1e-6


def _xz(payload: dict[str, Any], key: str) -> tuple[float, float]:
    """Pull a nested `{x, z}` position, defaulting to the origin when absent."""
    point = payload.get(key) or {}
    return float(point.get("x", 0.0)), float(point.get("z", 0.0))


@dataclass(slots=True)
class _Rows:
    """Accumulators, one per bundle field."""

    heroes: list = None  # type: ignore[assignment]
    waypoints: list = None  # type: ignore[assignment]
    waypoint_xz: list = None  # type: ignore[assignment]
    fog: list = None  # type: ignore[assignment]
    replication: list = None  # type: ignore[assignment]
    turrets: list = None  # type: ignore[assignment]
    minions: list = None  # type: ignore[assignment]
    neutrals: list = None  # type: ignore[assignment]
    casts: list = None  # type: ignore[assignment]
    attacks: list = None  # type: ignore[assignment]
    damage: list = None  # type: ignore[assignment]
    deaths: list = None  # type: ignore[assignment]
    items: list = None  # type: ignore[assignment]
    barracks: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name in PACKET_KINDS:
            setattr(self, name, [])


def read_shard_line(events: list[dict[str, Any]], meta: MatchMeta) -> PacketBundle:
    """Convert one match's raw packet list into a `PacketBundle`."""
    rows = _Rows()
    clock = 0.0

    for seq, event in enumerate(events):
        kind = next(iter(event))
        payload = event[kind]
        raw_time = float(payload.get("time", 0.0))
        # The running clock only advances on packets whose own timestamp is
        # trustworthy, so a denormal `SpawnMinion.time` cannot drag it backwards.
        if raw_time > MIN_VALID_TIME and math.isfinite(raw_time):
            clock = max(clock, raw_time)

        if kind == "CreateHero":
            rows.heroes.append(
                (
                    raw_time,
                    payload["net_id"],
                    str(payload.get("name", ""))[:24],
                    str(payload.get("champion", ""))[:24],
                    seq,
                )
            )

        elif kind in ("WaypointGroup", "WaypointGroupWithSpeed"):
            # The dict key is the LIST LENGTH, not a net_id, movement orders carry no
            # entity attribution at all, which is the project's first hard problem.
            for _, points in (payload.get("waypoints") or {}).items():
                offset = len(rows.waypoint_xz)
                for point in points:
                    rows.waypoint_xz.append((point["x"], point["z"]))
                rows.waypoints.append(
                    (raw_time, offset, len(points), kind.endswith("WithSpeed"), seq)
                )

        elif kind in ("EnterFog", "LeaveFog"):
            rows.fog.append((raw_time, payload["net_id"], kind == "LeaveFog", seq))

        elif kind == "Replication":
            for net_id, data in (payload.get("net_id_to_replication_datas") or {}).items():
                value = data.get("data") or {}
                type_name = next(iter(value), "")
                raw_value = value.get(type_name)
                rows.replication.append(
                    (
                        raw_time,
                        int(net_id),
                        int(data.get("primary_index", 0)),
                        int(data.get("secondary_index", 0)),
                        str(data.get("name", ""))[:32],
                        # A null value is NaN, never zero. 199 entries in the first real
                        # match arrive as `{"Float": null}`, and a null `mHP` read as
                        # zero is a death that never happened, which the kill inference
                        # would then attribute to whoever last dealt damage.
                        float("nan") if raw_value is None else float(raw_value),
                        type_name == "Float",
                        seq,
                    )
                )

        elif kind == "CreateTurret":
            rows.turrets.append(
                (
                    raw_time,
                    payload["net_id"],
                    payload.get("owner_net_id", 0),
                    str(payload.get("name", ""))[:40],
                    seq,
                )
            )

        elif kind == "SpawnMinion":
            x, z = _xz(payload, "position1")
            valid = raw_time > MIN_VALID_TIME
            rows.minions.append(
                (
                    raw_time if valid else clock,
                    payload["net_id"],
                    x,
                    z,
                    str(payload.get("name", ""))[:28],
                    str(payload.get("skin_name", ""))[:32],
                    payload.get("targetable_on_client", 0) or 0,
                    valid,
                    seq,
                )
            )

        elif kind == "CreateNeutral":
            x, z = _xz(payload, "position1")
            rows.neutrals.append(
                (
                    raw_time,
                    payload["net_id"],
                    x,
                    z,
                    str(payload.get("name", ""))[:32],
                    payload.get("camp_id", -1),
                    payload.get("neutral_type", -1),
                    seq,
                )
            )

        elif kind == "CastSpellAns":
            sx, sz = _xz(payload, "source_position")
            tx, tz = _xz(payload, "target_position")
            rows.casts.append(
                (
                    raw_time,
                    payload.get("caster_net_id", 0),
                    str(payload.get("spell_name", ""))[:32],
                    payload.get("spell_hash", 0) or 0,
                    sx,
                    sz,
                    tx,
                    tz,
                    payload.get("slot", -1),
                    seq,
                )
            )

        elif kind == "BasicAttackPos":
            sx, sz = _xz(payload, "source_position")
            tx, tz = _xz(payload, "target_position")
            targets = payload.get("target_net_ids") or []
            rows.attacks.append(
                (
                    raw_time,
                    payload.get("source_net_id", 0) or payload.get("caster_net_id", 0),
                    # `target_net_id` is the attacked unit; the reveal rule needs it,
                    # because a champion is revealed only when it attacks an ENEMY.
                    payload.get("target_net_id") or (targets[0] if targets else 0),
                    sx,
                    sz,
                    tx,
                    tz,
                    seq,
                )
            )

        elif kind == "UnitApplyDamage":
            rows.damage.append(
                (
                    raw_time,
                    payload.get("source_net_id", 0),
                    payload.get("target_net_id", 0),
                    float(payload.get("damage", 0.0)),
                    seq,
                )
            )

        elif kind in ("NPCDieMapView", "NPCDieMapViewBroadcast"):
            rows.deaths.append(
                (
                    raw_time,
                    payload.get("killer_net_id", 0),
                    payload.get("killed_net_id", 0),
                    kind.endswith("Broadcast"),
                    seq,
                )
            )

        elif kind == "UseItem":
            rows.items.append((raw_time, payload["net_id"], payload.get("slot", -1), seq))

        elif kind == "BarrackSpawnUnit":
            # `minion_type` and `minion_level` are dropped: the former arrives as
            # 2^64-1, which does not survive the cast, and neither is needed once the
            # barrack itself identifies the lane and the side.
            rows.barracks.append(
                (
                    raw_time,
                    payload["minion_net_id"],
                    payload["barrack_net_id"],
                    payload.get("wave_count", -1),
                    seq,
                )
            )

    arrays: dict[str, np.ndarray] = {}
    for name, dtype in PACKET_KINDS.items():
        source = getattr(rows, name)
        out = np.empty(len(source), dtype=dtype)
        for n, row in enumerate(source):
            out[n] = row
        arrays[name] = out

    return PacketBundle(meta=meta, **arrays)


class ReplaySource:
    """Decoded replay shards, as a `PacketSource`.

    Matches are addressed by `"<shard stem>:<line>"`, which is stable across runs and
    carries its own provenance. Lines are indexed lazily on first use. A shard is 86 MB
    gzipped and 2 GB expanded, so nothing is held that is not asked for.
    """

    def __init__(self, shard: Path | str, limit: int | None = None) -> None:
        self.shard = Path(shard)
        if not self.shard.exists():
            raise FileNotFoundError(
                f"no shard at {self.shard}. Fetch one with:\n"
                '  uv run python -c "from huggingface_hub import hf_hub_download as d; '
                "d(repo_id='maknee/league-of-legends-decoded-replay-packets', "
                "repo_type='dataset', filename='12_22/batch_001.jsonl.gz', "
                "local_dir='data/raw')\""
            )
        self.limit = limit
        self._ids: list[str] | None = None

    @property
    def patch(self) -> str:
        """From the shard's parent directory. The packets themselves carry no patch."""
        return self.shard.parent.name or "unknown"

    def match_ids(self) -> list[str]:
        if self._ids is None:
            with gzip.open(self.shard, "rt") as fh:
                n = sum(1 for _ in fh)
            if self.limit is not None:
                n = min(n, self.limit)
            self._ids = [f"{self.shard.stem.replace('.jsonl', '')}:{i}" for i in range(n)]
        return list(self._ids)

    def _bundle(self, index: int, line: str) -> PacketBundle:
        events = json.loads(line)["events"]
        duration = max(
            (
                float(e[next(iter(e))].get("time", 0.0))
                for e in events
                if float(e[next(iter(e))].get("time", 0.0)) > MIN_VALID_TIME
            ),
            default=0.0,
        )
        stem = self.shard.stem.replace(".jsonl", "")
        meta = MatchMeta(
            match_id=f"{self.patch}/{stem}:{index}",
            source=str(self.shard),
            duration=duration,
            n_packets=len(events),
            patch=self.patch,
            # Stated rather than implied: the corpus carries no match id, region,
            # rank, win/loss or duration, so the identifier above is constructed.
            extra={"identifier": "constructed from shard and line", "line": index},
        )
        return read_shard_line(events, meta)

    def read(self, match_id: str) -> PacketBundle:
        index = int(match_id.rsplit(":", 1)[1])
        with gzip.open(self.shard, "rt") as fh:
            for n, line in enumerate(fh):
                if n == index:
                    return self._bundle(index, line)
        raise KeyError(f"{match_id} is not in {self.shard}")

    def read_all(self, limit: int | None = None) -> Iterator[PacketBundle]:
        """Every match in the shard, in one pass over the file.

        `read` seeks by decompressing from the start, so reading N matches one at a time
        is quadratic in the shard, and a shard is 83 MB gzipped against 2 GB expanded, so
        that is not a small constant. Measuring anything across a whole shard goes through
        here instead.
        """
        stop = self.limit if limit is None else min(limit, self.limit or limit)
        with gzip.open(self.shard, "rt") as fh:
            for n, line in enumerate(fh):
                if stop is not None and n >= stop:
                    return
                yield self._bundle(n, line)
