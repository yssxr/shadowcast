"""Which vision source is responsible for each disagreement.

`fog_oracle` says how often the reconstruction is wrong and `real_fog` says whether the
positions involved were stale. Neither says *what to fix*, and on real data those are
different questions: at essentially exact positions the reconstruction still calls 31.8%
of hidden champions visible, and a false positive that size has to come from some source
being modelled when the game did not grant it.

So this attributes every false positive to the source class that covers the cell. The
number that matters is not "how often was a turret in range" — several classes usually
overlap — but **how often a class was the ONLY thing in range**, because that is the
subset a fix to that class would actually move. A source that is always accompanied by
another can be entirely wrong and change nothing.

False negatives get the same treatment in reverse: nothing was in range, so the question
is what would have had to be there, which `real_fog` answers geometrically instead.

The occlusion test is the real field-of-view table, not a distance check — a turret
1,000 units away behind a wall grants nothing and must not be blamed for anything.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast import sr
from shadowcast.fov.union import assemble, mask_bit, mask_to_bool, new_mask
from shadowcast.geom.grid import world_to_cell
from shadowcast.l1_events.schema import UNKNOWN, MatchEvents
from shadowcast.l2_reconstruct.vision import VisionStream
from shadowcast.validate.fog_oracle import _oracle_timeline

__all__ = ["SOURCE_CLASSES", "BlameReport", "blame_false_positives"]

#: Every class of vision source, in the order they are reported.
SOURCE_CLASSES = ("turret", "ward", "champion", "minion", "reveal")


@dataclass(frozen=True, slots=True)
class BlameReport:
    false_positives: int
    #: How often each class covered the cell at all.
    covered: Counter
    #: How often each class was the ONLY class covering it — the actionable subset.
    sole: Counter
    stats: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        n = max(1, self.false_positives)
        return {
            "false_positives": self.false_positives,
            "covered": {k: round(self.covered[k] / n, 4) for k in SOURCE_CLASSES},
            "sole": {k: round(self.sole[k] / n, 4) for k in SOURCE_CLASSES},
            **self.stats,
        }


def _sees(table, terrain, si: int, sj: int, radius: float, ti: int, tj: int) -> bool:
    """Does a source at `(si, sj)` with this radius see `(ti, tj)`?

    One source into a scratch mask and one bit read. Slower per query than assembling a
    whole team's mask, and the right shape here because a blame query is about a single
    cell rather than about the map.
    """
    if not (0 <= si < terrain.grid and 0 <= sj < terrain.grid):
        return False
    scratch = new_mask(terrain.grid)
    assemble(table, terrain, [(si, sj, radius, int(terrain.brush_id[sj, si]))], out=scratch)
    return mask_bit(scratch, ti, tj)


def blame_false_positives(
    events: MatchEvents, attribution, terrain, table, stride: int = 16
) -> BlameReport:
    """Attribute every false positive to the source classes that could explain it."""
    n_ticks, n_slots = attribution.pos.shape[:2]
    pos, valid = attribution.pos, attribution.valid
    team = events.heroes["team"].astype(np.int64)
    oracle = _oracle_timeline(events, n_ticks, C.TICK_DT)

    stream = VisionStream(events, attribution, terrain, table)
    front = stream._front
    covered: Counter = Counter()
    sole: Counter = Counter()
    total = 0

    for tick, mask_order, mask_chaos in stream.masks():
        if tick % stride:
            continue
        t = tick * C.TICK_DT
        bools = (mask_to_bool(mask_order, terrain.grid), mask_to_bool(mask_chaos, terrain.grid))

        for slot in range(n_slots):
            own = int(team[slot])
            if own == UNKNOWN or not valid[tick, slot]:
                continue
            observer = 1 - own
            x, z = pos[tick, slot]
            ti, tj = world_to_cell(float(x), float(z))
            if not (0 <= ti < terrain.grid and 0 <= tj < terrain.grid):
                continue
            # A false positive: we claim vision the stream says the team did not have.
            if not bools[observer][tj, ti] or oracle[tick, slot]:
                continue
            total += 1

            hits: set[str] = set()
            for site in events.turret_sites:
                if int(site["team"]) != observer or not np.isfinite(site["x"]):
                    continue
                if t >= float(site["destroyed_t"]):
                    continue
                si, sj = world_to_cell(float(site["x"]), float(site["z"]))
                if _sees(table, terrain, si, sj, C.SIGHT_TURRET, ti, tj):
                    hits.add("turret")
                    break
            for ward in events.wards:
                if int(ward["team"]) != observer or not (ward["t0"] <= t <= ward["t1"]):
                    continue
                si, sj = world_to_cell(float(ward["x"]), float(ward["z"]))
                if _sees(table, terrain, si, sj, float(ward["sight"]), ti, tj):
                    hits.add("ward")
                    break
            for other in range(n_slots):
                if int(team[other]) != observer or not valid[tick, other]:
                    continue
                si, sj = world_to_cell(float(pos[tick, other, 0]), float(pos[tick, other, 1]))
                if _sees(table, terrain, si, sj, C.SIGHT_CHAMPION, ti, tj):
                    hits.add("champion")
                    break
            for wave in events.minion_waves:
                if int(wave["team"]) != observer:
                    continue
                lane = str(wave["lane"])
                p = sr.minion_clump_position(
                    lane,
                    observer,
                    float(wave["t0"]),
                    t,
                    float(wave["t1"]),
                    front_s=float(front[lane][tick]),
                )
                if p is None:
                    continue
                si, sj = world_to_cell(float(p[0]), float(p[1]))
                if _sees(table, terrain, si, sj, C.SIGHT_MINION, ti, tj):
                    hits.add("minion")
                    break

            # Reveals are whatever is left: the base layer above is everything except
            # them, so a covered cell no base source explains was revealed by an attack.
            if not hits:
                hits.add("reveal")

            for name in hits:
                covered[name] += 1
            if len(hits) == 1:
                sole[next(iter(hits))] += 1

    return BlameReport(
        false_positives=total,
        covered=covered,
        sole=sole,
        stats={
            "stride": stride,
            "turrets_destroyed": int(np.isfinite(events.turret_sites["destroyed_t"]).sum()),
        },
    )
