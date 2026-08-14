"""Inferring champion deaths and who caused them.

There is no death packet. `HeroDie` is declared in the published schema and fires zero
times across 965,768 real packets, and no champion net_id ever appears as the victim in
the NPC death packets either — checked across 45,851 of them. So a champion death has to
be inferred, and both halves of it are guesses that carry a confidence.

**The death itself** is a health replication reaching zero. That is about as direct as an
inference gets, and the only real hazard is a champion's health legitimately touching
zero without dying, which does not happen.

**The killer** is whoever dealt damage to that champion most recently before it died.
There is no killing-blow flag, so this is genuinely uncertain: an execution by a minion
or a turret leaves no champion as the last damager, and a kill stolen in the final
instant looks identical to one earned over the preceding five seconds. The confidence
reported is the killer's share of the damage in the window, so a clean solo kill scores
near 1.0 and a chaotic teamfight scores low — which is the right shape, because that is
exactly when the attribution is least trustworthy.

**The respawn** is observed rather than computed. League's respawn timer is a function of
level and game clock that we could evaluate, but the stream already answers the question:
the champion's next labelled observation after the death is when it was demonstrably
alive again. Anchors arrive roughly every 0.75 s, so that is a tight bound and it needs no
formula that might be wrong for the patch.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from shadowcast.l1_events.schema import DEATH, UNKNOWN, MatchEvents

__all__ = ["DeathResolution", "resolve_deaths"]

#: How far back to look for the damage that caused a death.
_DAMAGE_WINDOW = 5.0
#: A death this soon after a previous one for the same champion is the same event seen
#: twice, not a second death — health can be replicated at zero more than once.
_DEDUPE_WINDOW = 3.0


@dataclass(frozen=True, slots=True)
class DeathResolution:
    deaths: np.ndarray  # DEATH dtype
    stats: dict[str, object]

    @property
    def n_deaths(self) -> int:
        return int(self.deaths.size)

    @property
    def killers_identified(self) -> int:
        return int((self.deaths["killer"] != UNKNOWN).sum()) if self.deaths.size else 0


def resolve_deaths(events: MatchEvents, window: float = _DAMAGE_WINDOW) -> DeathResolution:
    """Infer deaths from health, killers from damage, and respawns from anchors."""
    hp = events.hp
    if hp.size == 0:
        return DeathResolution(np.empty(0, dtype=DEATH), {"reason": "no health replication"})

    zeros = hp[hp["value"] <= 0.0]
    zeros = zeros[np.argsort(zeros["t"], kind="stable")]

    rows: list[tuple[float, int, int, float, float]] = []
    last_death_t: dict[int, float] = {}
    unattributed = 0
    duplicates = 0

    for rec in zeros:
        slot = int(rec["slot"])
        t = float(rec["t"])
        if slot in last_death_t and t - last_death_t[slot] < _DEDUPE_WINDOW:
            duplicates += 1
            continue
        last_death_t[slot] = t

        killer = UNKNOWN
        confidence = 0.0
        if events.damage.size:
            dmg = events.damage
            inwin = dmg[(dmg["target"] == slot) & (dmg["t"] >= t - window) & (dmg["t"] <= t + 1e-9)]
            if inwin.size:
                # Last damager, with confidence set by their share of the window's
                # damage — low in a teamfight, which is exactly when it is least sure.
                last = inwin[np.argmax(inwin["t"])]
                killer = int(last["source"])
                total = float(inwin["amount"].sum())
                mine = float(inwin["amount"][inwin["source"] == killer].sum())
                confidence = mine / total if total > 0 else 0.0
        if killer == UNKNOWN:
            unattributed += 1

        # Respawn: the champion's next labelled observation. Observed, not computed.
        later = events.anchors[(events.anchors["slot"] == slot) & (events.anchors["t"] > t)]
        respawn = float(later["t"].min()) if later.size else np.nan

        rows.append((t, slot, killer, respawn, confidence))

    out = np.empty(len(rows), dtype=DEATH)
    for n, r in enumerate(rows):
        out[n] = r

    return DeathResolution(
        deaths=out,
        stats={
            "health_zero_rows": int(zeros.size),
            "duplicate_zeros_dropped": duplicates,
            "deaths": int(out.size),
            "killer_unattributed": unattributed,
            "mean_killer_confidence": round(
                float(out["killer_confidence"].mean()) if out.size else 0.0, 3
            ),
            "respawn_unobserved": int(np.isnan(out["respawn_t"]).sum()) if out.size else 0,
        },
    )


def with_deaths(events: MatchEvents, resolution: DeathResolution) -> MatchEvents:
    return dataclasses.replace(events, deaths=resolution.deaths)
