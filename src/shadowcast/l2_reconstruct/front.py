"""Where each lane's minion waves are actually meeting, over time.

Minions are the largest vision source in the game and the only one whose position is
modelled rather than observed: they have no labelled position packets, and movement
orders carry no entity id, so nothing in the corpus says where a minion stands. What the
corpus does say is when each one spawned, which lane it belongs to and when it died.

That leaves one free parameter — the arclength fraction along the lane where the two
waves meet and stop advancing — and the obvious answer, the midpoint, is right on average
and wrong at every individual moment. MEASURED on a real match, the front sits a median
**1,442 units from the midpoint on top, 771 on mid and 1,640 on bot**. A minion sees 1,200
units, so a front-line error of that size does not merely blur the minion's vision, it
puts it somewhere else entirely.

**The evidence is champions.** A champion trading damage with a lane minion is standing at
the front, and champion positions are recovered by attribution. So `MINION_CONTACT` rows
become arclength observations, and this module turns the scatter into a track.

Two properties are deliberate:

- **It falls back rather than extrapolates.** With no evidence near a time, the estimate
  returns to the midpoint instead of holding the last reading. Lanes go quiet for real
  reasons — a wave crashes, both laners recall, the lane is pushed and abandoned — and a
  stale front held for forty seconds is a confident lie, while the midpoint is at worst
  the average truth.
- **It is smooth in time.** A front that jumps between adjacent ticks makes minion vision
  flicker, which shows up as spurious visibility transitions rather than as position
  error. The exponential kernel is wide enough that a single stray contact cannot move it
  far — one observation among twenty shifts the estimate by 5% of its distance.

The estimator is not told which team is pushing, and does not need to be: both waves stop
at the same place, so the front is a property of the lane rather than of a side.
"""

from __future__ import annotations

import numpy as np

from shadowcast import constants as C
from shadowcast import sr
from shadowcast.l1_events.schema import UNKNOWN, MatchEvents

__all__ = ["FRONT_HALF_LIFE", "estimate_front"]

#: Time constant of the exponential kernel, in seconds. CHOSEN at roughly one wave
#: interval (30 s): shorter and the track chases individual last-hits, longer and it
#: cannot follow a lane state that genuinely changes between waves.
FRONT_HALF_LIFE = 20.0

#: Weight given to the midpoint as a prior observation, in units of contacts.
#:
#: The midpoint is not a fallback, it is a *reason*: both waves spawn simultaneously and
#: move at the same speed, so absent any evidence they meet in the middle. Treating it as
#: `_PRIOR_WEIGHT` observations at `MEETING_S` makes the estimate a posterior mean —
#: shrunk hard toward the midpoint when contacts are sparse, essentially unshrunk when
#: they are dense — instead of a raw average that trusts three last-hits as much as sixty.
#:
#: MEASURED, and this is what the number is for: a synthetic match whose waves genuinely
#: do meet at 0.500 yields about 3.5 units of kernel weight per lane, while a real match
#: yields about 66. At a prior weight of 8 the synthetic estimate shrinks by 70% (its
#: deviation from the midpoint is noise, and the estimator is told so by the sparsity)
#: while the real one keeps 89% of its deviation. Without the prior the synthetic noise
#: alone cost 1.1 points of fog agreement.
_PRIOR_WEIGHT = 8.0

#: How far out to look for contacts, in multiples of the half-life. Beyond this the
#: kernel weight is under 1% and the samples only cost time.
_KERNEL_SPAN = 5.0


def estimate_front(
    events: MatchEvents, pos: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-lane front-line arclength for every tick, keyed by lane name.

    `pos` and `valid` are the attribution outputs, `(n_ticks, n_slots, 2)` and
    `(n_ticks, n_slots)`. A contact whose champion has no position claim at that tick is
    dropped rather than guessed — it is evidence about a position we do not have.
    """
    n_ticks = int(pos.shape[0])
    default = np.full(n_ticks, sr.MEETING_S, dtype=np.float64)
    out = {lane: default.copy() for lane in sr.LANES}
    if events.minion_contacts.size == 0:
        return out

    tick_t = np.arange(n_ticks) * C.TICK_DT
    for lane in sr.LANES:
        rows = events.minion_contacts[events.minion_contacts["lane"] == lane]
        if rows.size == 0:
            continue

        ticks = np.clip(np.round(rows["t"] * C.TICK_HZ).astype(np.int64), 0, n_ticks - 1)
        slots = rows["slot"].astype(np.int64)
        keep = (slots != UNKNOWN) & valid[ticks, slots]
        if not keep.any():
            continue
        ticks, slots = ticks[keep], slots[keep]

        # Project each contact onto the lane once, not once per tick.
        obs_t = tick_t[ticks]
        obs_s = np.array([sr.arclength_fraction(lane, pos[tk, sl]) for tk, sl in zip(ticks, slots)])

        order = np.argsort(obs_t, kind="stable")
        obs_t, obs_s = obs_t[order], obs_s[order]

        # Only contacts inside the kernel span matter, so walk a window over the sorted
        # observations rather than weighting all of them against every tick.
        span = _KERNEL_SPAN * FRONT_HALF_LIFE
        lo = np.searchsorted(obs_t, tick_t - span, side="left")
        hi = np.searchsorted(obs_t, tick_t + span, side="right")
        track = out[lane]
        for k in range(n_ticks):
            a, b = int(lo[k]), int(hi[k])
            if b <= a:
                continue
            w = np.exp(-np.abs(obs_t[a:b] - tick_t[k]) / FRONT_HALF_LIFE)
            total = float(w.sum()) + _PRIOR_WEIGHT
            track[k] = float((w @ obs_s[a:b] + _PRIOR_WEIGHT * sr.MEETING_S) / total)

        # A front outside the lane's own body is not a front — it is a champion who was
        # standing somewhere odd when the damage landed. Clamping keeps a base skirmish
        # from dragging the meeting point into a fountain.
        np.clip(track, C.FRONT_MIN_S, C.FRONT_MAX_S, out=track)
    return out
