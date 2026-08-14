"""Independent visibility oracle for the synthetic generator.

This computes the fog transitions the synthetic stream will publish, and it must share
no code with `fov/` — otherwise validating L2's masks against these events would be a
tautology dressed up as a test.

It is independent along three axes, not just one:

1. **Different question.** `fov/` computes a whole visible *region* from a source. This
   asks only "can observer at A see the champion at B", which is all a fog event
   asserts. No mask is ever built.
2. **Different algorithm.** Octant sweep with accumulated slope intervals there;
   per-pair segment marching here.
3. **Different coordinates.** `fov/` snaps sources to cell centres. This marches
   between the entities' true continuous positions, so cell quantisation enters at a
   different place and in a different way.

The ray-march logic is deliberately duplicated rather than factored out of
`fov/reference.py`. Sharing it would import exactly the correlation the oracle exists
to avoid, and twenty lines is a cheap price for that.

Everything here works in **cell units** — world coordinates divided by the cell size —
so the terrain lookup is a truncation and the radius test is a plain comparison.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["compute_visibility", "segment_clear", "transitions_from_visibility"]

#: March step, in cells. Fine enough that the ray cannot tunnel through a one-cell
#: wall, which is the only failure mode that would matter here.
_STEP = 0.1


@njit(cache=True)
def segment_clear(
    blocks_vision: np.ndarray,
    brush_id: np.ndarray,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
    src_brush: int,
) -> bool:
    """Is the open segment from (x0, z0) to (x1, z1) free of occluders?

    Coordinates are in cell units. Endpoints are excluded: a source cannot occlude
    itself, and an occluder at the target does not prevent seeing the target.
    """
    h, w = blocks_vision.shape
    dx = x1 - x0
    dz = z1 - z0
    length = np.sqrt(dx * dx + dz * dz)
    if length <= _STEP:
        return True

    i0 = int(x0)
    j0 = int(z0)
    i1 = int(x1)
    j1 = int(z1)

    steps = int(length / _STEP)
    for s in range(1, steps + 1):
        t = (s * _STEP) / length
        if t >= 1.0:
            break
        px = x0 + dx * t
        pz = z0 + dz * t
        ci = int(px)
        cj = int(pz)
        if (ci == i0 and cj == j0) or (ci == i1 and cj == j1):
            continue
        if ci < 0 or ci >= w or cj < 0 or cj >= h:
            return False
        if blocks_vision[cj, ci]:
            return False
        b = brush_id[cj, ci]
        if b >= 0 and b != src_brush:
            return False
    return True


@njit(cache=True)
def compute_visibility(
    blocks_vision: np.ndarray,
    brush_id: np.ndarray,
    champ_x: np.ndarray,  # f8[n_ticks, n_champs] in cell units
    champ_z: np.ndarray,
    champ_brush: np.ndarray,  # i2[n_ticks, n_champs]
    champ_alive: np.ndarray,  # u1[n_ticks, n_champs]
    champ_team: np.ndarray,  # u1[n_champs]
    src_off: np.ndarray,  # i8[n_ticks * 2]  index into the flat source arrays
    src_n: np.ndarray,  # i4[n_ticks * 2]
    src_x: np.ndarray,  # f8[total_sources] in cell units
    src_z: np.ndarray,
    src_r: np.ndarray,  # f8[total_sources] radius in cells
    src_brush: np.ndarray,  # i2[total_sources]
) -> np.ndarray:
    """`visible[tick, observer_team, champ]` — 1 where that team can see that champion.

    Own-team entries are set to 1 unconditionally: a team always sees its own members,
    and that is precisely the fact that lets the observing team of a real fog event be
    recovered from the champion it names.
    """
    n_ticks, n_champs = champ_x.shape
    out = np.zeros((n_ticks, 2, n_champs), dtype=np.uint8)

    for t in range(n_ticks):
        for c in range(n_champs):
            own = champ_team[c]
            out[t, own, c] = 1
            if champ_alive[t, c] == 0:
                # A dead champion is at the fountain and is public knowledge, but it
                # is not *observed*; the caller decides what that means for darkness.
                continue

            obs = 1 - own
            slot = t * 2 + obs
            lo = src_off[slot]
            hi = lo + src_n[slot]
            tx = champ_x[t, c]
            tz = champ_z[t, c]
            tb = champ_brush[t, c]

            for s in range(lo, hi):
                # Brush is opaque inward, so a source outside the target's brush
                # cannot see it at all, however clear the line.
                sb = src_brush[s]
                if tb >= 0 and tb != sb:
                    continue
                dx = tx - src_x[s]
                dz = tz - src_z[s]
                r = src_r[s]
                if dx * dx + dz * dz > r * r:
                    continue
                if segment_clear(blocks_vision, brush_id, src_x[s], src_z[s], tx, tz, sb):
                    out[t, obs, c] = 1
                    break

    return out


def transitions_from_visibility(
    visible: np.ndarray,
    champ_team: np.ndarray,
    net_ids: np.ndarray,
    tick_dt: float,
) -> np.ndarray:
    """Turn a visibility timeline into `EnterFog`/`LeaveFog` rows.

    Only cross-team transitions produce events, because a team never loses sight of
    its own members and the real stream contains no such events either.

    The published rows carry `(t, net_id, leaving)` and nothing else — no observer
    field — exactly matching the real packets. That is deliberate: the observing team
    has to be *re-derived* downstream from the champion's own team, so the code doing
    that derivation is exercised by synthetic data too.
    """
    from shadowcast.packets.source import FOG

    rows: list[tuple[float, int, int]] = []
    n_ticks, _, n_champs = visible.shape
    for c in range(n_champs):
        obs = 1 - int(champ_team[c])
        seen = visible[:, obs, c].astype(bool)
        # Champions start the game visible to nobody but their own team; treat tick 0
        # as a transition if they are visible then, so the timeline is self-describing.
        prev = False
        for t in range(n_ticks):
            if bool(seen[t]) != prev:
                rows.append((t * tick_dt, int(net_ids[c]), 1 if seen[t] else 0))
                prev = bool(seen[t])

    out = np.empty(len(rows), dtype=FOG)
    for n, (t, nid, leaving) in enumerate(rows):
        out[n]["t"] = t
        out[n]["net_id"] = nid
        out[n]["leaving"] = leaving
    # `seq` is stamped later, once every packet kind has been interleaved.
    out["seq"] = -1
    out.sort(order="t", kind="stable")
    return out
