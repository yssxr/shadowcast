"""Negative information: what not seeing someone tells you.

This is the module the project exists for. Every League vision tool in existence draws a
circle growing from a last known position. None of them subtract the region the team is
currently looking at — which is strange, because that is where most of the information
is. If Blue holds the whole river and does not see Red's jungler, Red's jungler is not in
the river, and after twenty seconds of that the surviving belief is a thin terrain-shaped
sliver rather than a fat disc.

The update is one line of arithmetic:

    logw[particle] += log1p(-p_d)   for every particle inside the visible region

and the whole design question is what `p_d` should be.

**It must not be 1.** A hard kill is the obvious implementation and it is wrong, not as a
matter of taste but of arithmetic. Our visible region is a reconstruction: source
positions snap to 28.8-unit cells, recursive shadowcasting is a few percent permissive at
wall corners, trajectories carry their own timing error, and a ward's exact expiry is
modelled. A particle killed by a mask that is wrong at its edge can never come back, so
one bad cell at one tick permanently deletes the correct hypothesis, and the filter
reports high confidence in a wrong answer — the single worst failure mode available to a
Bayesian filter, because it looks like success.

So `p_d` is 0.98 in the interior and 0.75 in the outer two-cell ring, and the ring is
where every one of those error sources lives. A particle sitting on the boundary is
downweighted by a factor of four per tick rather than annihilated, which is enough to
resolve within a second of genuine visibility and recoverable if we were wrong.

`p_d = 1.0` stays available as an ablation, because "we chose the soft update" is an
assertion and "here is what the hard update costs" is a measurement.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "collapse_to_cell",
    "detection_field",
    "effective_sample_size",
    "negative_update",
    "systematic_resample",
]


@njit(cache=True)
def detection_field(
    cell: np.ndarray,
    mask: np.ndarray,
    grid: int,
    ring: int,
    pd_interior: float,
    pd_edge: float,
    out: np.ndarray,
) -> None:
    """Per-particle detection probability against a packed visibility mask.

    Zero where the particle is outside the visible region — we are not looking there, so
    not seeing them is not evidence.

    Interior versus edge is decided by probing the eight cells at Chebyshev distance
    `ring` rather than by eroding the mask. Two reasons: an erosion of a 512² mask at
    every tick for both teams is 3.8 billion cell operations a match, against 230 million
    bit reads this way; and the probe is exact for a locally convex region and only
    slightly permissive otherwise, which errs toward calling a cell an edge — the
    cautious direction, since edge cells get the weaker update.
    """
    n = cell.shape[0]
    words = mask.shape[1]
    for p in range(n):
        c = cell[p]
        j = c // grid
        i = c - j * grid
        if j < 0 or j >= grid or i < 0 or i >= grid:
            out[p] = 0.0
            continue
        if not ((mask[j, i >> 6] >> np.uint64(i & 63)) & np.uint64(1)):
            out[p] = 0.0
            continue

        interior = True
        for dj in (-ring, 0, ring):
            for di in (-ring, 0, ring):
                if dj == 0 and di == 0:
                    continue
                nj = j + dj
                ni = i + di
                if nj < 0 or nj >= grid or ni < 0 or ni >= grid:
                    # Off the map counts as visible: the map edge is not a place an
                    # enemy can hide, so treating it as darkness would invent an edge
                    # ring around the whole world.
                    continue
                if (ni >> 6) >= words:
                    continue
                if not ((mask[nj, ni >> 6] >> np.uint64(ni & 63)) & np.uint64(1)):
                    interior = False
                    break
            if not interior:
                break
        out[p] = pd_interior if interior else pd_edge


@njit(cache=True)
def negative_update(logw: np.ndarray, pd: np.ndarray) -> None:
    """`logw += log1p(-p_d)`, in place.

    Log weights rather than linear ones, and that is not fastidiousness. A particle in a
    permanently visible region accumulates `log(0.02)` per tick; in float32 linear
    weights it underflows to exactly zero inside about 200 ticks — twenty-five seconds —
    after which it is indistinguishable from a particle that was never possible, and the
    normalisation silently divides by zero when *every* particle gets there.
    """
    for p in range(logw.shape[0]):
        d = pd[p]
        if d <= 0.0:
            continue
        if d >= 1.0:
            logw[p] = -np.inf  # the hard-kill ablation, reachable only when pd == 1
        else:
            logw[p] += np.log1p(-d)


def collapse_to_cell(cell: np.ndarray, logw: np.ndarray, observed: int) -> None:
    """A sighting is exact, so the posterior is a point mass.

    Champions are rendered at their true position the moment they enter vision — there is
    no measurement noise in a replay, only the 28.8-unit quantisation the cell already
    represents. Spreading particles around a sighting would be inventing uncertainty that
    the observer did not have, and it would show up directly as miscalibration: the 50%
    credible region would contain the truth far more than half the time.
    """
    cell.fill(observed)
    logw.fill(0.0)


@njit(cache=True)
def effective_sample_size(logw: np.ndarray) -> float:
    """Kish's ESS, computed in the log domain.

    The max-subtraction is what keeps this finite: raw `exp(logw)` underflows long before
    the ratio it is part of becomes meaningless, so the naive version reports ESS = nan
    exactly when the filter most needs to notice it is in trouble.
    """
    n = logw.shape[0]
    m = -np.inf
    for p in range(n):
        if logw[p] > m:
            m = logw[p]
    if not np.isfinite(m):
        return 0.0
    s1 = 0.0
    s2 = 0.0
    for p in range(n):
        w = np.exp(logw[p] - m)
        s1 += w
        s2 += w * w
    if s2 <= 0.0:
        return 0.0
    return s1 * s1 / s2


@njit(cache=True)
def systematic_resample(logw: np.ndarray, u: float, out: np.ndarray) -> None:
    """Systematic resampling: one uniform draw, `n` evenly spaced strata.

    Chosen over multinomial because it has strictly lower variance for the same cost and
    — the reason that matters here — it is deterministic given one number, so a filter
    run twice with the same seed produces bit-identical particle sets. The
    information-barrier test asserts exactly that, and multinomial resampling would make
    the assertion untestable without weakening it to a statistical comparison.
    """
    n = logw.shape[0]
    m = -np.inf
    for p in range(n):
        if logw[p] > m:
            m = logw[p]
    total = 0.0
    for p in range(n):
        total += np.exp(logw[p] - m)
    if total <= 0.0 or not np.isfinite(total):
        for p in range(n):
            out[p] = p
        return

    src = 0
    acc = np.exp(logw[0] - m) / total
    for p in range(n):
        target = (u + p) / n
        while acc < target and src < n - 1:
            src += 1
            acc += np.exp(logw[src] - m) / total
        out[p] = src
