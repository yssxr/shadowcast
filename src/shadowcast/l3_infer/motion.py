"""How a belief moves when nobody is looking.

Six motion models, which are the six baselines' only substantive difference. They split
into two families, and the split is not cosmetic:

**Re-derived models** (`uniform`, `disc`, `geodisc`) ignore their own previous state and
recompute the belief from the last sighting and the elapsed time. They are closed forms
pretending to be filters, which is the honest way to represent "grow a circle from where
we last saw them". The thing every existing vision overlay does.

**Propagated models** (`constant_velocity`, `navmesh_diffusion`, `navmesh_behavioural`)
carry state forward tick by tick, so evidence accumulated at tick 40 is still shaping the
distribution at tick 400. Only these can express negative information, because only these
have anything for it to act on.

Particles are **cell indices, not points.** That decision buys three things. The negative
update becomes a mask bit lookup instead of an interpolation, so "is this particle inside
the visible region" is exact rather than nearly-exact. Entropy, credible area and the
export mixture are all grid-shaped anyway. And: the reason that settled it. A lattice
walk has a transition matrix you can write down, which is what lets `single_step_matrix`
below be checked against an exact Bayesian forward pass using *the shipped kernel* rather
than a stand-in written for the test.

The cost is that a constant-velocity baseline has to carry a float velocity alongside its
cell index. That is a small ugliness in one baseline, paid once.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from shadowcast import constants as C
from shadowcast.geom.path import DIAG_COST, STEP_COST, diagonal_ok

__all__ = [
    "MOVES",
    "STAY",
    "propose_cells",
    "refresh_goals",
    "role_targets",
    "single_step_matrix",
    "sub_steps_per_tick",
]

#: (dj, di, weight) for the eight lattice moves, in the order the kernel scans them.
#: The weight is `STEP_COST / cost`, i.e. exactly 1/sqrt(2) for a diagonal. Uniform
#: weights would make the walk measurably faster along diagonals than along axes, which
#: on a map whose corridors run diagonally would bias the belief in a direction that has
#: nothing to do with League and everything to do with our lattice.
MOVES = np.array(
    [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, STEP_COST / DIAG_COST),
        (-1, 1, STEP_COST / DIAG_COST),
        (1, -1, STEP_COST / DIAG_COST),
        (1, 1, STEP_COST / DIAG_COST),
    ],
    dtype=np.float64,
)

#: Heading sentinel for a particle that has not moved yet.
STAY = np.int8(-1)


def sub_steps_per_tick(
    tick_hz: int = C.TICK_HZ,
    speed: float = C.MOVE_SPEED_DEFAULT,
    cell_size: float = C.GRID_CELL_SIZE,
) -> int:
    """Lattice steps per simulation tick, from the speed the walk should represent.

    At 8 Hz and 335 u/s a champion covers 41.9 units, 1.45 cells: so two sub-steps
    with a stay probability lands the right displacement rather than a round number
    chosen for looking tidy.
    """
    return max(1, round(speed / tick_hz / cell_size))


# ---------------------------------------------------------------------------
# The kernel, and its analytic form
# ---------------------------------------------------------------------------
@njit(cache=True)
def propose_cells(
    cell: np.ndarray,
    heading: np.ndarray,
    walkable: np.ndarray,
    n_sub: int,
    p_stay: float,
    persistence: float,
    goal: np.ndarray,
    goal_beta: float,
    rnd: np.ndarray,
    out_cell: np.ndarray,
    out_heading: np.ndarray,
) -> None:
    """One tick of the navmesh walk, in place into `out_cell` / `out_heading`.

    Three effects, each disableable to zero so the baselines really are the same code:

    `p_stay`: champions do stand still: recalling, holding a brush, hitting a camp.
    `persistence`: and when they do move, repeating the previous heading is more likely
    than a fresh isotropic draw.
    `goal_beta`: and, mostly, they are walking somewhere specific. `goal[p]` is that
    somewhere, managed by the caller; the kernel only biases each step toward it.

    **The goal term is not a refinement, it is the model.** An unbiased walk was tried
    first and it cannot reproduce how far champions actually travel: measured against
    synthetic ground truth, the median twenty-second displacement is 1,395 units, and no
    combination of sub-steps, stay probability and heading persistence gets a diffusive
    walk past 900. A random walk is recurrent. It wanders back over itself, while a
    champion crossing the map does not. Raising persistence far enough to fix the long
    horizon breaks the short one, because a straight-line walk at champion speed covers
    1,600 units in the two seconds where the truth is 268.

    Randomness is pre-drawn into `rnd` of shape `(n_particles, n_sub)` rather than
    generated here. Numba's RNG state is not the same object as NumPy's, so a kernel
    that draws its own noise cannot be compared bit-for-bit against a NumPy reference,
    and the information-barrier test needs exactly that comparison.
    """
    grid = walkable.shape[0]
    n = cell.shape[0]
    weights = np.empty(9, dtype=np.float64)

    for p in range(n):
        c = cell[p]
        h = heading[p]
        g = goal[p]
        for s in range(n_sub):
            j = c // grid
            i = c - j * grid
            gj = g // grid - j
            gi = g - (g // grid) * grid - i
            gnorm = np.sqrt(float(gi * gi + gj * gj))

            move_total = 0.0
            for m in range(8):
                dj = int(MOVES[m, 0])
                di = int(MOVES[m, 1])
                nj = j + dj
                ni = i + di
                if nj < 0 or nj >= grid or ni < 0 or ni >= grid:
                    weights[m] = 0.0
                    continue
                if not walkable[nj, ni] or not diagonal_ok(walkable, j, i, dj, di):
                    weights[m] = 0.0
                    continue
                w = MOVES[m, 2]
                if persistence > 0.0 and m == h:
                    w *= 1.0 + persistence
                if goal_beta > 0.0 and gnorm > 0.0:
                    # Cosine between the candidate direction and the direction to the
                    # goal. MOVES is (dj, di) = (dz, dx), so di pairs with x.
                    dot = (di * gi + dj * gj) / gnorm
                    norm = 1.0 if (dj == 0 or di == 0) else 0.7071067811865476
                    w *= np.exp(goal_beta * dot * norm)
                weights[m] = w
                move_total += w

            # Stay gets whatever weight makes its probability exactly `p_stay`, rather
            # than a fixed weight competing against however many moves the terrain
            # happens to allow. Otherwise a particle in a corridor stands still far more
            # often than one in the open. An accident of the lattice masquerading as a
            # behavioural claim.
            # `p_stay` must be in [0, 1); a belief that never moves is not a model.
            weights[8] = move_total * p_stay / (1.0 - p_stay)
            total = move_total + weights[8]

            if move_total <= 0.0:
                # Fully enclosed: the only legal move is to stay. Reachable in principle
                # at the bottom of a one-cell pocket, and cheaper to handle than to prove
                # impossible.
                out_cell[p] = c
                out_heading[p] = STAY
                h = STAY
                continue

            u = rnd[p, s] * total
            acc = 0.0
            chosen = 8
            for m in range(9):
                acc += weights[m]
                if u < acc:
                    chosen = m
                    break
            if chosen == 8:
                h = STAY
            else:
                dj = int(MOVES[chosen, 0])
                di = int(MOVES[chosen, 1])
                c = (j + dj) * grid + (i + di)
                h = chosen
        out_cell[p] = c
        out_heading[p] = h


def single_step_matrix(walkable: np.ndarray, p_stay: float) -> np.ndarray:
    """The transition matrix of one `propose_cells` sub-step, drift and persistence off.

    **This is the specification of the kernel, not a re-implementation of it.** It exists
    so that `tests/test_pf.py` can compare the particle filter against an exact Bayesian
    forward pass over every state, and `tests/test_motion.py` can check that the kernel's
    empirical transition frequencies match this matrix. If the two ever drift apart, both
    tests fail, which is the point. A filter validated against a hand-written toy model
    proves only that the toy model was written to agree.

    `p_stay` must be in [0, 1), as in the kernel.

    Dense, so only usable on small grids. That is fine: it is only ever used on the 16×16
    world the exact test runs in.
    """
    grid = walkable.shape[0]
    n = grid * grid
    if n > 4096:
        raise ValueError(
            f"single_step_matrix is dense and {n}x{n} would be "
            f"{n * n * 8 / 1e9:.1f} GB; it is meant for the exact-inference test only"
        )
    t = np.zeros((n, n), dtype=np.float64)
    for j in range(grid):
        for i in range(grid):
            c = j * grid + i
            if not walkable[j, i]:
                t[c, c] = 1.0  # absorbing, and unreachable, so it never matters
                continue
            weights = np.zeros(n)
            move_total = 0.0
            for dj, di, w in MOVES:
                nj, ni = j + int(dj), i + int(di)
                if not (0 <= nj < grid and 0 <= ni < grid):
                    continue
                if not walkable[nj, ni] or not diagonal_ok(walkable, j, i, int(dj), int(di)):
                    continue
                weights[nj * grid + ni] += w
                move_total += w
            if move_total <= 0.0:
                t[c, c] = 1.0
            else:
                weights[c] += move_total * p_stay / (1.0 - p_stay)
                t[c] = weights / weights.sum()
    return t


# ---------------------------------------------------------------------------
# The behavioural prior
# ---------------------------------------------------------------------------
#: Where each resolved role tends to go.
#:
#: **Keyed on `roles.ROLES` verbatim**, and that is the point rather than a style note.
#: An earlier version matched `"jng"` and `"sup"` while the resolver emits `"jungle"` and
#: `"support"`, so two of every five enemies fell silently through to the catch-all. And
#: one of them was the jungler, the champion who spends the most time in fog and the one
#: this whole tool exists to locate. Nothing failed; the prior was simply absent for 40%
#: of the targets, and the only symptom was a belief that was confident in the wrong
#: places. The mapping is now asserted against `ROLES` at import.
_ROLE_LANES: dict[str, str] = {"top": "top", "mid": "mid", "bot": "bot", "support": "bot"}
_JUNGLE_ROLE = "jungle"


def role_targets(role: str, terrain, team: int | None = None) -> np.ndarray:
    """Cells a champion of this role plausibly walks toward.

    Roles are resolved upstream and a role is public information, everyone watching can
    see who went to the jungle, so conditioning the prior on it leaks nothing.

    This is a *random-waypoint* mobility model, which is the standard treatment of
    purposeful movement and, more to the point, is how champions actually move: pick a
    destination, walk there, pick another.
    """
    from shadowcast import sr
    from shadowcast.geom.grid import world_to_cell

    if role in _ROLE_LANES:
        pts = [np.asarray(sr.LANES[_ROLE_LANES[role]], dtype=np.float64)]
    elif role == _JUNGLE_ROLE:
        pts = [np.asarray(r, dtype=np.float64) for r in sr.JUNGLE_ROUTES.values()]
    else:
        # Genuinely unresolved: everywhere anyone goes, which is a weaker prior rather
        # than a wrong one.
        pts = [np.asarray(v, dtype=np.float64) for v in sr.LANES.values()]
        pts += [np.asarray(r, dtype=np.float64) for r in sr.JUNGLE_ROUTES.values()]
    # Champions recall. The generator sends them back every 190 seconds for 26, and real
    # players do it constantly, so a belief whose goal set contains only lanes and camps
    # walks the whole cloud up the lane while the champion is walking down it. Adding the
    # champion's OWN fountain is not a synthetic detail; it is a behaviour the model was
    # simply missing.
    if team is not None:
        pts.append(np.asarray([sr.FOUNTAINS[team]], dtype=np.float64))
    pts = np.concatenate(pts)

    cells = []
    for x, z in pts:
        i, j = world_to_cell(float(x), float(z))
        if 0 <= i < terrain.grid and 0 <= j < terrain.grid and terrain.walkable[j, i]:
            cells.append(j * terrain.grid + i)
    if not cells:
        cells = [int(np.flatnonzero(terrain.walkable.reshape(-1))[0])]
    return np.unique(np.asarray(cells, dtype=np.int32))


def refresh_goals(
    cell: np.ndarray,
    goal: np.ndarray,
    targets: np.ndarray,
    grid: int,
    arrive_cells: float,
    rng: np.random.Generator,
) -> int:
    """Give a new destination to every particle that reached its old one.

    Done in NumPy rather than inside the kernel so that all randomness stays on one
    generator. The barrier test asserts two runs of the same seed are bit-identical, and
    a second RNG inside Numba would make that impossible to guarantee.
    """
    dj = goal // grid - cell // grid
    di = goal % grid - cell % grid
    arrived = (dj * dj + di * di) <= arrive_cells * arrive_cells
    n = int(arrived.sum())
    if n:
        goal[arrived] = targets[rng.integers(0, targets.size, size=n)]
    return n
