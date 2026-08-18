"""Tests for the belief filter, its motion kernel, and the negative update.

The centrepiece is `test_filter_matches_exact_bayes`. Everything else here checks a
component; that one checks whether the filter is doing *Bayesian inference* or merely
something plausible-looking, and it is the only test in the project capable of telling
the difference. On a 16×16 world there are 256 states, so the exact posterior can be
computed by matrix multiplication and compared against the particle cloud directly.

The comparison is meaningful only because `motion.single_step_matrix` is the analytic
form of the shipped kernel rather than a model written for the test, `test_motion.py`
pins the two together empirically, and if they ever drift apart both files fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.config import FilterSpec
from shadowcast.geom.bitset import pack_rows
from shadowcast.l3_infer import motion, observation

# ---------------------------------------------------------------------------
# A toy world small enough to enumerate
# ---------------------------------------------------------------------------
TOY = 16


def toy_walkable(seed: int = 3) -> np.ndarray:
    """A 16×16 world with a wall down the middle and a one-cell door.

    Deliberately not open ground. A wall with a door is where a diffusion model and an
    exact posterior disagree most: probability has to funnel, and any error in the
    transition kernel shows up as mass on the wrong side.
    """
    w = np.ones((TOY, TOY), dtype=bool)
    w[:, TOY // 2] = False
    w[TOY // 2, TOY // 2] = True  # the door
    rng = np.random.default_rng(seed)
    for _ in range(8):
        j, i = rng.integers(0, TOY, 2)
        if i != TOY // 2:
            w[j, i] = False
    return w


def toy_visible(walkable: np.ndarray, half: str) -> np.ndarray:
    """A packed mask covering one half of the toy world.

    This is the observer looking at a region. The thing the negative update consumes.
    """
    vis = np.zeros((TOY, TOY), dtype=bool)
    if half == "left":
        vis[:, : TOY // 2] = walkable[:, : TOY // 2]
    else:
        vis[:, TOY // 2 + 1 :] = walkable[:, TOY // 2 + 1 :]
    return pack_rows(vis)


def walk(
    cell,
    walkable,
    n_sub=1,
    p_stay=C.PARTICLE_STAY_PROB,
    persistence=0.0,
    goal=None,
    goal_beta=0.0,
    steps=1,
    rng=None,
    targets=None,
    arrive=C.GOAL_ARRIVE_CELLS,
):
    """Drive the kernel, so a signature change breaks one place rather than eight."""
    grid = walkable.shape[0]
    rng = rng if rng is not None else np.random.default_rng(0)
    heading = np.full(cell.size, motion.STAY, dtype=np.int8)
    goal = cell.copy() if goal is None else goal
    for _ in range(steps):
        if targets is not None:
            motion.refresh_goals(cell, goal, targets, grid, arrive, rng)
        motion.propose_cells(
            cell,
            heading,
            walkable,
            n_sub,
            p_stay,
            persistence,
            goal,
            goal_beta,
            rng.uniform(size=(cell.size, n_sub)),
            cell,
            heading,
        )
    return cell


# ---------------------------------------------------------------------------
# The kernel's analytic form
# ---------------------------------------------------------------------------
def test_transition_matrix_rows_are_distributions():
    t = motion.single_step_matrix(toy_walkable(), C.PARTICLE_STAY_PROB)
    assert np.allclose(t.sum(axis=1), 1.0)
    assert (t >= 0.0).all()


def test_stay_probability_is_what_it_says_regardless_of_terrain():
    """The whole point of normalising the stay weight against the move weights.

    A fixed stay weight competing against however many moves the terrain allows would
    make a particle in a corridor stand still far more often than one in the open. An
    accident of the lattice that would read as a behavioural claim.
    """
    walkable = toy_walkable()
    t = motion.single_step_matrix(walkable, 0.15)
    diag = np.array(
        [t[c, c] for c in range(TOY * TOY) if walkable[c // TOY, c % TOY] and t[c].argmax() != c]
    )
    open_cells = [
        c
        for c in range(TOY * TOY)
        if walkable[c // TOY, c % TOY]
        and 0 < c // TOY < TOY - 1
        and 0 < c % TOY < TOY - 1
        and walkable[c // TOY - 1 : c // TOY + 2, c % TOY - 1 : c % TOY + 2].all()
    ]
    assert diag.size > 0
    for c in open_cells:
        assert t[c, c] == pytest.approx(0.15, abs=1e-12)


def test_kernel_frequencies_match_the_matrix():
    """The kernel and its specification, checked against each other on real samples.

    Without this, `test_filter_matches_exact_bayes` would prove only that the analytic
    matrix agrees with itself.
    """
    walkable = toy_walkable()
    t = motion.single_step_matrix(walkable, C.PARTICLE_STAY_PROB)
    start = TOY // 4 * TOY + TOY // 4
    assert walkable[start // TOY, start % TOY]

    n = 200_000
    cell = np.full(n, start, dtype=np.int32)
    walk(cell, walkable, rng=np.random.default_rng(11))
    empirical = np.bincount(cell, minlength=TOY * TOY) / n
    # 200k samples put the standard error on any one cell below 0.0011.
    assert np.abs(empirical - t[start]).max() < 0.005


#: MEASURED: median displacement of a champion over each horizon, from synthetic ground
#: truth. The motion constants were fitted to these, so the test below is what keeps the
#: two from drifting apart, change a constant and this fails with the number it broke.
TRUTH_DISPLACEMENT = {2.0: 268.1, 5.0: 565.2, 10.0: 976.5, 20.0: 1394.9}


def _displacement_curve(terrain, goal_beta, n=3000, seed=1):
    """Median displacement of the walk at each measured horizon, in world units."""
    grid = terrain.grid
    cells = np.flatnonzero(terrain.walkable.reshape(-1)).astype(np.int32)
    targets = motion.role_targets("unknown", terrain)
    rng = np.random.default_rng(seed)
    start = rng.choice(cells, size=n).astype(np.int32)
    cell = start.copy()
    heading = np.full(n, motion.STAY, dtype=np.int8)
    goal = targets[rng.integers(0, targets.size, size=n)].astype(np.int32)

    marks = {int(h * C.TICK_HZ): h for h in TRUTH_DISPLACEMENT}
    out = {}
    for t in range(1, max(marks) + 1):
        if goal_beta > 0.0:
            motion.refresh_goals(cell, goal, targets, grid, C.GOAL_ARRIVE_CELLS, rng)
        motion.propose_cells(
            cell,
            heading,
            terrain.walkable,
            C.MOTION_SUB_STEPS,
            C.PARTICLE_STAY_PROB,
            C.HEADING_PERSISTENCE,
            goal,
            goal_beta,
            rng.uniform(size=(n, C.MOTION_SUB_STEPS)),
            cell,
            heading,
        )
        if t in marks:
            dj = cell // grid - start // grid
            di = cell % grid - start % grid
            out[marks[t]] = float(np.median(np.hypot(dj, di)) * C.GRID_CELL_SIZE)
    return out


def test_the_walk_travels_as_far_as_champions_actually_do(terrain):
    """The fit, pinned. This is why the motion constants have the values they have.

    Tolerances are asymmetric because the error is: the model tracks the truth to within
    a few percent out to ten seconds and then overshoots by 28% at twenty, because real
    champions reverse course, recall, then walk back, and a random-waypoint model does
    not. Overshooting means the belief is slightly too spread rather than too confident,
    which is the safe direction, but it is a real limit and is stated as one.
    """
    curve = _displacement_curve(terrain, C.GOAL_BETA)
    for horizon in (2.0, 5.0, 10.0):
        ratio = curve[horizon] / TRUTH_DISPLACEMENT[horizon]
        assert 0.8 < ratio < 1.2, f"{horizon}s: {curve[horizon]:.0f}u vs truth"
    assert 0.8 < curve[20.0] / TRUTH_DISPLACEMENT[20.0] < 1.5


def test_a_diffusive_walk_cannot_reach_the_measured_displacement(terrain):
    """The measurement that rejected pure diffusion as the motion model.

    An unbiased walk is recurrent. It wanders back over its own path, so displacement
    grows like the square root of time, while a champion crossing the map does not. No
    setting of sub-steps, stay probability or heading persistence got a diffusive walk
    past 900 units at twenty seconds against a truth of 1,395. That gap is the entire
    reason `navmesh_behavioural` carries a destination, and the ablation reports what it
    is worth.
    """
    curve = _displacement_curve(terrain, 0.0)
    assert curve[20.0] < 0.75 * TRUTH_DISPLACEMENT[20.0]
    assert curve[20.0] < _displacement_curve(terrain, C.GOAL_BETA)[20.0]


def test_the_walk_never_leaves_the_navmesh():
    """Including with a goal on the far side of a wall, which is the case that bites.

    The goal bias only reweights candidate moves; it can never propose an illegal one.
    If it could, the belief would put mass inside terrain and every downstream area and
    entropy figure would be inflated by cells no champion can stand in.
    """
    walkable = toy_walkable()
    cells = np.flatnonzero(walkable.reshape(-1)).astype(np.int32)
    rng = np.random.default_rng(2)
    cell = rng.choice(cells, size=5000).astype(np.int32)
    goal = rng.choice(cells, size=5000).astype(np.int32)
    flat = walkable.reshape(-1)
    for _ in range(40):
        walk(
            cell,
            walkable,
            n_sub=2,
            persistence=C.HEADING_PERSISTENCE,
            goal=goal,
            goal_beta=C.GOAL_BETA,
            rng=rng,
        )
        assert flat[cell].all()


def test_goals_refresh_only_on_arrival():
    grid = 64
    targets = np.array([10 * grid + 10, 50 * grid + 50], dtype=np.int32)
    cell = np.array([10 * grid + 10, 30 * grid + 30], dtype=np.int32)
    goal = np.array([10 * grid + 11, 50 * grid + 50], dtype=np.int32)
    n = motion.refresh_goals(cell, goal, targets, grid, 2.0, np.random.default_rng(0))
    assert n == 1
    assert goal[1] == 50 * grid + 50  # far away, untouched
    assert goal[0] in targets


# ---------------------------------------------------------------------------
# The test that validates the negative update as Bayesian
# ---------------------------------------------------------------------------
def _exact_forward(walkable, visible_seq, p_stay, pd_map, start_cell, n_sub):
    """The posterior, computed exactly over all 256 states.

    At each tick: propagate through the transition matrix, then multiply by the
    likelihood of *not having been seen*, which is `1 - p_d(cell)` inside the visible
    region and 1 outside it. That likelihood is the negative update, written the way a
    textbook would write it.
    """
    n = walkable.size
    t = np.linalg.matrix_power(motion.single_step_matrix(walkable, p_stay), n_sub)
    b = np.zeros(n)
    b[start_cell] = 1.0
    for visible in visible_seq:
        b = b @ t
        b = b * (1.0 - np.where(visible, pd_map, 0.0).reshape(-1))
        s = b.sum()
        assert s > 0, "the exact filter died; the scenario is degenerate"
        b /= s
    return b


def test_filter_matches_exact_bayes():
    """Total variation < 0.05 between the particle cloud and the exact posterior.

    If the negative update were merely plausible rather than correct. A hard kill, a
    wrong ordering of predict and update, a likelihood applied to the wrong particles,
    this diverges. Nothing else in the suite can catch that, because a wrong-but-smooth
    posterior looks exactly as reasonable as a right one when drawn on a map.
    """
    walkable = toy_walkable()
    flat = walkable.reshape(-1)
    start = int(np.flatnonzero(flat)[0])
    p_stay = C.PARTICLE_STAY_PROB
    n_sub = 2
    n_ticks = 25

    # The observer sweeps one half, then the other. Alternating is what makes the test
    # bite: a filter that merely reweights toward the unobserved side would pass a
    # single-region test by accident.
    packed = [toy_visible(walkable, "left"), toy_visible(walkable, "right")]
    bool_masks = [
        np.array(
            [
                [bool((m[j, i >> 6] >> np.uint64(i & 63)) & np.uint64(1)) for i in range(TOY)]
                for j in range(TOY)
            ]
        )
        for m in packed
    ]
    pd_map = np.full((TOY, TOY), C.PD_INTERIOR)

    # The exact model has a single detection probability, so the filter is run with the
    # edge ring switched off, comparing against a two-valued p_d would be comparing
    # against a different model and the discrepancy would be the test's fault.
    order = [i % 2 for i in range(n_ticks)]
    exact = _exact_forward(walkable, [bool_masks[k] for k in order], p_stay, pd_map, start, n_sub)

    p = 20_000
    rng = np.random.default_rng(17)
    cell = np.full(p, start, dtype=np.int32)
    heading = np.full(p, motion.STAY, dtype=np.int8)
    logw = np.zeros(p)
    pd = np.empty(p)
    idx = np.empty(p, dtype=np.int32)

    for k in order:
        walk(cell, walkable, n_sub=n_sub, p_stay=p_stay, rng=rng)
        observation.detection_field(cell, packed[k], TOY, 0, C.PD_INTERIOR, C.PD_INTERIOR, pd)
        observation.negative_update(logw, pd)
        if observation.effective_sample_size(logw) < 0.5 * p:
            observation.systematic_resample(logw, float(rng.uniform()), idx)
            cell = cell[idx]
            heading = heading[idx]
            logw.fill(0.0)

    w = np.exp(logw - logw.max())
    w /= w.sum()
    empirical = np.bincount(cell, weights=w, minlength=walkable.size)
    tv = 0.5 * np.abs(empirical - exact).sum()
    assert tv < 0.05, f"total variation {tv:.4f} against the exact posterior"


def test_exact_bayes_error_falls_at_the_monte_carlo_rate():
    """Convergence, which is a far stronger claim than any single threshold.

    A filter that is *biased*, targeting a subtly different posterior, plateaus at a
    non-zero total variation no matter how many particles it is given, and would sail
    past a fixed 0.05 gate at P = 20,000 while being wrong. A filter that is merely
    *noisy* has error falling as one over the square root of the particle count.

    MEASURED: 0.115 at P=2,000, 0.030 at P=20,000, 0.012 at P=100,000, ratios of 3.9
    and 2.4 against the sqrt(10)=3.2 and sqrt(5)=2.2 that pure sampling error predicts.
    """
    walkable = toy_walkable()
    start = int(np.flatnonzero(walkable.reshape(-1))[0])
    order = [i % 2 for i in range(25)]
    packed = [toy_visible(walkable, "left"), toy_visible(walkable, "right")]
    bool_masks = [_unpack(m) for m in packed]
    exact = _exact_forward(
        walkable,
        [bool_masks[k] for k in order],
        C.PARTICLE_STAY_PROB,
        np.full((TOY, TOY), C.PD_INTERIOR),
        start,
        2,
    )
    tv = {p: _tv_against(exact, walkable, packed, order, p) for p in (2_000, 20_000)}
    assert tv[20_000] < tv[2_000] / 2.0, tv


def _unpack(mask):
    return np.array(
        [
            [bool((mask[j, i >> 6] >> np.uint64(i & 63)) & np.uint64(1)) for i in range(TOY)]
            for j in range(TOY)
        ]
    )


def _tv_against(exact, walkable, packed, order, p, seed=17):
    rng = np.random.default_rng(seed)
    start = int(np.flatnonzero(walkable.reshape(-1))[0])
    cell = np.full(p, start, dtype=np.int32)
    logw = np.zeros(p)
    pd = np.empty(p)
    idx = np.empty(p, dtype=np.int32)
    for k in order:
        walk(cell, walkable, n_sub=2, p_stay=C.PARTICLE_STAY_PROB, rng=rng)
        observation.detection_field(cell, packed[k], TOY, 0, C.PD_INTERIOR, C.PD_INTERIOR, pd)
        observation.negative_update(logw, pd)
        if observation.effective_sample_size(logw) < 0.5 * p:
            observation.systematic_resample(logw, float(rng.uniform()), idx)
            cell = cell[idx]
            logw.fill(0.0)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    empirical = np.bincount(cell, weights=w, minlength=walkable.size)
    return 0.5 * float(np.abs(empirical - exact).sum())


def test_hard_kill_is_measurably_worse_than_the_soft_update():
    """`p_d = 1` is the obvious implementation, and it destroys the filter.

    Kept as an executable record of why the soft update exists: with a mask that is a
    little wrong, a hard kill deletes the correct hypothesis permanently and the filter
    ends up confidently elsewhere.
    """
    walkable = toy_walkable()
    flat = walkable.reshape(-1)
    start = int(np.flatnonzero(flat)[0])
    packed = toy_visible(walkable, "left")
    p = 4000
    survivors = {}
    for pd_value in (C.PD_INTERIOR, 1.0):
        rng = np.random.default_rng(23)
        cell = np.full(p, start, dtype=np.int32)
        logw = np.zeros(p)
        pd = np.empty(p)
        for _ in range(30):
            walk(cell, walkable, n_sub=2, rng=rng)
            observation.detection_field(cell, packed, TOY, 0, pd_value, pd_value, pd)
            observation.negative_update(logw, pd)
        survivors[pd_value] = float(np.isfinite(logw).mean())
    assert survivors[1.0] < survivors[C.PD_INTERIOR]
    assert survivors[C.PD_INTERIOR] == 1.0


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def test_effective_sample_size_bounds():
    n = 500
    assert observation.effective_sample_size(np.zeros(n)) == pytest.approx(n)
    spike = np.full(n, -np.inf)
    spike[0] = 0.0
    assert observation.effective_sample_size(spike) == pytest.approx(1.0)


def test_ess_survives_weights_that_would_underflow_in_linear_space():
    """A particle held in vision for 25 seconds accumulates log(0.02) per tick.

    In float32 linear weights that underflows to exactly zero, and the naive ESS returns
    nan at precisely the moment the filter most needs to notice it is in trouble.
    """
    logw = np.linspace(-8000.0, -7000.0, 400)
    ess = observation.effective_sample_size(logw)
    assert np.isfinite(ess)
    assert 1.0 <= ess <= 400.0


def test_systematic_resample_reproduces_the_weight_distribution():
    rng = np.random.default_rng(4)
    n = 20_000
    logw = np.log(rng.uniform(0.01, 1.0, n))
    idx = np.empty(n, dtype=np.int32)
    observation.systematic_resample(logw, 0.5, idx)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    counts = np.bincount(idx, minlength=n) / n
    assert np.abs(counts - w).max() < 3.0 / n
    assert idx.min() >= 0
    assert idx.max() < n


def test_systematic_resample_is_deterministic_given_one_number():
    """Which is what makes the barrier test able to assert bit-identity."""
    logw = np.log(np.random.default_rng(1).uniform(0.01, 1.0, 1000))
    a = np.empty(1000, dtype=np.int32)
    b = np.empty(1000, dtype=np.int32)
    observation.systematic_resample(logw, 0.37, a)
    observation.systematic_resample(logw, 0.37, b)
    assert np.array_equal(a, b)


def test_detection_field_separates_interior_from_edge():
    vis = np.zeros((TOY, TOY), dtype=bool)
    vis[4:12, 4:12] = True
    packed = pack_rows(vis)
    cells = np.array(
        [
            8 * TOY + 8,  # deep interior
            4 * TOY + 4,  # corner of the visible block
            5 * TOY + 5,  # one cell in, still inside the 2-cell ring
            0 * TOY + 0,  # not visible at all
        ],
        dtype=np.int32,
    )
    pd = np.empty(4)
    observation.detection_field(cells, packed, TOY, 2, 0.98, 0.75, pd)
    assert pd[0] == pytest.approx(0.98)
    assert pd[1] == pytest.approx(0.75)
    assert pd[2] == pytest.approx(0.75)
    assert pd[3] == 0.0


def test_detection_probability_is_zero_outside_the_visible_region():
    """Not looking somewhere is not evidence about it. The sign error that would
    invert the whole model."""
    packed = pack_rows(np.zeros((TOY, TOY), dtype=bool))
    cells = np.arange(TOY * TOY, dtype=np.int32)
    pd = np.empty(cells.size)
    observation.detection_field(cells, packed, TOY, 2, 0.98, 0.75, pd)
    assert (pd == 0.0).all()
    logw = np.zeros(cells.size)
    observation.negative_update(logw, pd)
    assert (logw == 0.0).all()


def test_filter_spec_refuses_an_entropy_lattice_it_cannot_support():
    with pytest.raises(ValueError, match="particle"):
        FilterSpec(entropy_lattice=64, particles=64)


def test_filter_spec_refuses_a_frozen_belief():
    with pytest.raises(ValueError, match="p_stay"):
        FilterSpec(p_stay=1.0)


def test_every_resolved_role_has_its_own_prior(terrain):
    """The vocabularies on both sides of the behavioural prior must agree.

    They did not. The resolver emits `("top", "jungle", "mid", "bot", "support")` and the
    motion model matched `"jng"` and `"sup"`, so junglers and supports fell through to the
    catch-all target set, two of every five enemies, including the champion who spends
    the most time in fog. Nothing raised, nothing failed a test, and the only symptom was
    a belief that was confidently wrong more often than it should have been.

    This asserts each role gets a *distinct* target set, which a silent fallthrough
    cannot satisfy.
    """
    from shadowcast.l1_events.resolve.roles import ROLES

    catch_all = motion.role_targets("this-is-not-a-role", terrain)
    for role in ROLES:
        targets = motion.role_targets(role, terrain)
        assert targets.size > 0, role
        assert not np.array_equal(targets, catch_all), (
            f"role {role!r} fell through to the catch-all target set"
        )
