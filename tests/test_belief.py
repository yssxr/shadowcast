"""The belief state end to end: the barrier, the ablation, and the sanity checks.

Two tests here carry the milestone.

`test_perturbing_unobserved_truth_changes_nothing` is the information-barrier leak
detector. It moves every unobserved enemy two thousand units and asserts the filter's
output is bit-identical. Without it, the project would eventually publish a number that
is far too good and have no way of knowing why. A leak does not crash, does not look
wrong, and improves every metric at once.

`test_negative_information_beats_the_same_model_without_it` is the thesis. If it fails,
the central claim is empty, and the test is written so that it can say so.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from shadowcast.config import BASELINES, THESIS_PAIR, FilterSpec
from shadowcast.l1_events.normalise import normalise
from shadowcast.l1_events.resolve import attribute, resolve_all
from shadowcast.l2_reconstruct.vision import VisionStream
from shadowcast.l3_infer.baselines import ablate, run_model
from shadowcast.l3_infer.metrics import CALIBRATION_LEVELS, LatticeIndex
from shadowcast.l3_infer.policy import NO_CELL, observe
from shadowcast.l3_infer.reachability import ReachabilityIndex

#: Short by the standards of a match, long enough that champions leave vision and come
#: back many times: which is the only regime in which any of this is being tested.
_DURATION = 200.0

#: The ablation runs seven models over the whole match, so the suite scores every fourth
#: tick. Scoring is a per-tick histogram over 358 bins and contributes nothing to what is
#: being tested; the belief itself still steps at the full rate.
_SCORE_STRIDE = 4


@dataclasses.dataclass(frozen=True)
class Match:
    """One reconstructed match and everything downstream needs from it."""

    events: object
    attribution: object
    obs: object
    public: object
    truth: object
    masks: object  # a factory: a mask stream is consumed once, so each run needs its own
    table: object


@pytest.fixture(scope="module")
def match(terrain, fov_table):
    """A reconstructed match, with observations gated through its own vision."""
    from shadowcast.packets.synth import Pathologies, ScenarioSpec, SyntheticSource

    src = SyntheticSource(
        terrain, ScenarioSpec(seed=7, duration=_DURATION, pathologies=Pathologies.all())
    )
    bundle, _ = src.generate(src.match_ids()[0])
    events = normalise(bundle, terrain)
    at = attribute(events)
    events, _ = resolve_all(events, at)

    obs, public, truth = observe(events, at, VisionStream(events, at, terrain, fov_table))
    return Match(
        events=events,
        attribution=at,
        obs=obs,
        public=public,
        truth=truth,
        masks=lambda: VisionStream(events, at, terrain, fov_table).masks(),
        table=fov_table,
    )


@pytest.fixture(scope="module")
def lattice(terrain):
    return LatticeIndex(terrain)


@pytest.fixture(scope="module")
def reach(terrain):
    return ReachabilityIndex(terrain)


@pytest.fixture(scope="module")
def ablation(terrain, match, lattice):
    return ablate(
        terrain,
        match.obs,
        match.public,
        match.truth,
        match.masks,
        lattice=lattice,
        stride=_SCORE_STRIDE,
    )


# ---------------------------------------------------------------------------
# The information barrier
# ---------------------------------------------------------------------------
def test_observations_are_empty_wherever_nothing_was_seen(match):
    """`cell` must be the sentinel wherever `seen` is false, not a stale position.

    This is the leak that would be easiest to introduce and hardest to notice: fill the
    cell in unconditionally, and any consumer that reads it without checking the flag is
    reading the enemy's true position out of fog.
    """
    assert (match.obs.cell[~match.obs.seen] == NO_CELL).all()
    assert (match.obs.cell[match.obs.seen] >= 0).all()


def _shift_the_hidden(match, terrain, delta=2000.0):
    """Move every champion who is currently in fog, without changing what anyone sees.

    Two conditions, and both are needed for the perturbation to be information-neutral:
    the champion must be unseen where they are, and unseen where they are moved to. A
    shift that pulls someone into view changes what the observer legitimately knows, and
    a test that allowed it would be measuring a real effect and calling it a leak.
    """
    from shadowcast.fov.union import mask_bit
    from shadowcast.geom.grid import world_to_cell

    at = match.attribution
    shifted = at.pos.copy()
    team = match.events.heroes["team"].astype(int)
    grid = terrain.grid
    moved = 0

    for tick, mask_order, mask_chaos in match.masks():
        if tick >= at.pos.shape[0]:
            break
        masks = (mask_order, mask_chaos)
        for slot in range(team.size):
            if not at.valid[tick, slot]:
                continue
            observer = 1 - int(team[slot])
            x, z = at.pos[tick, slot]
            i, j = world_to_cell(float(x), float(z))
            if not (0 <= i < grid and 0 <= j < grid) or mask_bit(masks[observer], i, j):
                continue
            i2, j2 = world_to_cell(float(x) + delta, float(z) + delta)
            if not (0 <= i2 < grid and 0 <= j2 < grid) or mask_bit(masks[observer], i2, j2):
                continue
            shifted[tick, slot] = (x + delta, z + delta)
            moved += 1
    return shifted, moved


def test_perturbing_unobserved_truth_changes_nothing(terrain, match, reach):
    """The leak detector. Move every hidden enemy 2,000 units; nothing may change.

    **The vision masks are held fixed on purpose.** A mask is the observer's own
    information, and moving a hidden enemy does not change what the observer is looking
    at, but it does change where *that champion's team* has vision, so regenerating the
    masks from perturbed positions would alter the observations legitimately and the test
    would be measuring the game rather than the barrier.

    The check has two halves. That `observe` yields an identical `Observation` is a claim
    about the gating. That the filter's entire output stream is bit-identical is a claim
    about the filter, and it is the half that would catch someone reaching for a
    `TruthTable` later.
    """
    from shadowcast.l3_infer.pf import BeliefFilter

    shifted, moved = _shift_the_hidden(match, terrain)
    assert moved > 1000, f"only {moved} positions were perturbed; the test proves little"
    perturbed = dataclasses.replace(match.attribution, pos=shifted)

    # Gated through the ORIGINAL vision, which is the whole point.
    obs2, public2, truth2 = observe(
        match.events,
        perturbed,
        VisionStream(match.events, match.attribution, terrain, match.table),
    )
    assert np.array_equal(obs2.seen, match.obs.seen)
    assert np.array_equal(obs2.cell, match.obs.cell)
    assert np.array_equal(public2.alive, match.public.alive)
    # And the truth really did move, so the inputs were not accidentally identical.
    assert not np.array_equal(truth2.cell, match.truth.cell)

    spec = BASELINES["full"]
    runs = [
        [
            (b.cell.copy(), b.logw.copy())
            for b in BeliefFilter(spec, terrain, reach=reach).run(o, p, match.masks())
        ]
        for o, p in ((match.obs, match.public), (obs2, public2))
    ]
    assert len(runs[0]) == len(runs[1])
    for (c0, w0), (c1, w1) in zip(runs[0], runs[1]):
        assert np.array_equal(c0, c1)
        assert np.array_equal(w0, w1)


# ---------------------------------------------------------------------------
# The thesis
# ---------------------------------------------------------------------------
def test_negative_information_beats_the_same_model_without_it(ablation):
    """`behavioural` versus `full`: identical in every field but the observation model.

    If this fails, negative information is contributing nothing and the project's
    central claim is empty. That would be a result worth publishing, not a test to
    relax, so the assertion message reports the gap rather than hiding it.
    """
    a, b = THESIS_PAIR
    before, after = ablation.scores[a], ablation.scores[b]
    assert dataclasses.replace(before.spec, obs=after.spec.obs) == after.spec, (
        "the thesis pair must differ in exactly one field, or the comparison proves nothing"
    )
    assert ablation.thesis_holds, (
        f"negative information did not help: {a} NLL {before.nll:.4f} vs {b} NLL {after.nll:.4f}"
    )


def test_negative_information_sharpens_the_belief(ablation):
    """It should also shrink the credible region, not only improve the likelihood.

    A model can win on NLL by being better calibrated while staying just as vague. The
    point of the negative update is that it eliminates territory.
    """
    a, b = THESIS_PAIR
    assert ablation.scores[b].credible_area_ku2 < ablation.scores[a].credible_area_ku2


def test_the_shipped_models_beat_the_uniform_prior(ablation):
    """A model that cannot beat "somewhere on the map" is not a model.

    Asserted for the ones that carry the argument rather than for all six. MEASURED:
    plain `navmesh_diffusion` does NOT reliably beat a uniform prior on a short window,
    it is the sharpest model in the table (1.6% of the map) and the least likely to have
    the truth inside, which is what a likelihood score is supposed to punish. That is a
    result about diffusion without a prior, not a bug, and it is reported in
    `docs/validation.md` rather than asserted away.
    """
    baseline = ablation.scores["uniform"].nll
    for name in ("geodisc", "behavioural", "full"):
        assert ablation.scores[name].nll < baseline, f"{name} is no better than uniform"


def test_the_navmesh_is_worth_something(ablation):
    """`disc` versus `geodisc`: the same growing ball, Euclidean against geodesic.

    This is the cheapest possible check that terrain matters at all, and it is the one
    number that justifies parsing a navgrid instead of drawing circles.
    """
    assert ablation.scores["geodisc"].nll < ablation.scores["disc"].nll


def test_the_ablation_pairs_differ_in_exactly_one_field():
    """Guards the table's whole logic. Two changes at once and it isolates nothing."""
    pairs = [("diffusion", "behavioural"), THESIS_PAIR]
    for a, b in pairs:
        sa, sb = BASELINES[a], BASELINES[b]
        differing = [
            f.name
            for f in dataclasses.fields(FilterSpec)
            if getattr(sa, f.name) != getattr(sb, f.name)
        ]
        assert len(differing) == 1, f"{a} vs {b} differ in {differing}"


def test_the_ranking_is_not_an_artefact_of_smoothing(terrain, match, reach):
    """Re-rank the models with the scoring smoothing switched off.

    The smoothed histogram is part of the metric's definition and exists for a real
    statistical reason, but a result that only appears under smoothing would be a
    property of the estimator rather than of the models. The ordering must survive it.
    """
    raw = LatticeIndex(terrain, smoothing=0.0)
    scores = {
        name: run_model(
            name,
            BASELINES[name],
            terrain,
            match.obs,
            match.public,
            match.truth,
            match.masks,
            raw,
            reach=reach,
            stride=16,
        ).nll
        for name in ("geodisc", "diffusion", "behavioural", "full")
    }
    assert scores["full"] < scores["behavioural"] < scores["diffusion"]
    assert scores["full"] < scores["geodisc"]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_coverage_rises_with_the_credible_level(ablation):
    """The one property calibration cannot be excused from.

    A 90% region contains everything a 75% region does, so its coverage cannot be lower.
    A violation would mean the credible regions are not nested, i.e. the highest-density
    region is being built wrong, which no amount of modelling uncertainty explains.
    """
    for name, score in ablation.scores.items():
        values = [score.coverage[q] for q in sorted(CALIBRATION_LEVELS)]
        assert values == sorted(values), f"{name}: {score.coverage}"


def test_the_full_model_is_overconfident_and_this_is_tracked(ablation):
    """**A known, open defect, pinned so it cannot quietly get worse.**

    Coverage was 55.6% / 87.1% / 91.5% at the 75/90/95% levels while the synthetic
    scenario had enemies visible 84% of the time. Fixing the fog-attack reveal. It was
    firing on attacks that had no target, dropped visibility to a realistic 42%, and the
    same measurement is now 29.8% / 39.1% / 47.1%.

    Nothing about the filter changed. Longer darkness episodes simply exposed that the
    propagated models concentrate faster than the truth disperses, which short episodes
    hid. A geodesic disc, which is enormously vague and well calibrated, now beats the
    full model on likelihood over a whole match.

    This asserts the two things that must still hold. The belief is informative, and it
    is better than having no belief, plus a floor on coverage so a regression is caught.
    It deliberately does NOT assert the model is calibrated, because it is not.
    """
    score = ablation.scores["full"]
    assert score.credible_area_map_fraction < 0.1, "the belief should still be informative"
    assert score.coverage[0.9] > 0.25, (
        f"coverage at 90% fell to {score.coverage[0.9]:.1%}; the known gap is ~39%"
    )
    assert score.nll < ablation.scores["uniform"].nll


def test_negative_information_does_not_cost_calibration(ablation):
    """It should not buy its likelihood by becoming more overconfident.

    A weaker claim than the one this test used to make. Negative information DID improve
    calibration measurably, 0.208 against 0.181: while the scenario had enemies visible
    84% of the time. With realistic darkness the two are a tie (0.371 against 0.370), so
    the honest assertion is that the negative update improves the likelihood without
    making the belief more overconfident, which is the failure mode that would matter.
    """
    a, b = THESIS_PAIR
    before, after = ablation.scores[a], ablation.scores[b]
    assert after.nll < before.nll
    assert after.calibration_error <= before.calibration_error + 0.02


def test_vagueness_calibrates_easily(ablation):
    """A result worth stating rather than hiding.

    `geodisc` has a far better calibration error than the full model over a credible
    region an order of magnitude larger. It buys that purely by being uninformative, and
    on a full-length match it now also wins on likelihood, which is the open defect
    recorded above. Calibration alone would rank it first; area alone would rank it last.
    Neither is reported without the other.
    """
    naive, full = ablation.scores["geodisc"], ablation.scores["full"]
    assert naive.calibration_error < full.calibration_error
    assert naive.credible_area_ku2 > full.credible_area_ku2


def test_calibration_is_reported_for_every_model(ablation):
    for score in ablation.scores.values():
        assert set(score.coverage) == set(CALIBRATION_LEVELS)
        assert np.isfinite(score.calibration_error)


# ---------------------------------------------------------------------------
# Sanity: things that must hold if the model means anything
# ---------------------------------------------------------------------------
def test_only_unseen_living_enemies_are_scored(ablation):
    """A seen enemy's belief is a point mass by construction.

    Including those ticks would average in a perfect score for every moment the question
    was not being asked, and since visibility runs 25-40% that alone would move every
    model a third of the way toward looking good.
    """
    for score in ablation.scores.values():
        assert score.scored_ticks > 0
        assert score.darkness_strict <= score.darkness_naive


def test_darkness_excludes_dead_time(match):
    """Dead time is not darkness. A dead enemy's position is known.

    Getting this backwards makes a team look informationally dominant exactly when it is
    winning fights, which inverts the whole analysis.
    """
    obs, public = match.obs, match.public
    dead = ~public.alive
    if not dead.any():
        pytest.skip("no deaths in this scenario")
    naive = float((~obs.seen).mean())
    strict = float((~obs.seen & public.alive).sum() / public.alive.sum())
    assert strict < naive


def test_entropy_never_exceeds_the_lattice_ceiling(ablation, lattice):
    """Entropy is measured against a frozen lattice, so it has a hard maximum.

    Exceeding it would mean the estimator, not the game, is producing the number, which
    is the failure the 32² lattice choice exists to prevent.
    """
    for score in ablation.scores.values():
        assert 0.0 <= score.entropy_bits <= lattice.max_bits + 1e-9


def test_credible_area_is_a_fraction_of_a_real_map(ablation):
    for score in ablation.scores.values():
        assert 0.0 < score.credible_area_map_fraction <= 1.0


def test_the_filter_is_deterministic_under_a_fixed_seed(terrain, match, lattice, reach):
    spec = dataclasses.replace(BASELINES["full"], seed=11)
    args = (terrain, match.obs, match.public, match.truth, match.masks, lattice)
    a = run_model("a", spec, *args, reach=reach, stride=8)
    b = run_model("b", spec, *args, reach=reach, stride=8)
    assert a.nll == b.nll
    assert a.entropy_bits == b.entropy_bits
    assert a.credible_area_ku2 == b.credible_area_ku2


def test_depletion_is_rare(ablation):
    """Reinitialisation throws away accumulated negative information.

    Frequent depletion is a QA signal rather than a counter: it means the vision masks,
    the trajectories or the detection probability are wrong, and the filter is papering
    over it. Rare depletion means the soft update is doing its job.
    """
    score = ablation.scores["full"]
    ticks = score.scored_ticks * _SCORE_STRIDE
    assert score.depletion_events < 0.02 * max(ticks, 1), score.depletion_events
