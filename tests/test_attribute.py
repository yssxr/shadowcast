"""Tests for movement-order attribution and trajectory reconstruction.

The headline numbers are asserted here rather than described in prose, and they are
chosen to measure what actually matters. Raw attribution accuracy is the wrong headline:
a misattributed order's true and assigned owners sit a median of zero units apart,
because the assignment is only ambiguous when the champions are in the same place. So
the gates are on *harmful* misattribution, on the margin-conditioned accuracy, and on
trajectory error — all of which are consequences a later layer would actually feel.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.l1_events.normalise import normalise
from shadowcast.l1_events.resolve.attribute import (
    AttributionSpec,
    attribute,
    with_owners,
)
from shadowcast.l1_events.schema import UNKNOWN, FrameCalibration


def _exact_frame(truth) -> FrameCalibration:
    """Normalise with the true frame offset, isolating attribution from calibration.

    Frame error propagates directly into every position, so mixing the two would make an
    attribution gate secretly a calibration gate.
    """
    return FrameCalibration(
        offset=truth.spec.waypoint_offset,
        walkable_fraction=1.0,
        plateau_width=C.GRID_CELL_SIZE,
        baseline_fraction=0.9,
        n_samples=1,
    )


@pytest.fixture(scope="module")
def run_clean(synth_clean, terrain):
    bundle, truth = synth_clean
    events = normalise(bundle, terrain, frame=_exact_frame(truth))
    return events, truth, attribute(events), bundle


@pytest.fixture(scope="module")
def run_dirty(synth_dirty, terrain):
    bundle, truth = synth_dirty
    events = normalise(bundle, terrain, frame=_exact_frame(truth))
    return events, truth, attribute(events), bundle


def _true_owners(events, bundle, truth) -> np.ndarray:
    """Ground-truth owner per row of `events.orders`.

    Matched through `seq` rather than by position, because normalisation re-sorts the
    orders and the arrival-jitter pathology permutes them.
    """
    by_seq = {int(s): int(o) for s, o in zip(bundle.waypoints["seq"], truth.order_owner)}
    return np.array([by_seq[int(s)] for s in events.orders["seq"]], dtype=np.int64)


def _separation(truth, events, wrong_idx, assigned) -> np.ndarray:
    """How far apart the true and assigned owners were when each order was issued."""
    dt = truth.spec.dt
    out = []
    for oi in wrong_idx:
        tick = min(int(events.orders["t"][oi] / dt), truth.pos.shape[0] - 1)
        true_slot = int(_TRUE[oi])
        out.append(float(np.hypot(*(truth.pos[tick, true_slot] - truth.pos[tick, assigned[oi]]))))
    return np.array(out)


_TRUE: np.ndarray = np.empty(0)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["run_clean", "run_dirty"])
def test_almost_every_order_is_attributed(which, request):
    _, _, at, _ = request.getfixturevalue(which)
    assert at.attributed_fraction > 0.99


@pytest.mark.parametrize("which", ["run_clean", "run_dirty"])
def test_orders_are_spread_across_all_champions(which, request):
    _, _, at, _ = request.getfixturevalue(which)
    counts = np.array(at.stats["orders_per_slot"])
    assert counts.size == C.N_HEROES
    assert counts.min() > 200
    assert counts.max() / counts.min() < 2.0


# ---------------------------------------------------------------------------
# Accuracy — the metrics that matter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["run_clean", "run_dirty"])
def test_harmful_misattribution_is_rare(which, request):
    """The gate that actually matters.

    A misattribution only has consequences if the two champions were somewhere
    different. Measured median separation between true and assigned owner is zero, so
    raw accuracy overstates the damage by more than an order of magnitude.
    """
    global _TRUE
    events, truth, at, bundle = request.getfixturevalue(which)
    _TRUE = _true_owners(events, bundle, truth)

    assigned = at.owner.astype(np.int64)
    known = at.owner != UNKNOWN
    wrong = np.flatnonzero(known & (assigned != _TRUE))
    sep = _separation(truth, events, wrong, assigned)

    harmful = int((sep >= 300.0).sum())
    assert harmful / max(1, int(known.sum())) < 0.01, (
        f"{harmful} harmful misattributions of {known.sum()} assigned"
    )
    assert float(np.median(sep)) < 100.0, "misattributions should be between co-located champions"


@pytest.mark.parametrize("which", ["run_clean", "run_dirty"])
def test_raw_accuracy_is_reasonable(which, request):
    """A floor on the raw figure, so a genuine regression is still caught."""
    global _TRUE
    events, truth, at, bundle = request.getfixturevalue(which)
    _TRUE = _true_owners(events, bundle, truth)
    known = at.owner != UNKNOWN
    accuracy = float((at.owner.astype(np.int64)[known] == _TRUE[known]).mean())
    assert accuracy > 0.95, f"attribution accuracy {accuracy:.3%}"


@pytest.mark.parametrize("which", ["run_clean", "run_dirty"])
def test_the_margin_predicts_correctness_without_ground_truth(which, request):
    """The confidence signal, and the reason it exists.

    `order_margin` is computed from costs alone, yet conditioning on it lifts accuracy
    from ~97% to ~99.7%. That is what lets a later layer discount uncertain positions
    instead of treating every attribution as equally reliable.
    """
    global _TRUE
    events, truth, at, bundle = request.getfixturevalue(which)
    _TRUE = _true_owners(events, bundle, truth)
    assigned = at.owner.astype(np.int64)

    known = at.owner != UNKNOWN
    raw = float((assigned[known] == _TRUE[known]).mean())

    confident = at.confident(100.0)
    assert confident.mean() > 0.85, "margin filter should retain most orders"
    conf_acc = float((assigned[confident] == _TRUE[confident]).mean())
    assert conf_acc > 0.99
    assert conf_acc > raw, "the margin must actually be informative"

    very = at.confident(300.0)
    assert float((assigned[very] == _TRUE[very]).mean()) >= conf_acc


def test_low_margin_orders_are_the_co_located_ones(run_clean):
    """Confirms the mechanism, not just the correlation."""
    global _TRUE
    events, truth, at, bundle = run_clean
    _TRUE = _true_owners(events, bundle, truth)
    dt = truth.spec.dt

    def nearest_teammate(order_index: int) -> float:
        tick = min(int(events.orders["t"][order_index] / dt), truth.pos.shape[0] - 1)
        slot = int(_TRUE[order_index])
        d = np.hypot(*(truth.pos[tick] - truth.pos[tick, slot]).T)
        d[slot] = np.inf
        d[truth.team != truth.team[slot]] = np.inf
        return float(d.min())

    rng = np.random.default_rng(0)
    known = np.flatnonzero(at.owner != UNKNOWN)
    low = [i for i in known if at.order_margin[i] < 50.0]
    high = [i for i in known if at.order_margin[i] > 500.0]
    low = rng.choice(low, size=min(120, len(low)), replace=False)
    high = rng.choice(high, size=min(120, len(high)), replace=False)

    low_sep = np.median([nearest_teammate(int(i)) for i in low])
    high_sep = np.median([nearest_teammate(int(i)) for i in high])
    assert low_sep < high_sep, f"low-margin {low_sep:.0f}u vs high-margin {high_sep:.0f}u"


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
def test_clean_trajectory_is_exact_where_the_frame_is_exact(run_clean):
    """With no pathologies and the true frame offset, integration should be perfect.

    This is the test that pins the algebra: median error at float32 noise means the
    order model, the tick rate, the speed handling and the coordinate frame all agree
    with the generator exactly.
    """
    _, truth, at, _ = run_clean
    n = min(at.pos.shape[0], truth.pos.shape[0])
    err = np.linalg.norm(at.pos[:n] - truth.pos[:n], axis=2)
    ok = at.valid[:n] & np.isfinite(err)
    assert np.median(err[ok]) < 0.05, f"median trajectory error {np.median(err[ok]):.4f}u"
    assert np.nanmedian(at.anchor_residual) < 0.05


@pytest.mark.parametrize("which", ["run_clean", "run_dirty"])
def test_trajectory_error_stays_bounded(which, request):
    _, truth, at, _ = request.getfixturevalue(which)
    n = min(at.pos.shape[0], truth.pos.shape[0])
    err = np.linalg.norm(at.pos[:n] - truth.pos[:n], axis=2)
    ok = at.valid[:n] & np.isfinite(err)
    assert ok.mean() > 0.97, "too many ticks without a position estimate"
    assert np.percentile(err[ok], 99) < 800.0
    assert err[ok].max() < 3000.0


def test_frame_error_propagates_into_trajectory_error(synth_clean, terrain):
    """Trajectory accuracy cannot beat frame calibration, and the relationship is exact.

    Offsetting the frame by a known amount in both axes must shift every position by
    sqrt(2) times that amount. Worth pinning because it explains a median residual that
    would otherwise look like an integration bug.
    """
    bundle, truth = synth_clean
    shift = 5.0
    skewed = dataclasses.replace(_exact_frame(truth), offset=truth.spec.waypoint_offset - shift)
    at = attribute(normalise(bundle, terrain, frame=skewed))
    # The ANCHOR residual is the right quantity here, not the order residual. Order
    # residual is the skeleton-to-order-start distance, so it is dominated by the
    # skeleton's own interpolation error (~20 units) and says nothing about the frame.
    # The anchor residual compares an order-integrated position against a world-framed
    # observation, so it isolates the shift.
    assert np.nanmedian(at.anchor_residual) == pytest.approx(shift * np.sqrt(2), abs=0.2)


def test_speed_changes_are_followed(run_clean):
    """Boots at eight minutes must show up, or the reconstructor is assuming a speed."""
    _, truth, at, _ = run_clean
    dt = truth.spec.dt
    early = at.speed[int(100 / dt)]
    late = at.speed[int(700 / dt)]
    assert late.max() > early.max()
    assert late.max() == pytest.approx(380.0)


def test_death_marks_the_estimate_unknown_rather_than_guessing(run_clean):
    """No respawn timer exists in the stream, so the honest response is to stop claiming.

    Anchors are dense enough that the next one re-establishes the position quickly.
    """
    _, truth, at, _ = run_clean
    dt = truth.spec.dt
    assert at.stats["deaths_seen"] == truth.kills.size
    for row in truth.kills:
        victim = int(row["victim"])
        tick = int(row["t"] / dt) + 2
        assert not at.valid[tick, victim], "a dead champion should have no position claim"


# ---------------------------------------------------------------------------
# Ablations, so the rejected ideas stay rejected for a reason
# ---------------------------------------------------------------------------
def test_direction_term_is_off_by_default_and_hurts_when_on(run_dirty):
    """A term that seemed obviously right and measurably was not.

    While five champions share a fountain, where they are heading is the only thing that
    separates them — so weighting order direction against champion heading looks like
    the answer. It is not: the heading is interpolated between anchors and therefore
    noisy, and a champion's instantaneous heading routinely disagrees with its next
    order's opening direction because it stops and turns.
    """
    events, truth, at, bundle = run_dirty
    assert AttributionSpec().direction_weight == 0.0

    with_dir = attribute(events, AttributionSpec(direction_weight=600.0))
    true_owner = _true_owners(events, bundle, truth)

    def accuracy(a):
        known = a.owner != UNKNOWN
        return float((a.owner.astype(np.int64)[known] == true_owner[known]).mean())

    assert accuracy(with_dir) < accuracy(at), "direction term should measurably hurt"


def test_iterating_does_not_help(run_clean):
    """The other rejected idea: reassigning against the integrated trajectory.

    It reintroduces exactly the feedback the anchor skeleton exists to remove — a wrong
    assignment corrupts the estimate that later assignments are judged against — and the
    tail gets worse while accuracy stays flat.
    """
    events, truth, at, _ = run_clean
    assert AttributionSpec().iterations == 1
    iterated = attribute(events, AttributionSpec(iterations=3))
    n = min(at.pos.shape[0], truth.pos.shape[0])

    def p99(a):
        err = np.linalg.norm(a.pos[:n] - truth.pos[:n], axis=2)
        ok = a.valid[:n] & np.isfinite(err)
        return float(np.percentile(err[ok], 99))

    assert p99(iterated) > p99(at), "iteration should measurably widen the error tail"


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def test_with_owners_writes_ownership_back(run_clean):
    events, _, at, _ = run_clean
    assert not events.orders_attributed
    filled = with_owners(events, at)
    np.testing.assert_array_equal(filled.orders["owner"], at.owner)
    assert filled.orders_attributed == (at.stats["unattributed"] == 0)
    for slot in range(C.N_HEROES):
        assert filled.orders_of(slot).size == at.stats["orders_per_slot"][slot]
    # The original must be untouched: MatchEvents is treated as immutable.
    assert (events.orders["owner"] == UNKNOWN).all()


def test_attribution_is_deterministic(run_clean):
    events, _, at, _ = run_clean
    again = attribute(events)
    np.testing.assert_array_equal(again.owner, at.owner)
    np.testing.assert_allclose(np.nan_to_num(again.pos), np.nan_to_num(at.pos))


def test_handles_a_match_with_no_orders(synth_clean, terrain):
    bundle, truth = synth_clean
    events = normalise(bundle, terrain, frame=_exact_frame(truth))
    empty = dataclasses.replace(events, orders=events.orders[:0])
    at = attribute(empty)
    assert at.owner.size == 0
    assert at.attributed_fraction == 0.0
    # Anchors alone still pin positions, so the trajectory is not empty.
    assert at.valid.any()
