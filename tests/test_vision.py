"""Tests for vision assembly and the fog-agreement measurement.

This is the project's central validation, so the tests here are as much about *how the
number is read* as about its value. Two things in particular:

The headline is reported as a decomposition, not a single figure. Substituting true
positions isolates the irreducible floor — cell snapping, shadowcasting's permissiveness,
ward and minion modelling — from what the reconstruction itself costs. A single percentage
conflates the two and cannot tell a modelling limit from a bug.

And the plan's original ≥99.9% gate was mis-specified. That figure came from comparing
field-of-view geometry against a ray-march reference at *identical* positions, which does
reach ~100% (see `test_fov.py`). This comparison is a whole reconstruction against
continuous-position ground truth through a 28.8-unit grid, where 99.9% was never
reachable. The gates below are what the measurement can actually support.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.fov.union import mask_bit, mask_popcount, mask_to_bool
from shadowcast.l1_events.normalise import normalise
from shadowcast.l1_events.resolve import attribute, resolve_all
from shadowcast.l2_reconstruct.vision import VisionStream
from shadowcast.validate.fog_oracle import REGIONS, validate_fog

#: The synthetic match is shortened for the test suite. Agreement is stable across
#: lengths (96.9% over 900 s, 97.5% over 200 s), so this trades nothing for speed.
_DURATION = 200.0


@pytest.fixture(scope="module")
def pipeline(terrain, fov_table):
    """A fully reconstructed match, plus its truth and a truth-substituted attribution."""
    from shadowcast.packets.synth import Pathologies, ScenarioSpec, SyntheticSource

    src = SyntheticSource(
        terrain, ScenarioSpec(seed=7, duration=_DURATION, pathologies=Pathologies.all())
    )
    bundle, truth = src.generate(src.match_ids()[0])
    events = normalise(bundle, terrain)
    at = attribute(events)
    events, info = resolve_all(events, at.pos, at.valid)

    n = min(at.pos.shape[0], truth.pos.shape[0])
    truth_at = dataclasses.replace(at, pos=truth.pos[:n].copy(), valid=truth.alive[:n].astype(bool))
    return events, truth, at, truth_at, info, fov_table


@pytest.fixture(scope="module")
def agreement(pipeline, terrain):
    """The fog agreement, computed once and shared.

    Every gate below reads the same two measurements. Recomputing them per test was
    costing more than the rest of the suite combined.
    """
    events, _, at, truth_at, _, table = pipeline
    return (
        validate_fog(events, at, terrain, table),
        validate_fog(events, truth_at, terrain, table),
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def test_vision_requires_resolved_teams(terrain, fov_table, synth_clean):
    """A mask is per team. A combined mask would look fine and mean nothing."""
    bundle, _ = synth_clean
    events = normalise(bundle, terrain)
    at = attribute(events)
    with pytest.raises(ValueError, match="per team"):
        VisionStream(events, at, terrain, fov_table)


def test_masks_are_yielded_for_every_tick(pipeline, terrain):
    events, _, at, _, _, table = pipeline
    stream = VisionStream(events, at, terrain, table)
    ticks = [tick for tick, _, _ in stream.masks()]
    assert ticks == list(range(at.pos.shape[0]))


def test_mask_buffers_are_reused_unless_a_copy_is_asked_for(pipeline, terrain):
    """The whole point of streaming: holding every mask would be 472 MB.

    Reuse is a footgun if undocumented, so it is asserted here — a consumer that stashes
    a yielded array without asking for a copy gets the last tick's contents.
    """
    events, _, at, _, _, table = pipeline
    stream = VisionStream(events, at, terrain, table)
    kept = []
    for tick, m0, _ in stream.masks():
        kept.append(m0)
        if tick >= 3:
            break
    assert all(k is kept[0] for k in kept), "buffers should be the same object"

    stream2 = VisionStream(events, at, terrain, table)
    copies = []
    for tick, m0, _ in stream2.masks(copy=True):
        copies.append(m0)
        if tick >= 3:
            break
    assert len({id(c) for c in copies}) == len(copies)


def test_both_teams_have_vision_from_the_first_tick(pipeline, terrain):
    """Turrets are static and always present, so no tick should be totally blind."""
    events, _, at, _, _, table = pipeline
    stream = VisionStream(events, at, terrain, table)
    for tick, m0, m1 in stream.masks():
        assert mask_popcount(m0) > 0
        assert mask_popcount(m1) > 0
        if tick > 20:
            break


def test_a_ward_adds_vision_for_its_lifetime_and_no_longer(pipeline, terrain):
    """Ward lifetimes drive the semi-static layer, and ward yield depends on them."""
    events, _, at, _, _, table = pipeline
    ward = events.wards[np.argsort(events.wards["t0"])][0]
    from shadowcast.geom.grid import world_to_cell

    i, j = world_to_cell(float(ward["x"]), float(ward["z"]))
    team = int(events.heroes["team"][int(ward["owner_slot"])])
    dt = 1.0 / C.TICK_HZ
    before = max(0, round(float(ward["t0"]) / dt) - 12)
    during = round((float(ward["t0"]) + 3.0) / dt)

    seen = {}
    stream = VisionStream(events, at, terrain, table)
    for tick, m0, m1 in stream.masks():
        if tick in (before, during):
            seen[tick] = mask_bit((m0, m1)[team], i, j)
        if tick > during:
            break
    # The ward's own cell must be visible while it lives. It may already be visible
    # beforehand from other sources, so only the "during" direction is asserted.
    assert seen[during]


def test_turret_sites_without_a_position_grant_no_vision(pipeline, terrain):
    """A turret whose position could not be recovered must not be placed somewhere.

    Inventing a location would put vision on the map that nothing in the data supports.
    """
    events, _, at, _, _, table = pipeline
    sites = events.turret_sites.copy()
    sites["x"][0] = np.nan
    stripped = dataclasses.replace(events, turret_sites=sites)
    a = VisionStream(stripped, at, terrain, table)
    b = VisionStream(events, at, terrain, table)
    _, a0, a1 = next(iter(a.masks()))
    _, b0, b1 = next(iter(b.masks()))
    assert mask_popcount(a0) + mask_popcount(a1) < mask_popcount(b0) + mask_popcount(b1)


def test_the_table_covers_almost_every_source(pipeline, terrain):
    """Live field-of-view fallbacks should be rare, or the table is not earning its keep."""
    events, _, at, _, _, table = pipeline
    stream = VisionStream(events, at, terrain, table)
    for _ in stream.masks():
        pass
    counts = stream.counts()
    total = counts.champion_ticks + counts.minion_ticks + counts.reveal_ticks
    assert counts.live_fallbacks / max(1, total) < 0.01


# ---------------------------------------------------------------------------
# Reveal-on-attack — two bugs found here, both now pinned
# ---------------------------------------------------------------------------
def test_reveal_is_gated_on_having_been_in_fog(pipeline, terrain):
    """Applying the reveal unconditionally is catastrophically wrong.

    The tempting argument is that a reveal centred on an already-visible champion lies
    inside vision the observer had anyway, so the gate cannot matter. It does: a champion
    that attacks while visible and then walks into fog would keep being revealed for 4.5
    seconds it never earned. With attacks every ~1.5 seconds that is not a rounding error
    — measured, it took fog agreement from 98.8% to 43.4% with a 56.6% false-positive
    rate.
    """
    events, _, at, _, _, table = pipeline

    class Ungated(VisionStream):
        def _reveal_sources(self, tick, team):
            from shadowcast.geom.grid import world_to_cell

            t = tick * self.dt
            out = []
            window = self._attacks[
                (self._attacks["t"] > t - C.FOG_ATTACK_REVEAL_DURATION) & (self._attacks["t"] <= t)
            ]
            for row in window:
                slot = int(row["slot"])
                if not (0 <= slot < self.team.size) or int(self.team[slot]) == team:
                    continue
                i, j = world_to_cell(float(row["x"]), float(row["z"]))
                if 0 <= i < self.grid and 0 <= j < self.grid:
                    out.append((i, j, C.FOG_ATTACK_REVEAL_RADIUS, int(self.terrain.brush_id[j, i])))
            return out

    def lit(stream_cls):
        stream = stream_cls(events, at, terrain, table)
        total = 0
        for tick, m0, m1 in stream.masks():
            if tick % 40:
                continue
            total += mask_popcount(m0) + mask_popcount(m1)
        return total

    assert lit(Ungated) > lit(VisionStream), "the gate must actually restrict vision"


def test_reveal_is_an_area_at_the_attack_position_not_a_follow(pipeline, terrain):
    """The rule reveals "a 400-radius area centred on top of the attacker", statically.

    Modelling it as revealing the *champion* for the duration is a different rule, and a
    materially wrong one — a champion that attacks and walks away stays revealed under
    that version. It cost 11 points of agreement in false negatives before being caught.
    Asserted structurally: reveal sources sit at the recorded attack positions.
    """
    events, _, at, _, _, table = pipeline
    stream = VisionStream(events, at, terrain, table)
    from shadowcast.geom.grid import world_to_cell

    found = 0
    for tick, _, _ in stream.masks():
        for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
            for i, j, radius, _brush in stream._reveal_sources(tick, team):
                assert radius == C.FOG_ATTACK_REVEAL_RADIUS
                cells = {world_to_cell(float(r["x"]), float(r["z"])) for r in stream._attacks}
                assert (i, j) in cells, "a reveal is not at any recorded attack position"
                found += 1
        if found > 5:
            return
    assert found > 0, "no reveals were produced, so the rule is untested"


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_fog_agreement_with_true_positions(agreement):
    """The irreducible floor: how well the vision model can possibly do.

    Everything the reconstruction might get wrong is removed by substituting truth, so
    what remains is cell snapping (sources land on 28.8-unit cell centres, the oracle uses
    continuous positions), shadowcasting's known permissiveness at shadow edges, and the
    ward and minion models.
    """
    _, res = agreement
    # 98.8% on the adversarial stream, 99.4% on a clean one. The gate sits under the
    # adversarial figure: pathologies degrade ward timing and speed replication, which
    # shows up here even with positions handed over.
    assert res.rate > 0.98, res.describe()
    assert res.false_positive_rate < 0.01
    assert res.false_negative_rate < 0.01
    assert res.timing()["within_150ms"] > 0.90
    assert res.timing()["abs_median_s"] < 0.2


def test_fog_agreement_with_reconstructed_positions(agreement):
    """What actually ships. Lower than the floor, and the gap is the reconstruction's cost."""
    res, _ = agreement
    assert res.rate > 0.96, res.describe()
    assert res.false_positive_rate < 0.04
    assert res.false_negative_rate < 0.03


def test_reconstruction_costs_a_bounded_amount_of_agreement(agreement):
    """The decomposition, asserted as a relationship rather than two loose numbers."""
    reconstructed, floor = agreement
    assert floor.rate > reconstructed.rate, "truth should not be worse than reconstruction"
    assert floor.rate - reconstructed.rate < 0.05, (
        f"reconstruction costs {floor.rate - reconstructed.rate:.2%} of agreement"
    )


def test_false_positives_and_negatives_are_reported_separately(agreement):
    """They have different causes and opposite consequences for every metric.

    A false positive says a team saw an enemy it did not, which understates darkness and
    entropy. A false negative does the reverse. One combined percentage hides both.
    """
    res, _ = agreement
    assert res.false_positive + res.false_negative + res.agree == res.compared
    assert res.false_positive > 0
    assert res.false_negative > 0
    d = res.describe()
    assert "false_positive_rate" in d
    assert "false_negative_rate" in d


def test_agreement_is_broken_down_by_region(agreement):
    """Where errors land distinguishes a modelling limit from a bug.

    Brush-adjacent disagreement is expected — brush is a conditional occluder and the grid
    quantises its boundary. Open-lane disagreement is not, and lane must therefore be the
    strongest region.
    """
    res, _ = agreement
    rates = res.region_rates()
    assert set(rates) == set(REGIONS)
    assert all(n > 0 for n, _ in res.by_region.values())
    assert rates["lane"] == max(rates.values()), rates
    assert rates["brush_adjacent"] < rates["lane"]


def test_transition_counts_are_comparable(agreement):
    """Getting the state right while flickering would score well and be useless.

    An early version produced three times too many transitions while still reporting
    reasonable state agreement.
    """
    res, _ = agreement
    ours = res.stats["our_transitions"]
    theirs = res.stats["oracle_transitions"]
    assert 0.6 < ours / theirs < 1.6, f"{ours} transitions vs {theirs}"


def test_validation_reports_its_own_coverage(pipeline, agreement):
    """A comparison that silently skipped most ticks would look like a strong result."""
    events = pipeline[0]
    res, _ = agreement
    possible = res.stats["ticks"] * events.n_heroes
    assert res.compared > 0.9 * (possible - res.stats["no_position_claim"])


def test_stride_samples_without_changing_the_answer(pipeline, terrain):
    events, _, at, _, _, table = pipeline
    full = validate_fog(events, at, terrain, table)
    strided = validate_fog(events, at, terrain, table, stride=8)
    assert strided.compared < full.compared / 4
    assert abs(strided.rate - full.rate) < 0.02


def test_masks_agree_with_a_direct_bit_read(pipeline, terrain):
    """`mask_bit` is used on the hot path; it must match unpacking."""
    events, _, at, _, _, table = pipeline
    stream = VisionStream(events, at, terrain, table)
    rng = np.random.default_rng(0)
    for tick, m0, _ in stream.masks():
        if tick != 40:
            continue
        bools = mask_to_bool(m0, terrain.grid)
        for _ in range(200):
            i = int(rng.integers(0, terrain.grid))
            j = int(rng.integers(0, terrain.grid))
            assert mask_bit(m0, i, j) == bool(bools[j, i])
        break
