"""Tests for L1 normalisation.

Each test corresponds to one measured defect in the corpus that this layer repairs.
The interesting ones are the frame calibration (which recovers a constant nobody
documented) and the fog dedupe (which has to discard 85% of the fog stream without
losing a single real transition).
"""

from __future__ import annotations

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.l1_events.normalise import calibrate_waypoint_frame, normalise
from shadowcast.l1_events.schema import (
    ANCHOR_ATTACK,
    ANCHOR_CAST,
    UNKNOWN,
    MatchEvents,
)
from shadowcast.packets.synth import ScenarioSpec, SyntheticSource


@pytest.fixture(scope="module")
def events_clean(synth_clean, terrain):
    bundle, truth = synth_clean
    return normalise(bundle, terrain), truth


@pytest.fixture(scope="module")
def events_dirty(synth_dirty, terrain):
    bundle, truth = synth_dirty
    return normalise(bundle, terrain), truth


# ---------------------------------------------------------------------------
# Frame calibration
# ---------------------------------------------------------------------------
def test_frame_offset_is_recovered_within_a_cell(events_dirty):
    """The offset nobody documented, recovered from walkability alone.

    Waypoints are map-centred while every other position is world-framed. The true
    offset is deliberately not the 7500 a calibrator would guess first, so passing this
    requires the search to actually work.
    """
    events, truth = events_dirty
    error = abs(events.frame.offset - truth.spec.waypoint_offset)
    assert error <= C.GRID_CELL_SIZE / 2, f"frame offset off by {error:.1f} units"


def test_calibration_beats_the_naive_guess(events_dirty):
    """Distinguishes "calibration worked" from "calibration did nothing"."""
    events, _ = events_dirty
    assert events.frame.walkable_fraction > events.frame.baseline_fraction
    assert events.frame.walkable_fraction > 0.99


def test_calibration_plateau_is_one_cell_wide(events_dirty):
    """The method's resolution limit, asserted rather than left implicit.

    Offsets differing by less than a cell assign every waypoint to the same cell and
    score identically, so the plateau cannot be narrower than a cell however much data
    is used. A much wider plateau would mean the signal had gone weak; a much narrower
    one would mean the score is not doing what this claims.
    """
    events, _ = events_dirty
    assert events.frame.plateau_width == pytest.approx(C.GRID_CELL_SIZE, abs=2.0)
    assert events.frame.well_determined


def test_calibration_handles_an_empty_stream(terrain):
    from shadowcast.packets.source import WAYPOINT_XZ

    cal = calibrate_waypoint_frame(np.empty(0, dtype=WAYPOINT_XZ), terrain)
    assert cal.offset == C.WAYPOINT_OFFSET_GUESS
    assert cal.n_samples == 0
    assert not cal.well_determined


def test_calibrated_positions_land_on_walkable_ground(events_clean, terrain):
    from shadowcast.geom.grid import world_to_cell_array

    events, _ = events_clean
    i, j = world_to_cell_array(events.order_xz["x"], events.order_xz["z"])
    inside = (i >= 0) & (i < terrain.grid) & (j >= 0) & (j < terrain.grid)
    assert inside.mean() > 0.999
    assert terrain.walkable[j[inside], i[inside]].mean() > 0.99


def test_frame_can_be_supplied_to_reuse_across_matches(synth_clean, terrain):
    """The offset is a property of the decoder, not of a match, so one calibration
    should serve a whole shard."""
    from shadowcast.l1_events.schema import FrameCalibration

    bundle, _ = synth_clean
    fixed = FrameCalibration(
        offset=1234.0, walkable_fraction=0.5, plateau_width=1.0, baseline_fraction=0.4, n_samples=10
    )
    events = normalise(bundle, terrain, frame=fixed)
    assert events.frame.offset == 1234.0
    assert events.order_xz["x"][0] == pytest.approx(bundle.waypoint_xz["x"][0] + 1234.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
def test_keyframe_repeated_creates_collapse_to_one_row_per_champion(events_dirty):
    events, truth = events_dirty
    assert events.n_heroes == C.N_HEROES
    assert np.unique(events.heroes["net_id"]).size == C.N_HEROES
    assert sorted(events.heroes["net_id"].tolist()) == sorted(truth.net_ids.tolist())


def test_slots_are_assigned_by_sorted_net_id(events_clean):
    """Stability matters: every downstream array is indexed by slot."""
    events, _ = events_clean
    assert events.heroes["slot"].tolist() == list(range(C.N_HEROES))
    assert (np.diff(events.heroes["net_id"]) > 0).all()


def test_champion_identity_survives_normalisation(events_clean):
    events, truth = events_clean
    for row in events.heroes:
        idx = int(np.flatnonzero(truth.net_ids == row["net_id"])[0])
        assert str(row["champion"]) == truth.champion[idx]


def test_unresolved_fields_are_marked_unknown(events_clean):
    """Team, role and order ownership are inferences, not readings.

    They stay UNKNOWN until a resolver runs, so no consumer can mistake an
    unresolved value for a real one — which matters because the corpus omits all
    three entirely.
    """
    events, _ = events_clean
    assert (events.heroes["team"] == UNKNOWN).all()
    assert (events.heroes["role"] == "").all()
    assert (events.orders["owner"] == UNKNOWN).all()
    assert (events.fog["observer_team"] == UNKNOWN).all()
    assert not events.teams_resolved
    assert not events.roles_resolved
    assert not events.orders_attributed


# ---------------------------------------------------------------------------
# Fog dedupe
# ---------------------------------------------------------------------------
def test_fog_dedupe_discards_duplicates_without_losing_transitions(events_dirty, synth_dirty):
    """The corpus's largest defect, and the first thing any consumer must handle.

    LeaveFog is 65-70% of all real packets, repeated up to twenty times at a single
    timestamp. Dedupe has to throw away the vast majority while keeping every genuine
    state change — so the output is compared against the oracle's own timeline, not
    merely checked for plausibility.
    """
    events, truth = events_dirty
    bundle, _ = synth_dirty
    assert events.stats["fog_dedupe_ratio"] > 3.0, "duplication was not exercised"
    assert events.fog.size < bundle.fog.size

    # The timeline's implicit initial state is "not visible", so a champion emits a
    # row at tick 0 only if it is somehow visible then. None are: every champion
    # starts at its own fountain, where the enemy cannot see it.
    expected = 0
    for c in range(C.N_HEROES):
        obs = 1 - int(truth.team[c])
        seen = truth.visible[:, obs, c].astype(bool)
        expected += int(seen[0]) + int((seen[1:] != seen[:-1]).sum())
    assert events.fog.size == expected


def test_fog_timeline_alternates_per_champion(events_dirty):
    events, _ = events_dirty
    for slot in range(C.N_HEROES):
        seq = events.fog[events.fog["slot"] == slot]
        assert seq.size > 0
        v = seq["visible"].astype(np.int8)
        assert not (np.diff(v) == 0).any(), f"slot {slot} repeats a transition"


def test_fog_is_only_about_champions(events_dirty):
    events, _ = events_dirty
    assert (events.fog["slot"] >= 0).all()
    assert (events.fog["slot"] < C.N_HEROES).all()


def test_fog_is_time_sorted(events_dirty):
    events, _ = events_dirty
    assert (np.diff(events.fog["t"]) >= 0).all()


# ---------------------------------------------------------------------------
# Turret positions
# ---------------------------------------------------------------------------
def test_turret_positions_are_recovered_from_attack_packets(events_clean, terrain):
    """`CreateTurret` has no coordinates, so this is the only route to turret vision.

    It is also what makes champion teams resolvable at all: the turret name encodes
    T1/T2 while CreateHero encodes nothing, so a turret with both a team and a position
    is the anchor everything else hangs from.
    """
    from shadowcast import sr

    events, _ = events_clean
    assert events.turret_sites.size == len(sr.TURRETS)
    assert (events.turret_sites["n_obs"] > 0).all(), "a turret never attacked"
    assert np.isfinite(events.turret_sites["x"]).all()

    expected, _ = sr.snap_polyline(terrain, np.array([p for _, _, p in sr.TURRETS]))
    for site in events.turret_sites:
        d = np.hypot(expected[:, 0] - site["x"], expected[:, 1] - site["z"]).min()
        assert d < 1.0, f"turret {site['name']} recovered {d:.1f} units from any real turret"


def test_turret_teams_come_from_their_names(events_clean):
    events, _ = events_clean
    assert (events.turret_sites["team"] != UNKNOWN).all()
    teams, counts = np.unique(events.turret_sites["team"], return_counts=True)
    assert set(teams.tolist()) == {C.TEAM_ORDER, C.TEAM_CHAOS}
    assert counts[0] == counts[1], "turrets should split evenly between teams"


def test_turret_sites_without_attacks_are_marked_rather_than_invented(terrain):
    """A turret that never shoots has no recoverable position, and says so."""
    src = SyntheticSource(terrain, ScenarioSpec(seed=2, duration=20.0))
    bundle, _ = src.generate(src.match_ids()[0])
    events = normalise(bundle, terrain)
    silent = events.turret_sites[events.turret_sites["n_obs"] == 0]
    if silent.size:
        assert np.isnan(silent["x"]).all()


# ---------------------------------------------------------------------------
# Wards
# ---------------------------------------------------------------------------
def test_ward_placements_are_recovered_with_correct_times(events_dirty):
    """Placement times come from stream position, because `SpawnMinion.time` is garbage.

    Denormal-float noise in the real field, so a reader that trusts it dates every ward
    to the first instant of the match.
    """
    events, truth = events_dirty
    assert events.wards.size == truth.wards.size
    for ward in events.wards:
        d = np.hypot(truth.wards["x"] - ward["x"], truth.wards["z"] - ward["z"])
        match = truth.wards[int(np.argmin(d))]
        assert d.min() < 1.0
        assert ward["t0"] == pytest.approx(match["t0"], abs=0.5)


def test_observed_ward_destruction_is_distinguished_from_a_modelled_expiry(events_dirty):
    """`t1_known` matters because ward yield integrates over lifetime.

    A modelled endpoint carries more uncertainty than an observed one, and the real
    stream contains both.
    """
    events, truth = events_dirty
    assert int(truth.wards["silent_expiry"].sum()) == 1
    assert int((events.wards["t1_known"] == 0).sum()) == 1
    for ward in events.wards[events.wards["t1_known"] == 1]:
        d = np.hypot(truth.wards["x"] - ward["x"], truth.wards["z"] - ward["z"])
        assert ward["t1"] == pytest.approx(truth.wards[int(np.argmin(d))]["t1"], abs=0.5)


def test_modelled_expiry_uses_the_documented_duration(events_dirty):
    events, _ = events_dirty
    modelled = events.wards[events.wards["t1_known"] == 0]
    for ward in modelled:
        life = ward["t1"] - ward["t0"]
        if str(ward["kind"]) == "totem":
            assert C.WARD_TOTEM_DURATION_MIN <= life <= C.WARD_TOTEM_DURATION_MAX
        else:
            assert np.isinf(life)


def test_ward_owners_and_sight_radii_are_resolved(events_clean):
    events, _ = events_clean
    assert (events.wards["owner_slot"] >= 0).all()
    assert (events.wards["owner_slot"] < C.N_HEROES).all()
    for ward in events.wards:
        assert ward["sight"] == C.WARD_SIGHT_BY_KIND[str(ward["kind"])]


# ---------------------------------------------------------------------------
# Anchors and replication
# ---------------------------------------------------------------------------
def test_anchors_are_labelled_champion_positions(events_clean):
    """The only packets tying a position to a net_id, and therefore what makes
    anonymous movement orders attributable."""
    events, _ = events_clean
    assert events.anchors.size > 0
    assert (events.anchors["slot"] >= 0).all()
    assert set(np.unique(events.anchors["kind"]).tolist()) == {ANCHOR_CAST, ANCHOR_ATTACK}
    per_hero = events.anchors.size / C.N_HEROES
    # The real corpus gives 546-1,085 per champion per match.
    assert 400 < per_hero < 1400


def test_anchor_positions_match_the_truth_at_that_instant(events_clean):
    events, truth = events_clean
    dt = truth.spec.dt
    rng = np.random.default_rng(1)
    for k in rng.choice(events.anchors.size, size=400, replace=False):
        a = events.anchors[k]
        tick = round(float(a["t"]) / dt)
        if tick >= truth.pos.shape[0]:
            continue
        d = np.hypot(*(np.array([a["x"], a["z"]]) - truth.pos[tick, int(a["slot"])]))
        assert d < 0.05, f"anchor {d:.3f} units from truth"


def test_turret_and_ghost_anchors_are_excluded(events_clean, synth_dirty):
    """Turrets emit attacks and a pet emits casts; neither is a champion.

    A resolver that folded them into the champion anchor set would be tracking twelve
    targets and blaming the mismatch on the tracker.
    """
    events, truth = events_clean
    assert (events.anchors["slot"] < C.N_HEROES).all()
    bundle, _ = synth_dirty
    assert bundle.casts["caster_net_id"].max() > truth.net_ids.max()  # the ghost exists


def test_movement_speed_is_recovered(events_clean, events_dirty):
    events, _ = events_clean
    assert events.speed.size >= C.N_HEROES
    assert (events.speed["slot"] >= 0).all()
    values = np.unique(events.speed["value"])
    assert values.size >= 2, "a mid-game speed change should be visible"
    assert (values >= C.MOVE_SPEED_MIN).all()
    assert (values <= C.MOVE_SPEED_MAX).all()

    dirty, _ = events_dirty
    assert dirty.speed.size <= events.speed.size  # some replicas are dropped


def test_replication_is_matched_on_the_index_pair_when_the_name_is_empty(synth_clean, terrain):
    """57% of real replication entries have an empty name.

    A reader keying on the name alone silently discards the majority of them, including
    most movement-speed updates, and then the trajectory reconstructor integrates at the
    wrong speed with nothing to indicate why.
    """
    import dataclasses

    bundle, _ = synth_clean
    stripped = bundle.replication.copy()
    stripped["name"] = ""
    events = normalise(dataclasses.replace(bundle, replication=stripped), terrain)
    baseline = normalise(bundle, terrain)
    assert events.speed.size == baseline.speed.size
    np.testing.assert_allclose(np.sort(events.speed["value"]), np.sort(baseline.speed["value"]))


def test_health_is_recovered_for_the_death_inference(events_clean):
    """There is no death packet, so a death IS a health value reaching zero."""
    events, truth = events_clean
    zeros = events.hp[events.hp["value"] == 0.0]
    assert zeros.size == truth.kills.size
    assert (zeros["slot"] >= 0).all()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_round_trips_through_disk(events_clean, tmp_path):
    events, _ = events_clean
    back = MatchEvents.load(events.save(tmp_path / "events.npz"))
    assert back.match_id == events.match_id
    assert back.frame.offset == events.frame.offset
    for name in MatchEvents._ARRAYS:
        np.testing.assert_array_equal(getattr(back, name), getattr(events, name))
    assert back.describe() == events.describe()


def test_order_polyline_accessor_matches_the_payload(events_clean):
    events, _ = events_clean
    for n in (0, 5, events.orders.size - 1):
        row = events.orders[n]
        poly = events.order_polyline(n)
        assert poly.shape == (row["n"], 2)
        assert poly[0, 0] == events.order_xz["x"][row["off"]]


def test_describe_reports_the_shape_of_the_match(events_clean):
    events, _ = events_clean
    d = events.describe()
    assert d["heroes"] == C.N_HEROES
    assert d["orders"] > 1000
    assert d["turret_sites"] == 24
    assert d["wards"] == 10
