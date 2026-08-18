"""Tests for the packet-source seam, the synthetic generator, and the fog oracle.

The generator is the test harness for everything downstream, so it needs its own
tests more than most modules do: a bug here does not fail loudly, it makes some
later layer look wrong or, worse: look right.

Two properties matter above the rest. Movement orders must carry no entity id, or
the attribution layer is never exercised. And integrating the published orders must
reproduce the published truth, or the trajectory reconstructor is being asked to
invert something that was never consistent to begin with.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.packets.conformance import validate_bundle, validate_source
from shadowcast.packets.source import PACKET_KINDS, PacketSource
from shadowcast.packets.synth import ScenarioSpec, SyntheticSource, Truth
from shadowcast.packets.synth_fog import segment_clear


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------
def test_synthetic_source_satisfies_the_protocol(terrain):
    src = SyntheticSource(terrain, ScenarioSpec(duration=60.0))
    assert isinstance(src, PacketSource)
    assert len(src.match_ids()) == 1
    assert src.match_ids() == sorted(src.match_ids())


def test_movement_orders_carry_no_entity_id(synth_clean):
    """The defining defect of the real corpus, asserted structurally.

    In the real data the waypoint dict's key is the list length, not a net_id, 100%
    of 41,129 pairs checked. If a source ever supplied an id here, movement-order
    attribution would become dead code and every test that assumes attribution is
    hard would start passing for the wrong reason.
    """
    bundle, _ = synth_clean
    names = bundle.waypoints.dtype.names
    assert names is not None
    for forbidden in ("net_id", "entity", "entity_id", "owner"):
        assert forbidden not in names
    #  is the stream position, not an entity id: it says WHERE in the packet
    # order the row sat, which is what makes SpawnMinion's corrupt clock recoverable.
    assert set(names) == {"t", "off", "n", "with_speed", "seq"}


def test_bundle_dtypes_match_the_declared_kinds(synth_clean):
    bundle, _ = synth_clean
    for name, dtype in PACKET_KINDS.items():
        assert getattr(bundle, name).dtype == dtype, name


def test_no_death_packet_ever_names_a_champion(synth_clean):
    """Faithful to the corpus: 45,851 real death rows, zero champion victims.

    Anything downstream hoping to read champion deaths from here must find nothing,
    so that it is forced to infer them from health replication as it will have to on
    real data.
    """
    bundle, truth = synth_clean
    if bundle.deaths.size:
        assert not np.isin(bundle.deaths["killed_net_id"], truth.net_ids).any()


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------
def test_clean_stream_is_silent(synth_clean):
    bundle, _ = synth_clean
    rep = validate_bundle(bundle)
    assert rep.ok, rep.render()
    assert rep.warnings == [], rep.render()


def test_every_pathology_shows_up_as_a_warning_and_none_as_an_error(synth_dirty):
    """With everything injected the stream is still valid, just noisy.

    The error/warning split is the point: these defects are the corpus being itself,
    not bugs, so a reader must tolerate them. An error here would mean the generator
    produced something no real source would.
    """
    bundle, _ = synth_dirty
    rep = validate_bundle(bundle)
    assert rep.ok, rep.render()
    assert rep.warnings, "pathologies should be visible in the report"
    text = " ".join(rep.warnings)
    assert "out of order" in text  # reorder_window
    assert "never created" in text  # ghost_caster


def test_validate_source_covers_every_match(terrain):
    src = SyntheticSource(terrain, ScenarioSpec(duration=60.0), n_matches=2)
    reports = validate_source(src)
    assert len(reports) == 2
    assert all(r.ok for r in reports)


def test_conformance_rejects_a_source_that_labels_its_orders(synth_clean):
    """The check that keeps the seam honest, verified to actually fire."""
    import dataclasses

    bundle, _ = synth_clean
    labelled = np.empty(
        bundle.waypoints.size,
        dtype=np.dtype(
            [("t", "f8"), ("off", "i8"), ("n", "i4"), ("with_speed", "u1"), ("net_id", "u4")]
        ),
    )
    for f in ("t", "off", "n", "with_speed"):
        labelled[f] = bundle.waypoints[f]
    labelled["net_id"] = 1
    rep = validate_bundle(dataclasses.replace(bundle, waypoints=labelled))
    assert not rep.ok
    assert any("entity id" in e for e in rep.errors)


# ---------------------------------------------------------------------------
# The core property: orders reproduce truth
# ---------------------------------------------------------------------------
def test_each_order_starts_at_its_owners_true_position(synth_clean):
    """The consistency the trajectory reconstructor is asked to invert.

    Motion is defined BY the orders. A champion walks the polyline it published,
    so with no jitter injected the first waypoint of every order must equal the
    owner's true position at that tick, exactly. If this drifts, no downstream
    reconstruction can be correct, because truth and orders would disagree.
    """
    bundle, truth = synth_clean
    offset = np.array([truth.spec.waypoint_offset_x, truth.spec.waypoint_offset_z])
    worst = 0.0
    for n in range(bundle.waypoints.size):
        poly = bundle.order_polyline(n) + offset
        c = int(truth.order_owner[n])
        tick = int(truth.order_tick[n])
        worst = max(worst, float(np.hypot(*(poly[0] - truth.pos[tick, c]))))
    # Tolerance is float32 precision, not zero: waypoint coordinates are stored as
    # f4 to match the real payload and halve the bytes, and 14,000 units in float32
    # resolves to about 0.001. A 1e-6 threshold was below the representation itself.
    assert worst < 0.02, f"order start disagrees with truth by {worst:.4f} units"


def test_jitter_pathology_perturbs_order_starts_but_only_slightly(synth_dirty):
    """With jitter on, the disagreement must appear, and stay bounded.

    This models server-side smoothing, where waypoints[0] is near but not equal to
    the true position. The reconstructor's order residual is exactly this quantity,
    so it needs to be non-zero to be tested and small enough to be recoverable.
    """
    bundle, truth = synth_dirty
    offset = np.array([truth.spec.waypoint_offset_x, truth.spec.waypoint_offset_z])
    errs = []
    for n in range(bundle.waypoints.size):
        poly = bundle.order_polyline(n) + offset
        c = int(truth.order_owner[n])
        tick = int(truth.order_tick[n])
        errs.append(float(np.hypot(*(poly[0] - truth.pos[tick, c]))))
    errs = np.array(errs)
    assert errs.max() > 1.0, "jitter pathology had no effect"
    assert np.percentile(errs, 99) < 60.0, f"jitter p99 {np.percentile(errs, 99):.1f} too large"


def test_order_polylines_are_walkable(synth_clean, terrain):
    """Simplification must not cut wall corners.

    A shortcut through terrain would put the ground truth somewhere no champion
    could walk, and the belief filter, which constrains particles to the navmesh,
    would then look wrong precisely when it was right.
    """
    from shadowcast.geom.grid import world_to_cell_array
    from shadowcast.geom.path import chord_walkable

    bundle, truth = synth_clean
    offset = np.array([truth.spec.waypoint_offset_x, truth.spec.waypoint_offset_z])
    rng = np.random.default_rng(3)
    for n in rng.choice(bundle.waypoints.size, size=300, replace=False):
        poly = bundle.order_polyline(int(n)) + offset
        i, j = world_to_cell_array(poly[:, 0], poly[:, 1])
        i = np.clip(i, 0, terrain.grid - 1)
        j = np.clip(j, 0, terrain.grid - 1)
        for k in range(len(poly) - 1):
            assert chord_walkable(terrain.walkable, j[k], i[k], j[k + 1], i[k + 1]), (
                f"order {n} segment {k} crosses terrain"
            )


def test_true_positions_are_on_walkable_ground(synth_clean, terrain):
    from shadowcast.geom.grid import world_to_cell_array

    _, truth = synth_clean
    i, j = world_to_cell_array(truth.pos[:, :, 0].ravel(), truth.pos[:, :, 1].ravel())
    ok = terrain.walkable[np.clip(j, 0, terrain.grid - 1), np.clip(i, 0, terrain.grid - 1)]
    assert ok.all(), f"{(~ok).sum()} of {ok.size} truth positions are inside terrain"


def test_orders_are_evenly_spread_across_champions(synth_clean):
    """Guards a scenario bug that made the generator useless without failing.

    Issuing orders on completion rather than on a timer gave junglers ~950 orders
    against a laner's ~40, because a goal a few units away produces a one-cell path
    that finishes immediately. Attribution tested against that distribution would
    have been tested almost entirely on two champions.
    """
    _, truth = synth_clean
    counts = np.bincount(truth.order_owner, minlength=C.N_HEROES)
    assert counts.min() > 200, f"some champion barely moves: {counts}"
    assert counts.max() / counts.min() < 2.5, f"order counts too skewed: {counts}"


def test_implied_speed_is_plausible(synth_clean):
    """Champions must actually move.

    Long oscillation periods once left laners drifting ~10 units/second. They stood
    still, no order was long enough to emit, and median implied speed was zero.
    """
    _, truth = synth_clean
    d = np.linalg.norm(np.diff(truth.pos, axis=0), axis=2) / truth.spec.dt
    alive = truth.alive[1:].astype(bool) & truth.alive[:-1].astype(bool)
    speeds = d[alive]
    assert (speeds > 5).mean() > 0.3, "champions are stationary too often"
    assert speeds.max() <= 400.0, f"implied speed {speeds.max():.0f} exceeds any real speed"


# ---------------------------------------------------------------------------
# Wards, kills, replication
# ---------------------------------------------------------------------------
def test_wards_arrive_as_recognised_spawn_minion_units(synth_clean):
    """Faithful to the corpus: wards are not a packet type.

    They arrive as SpawnMinion rows identified by (name, skin_name), with the owner's
    hero net_id in `targetable_on_client` and destruction signalled by a WardCorpse.
    """
    bundle, truth = synth_clean
    pairs = [(str(a), str(b)) for a, b in zip(bundle.minions["name"], bundle.minions["skin_name"])]
    placements = [p for p in pairs if p in C.WARD_UNITS]
    corpses = [p for p in pairs if p == C.WARD_CORPSE_UNIT]
    assert len(placements) == truth.wards.size
    # One ward expires with no corpse when that pathology is off... it is off here,
    # so every ward should be accounted for.
    assert len(corpses) == truth.wards.size
    # Only WARD rows name an owner. Lane minions also arrive as SpawnMinion and carry
    # zero there, so the check has to select the ward rows rather than the whole array.
    is_ward = np.array([p in C.WARD_UNITS for p in pairs])
    owners = set(bundle.minions["targetable_on_client"][is_ward].tolist())
    assert owners <= set(truth.net_ids.tolist())


def test_silent_ward_expiry_drops_exactly_one_corpse(synth_dirty):
    bundle, truth = synth_dirty
    pairs = [(str(a), str(b)) for a, b in zip(bundle.minions["name"], bundle.minions["skin_name"])]
    corpses = sum(1 for p in pairs if p == C.WARD_CORPSE_UNIT)
    assert corpses == truth.wards.size - 1
    assert int(truth.wards["silent_expiry"].sum()) == 1


def test_ward_kinds_have_known_sight_radii(synth_clean):
    _, truth = synth_clean
    for kind in np.unique(truth.wards["kind"]):
        assert str(kind) in C.WARD_SIGHT_BY_KIND


def test_kills_appear_only_as_health_replication(synth_clean):
    """No death packet exists, so a death is a health value reaching zero."""
    bundle, truth = synth_clean
    hp_zero = bundle.replication[
        (bundle.replication["name"] == "mHP") & (bundle.replication["value"] == 0.0)
    ]
    assert hp_zero.size == truth.kills.size
    assert set(hp_zero["net_id"].tolist()) <= set(truth.net_ids.tolist())


def test_damage_precedes_each_kill(synth_clean):
    """Kill attribution will have to join a death to the last damage before it."""
    bundle, truth = synth_clean
    for row in truth.kills:
        killer_id = int(truth.net_ids[int(row["killer"])])
        victim_id = int(truth.net_ids[int(row["victim"])])
        window = bundle.damage[
            (bundle.damage["target_net_id"] == victim_id)
            & (bundle.damage["t"] >= row["t"] - 1.5)
            & (bundle.damage["t"] <= row["t"])
        ]
        assert window.size > 0, f"no damage before the kill at t={row['t']}"
        assert killer_id in window["source_net_id"].tolist()


def test_dead_champions_sit_at_their_fountain_then_respawn(synth_clean):
    from shadowcast import sr

    _, truth = synth_clean
    for row in truth.kills:
        tick = int(row["t"] / truth.spec.dt) + 2
        victim = int(row["victim"])
        assert truth.alive[tick, victim] == 0
        at = truth.pos[tick, victim]
        assert np.hypot(*(at - sr.FOUNTAINS[int(truth.team[victim])])) < 1.0
        back = int(row["respawn_t"] / truth.spec.dt) + 3
        if back < truth.alive.shape[0]:
            assert truth.alive[back, victim] == 1


def test_movement_speed_is_replicated_and_changes_mid_game(synth_clean):
    """The reconstructor must read speed from replication rather than assume it."""
    bundle, _ = synth_clean
    ms = bundle.replication[bundle.replication["name"] == "mMoveSpeed"]
    assert ms.size >= 2 * C.N_HEROES
    assert len(np.unique(ms["value"])) >= 2, "speed never changes, so a change is untested"
    assert (ms["primary"] == 32).all()
    assert (ms["secondary"] == 24).all()


def test_dropped_speed_replicas_pathology_loses_some(terrain):
    from shadowcast.packets.synth import Pathologies as P

    def n_speed(pathologies):
        src = SyntheticSource(
            terrain, ScenarioSpec(seed=3, duration=600.0, pathologies=pathologies)
        )
        b = src.read(src.match_ids()[0])
        r = b.replication
        return int(((r["name"] == "mMoveSpeed") & (r["t"] > 1.0)).sum())

    # A high drop rate makes the effect deterministic; the default 2% would need luck.
    assert n_speed(P.none()) > n_speed(P(drop_speed_replicas=1.0))


# ---------------------------------------------------------------------------
# Fog oracle
# ---------------------------------------------------------------------------
def test_fog_events_are_only_about_champions_and_only_cross_team(synth_clean):
    bundle, truth = synth_clean
    assert np.isin(bundle.fog["net_id"], truth.net_ids).all()
    # A team never loses sight of its own members, which is exactly what lets the
    # observing team be recovered from the champion an event names.
    for c in range(C.N_HEROES):
        own = int(truth.team[c])
        assert truth.visible[:, own, c].all()


def test_fog_transitions_alternate_per_champion(synth_clean):
    bundle, _ = synth_clean
    for nid in np.unique(bundle.fog["net_id"]):
        seq = bundle.fog[bundle.fog["net_id"] == nid]
        leaving = seq["leaving"][np.argsort(seq["t"], kind="stable")].astype(np.int8)
        assert not (np.diff(leaving) == 0).any(), f"entity {nid} repeats a transition"


def test_fog_rows_reconstruct_the_visibility_timeline(synth_clean):
    """Replaying the published events must recover the oracle's timeline.

    This is what a consumer will actually do, so a mismatch would mean the events
    understate or overstate what the oracle knew.
    """
    bundle, truth = synth_clean
    dt = truth.spec.dt
    n_ticks = truth.visible.shape[0]
    for c in range(C.N_HEROES):
        nid = int(truth.net_ids[c])
        obs = 1 - int(truth.team[c])
        rows = bundle.fog[bundle.fog["net_id"] == nid]
        rows = rows[np.argsort(rows["t"], kind="stable")]
        replay = np.zeros(n_ticks, dtype=bool)
        state = False
        cursor = 0
        for tick in range(n_ticks):
            t = tick * dt
            while cursor < rows.size and rows[cursor]["t"] <= t + 1e-9:
                state = bool(rows[cursor]["leaving"])
                cursor += 1
            replay[tick] = state
        np.testing.assert_array_equal(replay, truth.visible[:, obs, c].astype(bool))


def test_junglers_are_the_least_visible_role(synth_clean):
    """A sanity check on the scenario, and a preview of a headline finding.

    Junglers should live in the dark and supports should not. If this inverts, either
    the routes or the vision-source assembly is wrong.
    """
    _, truth = synth_clean
    seen = {
        truth.role[c]: truth.visible[:, 1 - int(truth.team[c]), c].mean() for c in range(C.N_HEROES)
    }
    jungle = min(
        truth.visible[:, 1 - int(truth.team[c]), c].mean()
        for c in range(C.N_HEROES)
        if truth.role[c] == "jungle"
    )
    others = [
        truth.visible[:, 1 - int(truth.team[c]), c].mean()
        for c in range(C.N_HEROES)
        if truth.role[c] != "jungle"
    ]
    assert jungle < min(others), f"jungler not the least visible: {seen}"


def test_oracle_agrees_with_shadowcasting_where_they_overlap(synth_clean, terrain, fov_table):
    """Cross-check between the two independent visibility implementations.

    The oracle marches segments between continuous positions; this builds a mask by
    octant sweep from snapped cell centres, using only the observing team's champions.
    Those are a subset of the oracle's sources, so the implication is one-directional:
    anything the champion-only mask lights, the oracle must also have seen.

    A preview of M5's fog-agreement gate, and the reason the oracle shares no code
    with `fov/`. A comparison against a re-run of the same algorithm would prove
    nothing.
    """
    from shadowcast.fov.union import assemble, mask_to_bool
    from shadowcast.geom.grid import world_to_cell

    _, truth = synth_clean
    rng = np.random.default_rng(17)
    ticks = rng.choice(np.arange(80, truth.visible.shape[0]), size=25, replace=False)

    checked = violations = 0
    for tick in ticks:
        tick = int(tick)
        for obs in (0, 1):
            sources = []
            for c in range(C.N_HEROES):
                if int(truth.team[c]) != obs or not truth.alive[tick, c]:
                    continue
                i, j = world_to_cell(*truth.pos[tick, c])
                if not (0 <= i < terrain.grid and 0 <= j < terrain.grid):
                    continue
                sources.append((i, j, C.SIGHT_CHAMPION, int(truth.brush[tick, c])))
            if not sources:
                continue
            mask = mask_to_bool(assemble(fov_table, terrain, sources), terrain.grid)

            for c in range(C.N_HEROES):
                if int(truth.team[c]) == obs or not truth.alive[tick, c]:
                    continue
                i, j = world_to_cell(*truth.pos[tick, c])
                if not (0 <= i < terrain.grid and 0 <= j < terrain.grid):
                    continue
                checked += 1
                if mask[j, i] and not truth.visible[tick, obs, c]:
                    violations += 1

    assert checked > 100, "not enough overlap sampled to conclude anything"
    # A handful of disagreements are expected: the oracle uses continuous positions
    # while the mask snaps to cell centres, so a champion near a shadow boundary can
    # fall on either side. Concentrated disagreement would mean a real inconsistency.
    rate = violations / checked
    assert rate < 0.02, f"{violations}/{checked} = {rate:.2%} disagreements between the two"


def test_segment_clear_closed_forms():
    """The oracle's ray march, against hand-computable cases."""
    blocks = np.zeros((32, 32), dtype=bool)
    brush = np.full((32, 32), -1, dtype=np.int16)

    assert segment_clear(blocks, brush, 4.5, 4.5, 20.5, 4.5, -1)

    blocks[4, 12] = True  # wall on the straight line between them
    assert not segment_clear(blocks, brush, 4.5, 4.5, 20.5, 4.5, -1)
    # A parallel line one row over is unaffected.
    assert segment_clear(blocks, brush, 5.5, 5.5, 20.5, 5.5, -1)

    blocks[:] = False
    brush[4, 12] = 3  # foreign brush blocks
    assert not segment_clear(blocks, brush, 4.5, 4.5, 20.5, 4.5, -1)
    # ...but not for an observer inside that same brush.
    assert segment_clear(blocks, brush, 4.5, 4.5, 20.5, 4.5, 3)


def test_segment_clear_ignores_its_own_endpoints():
    """A source cannot occlude itself, and a wall at the target does not hide it."""
    blocks = np.zeros((32, 32), dtype=bool)
    brush = np.full((32, 32), -1, dtype=np.int16)
    blocks[4, 4] = True
    blocks[4, 20] = True
    assert segment_clear(blocks, brush, 4.5, 4.5, 20.5, 4.5, -1)


def test_off_grid_blocks():
    blocks = np.zeros((32, 32), dtype=bool)
    brush = np.full((32, 32), -1, dtype=np.int16)
    assert not segment_clear(blocks, brush, 4.5, 4.5, -20.0, 4.5, -1)


# ---------------------------------------------------------------------------
# Determinism and persistence
# ---------------------------------------------------------------------------
def test_generation_is_deterministic(terrain):
    spec = ScenarioSpec(seed=11, duration=120.0)
    a_bundle, a_truth = SyntheticSource(terrain, spec).generate("synth-0011-000")
    b_bundle, b_truth = SyntheticSource(terrain, spec).generate("synth-0011-000")
    np.testing.assert_array_equal(a_truth.pos, b_truth.pos)
    np.testing.assert_array_equal(a_truth.visible, b_truth.visible)
    np.testing.assert_array_equal(a_bundle.fog, b_bundle.fog)
    np.testing.assert_array_equal(a_bundle.waypoint_xz, b_bundle.waypoint_xz)


def test_different_seeds_differ(terrain):
    _, a = SyntheticSource(terrain, ScenarioSpec(seed=1, duration=120.0)).generate("synth-0001-000")
    _, b = SyntheticSource(terrain, ScenarioSpec(seed=2, duration=120.0)).generate("synth-0002-000")
    assert not np.allclose(a.pos, b.pos)


def test_truth_round_trips_through_disk(synth_clean, tmp_path):
    _, truth = synth_clean
    path = truth.save(tmp_path / "truth.npz")
    back = Truth.load(path)
    np.testing.assert_array_equal(back.pos, truth.pos)
    np.testing.assert_array_equal(back.visible, truth.visible)
    np.testing.assert_array_equal(back.order_owner, truth.order_owner)
    np.testing.assert_array_equal(back.wards, truth.wards)
    assert back.role == truth.role
    assert back.champion == truth.champion
    assert back.spec == truth.spec


def test_waypoint_offset_is_perturbed_off_the_true_frame():
    """The calibration step must be tested, not trivially satisfied.

    If the synthetic frame offset were exactly the navgrid midpoint the calibrator starts from,
    a broken calibrator would still pass.
    """
    spec = ScenarioSpec()
    assert spec.waypoint_offset_x != C.WAYPOINT_OFFSET_X
    assert spec.waypoint_offset_z != C.WAYPOINT_OFFSET_Z
    assert abs(spec.waypoint_offset_x - C.WAYPOINT_OFFSET_X) < C.WAYPOINT_OFFSET_SEARCH
    assert abs(spec.waypoint_offset_z - C.WAYPOINT_OFFSET_Z) < C.WAYPOINT_OFFSET_SEARCH


def test_waypoints_are_map_centred(synth_clean):
    bundle, _ = synth_clean
    assert bundle.waypoint_xz["x"].min() < 0 < bundle.waypoint_xz["x"].max()
    assert bundle.waypoint_xz["z"].min() < 0 < bundle.waypoint_xz["z"].max()


def test_short_scenarios_simply_get_fewer_wards(terrain):
    src = SyntheticSource(terrain, ScenarioSpec(seed=5, duration=100.0))
    _, truth = src.generate(src.match_ids()[0])
    assert truth.wards.size >= 1
    assert (truth.wards["t0"] < 100.0).all()


def test_corrupt_minion_time_pathology(synth_clean, synth_dirty):
    clean, _ = synth_clean
    dirty, _ = synth_dirty
    assert (clean.minions["t_valid"] == 1).all()
    assert (dirty.minions["t_valid"] == 0).all()
    # Denormal garbage, exactly as the real field contains.
    assert dirty.minions["t"].max() < C.MIN_VALID_PACKET_TIME


def test_keyframe_creates_repeat_identity_without_changing_it(synth_dirty):
    bundle, truth = synth_dirty
    assert bundle.heroes.size > C.N_HEROES
    assert np.unique(bundle.heroes["net_id"]).size == C.N_HEROES
    for nid in truth.net_ids:
        rows = bundle.heroes[bundle.heroes["net_id"] == nid]
        assert np.unique(rows["champion"]).size == 1


@pytest.mark.parametrize("field_name", ["quantise_time", "reorder_window"])
def test_clean_stream_has_neither_quantisation_nor_reordering(synth_clean, field_name):
    bundle, _ = synth_clean
    # Strictly non-decreasing, and not snapped to a 30 Hz lattice.
    assert (np.diff(bundle.casts["t"]) >= 0).all()
    residual = np.abs(bundle.casts["t"] * 30.0 - np.round(bundle.casts["t"] * 30.0))
    assert residual.max() > 1e-6, "clean times look quantised"
