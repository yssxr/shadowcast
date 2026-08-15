"""Tests for the config specs and the staleness machinery."""

from __future__ import annotations

import dataclasses
import math

import pytest

from shadowcast import config as cfg
from shadowcast import constants as C


def test_grid_spec_derived_values():
    g = cfg.GridSpec()
    assert g.grid == 512
    assert g.cell_size == pytest.approx(C.GRID_CELL_SIZE)
    assert g.rmax_cells == C.RMAX_CELLS
    assert g.fov_window == C.FOV_WINDOW
    assert g.n_cells == 512 * 512


def test_tick_spec_derived_values():
    t = cfg.TickSpec()
    assert t.dt == pytest.approx(1 / 8)
    assert t.n_ticks == C.MATCH_TICKS


def test_specs_are_frozen():
    """A spec must not be mutable, or its hash could drift from its values."""
    g = cfg.GridSpec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.grid = 256  # type: ignore[misc]


def test_content_hash_is_stable_and_value_dependent():
    a = cfg.GridSpec()
    b = cfg.GridSpec()
    assert a.content_hash == b.content_hash
    assert cfg.GridSpec(grid=256).content_hash != a.content_hash
    # Different spec types with coincidentally identical fields must not collide.
    assert cfg.TickSpec().content_hash != cfg.ExportSpec().content_hash


def test_content_hash_is_order_independent():
    assert cfg.content_hash({"a": 1, "b": 2}) == cfg.content_hash({"b": 2, "a": 1})


def test_filter_spec_rejects_a_lattice_that_outruns_the_particle_budget():
    """The guard that stopped us shipping entropy-as-particle-count.

    Plug-in entropy of P particles saturates at log2(P). A 128^2 lattice admits up
    to 14 bits, so with 400 particles the reported entropy would be reporting the
    sample size rather than the belief. The spec refuses to be constructed.
    """
    with pytest.raises(ValueError, match="measure the particle count"):
        cfg.FilterSpec(entropy_lattice=128, particles=400)

    # The shipped configuration is inside the ceiling, with NO slack. An earlier
    # version of this assertion allowed two bits of headroom, which let a broken
    # configuration through: 32^2 has 890 walkable bins and so 9.80 bits, while
    # 400 particles cap the plug-in estimator at 8.64. Measured entropy of a
    # uniform belief came out at 8.74 bits — pinned, and reporting the particle
    # budget rather than the game.
    default = cfg.FilterSpec()
    assert 2 * math.log2(default.entropy_lattice) <= math.log2(default.particles)


def test_baselines_cover_the_ablation():
    assert set(cfg.BASELINES) == {
        "uniform",
        "disc",
        "geodisc",
        "cv",
        "diffusion",
        "behavioural",
        "full",
    }
    assert not cfg.BASELINES["behavioural"].uses_negative_information
    assert cfg.BASELINES["full"].uses_negative_information


def test_each_ablation_pair_isolates_exactly_one_change():
    """The property the whole table depends on, and which it used to violate.

    This test previously asserted `differing == {"motion", "obs"}` for the headline
    comparison — that is, it asserted the ablation did *not* isolate negative
    information, since a win could have come from the motion model instead. The fix was
    a seventh baseline, `behavioural`, which is `full` with the observation model turned
    down and nothing else changed.
    """
    for a, b in (("diffusion", "behavioural"), cfg.THESIS_PAIR):
        left = dataclasses.asdict(cfg.BASELINES[a])
        right = dataclasses.asdict(cfg.BASELINES[b])
        differing = {k for k in left if left[k] != right[k]}
        assert len(differing) == 1, f"{a} vs {b} differ in {sorted(differing)}"
    assert cfg.THESIS_PAIR == ("behavioural", "full")


def test_stage_header_round_trip():
    h = cfg.StageHeader(stage="terrain", stage_version=1, config_hash="abc", input_hash="def")
    again = cfg.StageHeader.from_dict(h.to_dict())
    assert again.stage == "terrain"
    assert again.config_hash == "abc"
    assert again.git_sha == h.git_sha


def test_stage_header_from_dict_ignores_unknown_keys():
    """Forward compatibility: a newer writer's extra keys must not crash a reader."""
    h = cfg.StageHeader.from_dict(
        {
            "stage": "fov",
            "stage_version": 2,
            "config_hash": "x",
            "input_hash": "y",
            "some_future_field": 123,
        }
    )
    assert h.stage == "fov"


def test_stale_config_raises():
    h = cfg.StageHeader(stage="fov", stage_version=1, config_hash="old", input_hash="in")
    with pytest.raises(cfg.StaleArtifactError, match="config"):
        h.validate_against(config_hash="new", input_hash="in")


def test_stale_input_raises():
    h = cfg.StageHeader(stage="fov", stage_version=1, config_hash="c", input_hash="old")
    with pytest.raises(cfg.StaleArtifactError, match="input"):
        h.validate_against(config_hash="c", input_hash="new")


def test_matching_header_validates_silently():
    h = cfg.StageHeader(stage="fov", stage_version=1, config_hash="c", input_hash="i")
    h.validate_against(config_hash="c", input_hash="i")


def test_fov_table_path_changes_with_terrain():
    """A terrain change must orphan the table by NAME, not merely flag it.

    If the path were stable, an interrupted rebuild could leave a half-written
    table where a valid one used to be. Keying the directory means the old table
    stays intact and the new one is written beside it.
    """
    g = cfg.GridSpec()
    t1 = cfg.TerrainSpec(navgrid_hash="aaaa")
    t2 = cfg.TerrainSpec(navgrid_hash="bbbb")
    assert cfg.fov_table_dir(g, t1) != cfg.fov_table_dir(g, t2)
    assert cfg.fov_table_dir(g, t1) == cfg.fov_table_dir(g, cfg.TerrainSpec(navgrid_hash="aaaa"))


def test_git_sha_is_reported():
    sha = cfg.git_sha()
    assert isinstance(sha, str)
    assert sha
    # In this repo it should resolve; "unknown" would mean the git probe broke.
    assert sha != "unknown"
