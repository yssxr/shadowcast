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

    # The shipped configuration is inside the ceiling.
    default = cfg.FilterSpec()
    assert 2 * math.log2(default.entropy_lattice) <= math.log2(default.particles) + 2.0


def test_baselines_cover_the_ablation():
    assert set(cfg.BASELINES) == {"uniform", "disc", "geodisc", "cv", "diffusion", "full"}
    # B3 vs Full must differ ONLY in the observation model, or the ablation does
    # not isolate negative information.
    b3 = dataclasses.asdict(cfg.BASELINES["diffusion"])
    full = dataclasses.asdict(cfg.BASELINES["full"])
    differing = {k for k in b3 if b3[k] != full[k]}
    assert differing == {"motion", "obs"}
    assert not cfg.BASELINES["diffusion"].uses_negative_information
    assert cfg.BASELINES["full"].uses_negative_information


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
