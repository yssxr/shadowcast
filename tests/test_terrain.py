"""Tests for terrain resampling and brush grouping."""

from __future__ import annotations

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.config import GridSpec, TerrainSpec
from shadowcast.geom.grid import cell_to_world
from shadowcast.terrain.terrain import NO_BRUSH, Terrain, build_terrain


def test_shape_and_dtypes(terrain):
    g = terrain.grid
    assert g == C.GRID
    assert terrain.walkable.shape == (g, g)
    assert terrain.blocks_vision.shape == (g, g)
    assert terrain.brush_id.shape == (g, g)
    assert terrain.walkable.dtype == bool
    assert terrain.blocks_vision.dtype == bool
    assert terrain.brush_id.dtype == np.int16


def test_resampling_is_exact(terrain, navgrid):
    """Every simulation cell must carry its containing navgrid cell's value.

    This is the claim that lets us skip area-majority rasterisation and topology
    repair entirely: the simulation grid (28.83 u) is FINER than the source
    (50 u), so this is an upsample and no information can be lost. If it were a
    downsample, a brush entrance could seal and vision would stop leaking through
    a real corridor — silently.

    Checked by going the long way round: take each simulation cell's world-space
    centre, independently locate the navgrid cell containing it, and compare.
    """
    rng = np.random.default_rng(7)
    js = rng.integers(0, terrain.grid, size=20000)
    is_ = rng.integers(0, terrain.grid, size=20000)

    x, z = cell_to_world(is_, js)
    gx = np.clip(((x - navgrid.min_x) / navgrid.cell_size).astype(np.int32), 0, navgrid.cells_x - 1)
    gz = np.clip(((z - navgrid.min_z) / navgrid.cell_size).astype(np.int32), 0, navgrid.cells_z - 1)

    np.testing.assert_array_equal(terrain.walkable[js, is_], navgrid.walkable[gz, gx])
    np.testing.assert_array_equal(terrain.blocks_vision[js, is_], navgrid.blocks_vision[gz, gx])
    # Brush drops the wall-overlapping cells; see test_brush_cells_are_walkable.
    np.testing.assert_array_equal(
        terrain.brush[js, is_], navgrid.brush[gz, gx] & ~navgrid.blocks_move[gz, gx]
    )


def test_vision_channel_stays_distinct_from_walkability(terrain):
    assert not np.array_equal(terrain.blocks_vision, ~terrain.walkable)
    # See-through only ever removes opacity from a wall, never adds it to open ground.
    assert not (terrain.blocks_vision & terrain.walkable).any()


def test_brush_groups_are_in_band(terrain):
    """Catches a raster that fuses patches or shatters one.

    Not an equality check against the wiki's 39: labelling gives 40 on this
    navgrid and we have not chased the one-patch difference down, so asserting 39
    would fail on correct output. What the band actually rules out is the two
    failure modes that break conditional-occluder semantics — far fewer groups
    means two brushes were welded together and standing in one reveals the other;
    far more means one brush was split and it stops behaving as a unit.
    """
    lo = C.SR_BRUSH_PATCHES_DOCUMENTED - 3
    hi = C.SR_BRUSH_PATCHES_DOCUMENTED + 6
    assert lo <= terrain.n_brush_groups <= hi
    assert terrain.n_brush_groups == C.SR_BRUSH_PATCHES_MEASURED


def test_brush_ids_are_dense_and_sentinel_is_negative(terrain):
    ids = np.unique(terrain.brush_id)
    assert ids[0] == NO_BRUSH
    np.testing.assert_array_equal(ids[1:], np.arange(terrain.n_brush_groups, dtype=np.int16))


def test_every_brush_group_is_connected(terrain):
    """A group must be one blob. If a label spans two blobs, the flood fill is wrong
    and a source in one would see through the other."""
    for gid in range(terrain.n_brush_groups):
        mask = terrain.brush_id == gid
        js, is_ = np.nonzero(mask)
        seen = np.zeros_like(mask)
        stack = [(int(js[0]), int(is_[0]))]
        seen[stack[0]] = True
        h, w = mask.shape
        while stack:
            j, i = stack.pop()
            for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nj, ni = j + dj, i + di
                if 0 <= nj < h and 0 <= ni < w and mask[nj, ni] and not seen[nj, ni]:
                    seen[nj, ni] = True
                    stack.append((nj, ni))
        assert int(seen.sum()) == int(mask.sum()), f"brush group {gid} is not connected"


def test_brush_cells_are_walkable(terrain):
    """`brush_id` is set only where brush semantics can actually apply.

    The navgrid disagrees: 251 of its 2,129 brush cells also carry NOT_PASSABLE,
    where the 50-unit raster straddles a brush/terrain boundary. Those cells are
    opaque whatever their brush flag says, so we drop them at build time — which
    both keeps this invariant and removes any chance of two patches being welded
    through a wall cell.
    """
    assert terrain.walkable[terrain.brush].all()


def test_wall_brush_cells_are_excluded_but_present_in_the_source(terrain, navgrid):
    """Pins the count, so a future navgrid with a different overlap is noticed."""
    overlap = navgrid.brush & navgrid.blocks_move
    assert int(overlap.sum()) == 251
    assert not (overlap & navgrid.see_through).any()
    # Dropping them must not change the grouping — verified for this navgrid.
    assert terrain.n_brush_groups == C.SR_BRUSH_PATCHES_MEASURED


def test_walkable_fraction_and_table_size(terrain):
    assert terrain.walkable_fraction == pytest.approx(0.6316, abs=1e-3)
    assert terrain.n_walkable == terrain.fov_table_rows()
    # The measured fraction is well above the 25-40% the table was budgeted
    # against, so the real table is ~286 MB rather than ~160 MB. Pinned here so a
    # future grid change makes the memory consequence visible immediately.
    assert 160_000 < terrain.n_walkable < 175_000


def test_walkable_cells_are_sorted_flat_indices(terrain):
    cells = terrain.walkable_cells()
    assert cells.dtype == np.int32
    assert (np.diff(cells) > 0).all()
    assert cells.size == terrain.n_walkable
    j, i = np.divmod(cells, terrain.grid)
    assert terrain.walkable[j, i].all()


def test_save_load_round_trip(terrain, tmp_path):
    path = terrain.save(tmp_path / "t.npz")
    back = Terrain.load(path)
    np.testing.assert_array_equal(back.walkable, terrain.walkable)
    np.testing.assert_array_equal(back.blocks_vision, terrain.blocks_vision)
    np.testing.assert_array_equal(back.brush_id, terrain.brush_id)
    assert back.n_brush_groups == terrain.n_brush_groups
    assert back.spec.content_hash == terrain.spec.content_hash
    assert back.header.input_hash == terrain.header.input_hash


def test_terrain_hash_tracks_the_navgrid_bytes(terrain):
    """The source file's bytes are part of terrain identity, so a different dump
    orphans every artifact built from the old one."""
    assert terrain.spec.navgrid_hash
    assert terrain.header.input_hash == terrain.spec.navgrid_hash
    assert TerrainSpec().content_hash != terrain.spec.content_hash


def test_see_through_ablation_changes_the_vision_channel(navgrid_path):
    """The ablation switch must actually do something, or measuring both ways is
    meaningless."""
    opaque = build_terrain(
        navgrid_path=navgrid_path,
        spec=TerrainSpec(see_through_transmits_vision=False),
    )
    transparent = build_terrain(navgrid_path=navgrid_path, spec=TerrainSpec())
    assert opaque.blocks_vision.sum() > transparent.blocks_vision.sum()
    np.testing.assert_array_equal(opaque.walkable, transparent.walkable)
    assert opaque.spec.content_hash != transparent.spec.content_hash


def test_build_reports_a_useful_error_when_the_navgrid_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="curl"):
        build_terrain(navgrid_path=tmp_path / "nope.aimesh_ngrid")


def test_grid_spec_is_recorded(terrain):
    assert terrain.grid_spec == GridSpec()
