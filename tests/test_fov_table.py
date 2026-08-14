"""Tests for the precomputed visibility table and the mask union kernel.

The table is the reason this project is tractable, and the reason it is dangerous:
a table that does not match its terrain yields masks that are subtly wrong and never
crash. So the tests here fall into two groups — does a row equal a fresh
computation, and does the table refuse to be used with the wrong inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.config import GridSpec, StaleArtifactError, TerrainSpec
from shadowcast.fov.shadowcast import fov_bool
from shadowcast.fov.table import MISS, build_table, load_table, row_words_for
from shadowcast.fov.union import (
    assemble,
    mask_popcount,
    mask_to_bool,
    new_mask,
    union_sources,
    union_sources_ref,
)
from shadowcast.geom.bitset import unpack_rows
from shadowcast.geom.grid import disc_mask


@pytest.fixture(scope="module")
def table_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("fov")


@pytest.fixture(scope="module")
def table(terrain, table_dir):
    """One table for the whole module — building it takes about four seconds."""
    return build_table(terrain, out_dir=table_dir)


def _row_window(table, row: int) -> np.ndarray:
    packed = np.asarray(table.rows[row])[: table.window * table.src_words]
    return unpack_rows(packed.reshape(table.window, table.src_words), table.window)


def test_row_words_alignment():
    """Rows start 64-byte aligned, which is what the blit's row indexing assumes."""
    assert row_words_for(107) == 216
    assert row_words_for(107) % 8 == 0
    assert row_words_for(107) >= 107 * 2


def test_geometry_matches_the_spec(table):
    assert table.window == C.FOV_WINDOW
    assert table.half == C.RMAX_CELLS
    assert table.src_words == 2
    assert table.row_words == 216
    assert table.nbytes == table.n_rows * 1728


def test_every_row_equals_a_fresh_computation(table, terrain):
    """The table is only useful if a row IS the field of view."""
    rng = np.random.default_rng(3)
    picks = rng.choice(terrain.walkable_cells(), size=150, replace=False)
    for k in picks:
        j, i = divmod(int(k), terrain.grid)
        brush = int(terrain.brush_id[j, i])
        row = table.lookup(int(k), brush, brush)
        assert row >= 0
        want = fov_bool(terrain, i, j, C.RMAX_UNITS, half=table.half)
        np.testing.assert_array_equal(_row_window(table, row), want)


def test_table_row_and_disc_reproduces_a_direct_computation(table, terrain):
    """Radius separability, exercised end to end THROUGH the table.

    `test_radius_monotonicity_is_exact` proves the property about the algorithm; this
    proves the stored bytes and the packed discs actually deliver it. A transposed
    disc, an off-by-one in the row stride, or a mismatched bit order would all pass
    the former and fail here.
    """
    rng = np.random.default_rng(4)
    picks = rng.choice(terrain.walkable_cells(), size=60, replace=False)
    for k in picks:
        j, i = divmod(int(k), terrain.grid)
        brush = int(terrain.brush_id[j, i])
        row = table.lookup(int(k), brush, brush)
        stored = _row_window(table, row)
        for radius in (C.SIGHT_WARD_TOTEM, C.SIGHT_MINION, C.SIGHT_CHAMPION):
            want = fov_bool(terrain, i, j, radius, half=table.half)
            got = stored & disc_mask(radius, window=table.window)
            np.testing.assert_array_equal(got, want)


def test_packed_discs_match_the_boolean_discs(table):
    for n, radius in enumerate(table.radii):
        got = unpack_rows(np.asarray(table.discs[n]), table.window)
        np.testing.assert_array_equal(got, disc_mask(radius, window=table.window))


def test_lookup_misses_for_vision_blocking_cells(table, terrain):
    """A source in a wall is a table miss, not a wrong answer.

    Walls are excluded from the table, and the live fallback is what makes that safe
    — wall-hop dashes and over-wall Farsight wards put real sources there.
    """
    j, i = np.unravel_index(int(np.flatnonzero(~terrain.walkable.ravel())[0]), terrain.walkable.shape)
    assert table.lookup(int(j) * terrain.grid + int(i), -1, -1) == MISS


def test_lookup_misses_when_source_brush_disagrees_with_the_cell(table, terrain):
    """The mechanism that replaced per-(cell, brush) variant rows.

    A champion 10 units inside a brush snaps to a cell that may be classified as
    non-brush. Brush transparency is a discrete switch, so using the stored row would
    be badly wrong rather than slightly wrong. Refusing the row costs one 34 µs live
    computation and removes an entire subsystem.
    """
    brush_cell = int(np.flatnonzero(terrain.brush.ravel())[0])
    j, i = divmod(brush_cell, terrain.grid)
    real = int(terrain.brush_id[j, i])
    assert table.lookup(brush_cell, real, real) >= 0
    assert table.lookup(brush_cell, -1, real) == MISS


def test_radius_index_requires_an_exact_match(table):
    """No nearest-radius fallback.

    A silently substituted radius shifts every vision boundary by cells and would
    read as a modelling difference rather than a lookup failure.
    """
    assert table.radius_index(C.SIGHT_CHAMPION) >= 0
    assert table.radius_index(C.SIGHT_CHAMPION - 0.5) == -1
    assert table.radius_index(1234.5) == -1


def test_index_covers_exactly_the_walkable_cells(table, terrain):
    hits = table.index >= 0
    np.testing.assert_array_equal(hits.reshape(terrain.grid, terrain.grid), terrain.walkable)
    assert int(hits.sum()) == table.n_rows
    # Row ids must be a permutation of range(n_rows), or two cells share a row.
    assert sorted(table.index[hits].tolist()) == list(range(table.n_rows))


def test_worst_scan_depth_is_recorded_with_headroom(table):
    depth = int(table.header.extra["worst_scan_depth"])
    frames = int(table.header.extra["scratch_frames"])
    assert 0 < depth < frames // 4


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
def test_load_refuses_a_table_built_from_different_terrain(terrain, table, table_dir):
    """The refusal that prevents the project's worst failure mode.

    Wrong-terrain masks still look like masks, still union, still render, and move
    the validation numbers a few percent in a way that reads as a modelling issue
    rather than a bug. Nothing else in the pipeline would notice.
    """
    import dataclasses

    other = dataclasses.replace(
        terrain, spec=dataclasses.replace(terrain.spec, navgrid_hash="ff" * 8)
    )
    with pytest.raises(StaleArtifactError, match="input"):
        load_table(other, table_dir=table_dir)


def test_load_refuses_a_table_built_with_a_different_grid(terrain, table, table_dir):
    with pytest.raises(StaleArtifactError, match="config"):
        load_table(terrain, grid_spec=GridSpec(rmax_units=1200.0), table_dir=table_dir)


def test_load_reports_a_missing_table_usefully(terrain, tmp_path):
    with pytest.raises(FileNotFoundError, match="shadowcast fov build"):
        load_table(terrain, table_dir=tmp_path / "absent")


def test_round_trip_through_disk(terrain, table, table_dir):
    reloaded = load_table(terrain, table_dir=table_dir)
    assert reloaded.n_rows == table.n_rows
    assert reloaded.window == C.FOV_WINDOW
    j, i = divmod(int(terrain.walkable_cells()[1000]), terrain.grid)
    brush = int(terrain.brush_id[j, i])
    row = reloaded.lookup(int(j) * terrain.grid + int(i), brush, brush)
    np.testing.assert_array_equal(
        _row_window(reloaded, row), fov_bool(terrain, i, j, C.RMAX_UNITS, half=reloaded.half)
    )


# ---------------------------------------------------------------------------
# Union kernel
# ---------------------------------------------------------------------------
def _sources(table, terrain, rng, n, positions=None):
    rows, oi, oj, ri = [], [], [], []
    coords = positions if positions is not None else [
        divmod(int(k), terrain.grid) for k in rng.choice(terrain.walkable_cells(), size=n, replace=False)
    ]
    for j, i in coords:
        if not (0 <= i < terrain.grid and 0 <= j < terrain.grid) or not terrain.walkable[j, i]:
            continue
        brush = int(terrain.brush_id[j, i])
        row = table.lookup(j * terrain.grid + i, brush, brush)
        if row < 0:
            continue
        rows.append(row)
        oi.append(i - table.half)
        oj.append(j - table.half)
        ri.append(table.radius_index(float(rng.choice(list(table.radii[:-1])))))
    return (
        np.array(rows, dtype=np.int32),
        np.array(oi, dtype=np.int32),
        np.array(oj, dtype=np.int32),
        np.array(ri, dtype=np.int32),
    )


def test_union_kernel_matches_reference_on_random_sources(table, terrain):
    rng = np.random.default_rng(11)
    args = _sources(table, terrain, rng, 40)
    a, b = new_mask(terrain.grid), new_mask(terrain.grid)
    rows = np.asarray(table.rows)
    union_sources(rows, table.discs, *args, table.window, table.src_words, a)
    union_sources_ref(rows, table.discs, *args, table.window, table.src_words, b)
    np.testing.assert_array_equal(a, b)
    assert mask_popcount(a) > 0


def test_union_kernel_matches_reference_at_edges_and_word_boundaries(table, terrain):
    """Border overhang and shift-zero, together.

    Every source within 53 cells of a map edge overhangs, and any source whose window
    origin is a multiple of 64 exercises the `v >> 64` undefined-behaviour path. Both
    are enumerated rather than sampled, because those are precisely the cases a
    random sweep misses.
    """
    rng = np.random.default_rng(12)
    coords = [
        (j, i)
        for j in (0, 1, 5, 53, 256, 458, 506, 511)
        for i in (0, 1, 53, 64, 128, 192, 256, 320, 384, 448, 511)
    ]
    args = _sources(table, terrain, rng, 0, positions=coords)
    assert len(args[0]) > 20, "fixture should retain a useful number of edge sources"
    a, b = new_mask(terrain.grid), new_mask(terrain.grid)
    rows = np.asarray(table.rows)
    union_sources(rows, table.discs, *args, table.window, table.src_words, a)
    union_sources_ref(rows, table.discs, *args, table.window, table.src_words, b)
    np.testing.assert_array_equal(a, b)


def test_union_ignores_miss_rows(table, terrain):
    rows = np.asarray(table.rows)
    out = new_mask(terrain.grid)
    union_sources(
        rows,
        table.discs,
        np.array([MISS, MISS], dtype=np.int32),
        np.array([10, 20], dtype=np.int32),
        np.array([10, 20], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        table.window,
        table.src_words,
        out,
    )
    assert mask_popcount(out) == 0


def test_union_is_a_union(table, terrain):
    """Adding a source may only add lit cells, never remove them."""
    rng = np.random.default_rng(13)
    args = _sources(table, terrain, rng, 6)
    rows = np.asarray(table.rows)
    out = new_mask(terrain.grid)
    seen = 0
    for n in range(len(args[0])):
        one = tuple(a[n : n + 1] for a in args)
        before = mask_to_bool(out, terrain.grid).copy()
        union_sources(rows, table.discs, *one, table.window, table.src_words, out)
        after = mask_to_bool(out, terrain.grid)
        assert (after | before == after).all(), "a source removed visibility"
        assert after.sum() >= seen
        seen = int(after.sum())


def test_assemble_equals_a_direct_field_of_view_union(table, terrain):
    """The dispatch path, checked against the obvious construction."""
    rng = np.random.default_rng(14)
    picks = rng.choice(terrain.walkable_cells(), size=10, replace=False)
    sources = []
    for k in picks:
        j, i = divmod(int(k), terrain.grid)
        sources.append((i, j, C.SIGHT_CHAMPION, int(terrain.brush_id[j, i])))

    got = mask_to_bool(assemble(table, terrain, sources), terrain.grid)

    want = np.zeros((terrain.grid, terrain.grid), dtype=bool)
    for i, j, radius, brush in sources:
        win = fov_bool(terrain, i, j, radius, half=table.half, src_brush=brush)
        x0, y0 = i - table.half, j - table.half
        sj_lo, sj_hi = max(0, -y0), min(table.window, terrain.grid - y0)
        si_lo, si_hi = max(0, -x0), min(table.window, terrain.grid - x0)
        want[y0 + sj_lo : y0 + sj_hi, x0 + si_lo : x0 + si_hi] |= win[
            sj_lo:sj_hi, si_lo:si_hi
        ]
    np.testing.assert_array_equal(got, want)


def test_assemble_takes_the_live_path_for_a_brush_mismatch(table, terrain):
    """A source claiming to be in brush sees more than one claiming not to be.

    Confirms the fallback is not merely reached but produces the brush-aware answer.
    """
    brush_cell = int(np.flatnonzero(terrain.brush.ravel())[0])
    j, i = divmod(brush_cell, terrain.grid)
    outside = assemble(table, terrain, [(i, j, C.SIGHT_CHAMPION, -1)])
    inside = assemble(table, terrain, [(i, j, C.SIGHT_CHAMPION, int(terrain.brush_id[j, i]))])
    assert mask_popcount(inside) > mask_popcount(outside)


def test_assemble_falls_back_for_an_unprecomputed_radius(table, terrain):
    """An arbitrary radius still works — it just does not use the table."""
    j, i = divmod(int(terrain.walkable_cells()[5000]), terrain.grid)
    got = mask_to_bool(
        assemble(table, terrain, [(i, j, 1111.0, int(terrain.brush_id[j, i]))]), terrain.grid
    )
    win = fov_bool(terrain, i, j, 1111.0, half=table.half)
    assert got.sum() == win.sum()
