"""Tests for the navgrid reader.

The flag block sits at a computed offset behind a 4.2 MB block of cell records
whose size is documented by community reverse-engineering rather than by Riot. If
that offset were wrong we would be reading arbitrary bytes, which would still
produce a plausible-looking terrain mask, because noise at the right density looks
like walls. So the exact flag populations are asserted as a regression, and the
geometry is asserted independently of them.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.terrain.navgrid import NavGridFormatError, read_navgrid


def test_header_matches_the_documented_geometry(navgrid):
    assert (navgrid.major, navgrid.minor) == (7, 1)
    assert (navgrid.cells_x, navgrid.cells_z) == (C.NAVGRID_CELLS_X, C.NAVGRID_CELLS_Z)
    assert navgrid.cell_size == C.NAVGRID_CELL_SIZE
    assert navgrid.min_x == pytest.approx(C.NAVGRID_MIN_X, abs=1e-4)
    assert navgrid.min_z == pytest.approx(C.NAVGRID_MIN_Z, abs=1e-4)
    assert navgrid.max_x == pytest.approx(C.NAVGRID_MAX_X, abs=1e-3)
    assert navgrid.max_z == pytest.approx(C.NAVGRID_MAX_Z, abs=1e-3)


def test_flag_populations_are_exactly_as_measured(navgrid):
    """Regression on the flag-block offset.

    These are the counts read from the real file. A shifted offset changes all
    three, and no other check in the suite would notice.
    """
    assert int(navgrid.blocks_move.sum()) == 32365
    assert int(navgrid.brush.sum()) == 2129
    assert int(navgrid.see_through.sum()) == 1819
    assert navgrid.flags.shape == (C.NAVGRID_CELLS_Z, C.NAVGRID_CELLS_X)


def test_unused_flags_are_absent(navgrid):
    """MARKED, PATHED_ON and ALWAYS_VISIBLE are documented but unset on SR.

    If any of them suddenly had a population, we would be misreading the block,
    they are the canary for a half-word offset error, which would smear real flags
    into neighbouring bit positions.
    """
    assert int((navgrid.flags & C.NGRID_MARKED).astype(bool).sum()) == 0
    assert int((navgrid.flags & C.NGRID_PATHED_ON).astype(bool).sum()) == 0
    assert int((navgrid.flags & C.NGRID_ALWAYS_VISIBLE).astype(bool).sum()) == 0


def test_vision_is_not_the_complement_of_walkability(navgrid):
    """The whole reason terrain has three channels rather than two.

    1,819 cells block movement but transmit vision. Deriving one from the other
    would make every wall diagonal opaque, which is precisely the artefact Riot
    added these cells to fix.
    """
    assert not np.array_equal(navgrid.blocks_vision, navgrid.blocks_move)
    assert int(navgrid.blocks_vision.sum()) == 32365 - 1683
    # blocks_vision must be a strict subset of blocks_move: see-through only ever
    # removes opacity, never adds it.
    assert not (navgrid.blocks_vision & ~navgrid.blocks_move).any()


def test_passable_see_through_cells_are_the_base_gates(navgrid):
    """136 cells are see-through AND passable. The base gates.

    Worth pinning because they are the one place our model knowingly diverges: the
    engine treats a gate as "see into and out of, but not through", and we treat it
    as fully transmitting. Should that ever matter, this is the population it
    concerns.
    """
    gates = navgrid.see_through & ~navgrid.blocks_move
    assert int(gates.sum()) == 136


def test_walkable_space_is_a_single_connected_component(navgrid):
    """No orphaned walkable pockets.

    Strong independent evidence the flags were read correctly: misread bytes would
    scatter walkable cells into the out-of-bounds corners, which would show up as
    dozens of disconnected components rather than one.
    """
    walkable = navgrid.walkable
    seen = np.zeros_like(walkable)
    start = (navgrid.cells_z // 2, navgrid.cells_x // 2)
    assert walkable[start], "map centre should be walkable"

    stack = [start]
    seen[start] = True
    h, w = walkable.shape
    while stack:
        z, x = stack.pop()
        for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nz, nx = z + dz, x + dx
            if 0 <= nz < h and 0 <= nx < w and walkable[nz, nx] and not seen[nz, nx]:
                seen[nz, nx] = True
                stack.append((nz, nx))

    assert int(seen.sum()) == int(walkable.sum())


def test_walkable_fraction_is_stable(navgrid):
    assert navgrid.walkable.mean() == pytest.approx(C.SR_WALKABLE_FRACTION_MEASURED, abs=1e-4)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def _header_bytes(**over) -> bytes:
    vals = {
        "major": 7,
        "minor": 1,
        "min_x": 0.0,
        "min_y": 0.0,
        "min_z": 0.0,
        "max_x": 14750.0,
        "max_y": 0.0,
        "max_z": 14800.0,
        "cell_size": 50.0,
        "cells_x": 295,
        "cells_z": 296,
    }
    vals.update(over)
    return struct.Struct("<BH3f3ffII").pack(
        vals["major"],
        vals["minor"],
        vals["min_x"],
        vals["min_y"],
        vals["min_z"],
        vals["max_x"],
        vals["max_y"],
        vals["max_z"],
        vals["cell_size"],
        vals["cells_x"],
        vals["cells_z"],
    )


def test_rejects_a_truncated_file(tmp_path):
    p = tmp_path / "short.ngrid"
    p.write_bytes(b"\x07\x01\x00")
    with pytest.raises(NavGridFormatError, match="too short"):
        read_navgrid(p)


def test_rejects_an_unsupported_major_version(tmp_path):
    p = tmp_path / "v5.ngrid"
    p.write_bytes(_header_bytes(major=5) + b"\x00" * 1000)
    with pytest.raises(NavGridFormatError, match="not supported"):
        read_navgrid(p)


def test_rejects_a_header_whose_span_disagrees_with_its_cell_count(tmp_path):
    """The check that would catch a misread header before it becomes bad terrain."""
    p = tmp_path / "bad_span.ngrid"
    p.write_bytes(_header_bytes(max_x=999.0) + b"\x00" * (295 * 296 * 50))
    with pytest.raises(NavGridFormatError, match="disagrees"):
        read_navgrid(p)


def test_rejects_a_file_too_small_for_its_flag_block(tmp_path):
    p = tmp_path / "no_flags.ngrid"
    p.write_bytes(_header_bytes() + b"\x00" * 100)
    with pytest.raises(NavGridFormatError, match="flag block needs"):
        read_navgrid(p)
