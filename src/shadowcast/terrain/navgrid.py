"""Reader for League of Legends `.aimesh_ngrid` navigation grids, format v7.1.

This is the engine's own terrain, not a trace of a minimap image. It gives exact
wall, brush and see-through masks at 50 world units per cell, which is why terrain
acquisition is not on this project's critical path at all.

Layout, confirmed byte-for-byte against the Season 10 Summoner's Rift dump:

    offset  size            field
    0       u8              major version (7)
    1       u16             minor version (1)
    3       f32[3]          min_pos (x, y, z) -- y is height and we discard it
    15      f32[3]          max_pos
    27      f32             cell_size (50.0)
    31      u32             cell_count_x (295)
    35      u32             cell_count_z (296)
    39      48 B * n_cells  per-cell records, row-major with index = z*count_x + x
    ...     u16 * n_cells   vision/pathing flags
    ...                     region layers and hint data (unused here)

Only the flags matter to us. The 48-byte cell records carry heights and
pathfinding hints; League's fog of war is a 2-D problem and Riot's own
implementation ignores height too.

Format documented by TheKillerey/MapgeoAddon's `navgrid.py` and Pupix's 010
template. Neither is normative, so the reader validates aggressively and the
expected flag population is asserted in tests. A silently wrong offset would
produce a plausible-looking mask.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shadowcast import constants as C

__all__ = ["NavGrid", "NavGridFormatError", "read_navgrid"]

_HEADER_STRUCT = struct.Struct("<BH3f3ffII")
_HEADER_SIZE = _HEADER_STRUCT.size  # 39
_CELL_RECORD_SIZE = 48
_SUPPORTED_MAJOR = 7


class NavGridFormatError(ValueError):
    """The file is not a navgrid we know how to read."""


@dataclass(frozen=True, slots=True)
class NavGrid:
    """A parsed navgrid. `flags` is indexed [z, x], matching the file's row order."""

    major: int
    minor: int
    min_x: float
    min_z: float
    max_x: float
    max_z: float
    cell_size: float
    cells_x: int
    cells_z: int
    flags: np.ndarray  # uint16[cells_z, cells_x]

    # ---- flag channels -------------------------------------------------
    @property
    def brush(self) -> np.ndarray:
        """HAS_GRASS. Brush is a conditional occluder, not a wall."""
        return (self.flags & C.NGRID_HAS_GRASS) != 0

    @property
    def blocks_move(self) -> np.ndarray:
        """NOT_PASSABLE. What pathfinding treats as solid."""
        return (self.flags & C.NGRID_NOT_PASSABLE) != 0

    @property
    def see_through(self) -> np.ndarray:
        """SEE_THROUGH: blocks movement, transmits vision.

        Riot stamped these along wall diagonals after S5 Worlds, where "some line
        of sight oddities with regards to these diagonals surfaced", and later
        around structures once players found they could hide behind them.

        This is the flag a naive implementation misses, and missing it does not
        merely lose detail. It reproduces the exact bug Riot patched, because
        deriving vision from walkability makes every wall diagonal opaque.
        """
        return (self.flags & C.NGRID_SEE_THROUGH) != 0

    @property
    def blocks_vision(self) -> np.ndarray:
        """The vision channel: solid, unless explicitly marked see-through.

        Note this is NOT the complement of walkability. On Summoner's Rift 1,819
        cells are see-through, of which 136 are see-through while *also* being
        passable -- those are the base gates, which the engine treats as
        "you can see into and out of, but not through". We do not model the gate's
        directional semantics in v1; treating them as vision-transmitting is the
        closer of the two available approximations.
        """
        return self.blocks_move & ~self.see_through

    @property
    def walkable(self) -> np.ndarray:
        return ~self.blocks_move

    @property
    def n_cells(self) -> int:
        return self.cells_x * self.cells_z

    def describe(self) -> dict[str, object]:
        return {
            "version": f"{self.major}.{self.minor}",
            "cells": f"{self.cells_x}x{self.cells_z}",
            "cell_size": self.cell_size,
            "bounds_x": (self.min_x, self.max_x),
            "bounds_z": (self.min_z, self.max_z),
            "n_wall": int(self.blocks_move.sum()),
            "n_brush": int(self.brush.sum()),
            "n_see_through": int(self.see_through.sum()),
            "n_blocks_vision": int(self.blocks_vision.sum()),
            "walkable_fraction": float(self.walkable.mean()),
        }


def read_navgrid(path: str | Path) -> NavGrid:
    """Parse a navgrid file.

    Validates the header against the geometry we expect rather than trusting it,
    because every downstream number depends on having read the flags from the
    right offset, and a wrong offset yields noise that still looks like terrain.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < _HEADER_SIZE:
        raise NavGridFormatError(f"{path}: only {len(raw)} bytes, too short for a header")

    (
        major,
        minor,
        min_x,
        _min_y,
        min_z,
        max_x,
        _max_y,
        max_z,
        cell_size,
        cells_x,
        cells_z,
    ) = _HEADER_STRUCT.unpack_from(raw, 0)

    if major != _SUPPORTED_MAJOR:
        raise NavGridFormatError(
            f"{path}: navgrid major version {major}.{minor} is not supported "
            f"(this reader implements {_SUPPORTED_MAJOR}.x). The cell record size and "
            "field order differ between major versions."
        )
    if not (0 < cells_x < 4096 and 0 < cells_z < 4096):
        raise NavGridFormatError(f"{path}: implausible cell counts {cells_x}x{cells_z}")
    if not (0 < cell_size < 1000):
        raise NavGridFormatError(f"{path}: implausible cell size {cell_size}")

    n = cells_x * cells_z
    flags_offset = _HEADER_SIZE + n * _CELL_RECORD_SIZE
    flags_end = flags_offset + n * 2
    if len(raw) < flags_end:
        raise NavGridFormatError(
            f"{path}: {len(raw)} bytes but the flag block needs {flags_end}. "
            "Either the cell record size differs in this file or the header is wrong."
        )

    # The grid's own extent should agree with cell_count * cell_size to within a
    # cell. If it does not, our offset arithmetic is built on a misread header.
    span_x = max_x - min_x
    span_z = max_z - min_z
    for axis, span, count in (("x", span_x, cells_x), ("z", span_z, cells_z)):
        expected = count * cell_size
        if abs(span - expected) > cell_size:
            raise NavGridFormatError(
                f"{path}: {axis} span {span:.1f} disagrees with "
                f"{count} cells * {cell_size} = {expected:.1f} by more than one cell"
            )

    flags = np.frombuffer(raw, dtype="<u2", count=n, offset=flags_offset).reshape(cells_z, cells_x)

    return NavGrid(
        major=major,
        minor=minor,
        min_x=float(min_x),
        min_z=float(min_z),
        max_x=float(max_x),
        max_z=float(max_z),
        cell_size=float(cell_size),
        cells_x=int(cells_x),
        cells_z=int(cells_z),
        flags=np.ascontiguousarray(flags),
    )


def default_navgrid_path() -> Path:
    from shadowcast.config import data_dir

    return data_dir() / "terrain" / "AIPath_SRX.aimesh_ngrid"
