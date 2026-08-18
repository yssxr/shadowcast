"""The precomputed visibility table. A cache, not a data structure.

For each walkable cell we store the field of view at RMAX, bit-packed over a
107x107 window. Because occlusion geometry is radius-independent, visibility at any
smaller radius is `FOV_max AND disc(r)`, so this one table serves every sight radius
in the game. That is what makes the whole approach tractable: the naive all-pairs
table at 512² would be 262144² bits ≈ 8.6 TB, and this is 286 MB.

**The table is a cache with a live-compute fallback, and that framing is the
important design decision.** `lookup` returns a sentinel on a miss and the caller
recomputes in ~34 µs. Coverage therefore becomes a pure performance knob and
correctness never depends on it, which dissolves several problems that would
otherwise need special cases:

- Sources in non-walkable cells: wall-hop dashes, Talon and Kayn traversal,
  Farsight wards thrown over walls. No need to dilate the source set into terrain,
  which would have nearly doubled the table.
- Sources whose *continuous* position sits in a brush their *cell* does not. At
  28.8 units per cell a champion 10 units inside a brush snaps to a non-brush cell,
  and since brush transparency is a discrete switch that error is large. The table
  stores the row for a cell's own brush id and refuses any other, so those sources
  take the live path, which removes the need for the per-(cell, brush) variant rows
  the design originally called for, an entire subsystem traded for a rare 34 µs.
- Runtime occluders: Trundle's pillar, Anivia's wall, Jarvan's flag.

Row layout is window-row-major with each 107-bit row padded to 2 uint64 words, and
the whole row padded to 216 words (1728 bytes) so every row starts 64-byte aligned.
That costs 1% over tight packing and buys a shifted-OR blit that needs no
cross-row bit juggling.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

from shadowcast import constants as C
from shadowcast.config import GridSpec, StageHeader, fov_table_dir
from shadowcast.fov.shadowcast import SCRATCH_FRAMES, fov_into
from shadowcast.geom.grid import radius_cells_sq
from shadowcast.terrain.terrain import Terrain

__all__ = [
    "MISS",
    "STAGE_VERSION",
    "FovTable",
    "build_table",
    "load_table",
    "row_words_for",
]

STAGE = "fov_table"
STAGE_VERSION = 1

#: `index` value meaning "no row for this cell; compute it live".
MISS = np.int32(-1)


def words_per_window_row(window: int) -> int:
    return (window + 63) // 64


def row_words_for(window: int, align_words: int = 8) -> int:
    """Words per table row, padded so each row starts 64-byte aligned."""
    raw = window * words_per_window_row(window)
    return int(math.ceil(raw / align_words) * align_words)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _fill_rows(
    cells: np.ndarray,  # int32[n] flat cell indices, ascending
    blocks_vision: np.ndarray,
    brush_id: np.ndarray,
    r2max: float,
    half: int,
    rows: np.ndarray,  # uint64[n, row_words]
    n_chunks: int,
    worst_per_chunk: np.ndarray,  # int64[n_chunks]
) -> None:
    """Compute and pack every table row.

    Scan-stack high-water marks go into a per-chunk array rather than a shared
    scalar: a `max` across prange iterations is not a reduction pattern Numba
    recognises, so writing to one variable would race and silently under-report.
    """
    n = cells.shape[0]
    grid = blocks_vision.shape[1]
    window = 2 * half + 1
    src_words = (window + 63) // 64

    for c in prange(n_chunks):
        lo = (n * c) // n_chunks
        hi = (n * (c + 1)) // n_chunks
        # Allocated per chunk rather than per cell: 165k allocations of an 11 KB
        # window would cost more than the field-of-view computations themselves.
        win = np.zeros((window, window), dtype=np.bool_)
        scratch = np.empty((SCRATCH_FRAMES, 3), dtype=np.float64)
        local_worst = 0

        for idx in range(lo, hi):
            k = cells[idx]
            j = k // grid
            i = k % grid
            win[:, :] = False
            depth = fov_into(
                win, blocks_vision, brush_id, i, j, brush_id[j, i], r2max, half, scratch
            )
            if depth > local_worst:
                local_worst = depth

            base = 0
            for wj in range(window):
                for wi in range(window):
                    if win[wj, wi]:
                        rows[idx, base + (wi >> 6)] |= np.uint64(1) << np.uint64(wi & 63)
                base += src_words

        worst_per_chunk[c] = local_worst


@dataclass(frozen=True, slots=True)
class FovTable:
    """A memory-mapped visibility table plus its cell index and radius discs."""

    grid: int
    half: int
    window: int
    src_words: int
    row_words: int
    index: np.ndarray  # int32[grid*grid], MISS where absent
    rows: np.ndarray  # uint64[n_rows, row_words], usually mmap'd
    radii: tuple[float, ...]
    discs: np.ndarray  # uint64[n_radii, window, src_words]
    header: StageHeader

    @property
    def n_rows(self) -> int:
        return int(self.rows.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.rows.shape[0]) * self.row_words * 8

    def radius_index(self, radius_units: float) -> int:
        """Index of a precomputed disc, or -1 if this radius was not precomputed.

        Radii are matched exactly rather than by nearest, because a silently
        substituted radius would shift every vision boundary by cells and look like
        a modelling difference rather than a lookup failure.
        """
        for n, r in enumerate(self.radii):
            if r == radius_units:
                return n
        return -1

    def lookup(self, cell: int, src_brush: int, cell_brush: int) -> int:
        """Row index for this source, or MISS if it must be computed live.

        `src_brush` is the brush the source is *actually* in, from its continuous
        position; `cell_brush` is what the snapped cell says. When they disagree the
        stored row is for the wrong occluder set, so the caller falls back.
        """
        if src_brush != cell_brush:
            return int(MISS)
        return int(self.index[cell])

    def describe(self) -> dict[str, object]:
        return {
            "grid": self.grid,
            "rows": f"{self.n_rows:,}",
            "window": f"{self.window}x{self.window}",
            "bytes_per_row": self.row_words * 8,
            "size_mb": round(self.nbytes / 1e6, 1),
            "radii": list(self.radii),
            "coverage": f"{self.n_rows / (self.grid * self.grid):.1%} of cells",
        }


#: Radii we precompute discs for: every sight radius in the game at this patch.
#: A radius absent from this list still works. It takes the live path.
DEFAULT_RADII: tuple[float, ...] = (
    C.FOG_ATTACK_REVEAL_RADIUS,
    C.SIGHT_GHOST_PORO,
    C.SIGHT_WARD_FARSIGHT,
    C.SIGHT_WARD_TOTEM,
    C.SIGHT_MINION,
    C.SIGHT_CHAMPION,
    C.RMAX_UNITS,
)


def _pack_discs(radii: tuple[float, ...], window: int, src_words: int) -> np.ndarray:
    from shadowcast.geom.bitset import pack_rows
    from shadowcast.geom.grid import disc_mask

    out = np.zeros((len(radii), window, src_words), dtype=np.uint64)
    for n, r in enumerate(radii):
        out[n] = pack_rows(disc_mask(r, window=window), row_words=src_words)
    return out


def build_table(
    terrain: Terrain,
    grid_spec: GridSpec | None = None,
    radii: tuple[float, ...] = DEFAULT_RADII,
    out_dir: Path | None = None,
    n_chunks: int = 64,
) -> FovTable:
    """Precompute the visibility table for a terrain and write it to disk."""
    grid = grid_spec if grid_spec is not None else GridSpec()
    if terrain.grid != grid.grid:
        raise ValueError(f"terrain grid {terrain.grid} != spec grid {grid.grid}")

    half = grid.rmax_cells
    window = 2 * half + 1
    src_words = words_per_window_row(window)
    row_words = row_words_for(window)

    cells = terrain.walkable_cells()
    dest = Path(out_dir) if out_dir else fov_table_dir(grid, terrain.spec)
    dest.mkdir(parents=True, exist_ok=True)

    # Written straight into a memmap: the table is 286 MB and there is no reason to
    # build it in RAM and then copy it out.
    rows = np.lib.format.open_memmap(
        dest / "rows.npy", mode="w+", dtype=np.uint64, shape=(cells.size, row_words)
    )
    worst_per_chunk = np.zeros(n_chunks, dtype=np.int64)
    _fill_rows(
        cells,
        terrain.blocks_vision,
        terrain.brush_id,
        radius_cells_sq(grid.rmax_units),
        half,
        np.asarray(rows),
        n_chunks,
        worst_per_chunk,
    )
    worst_depth = int(worst_per_chunk.max())
    if worst_depth >= SCRATCH_FRAMES:
        raise RuntimeError(
            f"scan stack exhausted while building the table (depth {worst_depth}); "
            "rows are incomplete. Raise SCRATCH_FRAMES."
        )
    rows.flush()

    index = np.full(grid.n_cells, MISS, dtype=np.int32)
    index[cells] = np.arange(cells.size, dtype=np.int32)
    np.save(dest / "index.npy", index)

    discs = _pack_discs(radii, window, src_words)
    np.save(dest / "discs.npy", discs)

    header = StageHeader(
        stage=STAGE,
        stage_version=STAGE_VERSION,
        config_hash=grid.content_hash,
        input_hash=terrain.spec.content_hash,
        extra={
            "half": half,
            "window": window,
            "src_words": src_words,
            "row_words": row_words,
            "radii": list(radii),
            "n_rows": int(cells.size),
            "worst_scan_depth": int(worst_depth),
            "scratch_frames": SCRATCH_FRAMES,
            "dir": str(dest),
        },
    )
    (dest / "meta.json").write_text(json.dumps(header.to_dict(), indent=2))

    return FovTable(
        grid=grid.grid,
        half=half,
        window=window,
        src_words=src_words,
        row_words=row_words,
        index=index,
        rows=rows,
        radii=tuple(radii),
        discs=discs,
        header=header,
    )


def load_table(
    terrain: Terrain,
    grid_spec: GridSpec | None = None,
    table_dir: Path | None = None,
) -> FovTable:
    """Memory-map an existing table, refusing one that does not match the terrain.

    The refusal is the point. A table built from different terrain produces masks
    that are subtly wrong and never crash. They still look like masks, still union,
    and shift the validation numbers by a few percent in a way that reads as a
    modelling issue. So the header is validated against the terrain's hash and a
    mismatch raises.
    """
    from shadowcast.config import StaleArtifactError

    grid = grid_spec if grid_spec is not None else GridSpec()
    src = Path(table_dir) if table_dir else fov_table_dir(grid, terrain.spec)
    meta_path = src / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no FOV table at {src}. Build it with `shadowcast fov build`.")
    header = StageHeader.from_dict(json.loads(meta_path.read_text()))
    header.validate_against(config_hash=grid.content_hash, input_hash=terrain.spec.content_hash)
    if header.stage_version != STAGE_VERSION:
        raise StaleArtifactError(
            f"table at {src} is stage version {header.stage_version}, "
            f"this build expects {STAGE_VERSION}"
        )

    extra = header.extra
    return FovTable(
        grid=grid.grid,
        half=int(extra["half"]),
        window=int(extra["window"]),
        src_words=int(extra["src_words"]),
        row_words=int(extra["row_words"]),
        index=np.load(src / "index.npy"),
        rows=np.load(src / "rows.npy", mmap_mode="r"),
        radii=tuple(float(r) for r in extra["radii"]),
        discs=np.load(src / "discs.npy"),
        header=header,
    )
