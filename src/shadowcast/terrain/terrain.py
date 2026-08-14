"""The simulation's terrain: three channels on the 512² grid, plus brush groups.

Built by *upsampling* the navgrid, and that direction matters. The navgrid is
50 world units per cell; our grid is 28.83. Every grid cell therefore falls
entirely inside exactly one navgrid cell, so point-sampling the containing cell is
exact — no information is lost, no brush entrance can be sealed, and no topology
repair is needed. The elaborate area-majority-plus-repair machinery that a
*downsample* would demand is simply not applicable here, and adding it would
invent precision the source does not have.

What the finer grid actually buys is rounder radius discs and less source-snapping
error, not more terrain detail. The effective resolution of this terrain is 50
units and the write-up should say so.
"""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shadowcast import constants as C
from shadowcast.config import GridSpec, StageHeader, TerrainSpec, file_hash
from shadowcast.terrain.navgrid import NavGrid, read_navgrid

__all__ = ["NO_BRUSH", "Terrain", "build_terrain"]

STAGE = "terrain"
STAGE_VERSION = 1

#: brush_id sentinel for "not in any brush".
NO_BRUSH = np.int16(-1)


@dataclass(frozen=True, slots=True)
class Terrain:
    """Terrain on the simulation grid. All arrays are [j, i] with j the z axis.

    Three channels, not two. `walkable` and `blocks_vision` are deliberately not
    complements of one another: League marks cells that block movement but
    transmit vision (see `NavGrid.see_through`), and conflating the two
    reintroduces the diagonal line-of-sight artefacts Riot patched after S5
    Worlds.
    """

    grid: int
    walkable: np.ndarray  # bool[grid, grid]
    blocks_vision: np.ndarray  # bool[grid, grid]
    brush_id: np.ndarray  # int16[grid, grid], NO_BRUSH where absent
    n_brush_groups: int
    header: StageHeader
    spec: TerrainSpec
    grid_spec: GridSpec

    # ---- derived ------------------------------------------------------
    @property
    def brush(self) -> np.ndarray:
        return self.brush_id != NO_BRUSH

    @property
    def n_walkable(self) -> int:
        return int(self.walkable.sum())

    @property
    def walkable_fraction(self) -> float:
        return float(self.walkable.mean())

    def walkable_cells(self) -> np.ndarray:
        """Flat indices of walkable cells, ascending. The FOV table's row set."""
        return np.flatnonzero(self.walkable.ravel()).astype(np.int32)

    def brush_at(self, i: int, j: int) -> int:
        return int(self.brush_id[j, i])

    def fov_table_rows(self) -> int:
        """How many rows a full FOV table would need for this terrain."""
        return self.n_walkable

    def describe(self) -> dict[str, object]:
        return {
            "grid": self.grid,
            "cell_size_units": self.grid_spec.cell_size,
            "walkable_cells": self.n_walkable,
            "walkable_fraction": self.walkable_fraction,
            "vision_blocking_cells": int(self.blocks_vision.sum()),
            "brush_cells": int(self.brush.sum()),
            "brush_groups": self.n_brush_groups,
            "terrain_hash": self.spec.content_hash,
            "source": self.spec.source,
        }

    # ---- persistence --------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            walkable=np.packbits(self.walkable, axis=None),
            blocks_vision=np.packbits(self.blocks_vision, axis=None),
            brush_id=self.brush_id,
            meta=np.frombuffer(
                json.dumps(
                    {
                        "grid": self.grid,
                        "n_brush_groups": self.n_brush_groups,
                        "header": self.header.to_dict(),
                        "spec": self.spec.to_dict(),
                        "grid_spec": self.grid_spec.to_dict(),
                    }
                ).encode("utf-8"),
                dtype=np.uint8,
            ),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Terrain:
        with np.load(path) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            grid = int(meta["grid"])
            n = grid * grid
            walkable = np.unpackbits(z["walkable"], count=n).astype(bool).reshape(grid, grid)
            blocks_vision = (
                np.unpackbits(z["blocks_vision"], count=n).astype(bool).reshape(grid, grid)
            )
            brush_id = z["brush_id"].astype(np.int16)
        return cls(
            grid=grid,
            walkable=walkable,
            blocks_vision=blocks_vision,
            brush_id=brush_id,
            n_brush_groups=int(meta["n_brush_groups"]),
            header=StageHeader.from_dict(meta["header"]),
            spec=TerrainSpec(**meta["spec"]),
            grid_spec=GridSpec(**meta["grid_spec"]),
        )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _navgrid_indices(grid_spec: GridSpec, ng: NavGrid) -> tuple[np.ndarray, np.ndarray]:
    """For each simulation cell centre, the navgrid cell containing it.

    Clamped, because the simulation grid spans a square region (so cells stay
    square) while the navgrid's extent is very slightly non-square: the grid
    overruns the navgrid's maximum x by 40 units, about 1.4 cells. Those cells sit
    outside the source data and clamping maps them onto the nearest real edge,
    which is a wall there in any case.
    """
    cs = grid_spec.cell_size
    centres = (np.arange(grid_spec.grid, dtype=np.float64) + 0.5) * cs
    gx = np.clip(
        ((centres + grid_spec.world_min_x - ng.min_x) / ng.cell_size).astype(np.int32),
        0,
        ng.cells_x - 1,
    )
    gz = np.clip(
        ((centres + grid_spec.world_min_z - ng.min_z) / ng.cell_size).astype(np.int32),
        0,
        ng.cells_z - 1,
    )
    return gx, gz


def _label_brush_groups(brush: np.ndarray, diagonal: bool = True) -> tuple[np.ndarray, int]:
    """Label connected components of the brush mask, 1-based; 0 means no brush.

    Brush identity is what makes the occluder conditional: a source inside brush B
    sees through B but not into any *other* brush. Getting the grouping wrong is
    therefore not a cosmetic problem. Two logically distinct patches labelled as
    one would make standing in either reveal both.

    MEASURED on the S10 Summoner's Rift navgrid: 4- and 8-connectivity give an
    identical 40 components, so no two patches are even diagonally adjacent and the
    connectivity choice is moot here. 8-connectivity is kept because it is the
    conservative direction for the failure we care about — it fuses rather than
    shatters, and a fused pair is caught by the count assertion while a shattered
    patch would silently make one brush behave as several.
    """
    h, w = brush.shape
    labels = np.zeros((h, w), dtype=np.int32)
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        neigh += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    n_groups = 0
    for sj in range(h):
        for si in range(w):
            if not brush[sj, si] or labels[sj, si]:
                continue
            n_groups += 1
            labels[sj, si] = n_groups
            q: deque[tuple[int, int]] = deque([(sj, si)])
            while q:
                j, i = q.popleft()
                for dj, di in neigh:
                    nj, ni = j + dj, i + di
                    if 0 <= nj < h and 0 <= ni < w and brush[nj, ni] and not labels[nj, ni]:
                        labels[nj, ni] = n_groups
                        q.append((nj, ni))
    return labels, n_groups


def build_terrain(
    navgrid_path: str | Path | None = None,
    grid_spec: GridSpec | None = None,
    spec: TerrainSpec | None = None,
) -> Terrain:
    """Parse the navgrid and resample it onto the simulation grid."""
    from shadowcast.terrain.navgrid import default_navgrid_path

    navgrid_path = Path(navgrid_path) if navgrid_path else default_navgrid_path()
    if not navgrid_path.exists():
        raise FileNotFoundError(
            f"{navgrid_path} not found. Fetch it with:\n"
            f"  mkdir -p {navgrid_path.parent} && curl -L -o {navgrid_path} \\\n"
            f'    "{C.NAVGRID_URL}"'
        )
    grid = grid_spec if grid_spec is not None else GridSpec()
    ng = read_navgrid(navgrid_path)

    src_hash = file_hash(navgrid_path)
    # The navgrid's own bytes are part of the terrain's identity, so a different
    # dump produces a different hash and orphans every artifact built from it.
    resolved = dataclasses.replace(
        spec if spec is not None else TerrainSpec(), navgrid_hash=src_hash
    )

    gx, gz = _navgrid_indices(grid, ng)
    # Outer product of the two index vectors: row j takes navgrid row gz[j],
    # column i takes navgrid column gx[i].
    sel = np.ix_(gz, gx)

    walkable = ng.walkable[sel]
    if resolved.see_through_transmits_vision:
        blocks_vision = ng.blocks_vision[sel]
    else:
        # Ablation: treat every wall as opaque, ignoring see-through. Kept so the
        # fog-agreement rate can be measured both ways rather than asserted.
        blocks_vision = ng.blocks_move[sel]
    # Brush that is also a wall is a wall.
    #
    # MEASURED: 251 of the navgrid's 2,129 HAS_GRASS cells also carry
    # NOT_PASSABLE, and none of those is see-through. They are the 50-unit raster
    # straddling a brush/terrain boundary — brush painted up against a wall.
    #
    # Such a cell is opaque no matter what its brush flag says, so its brush
    # membership can never affect a vision decision. Excluding it keeps the
    # invariant "brush_id is set only where brush semantics apply", and removes any
    # possibility of two patches being welded together through a wall cell, which
    # would make standing in one reveal the other. Verified not to change the
    # grouping: 40 components either way, under both 4- and 8-connectivity.
    brush_src = ng.brush[sel] & ~ng.blocks_move[sel]

    labels, n_groups = _label_brush_groups(brush_src)
    brush_id = (labels.astype(np.int16) - 1).astype(np.int16)  # 0 -> -1 == NO_BRUSH

    header = StageHeader(
        stage=STAGE,
        stage_version=STAGE_VERSION,
        config_hash=resolved.content_hash,
        input_hash=src_hash,
        extra={
            "navgrid": ng.describe(),
            "navgrid_path": str(navgrid_path),
        },
    )

    return Terrain(
        grid=grid.grid,
        walkable=np.ascontiguousarray(walkable),
        blocks_vision=np.ascontiguousarray(blocks_vision),
        brush_id=np.ascontiguousarray(brush_id),
        n_brush_groups=n_groups,
        header=header,
        spec=resolved,
        grid_spec=grid,
    )
