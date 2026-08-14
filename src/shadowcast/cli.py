"""`shadowcast` command line.

Every verb reads a versioned file and writes a versioned file. Nothing here holds
logic; the commands are thin wrappers so each stage stays independently testable
and independently rebuildable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from shadowcast import __version__
from shadowcast import constants as C

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Reconstructing what League of Legends teams could actually see.",
)
terrain_app = typer.Typer(no_args_is_help=True, help="Build and inspect map terrain.")
app.add_typer(terrain_app, name="terrain")
fov_app = typer.Typer(no_args_is_help=True, help="Build and verify the visibility table.")
app.add_typer(fov_app, name="fov")


def _echo_table(title: str, rows: dict[str, object]) -> None:
    typer.secho(title, bold=True)
    width = max((len(str(k)) for k in rows), default=0)
    for k, v in rows.items():
        typer.echo(f"  {k!s:<{width}}  {v}")


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"shadowcast {__version__}")


@terrain_app.command("build")
def terrain_build(
    navgrid: Annotated[
        Path | None,
        typer.Option(help="Path to a .aimesh_ngrid file. Defaults to data/terrain/."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Where to write the terrain .npz. Defaults to a hash-keyed path."),
    ] = None,
    no_see_through: Annotated[
        bool,
        typer.Option(
            "--no-see-through",
            help="Ablation: treat every wall as opaque, ignoring SEE_THROUGH cells. "
            "Exists so the fog-agreement rate can be measured both ways.",
        ),
    ] = False,
) -> None:
    """Parse the navgrid and resample it onto the simulation grid."""
    from shadowcast.config import GridSpec, TerrainSpec, terrain_path
    from shadowcast.terrain.terrain import build_terrain

    spec = TerrainSpec(see_through_transmits_vision=not no_see_through)
    terrain = build_terrain(navgrid_path=navgrid, grid_spec=GridSpec(), spec=spec)
    dest = Path(out) if out else terrain_path(terrain.spec)
    terrain.save(dest)

    _echo_table("navgrid", terrain.header.extra["navgrid"])  # type: ignore[arg-type]
    typer.echo("")
    _echo_table("terrain", terrain.describe())
    typer.echo("")

    lo = C.SR_BRUSH_PATCHES_DOCUMENTED - 3
    hi = C.SR_BRUSH_PATCHES_DOCUMENTED + 6
    n = terrain.n_brush_groups
    if lo <= n <= hi:
        typer.secho(
            f"  brush groups {n} within [{lo}, {hi}] of the documented "
            f"{C.SR_BRUSH_PATCHES_DOCUMENTED}",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            f"  brush groups {n} OUTSIDE [{lo}, {hi}] — the raster has either fused or "
            f"shattered brush patches, which breaks conditional-occluder semantics",
            fg=typer.colors.RED,
        )

    rows = terrain.fov_table_rows()
    import math

    words = math.ceil(math.ceil(C.FOV_WINDOW / 64) * C.FOV_WINDOW / 8) * 8
    typer.echo("")
    _echo_table(
        "implied FOV table",
        {
            "rows (walkable cells)": f"{rows:,}",
            "window": f"{C.FOV_WINDOW}x{C.FOV_WINDOW}",
            "bytes/row": f"{words * 8:,}",
            "total": f"{rows * words * 8 / 1e6:,.1f} MB",
        },
    )
    typer.echo("")
    typer.secho(f"wrote {dest}", fg=typer.colors.GREEN)


@terrain_app.command("show")
def terrain_show(
    path: Annotated[Path | None, typer.Argument(help="Terrain .npz.")] = None,
    width: Annotated[int, typer.Option(help="Characters across.")] = 96,
) -> None:
    """Render terrain as ASCII, for eyeballing that it is actually the right map.

    Matching cell counts can be a coincidence; a recognisable Summoner's Rift
    cannot. This is the check that caught nothing because the parse was right, and
    would have caught everything if it were not.
    """
    from shadowcast.config import TerrainSpec, terrain_path
    from shadowcast.terrain.terrain import Terrain, build_terrain

    if path is None:
        candidate = terrain_path(TerrainSpec())
        terrain = Terrain.load(candidate) if candidate.exists() else build_terrain()
    else:
        terrain = Terrain.load(path)

    g = terrain.grid
    step = max(1, g // width)
    lines = []
    # z descending, so the render matches the in-game minimap: blue base lower left.
    for j in range(g - 1, -1, -step * 2):
        row = []
        for i in range(0, g, step):
            blk = (slice(max(0, j - step * 2 + 1), j + 1), slice(i, i + step))
            if terrain.brush[blk].any():
                row.append('"')
            elif terrain.walkable[blk].mean() > 0.55:
                row.append(".")
            elif terrain.walkable[blk].any():
                row.append(":")
            else:
                row.append(" ")
        lines.append("".join(row))
    typer.echo("\n".join(lines))
    typer.echo("")
    typer.echo('legend: . walkable   : partial   " brush   (blank) vision-blocking')


def _load_terrain(explicit: Path | None = None):
    """Load the built terrain, or build it on the fly if it is absent."""
    from shadowcast.config import TerrainSpec, terrain_path
    from shadowcast.terrain.terrain import Terrain, build_terrain

    if explicit is not None:
        return Terrain.load(explicit)
    # A default TerrainSpec has no navgrid hash, so it cannot name the built file.
    # Build from the navgrid instead, which is cheap and always consistent.
    candidate = terrain_path(TerrainSpec()).parent
    built = sorted(candidate.glob("terrain_*.npz")) if candidate.exists() else []
    if len(built) == 1:
        return Terrain.load(built[0])
    return build_terrain()


@fov_app.command("build")
def fov_build(
    terrain_file: Annotated[
        Path | None, typer.Option("--terrain", help="Terrain .npz. Defaults to rebuilding.")
    ] = None,
    chunks: Annotated[
        int, typer.Option(help="Parallel work chunks. More than cores is fine.")
    ] = 64,
) -> None:
    """Precompute the visibility table.

    One row per walkable cell, holding field of view at RMAX. Every smaller sight
    radius is served by intersecting with a precomputed disc, which is why a single
    table suffices and why the naive all-pairs alternative (8.6 TB at this grid) is
    not needed.
    """
    import time

    from shadowcast.fov.table import build_table

    terrain = _load_terrain(terrain_file)
    typer.echo(f"terrain {terrain.spec.content_hash}: {terrain.n_walkable:,} walkable cells")
    start = time.perf_counter()
    table = build_table(terrain, n_chunks=chunks)
    elapsed = time.perf_counter() - start

    _echo_table("fov table", table.describe())
    typer.echo("")
    _echo_table(
        "build",
        {
            "elapsed": f"{elapsed:.1f}s",
            "per row": f"{elapsed / max(1, table.n_rows) * 1e6:.0f} us",
            "worst scan depth": (
                f"{table.header.extra['worst_scan_depth']} of "
                f"{table.header.extra['scratch_frames']} frames"
            ),
        },
    )
    typer.secho(f"\nwrote {table.header.extra.get('dir', '')}".rstrip(), fg=typer.colors.GREEN)


@fov_app.command("verify")
def fov_verify(
    samples: Annotated[int, typer.Option(help="Source cells to check.")] = 200,
    radius: Annotated[
        float | None, typer.Option(help="Radius to compare at. Defaults to champion sight.")
    ] = None,
) -> None:
    """Check the table against fresh computations, and shadowcasting against ray marching.

    Two independent checks. The first confirms the stored bytes are the field of view
    they claim to be. The second compares against a different *class* of algorithm —
    per-target ray marching rather than an octant sweep — so a shared mistake is
    unlikely. Disagreement there is expected in one direction only, and that
    direction is what gets reported: shadowcasting over-reports at shadow edges but
    must never lose vision.
    """
    import numpy as np

    from shadowcast.fov.reference import boundary_band, fov_reference
    from shadowcast.fov.shadowcast import fov_bool
    from shadowcast.fov.table import load_table
    from shadowcast.geom.bitset import unpack_rows
    from shadowcast.geom.grid import disc_mask

    terrain = _load_terrain()
    table = load_table(terrain)
    radius = radius if radius is not None else C.SIGHT_CHAMPION

    rng = np.random.default_rng(0)
    picks = rng.choice(terrain.walkable_cells(), size=samples, replace=False)

    row_mismatch = 0
    disc_mismatch = 0
    considered = permissive = restrictive = 0
    disc_r = disc_mask(radius, window=table.window)

    for k in picks:
        j, i = divmod(int(k), terrain.grid)
        brush = int(terrain.brush_id[j, i])
        row = table.lookup(int(k), brush, brush)
        packed = np.asarray(table.rows[row])[: table.window * table.src_words]
        stored = unpack_rows(packed.reshape(table.window, table.src_words), table.window)

        if not np.array_equal(stored, fov_bool(terrain, i, j, C.RMAX_UNITS, half=table.half)):
            row_mismatch += 1
        at_r = fov_bool(terrain, i, j, radius, half=table.half)
        if not np.array_equal(stored & disc_r, at_r):
            disc_mismatch += 1

        ref = fov_reference(terrain, i, j, radius, half=table.half)
        keep = disc_r & ~(boundary_band(at_r) | boundary_band(ref))
        considered += int(keep.sum())
        permissive += int((at_r & ~ref & keep).sum())
        restrictive += int((~at_r & ref & keep).sum())

    _echo_table(
        f"verify ({samples} sources, radius {radius:.0f})",
        {
            "rows == fresh computation": f"{samples - row_mismatch}/{samples}",
            "row & disc == fov(r)": f"{samples - disc_mismatch}/{samples}",
            "cells compared vs ray march": f"{considered:,}",
            "shadowcast over-reports": f"{permissive:,} ({permissive / max(1, considered):.4%})",
            "shadowcast under-reports": f"{restrictive:,}",
        },
    )
    typer.echo("")
    problems = row_mismatch or disc_mismatch or restrictive
    if problems:
        typer.secho(
            "FAIL — a row disagrees with a fresh computation, radius separability is "
            "broken, or vision is being lost relative to the reference",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    typer.secho(
        "OK — rows exact, radius separability exact, and no vision lost relative to "
        "the independent reference",
        fg=typer.colors.GREEN,
    )


@app.command()
def doctor() -> None:
    """Report versions, config hashes, and whether derived artifacts are stale."""
    import numpy

    from shadowcast.config import (
        ExportSpec,
        FilterSpec,
        GridSpec,
        TerrainSpec,
        TickSpec,
        fov_table_dir,
        git_sha,
        terrain_path,
    )

    try:
        import numba

        numba_version = numba.__version__
    except ImportError:  # pragma: no cover - the fallback path
        numba_version = "MISSING"

    _echo_table(
        "environment",
        {
            "shadowcast": __version__,
            "git": git_sha(),
            "numpy": numpy.__version__,
            "numba": numba_version,
        },
    )
    typer.echo("")

    grid, terrain_spec = GridSpec(), TerrainSpec()
    _echo_table(
        "config hashes",
        {
            "grid": grid.content_hash,
            "terrain": terrain_spec.content_hash,
            "tick": TickSpec().content_hash,
            "filter": FilterSpec().content_hash,
            "export": ExportSpec().content_hash,
        },
    )
    typer.echo("")

    # A TerrainSpec with no navgrid_hash cannot name the built terrain, so probe
    # the directory instead of guessing.
    terrain_dir = terrain_path(terrain_spec).parent
    built = sorted(terrain_dir.glob("terrain_*.npz")) if terrain_dir.exists() else []
    navgrid = terrain_dir / "AIPath_SRX.aimesh_ngrid"
    _echo_table(
        "artifacts",
        {
            "navgrid": "present" if navgrid.exists() else "MISSING — see README",
            "terrain": f"{len(built)} built" if built else "none — run `shadowcast terrain build`",
            "fov tables": _count_fov_tables(fov_table_dir(grid, terrain_spec).parent),
        },
    )


def _count_fov_tables(root: Path) -> str:
    if not root.exists():
        return "none — run `shadowcast fov build`"
    tables = [d for d in root.iterdir() if d.is_dir()]
    if not tables:
        return "none — run `shadowcast fov build`"
    total = sum(f.stat().st_size for d in tables for f in d.rglob("*") if f.is_file())
    return f"{len(tables)} built, {total / 1e6:,.1f} MB total"


if __name__ == "__main__":  # pragma: no cover
    app()
