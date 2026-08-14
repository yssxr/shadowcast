"""`shadowcast` command line.

Every verb reads a versioned file and writes a versioned file. Nothing here holds
logic; the commands are thin wrappers so each stage stays independently testable
and independently rebuildable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

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


def _echo_table(title: str, rows: dict[str, object]) -> None:
    typer.secho(title, bold=True)
    width = max((len(str(k)) for k in rows), default=0)
    for k, v in rows.items():
        typer.echo(f"  {str(k):<{width}}  {v}")


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"shadowcast {__version__}")


@terrain_app.command("build")
def terrain_build(
    navgrid: Annotated[
        Optional[Path],
        typer.Option(help="Path to a .aimesh_ngrid file. Defaults to data/terrain/."),
    ] = None,
    out: Annotated[
        Optional[Path],
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
    path: Annotated[Optional[Path], typer.Argument(help="Terrain .npz.")] = None,
    width: Annotated[int, typer.Option(help="Characters across.")] = 96,
) -> None:
    """Render terrain as ASCII, for eyeballing that it is actually the right map.

    Matching cell counts can be a coincidence; a recognisable Summoner's Rift
    cannot. This is the check that caught nothing because the parse was right, and
    would have caught everything if it were not.
    """
    from shadowcast.config import GridSpec, TerrainSpec, terrain_path
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
