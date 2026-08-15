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
def pipeline(
    seed: Annotated[int, typer.Option(help="Synthetic scenario seed.")] = 7,
    duration: Annotated[float, typer.Option(help="Match window, seconds.")] = 900.0,
    clean: Annotated[bool, typer.Option("--clean", help="Disable every stream pathology.")] = False,
    stride: Annotated[int, typer.Option(help="Sample every Nth tick when validating.")] = 1,
) -> None:
    """Run a synthetic match end to end and report the fog agreement.

    Reports the agreement twice: once with reconstructed positions (what would ship) and
    once with the true positions substituted. The gap between them is the reconstruction's
    cost, and the lower figure is the irreducible floor from cell snapping, shadowcasting's
    permissiveness and the ward and minion models. A single number cannot distinguish a
    modelling limit from a bug; the pair can.
    """
    import dataclasses
    import time

    from shadowcast.fov.table import load_table
    from shadowcast.l1_events import normalise
    from shadowcast.l1_events.resolve import attribute, resolve_all
    from shadowcast.packets.synth import Pathologies, ScenarioSpec, SyntheticSource
    from shadowcast.validate import validate_fog

    terrain = _load_terrain()
    table = load_table(terrain)

    pathologies = Pathologies.none() if clean else Pathologies.all()
    source = SyntheticSource(
        terrain, ScenarioSpec(seed=seed, duration=duration, pathologies=pathologies)
    )
    match_id = source.match_ids()[0]

    timings: dict[str, str] = {}

    def timed(label, fn):
        start = time.perf_counter()
        out = fn()
        timings[label] = f"{time.perf_counter() - start:.1f}s"
        return out

    bundle, truth = timed("generate", lambda: source.generate(match_id))
    events = timed("normalise", lambda: normalise(bundle, terrain))
    att = timed("attribute", lambda: attribute(events))
    events, info = timed("resolve", lambda: resolve_all(events, att))

    _echo_table("match", events.describe())
    typer.echo("")
    _echo_table("attribution", att.describe())
    typer.echo("")
    for name, block in info.items():
        _echo_table(name, block)  # type: ignore[arg-type]
        typer.echo("")

    reconstructed = timed(
        "validate", lambda: validate_fog(events, att, terrain, table, stride=stride)
    )
    n = min(att.pos.shape[0], truth.pos.shape[0])
    reference = validate_fog(
        events,
        dataclasses.replace(att, pos=truth.pos[:n].copy(), valid=truth.alive[:n].astype(bool)),
        terrain,
        table,
        stride=stride,
    )

    _echo_table(
        "fog agreement",
        {
            "compared": f"{reconstructed.compared:,}",
            "reconstructed positions": f"{reconstructed.rate:.3%}",
            "reconstructed false positive": f"{reconstructed.false_positive_rate:.3%}",
            "reconstructed false negative": f"{reconstructed.false_negative_rate:.3%}",
            "true positions (floor)": f"{reference.rate:.3%}",
            "floor false positive": f"{reference.false_positive_rate:.3%}",
            "floor false negative": f"{reference.false_negative_rate:.3%}",
            "reconstruction cost": f"{reference.rate - reconstructed.rate:.3%}",
        },
    )
    typer.echo("")
    _echo_table(
        "by region (reconstructed)",
        {k: f"{v:.3%}" for k, v in reconstructed.region_rates().items()},
    )
    typer.echo("")
    _echo_table(
        "transition timing",
        {
            "within 150 ms": f"{reconstructed.timing().get('within_150ms', 0):.1%}",
            "abs median": f"{reconstructed.timing().get('abs_median_s', 0):.3f}s",
            "abs p98": f"{reconstructed.timing().get('abs_p98_s', 0):.2f}s",
            "ours / oracle": (
                f"{reconstructed.stats['our_transitions']} / "
                f"{reconstructed.stats['oracle_transitions']}"
            ),
        },
    )
    typer.echo("")
    _echo_table("timings", timings)


@app.command()
def ablate(
    seed: Annotated[int, typer.Option(help="Synthetic scenario seed.")] = 7,
    duration: Annotated[float, typer.Option(help="Match window, seconds.")] = 900.0,
    clean: Annotated[bool, typer.Option("--clean", help="Disable every stream pathology.")] = False,
    stride: Annotated[int, typer.Option(help="Score every Nth tick.")] = 4,
    models: Annotated[str, typer.Option(help="Comma-separated model names.")] = "",
) -> None:
    """Run every belief model over one match and print the ablation table.

    Two adjacent rows carry the argument, and each differs from its neighbour in exactly
    one field of one frozen spec:

        diffusion -> behavioural    what the behavioural prior is worth
        behavioural -> full         what NEGATIVE INFORMATION is worth

    The second is the thesis. If `full` does not beat `behavioural`, negative information
    is contributing nothing and the central claim is empty — which this prints plainly
    rather than burying.
    """
    import time

    from shadowcast.config import BASELINES, THESIS_PAIR
    from shadowcast.fov.table import load_table
    from shadowcast.l1_events import normalise
    from shadowcast.l1_events.resolve import attribute, resolve_all
    from shadowcast.l2_reconstruct.vision import VisionStream
    from shadowcast.l3_infer.baselines import ablate as run_ablation
    from shadowcast.l3_infer.metrics import LatticeIndex
    from shadowcast.l3_infer.policy import observe
    from shadowcast.packets.synth import Pathologies, ScenarioSpec, SyntheticSource

    terrain = _load_terrain()
    table = load_table(terrain)
    pathologies = Pathologies.none() if clean else Pathologies.all()
    source = SyntheticSource(
        terrain, ScenarioSpec(seed=seed, duration=duration, pathologies=pathologies)
    )
    bundle, _ = source.generate(source.match_ids()[0])
    events = normalise(bundle, terrain)
    att = attribute(events)
    events, _ = resolve_all(events, att)

    start = time.perf_counter()
    obs, public, truth = observe(events, att, VisionStream(events, att, terrain, table))
    lattice = LatticeIndex(terrain)
    chosen = (
        {k: BASELINES[k] for k in models.split(",") if k.strip()} if models.strip() else BASELINES
    )
    result = run_ablation(
        terrain,
        obs,
        public,
        truth,
        lambda: VisionStream(events, att, terrain, table).masks(),
        models=chosen,
        lattice=lattice,
        stride=stride,
    )
    elapsed = time.perf_counter() - start

    _echo_table(
        "setup",
        {
            "lattice bins": f"{lattice.n_bins} ({lattice.lattice}^2, {lattice.cell_size:.0f}u)",
            "max entropy": f"{lattice.max_bits:.2f} bits",
            "particles": C.PARTICLES,
            "enemy visible": f"{obs.visible_fraction():.1%}",
            "scored ticks": f"{next(iter(result.scores.values())).scored_ticks:,}",
        },
    )
    typer.echo("")
    header = (
        f"{'model':<13}{'NLL':>8}{'H bits':>8}{'area ku2':>10}{'% map':>8}{'ECE':>7}{'depl':>6}"
    )
    typer.echo(typer.style(header, bold=True))
    for name, score in result.scores.items():
        typer.echo(
            f"{name:<13}{score.nll:>8.3f}{score.entropy_bits:>8.2f}"
            f"{score.credible_area_ku2:>10.2f}{score.credible_area_map_fraction:>8.2%}"
            f"{score.calibration_error:>7.3f}{score.depletion_events:>6}"
        )
    typer.echo("")
    _echo_table(
        "coverage (full)",
        {f"{q:.0%} region": f"{v:.1%}" for q, v in result.scores["full"].coverage.items()},
    )
    typer.echo("")
    a, b = THESIS_PAIR
    verdict = "HOLDS" if result.thesis_holds else "DOES NOT HOLD"
    _echo_table(
        "thesis",
        {
            "comparison": f"{a} -> {b} (negative information)",
            "NLL improvement": f"{result.thesis_delta:+.4f}",
            "verdict": verdict,
            "elapsed": f"{elapsed:.1f}s",
        },
    )


@app.command()
def inspect(
    shard: Annotated[
        Path, typer.Argument(help="A .jsonl.gz shard from the decoded replay corpus.")
    ] = Path("data/raw/12_22/batch_001.jsonl.gz"),
    matches: Annotated[int, typer.Option(help="How many matches to read.")] = 1,
) -> None:
    """Test the fog oracle against real packets.

    Everything downstream rests on one claim: a fog event naming champion C can only come
    from C's opponents, because a team always sees its own members. That makes the
    observer team derivable per event and gives a ground-truth visibility oracle for both
    sides — and it had never been checked against a real shard.

    The deciding test is geometric. If the fog stream is the opposing team's vision, a
    visible champion sits close to an ENEMY and a hidden one does not, while its distance
    to its own allies barely moves. Nothing else predicts that: interest culling around a
    camera would move both together.
    """
    from shadowcast.packets.inspect import CHAMPION_SIGHT, inspect_fog, read_matches

    if not shard.exists():
        typer.secho(f"no shard at {shard}", fg=typer.colors.RED)
        typer.echo("  fetch one with:")
        typer.echo('    uv run python -c "from huggingface_hub import hf_hub_download"')
        typer.echo(f"  see the README for the full command; expected at {shard}")
        raise typer.Exit(1)

    for n, match in enumerate(read_matches(shard, limit=matches)):
        report = inspect_fog(match)
        _echo_table(
            f"match {n + 1} — {shard.name}",
            {
                "packets": f"{match.n_packets:,}",
                "duration": f"{report.duration / 60:.1f} min",
                "heroes": report.n_heroes,
                "teams from the damage graph": (
                    "bipartite, 5/5" if report.bipartite else "NOT bipartite"
                ),
            },
        )
        typer.echo("")
        _echo_table(
            "fog transitions",
            {
                "raw EnterFog : LeaveFog": f"{report.raw_ratio:.2f} : 1",
                "after dedup, alternate": report.alternates,
                "transitions": f"{report.n_transitions:,}",
                "position packets while visible": (f"{report.position_packets_while_visible:.1%}"),
            },
        )
        typer.echo("")
        typer.secho("distance to the nearest champion, by fog state", bold=True)
        typer.echo(f"  {'':>9}{'nearest ally':>15}{'nearest enemy':>16}")
        typer.echo(
            f"  {'visible':>9}{report.visible_ally_distance:>13,.0f} u"
            f"{report.visible_enemy_distance:>14,.0f} u"
        )
        typer.echo(
            f"  {'in fog':>9}{report.hidden_ally_distance:>13,.0f} u"
            f"{report.hidden_enemy_distance:>14,.0f} u"
        )
        typer.echo(
            f"  ({report.samples:,} samples; champion sight radius is {CHAMPION_SIGHT:.0f} u)"
        )
        typer.echo("")
        if report.oracle_holds:
            typer.secho(
                "  the fog stream tracks ENEMY proximity, not isolation — the oracle holds",
                fg=typer.colors.GREEN,
                bold=True,
            )
        else:
            typer.secho(
                "  the fog stream does NOT behave like the opposing team's vision",
                fg=typer.colors.RED,
                bold=True,
            )


@app.command()
def diagnose(
    seed: Annotated[int, typer.Option(help="Synthetic scenario seed.")] = 7,
    duration: Annotated[float, typer.Option(help="Match window, seconds.")] = 900.0,
    model: Annotated[str, typer.Option(help="Which baseline to diagnose.")] = "full",
) -> None:
    """Ask HOW the belief is wrong, not just how much.

    A calibration number says the 90% region contains the truth 43% of the time; it does
    not say why, and the two possibilities need opposite fixes. If the truth is outside
    the cloud entirely, the filter is killing the correct hypothesis — a defect in the
    weights, the resampling or the negative update. If the truth is near the cloud but
    the cloud's mass has moved elsewhere, the machinery is fine and the motion model
    believes champions go somewhere they do not.
    """
    from shadowcast.config import BASELINES
    from shadowcast.fov.table import load_table
    from shadowcast.l1_events import normalise
    from shadowcast.l1_events.resolve import attribute, resolve_all
    from shadowcast.l2_reconstruct.vision import VisionStream
    from shadowcast.l3_infer.policy import observe
    from shadowcast.packets.synth import Pathologies, ScenarioSpec, SyntheticSource
    from shadowcast.validate.belief_diagnostic import diagnose_belief

    terrain = _load_terrain()
    table = load_table(terrain)
    source = SyntheticSource(
        terrain, ScenarioSpec(seed=seed, duration=duration, pathologies=Pathologies.all())
    )
    bundle, _ = source.generate(source.match_ids()[0])
    events = normalise(bundle, terrain)
    att = attribute(events)
    events, _ = resolve_all(events, att)
    obs, public, truth = observe(events, att, VisionStream(events, att, terrain, table))

    result = diagnose_belief(
        BASELINES[model],
        terrain,
        obs,
        public,
        truth,
        VisionStream(events, att, terrain, table).masks(),
    )

    _echo_table(
        f"belief diagnostic — {model}",
        {
            "scored moments": f"{result.scored:,}",
            "truth inside the cloud": f"{result.in_support:.1%}",
            "its density rank when inside": f"{result.median_rank:.2f} (0 = the peak)",
        },
    )
    typer.echo("")
    _echo_table(
        "distance from the truth",
        {
            f"nearest particle p{q}": f"{v:,.0f} u"
            for q, v in sorted(result.nearest_percentiles.items())
        }
        | {
            f"centre of mass p{q}": f"{v:,.0f} u"
            for q, v in sorted(result.centroid_percentiles.items())
        },
    )
    typer.echo("")
    typer.secho("by how long the enemy has been hidden", bold=True)
    typer.echo(f"  {'band':>8}  {'n':>6}  {'nearest':>9}  {'centroid':>9}")
    for label, n, near, cent in result.by_darkness:
        typer.echo(f"  {label:>8}  {n:>6,}  {near:>7,.0f} u  {cent:>7,.0f} u")
    typer.echo("")
    typer.secho(f"  {result.verdict}", fg=typer.colors.YELLOW, bold=True)


@app.command()
def export(
    seed: Annotated[int, typer.Option(help="Synthetic scenario seed.")] = 7,
    duration: Annotated[float, typer.Option(help="Match window, seconds.")] = 900.0,
    clean: Annotated[bool, typer.Option("--clean", help="Disable every stream pathology.")] = False,
    shard: Annotated[
        Path | None,
        typer.Option(help="Export a REAL match from this shard instead of a generated one."),
    ] = None,
    line: Annotated[int, typer.Option(help="Which match in the shard. Requires --shard.")] = 0,
    out: Annotated[
        Path | None, typer.Option(help="Artifact directory. Defaults to data/artifacts/<id>.")
    ] = None,
    codegen: Annotated[bool, typer.Option(help="Also regenerate the TypeScript reader.")] = True,
    web: Annotated[
        bool, typer.Option("--web", help="Write into web/public/ for the dev server.")
    ] = False,
) -> None:
    """Export one match as the artifact the static site reads.

    Two files: `meta.json` and `data.bin.gz`. Serve the payload with
    `Content-Encoding: gzip` and the browser inflates it during transfer, so the site
    ships no decompression code at all.

    With `--shard` the match comes from decoded replay packets rather than the generator,
    and `meta.provenance` says so. That field is not decoration: fog agreement is 98% on a
    generated match and 68% on a real one, so an unlabelled synthetic artifact shows the
    engine's geometry while looking like its accuracy.
    """
    import time

    from shadowcast.config import data_dir
    from shadowcast.fov.table import load_table
    from shadowcast.l1_events import normalise
    from shadowcast.l1_events.resolve import attribute, resolve_all
    from shadowcast.l2_reconstruct.vision import VisionStream
    from shadowcast.l3_infer.policy import observe
    from shadowcast.l4_export.artifact import write_artifact
    from shadowcast.l4_export.build import build_arrays
    from shadowcast.l4_export.spec import PROVENANCE_REAL, PROVENANCE_SYNTHETIC
    from shadowcast.l4_export.terrain_png import TERRAIN_PNG_NAME, write_terrain_png
    from shadowcast.l4_export.ts_codegen import write_typescript
    from shadowcast.packets.synth import Pathologies, ScenarioSpec, SyntheticSource

    terrain = _load_terrain()
    table = load_table(terrain)

    if shard is not None:
        from shadowcast.packets.replay import ReplaySource

        if not shard.exists():
            typer.secho(f"no shard at {shard}", fg=typer.colors.RED)
            raise typer.Exit(1)
        source = ReplaySource(shard, limit=line + 1)
        bundle = source.read(source.match_ids()[line])
        provenance = PROVENANCE_REAL
        # `/` and `:` both appear in a real match id and neither belongs in a path
        # segment — the id is `12_22/batch_001:0`, which would silently become a nested
        # directory the site cannot fetch back.
        match_id = bundle.meta.match_id
        slug = match_id.replace("/", "-").replace(":", "-")
        duration = bundle.meta.duration
    else:
        pathologies = Pathologies.none() if clean else Pathologies.all()
        source = SyntheticSource(
            terrain, ScenarioSpec(seed=seed, duration=duration, pathologies=pathologies)
        )
        match_id = source.match_ids()[0]
        bundle, _ = source.generate(match_id)
        provenance = PROVENANCE_SYNTHETIC
        slug = match_id

    events = normalise(bundle, terrain)
    att = attribute(events)
    events, _ = resolve_all(events, att)

    start = time.perf_counter()
    obs, public, _ = observe(events, att, VisionStream(events, att, terrain, table))
    built = build_arrays(
        events,
        att.pos,
        att.valid,
        obs,
        public,
        lambda: VisionStream(events, att, terrain, table).masks(),
        terrain,
    )
    root = Path("web/public") if web else data_dir()
    dest = Path(out) if out else root / "artifacts" / slug
    path, report = write_artifact(
        dest,
        match_id=match_id,
        duration=duration,
        provenance=provenance,
        dims=built.dims,
        arrays=built.arrays,
        heroes=[
            {
                "slot": int(h["slot"]),
                "name": str(h["name"]),
                "champion": str(h["champion"]),
                "team": int(h["team"]),
                "role": str(h["role"]),
            }
            for h in events.heroes
        ],
        events=_export_events(events),
        stats=built.stats,
    )
    elapsed = time.perf_counter() - start

    _echo_table(
        "sections (encoded, before gzip)",
        {k: f"{v / 1e3:,.0f} kB" for k, v in report["sections"].items()},
    )
    typer.echo("")
    _echo_table(
        "artifact",
        {
            "raw": f"{report['raw_bytes'] / 1e6:,.2f} MB",
            "gzipped": f"{report['gzipped_bytes'] / 1e6:,.2f} MB",
            "meta.json": f"{report['meta_bytes'] / 1e3:,.0f} kB",
            "total shipped": f"{report['total_bytes'] / 1e6:,.2f} MB",
            "compression": f"{report['ratio']}x",
            "mixture KL (mean)": f"{built.stats['mixture_kl_mean']:.4f} nats",
            "elapsed": f"{elapsed:.1f}s",
        },
    )
    budget = 2.0
    total_mb = report["total_bytes"] / 1e6
    colour = typer.colors.GREEN if total_mb <= budget else typer.colors.RED
    typer.secho(f"\n  {total_mb:.2f} MB against a {budget:.0f} MB budget", fg=colour, bold=True)
    typer.secho(f"  wrote {path}", fg=typer.colors.GREEN)

    terrain_path, terrain_report = write_terrain_png(terrain, root / TERRAIN_PNG_NAME)
    typer.secho(
        f"  wrote {terrain_path} ({terrain_report['bytes'] / 1e3:.0f} kB, shipped once)",
        fg=typer.colors.GREEN,
    )

    if codegen:
        ts, changed = write_typescript(Path("web/src/generated/artifact.ts"))
        typer.secho(
            f"  {'regenerated' if changed else 'unchanged'} {ts}",
            fg=typer.colors.YELLOW if changed else typer.colors.GREEN,
        )


def _export_events(events) -> dict[str, object]:
    """The event streams the site draws as ticks on a timeline."""
    import numpy as np

    return {
        "wards": [
            {
                "t0": round(float(w["t0"]), 2),
                "t1": round(float(w["t1"]), 2),
                "x": round(float(w["x"]), 1),
                "z": round(float(w["z"]), 1),
                "team": int(w["team"]),
                "owner": int(w["owner_slot"]),
                "sight": float(w["sight"]),
            }
            for w in events.wards
        ],
        # Turret sites, so the frontend can decide whether a sighting is attributable to
        # a ward or was covered by structure vision anyway. 24 rows, and without them the
        # ward-yield metric would credit wards for vision the turrets already had.
        "turrets": [
            {
                "x": round(float(s["x"]), 1),
                "z": round(float(s["z"]), 1),
                "team": int(s["team"]),
            }
            for s in events.turret_sites
            if np.isfinite(s["x"]) and np.isfinite(s["z"])
        ],
        "deaths": [
            {
                "t": round(float(d["t"]), 2),
                "victim": int(d["victim"]),
                "killer": int(d["killer"]),
                "respawn": round(float(d["respawn_t"]), 2),
                "confidence": round(float(d["killer_confidence"]), 3),
            }
            for d in events.deaths
        ],
    }


def _realfog_corpus(shard: Path, matches: int, stride: int, run_bundle, out: Path | None) -> None:
    """Fog agreement across a whole shard, reported as a spread rather than a number.

    A single match's agreement cannot say whether it is typical, and the project's central
    claim was quoted from exactly one for as long as it had a real figure at all. Matches
    that fail outright are reported, not skipped silently: a resolver that gives up on some
    matches and scores well on the rest is a selection effect, not a result.
    """
    import json
    import statistics
    import time

    from shadowcast.packets.replay import ReplaySource

    rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    start = time.perf_counter()

    typer.secho(f"reading {shard.name} in one pass, up to {matches} matches", bold=True)
    for bundle in ReplaySource(shard).read_all(limit=matches):
        label = bundle.meta.match_id.rsplit("/", 1)[-1]
        try:
            events, _, fog = run_bundle(bundle)
        except Exception as exc:
            failures.append((label, f"{type(exc).__name__}: {exc}"))
            typer.secho(f"  {label:<16} FAILED  {type(exc).__name__}", fg=typer.colors.RED)
            continue
        timing = fog.timing()
        row = {
            "match": label,
            "duration_min": round(bundle.meta.duration / 60, 1),
            "agreement": fog.rate,
            "false_positive": fog.false_positive_rate,
            "false_negative": fog.false_negative_rate,
            "compared": fog.compared,
            "lane_minions": int(events.minion_waves.size),
            "order_attribution": events.order_attribution_rate,
            "within_150ms": timing.get("within_150ms", float("nan")),
            **{f"region_{k}": v for k, v in fog.region_rates().items()},
        }
        rows.append(row)
        typer.echo(
            f"  {label:<16}{fog.rate:>8.2%}  FP {fog.false_positive_rate:>6.2%}  "
            f"FN {fog.false_negative_rate:>6.2%}  {bundle.meta.duration / 60:>5.1f} min"
        )

    if not rows:
        typer.secho("no match completed", fg=typer.colors.RED)
        raise typer.Exit(1)

    def spread(key: str) -> str:
        values = sorted(float(r[key]) for r in rows)  # type: ignore[arg-type]
        lo, hi = values[0], values[-1]
        med = statistics.median(values)
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        return f"{med:.2%}   [{lo:.2%}, {hi:.2%}]   sd {sd:.2%}"

    typer.echo("")
    _echo_table(
        f"across {len(rows)} matches — median, range, standard deviation",
        {
            "agreement": spread("agreement"),
            "false positive": spread("false_positive"),
            "false negative": spread("false_negative"),
            "order attribution": spread("order_attribution"),
        },
    )
    typer.echo("")
    _echo_table(
        "by region — median across matches",
        {
            k.removeprefix("region_"): f"{statistics.median(float(r[k]) for r in rows):.1%}"
            for k in rows[0]
            if k.startswith("region_")
        },
    )

    if failures:
        typer.echo("")
        typer.secho(f"{len(failures)} match(es) failed:", fg=typer.colors.RED, bold=True)
        for label, why in failures:
            typer.echo(f"  {label:<16}{why}")

    typer.echo("")
    typer.echo(f"  {len(rows)} matches in {time.perf_counter() - start:.0f}s, stride {stride}")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows, "failures": failures}, indent=2))
        typer.secho(f"wrote {out}", fg=typer.colors.GREEN)


@app.command()
def realfog(
    shard: Annotated[
        Path, typer.Argument(help="A .jsonl.gz shard from the decoded replay corpus.")
    ] = Path("data/raw/12_22/batch_001.jsonl.gz"),
    line: Annotated[int, typer.Option(help="Which match in the shard.")] = 0,
    matches: Annotated[
        int,
        typer.Option(
            help="Run this many matches from the shard and report the DISTRIBUTION. "
            "One match is a point estimate and cannot say whether it is typical."
        ),
    ] = 1,
    stride: Annotated[int, typer.Option(help="Sample every Nth tick.")] = 4,
    synthetic: Annotated[
        bool, typer.Option("--synthetic/--no-synthetic", help="Also run the same code on synth.")
    ] = True,
    out: Annotated[Path | None, typer.Option(help="Write the per-match rows here as JSON.")] = None,
) -> None:
    """Measure fog agreement on REAL packets, and decompose the disagreement.

    The headline number is the project's central claim submitting to the only ground truth
    that exists for it. The decomposition is what makes it actionable: agreement is split
    by how stale the positions involved are, separately for the champion being looked at
    and for the nearest champion doing the looking. Those two point at different repairs —
    a missing vision source versus a misplaced one — and a single percentage hides both.

    With `--matches N` the whole shard is read in one pass and the spread is reported
    instead of a single number, which is the only way to know whether one match's figure
    was typical.
    """

    from shadowcast.fov.table import load_table
    from shadowcast.l1_events import normalise
    from shadowcast.l1_events.resolve import attribute, resolve_all
    from shadowcast.packets.replay import ReplaySource
    from shadowcast.validate.fog_oracle import validate_fog
    from shadowcast.validate.real_fog import decompose_fog

    if not shard.exists():
        typer.secho(f"no shard at {shard}", fg=typer.colors.RED)
        typer.echo("  see the README for the download command")
        raise typer.Exit(1)

    terrain = _load_terrain()
    table = load_table(terrain)

    def run_bundle(bundle):
        events = normalise(bundle, terrain)
        att = attribute(events)
        events, _ = resolve_all(events, att)
        return events, att, validate_fog(events, att, terrain, table, stride=stride)

    def run(source, match_id: str):
        return run_bundle(source.read(match_id))

    if matches > 1:
        _realfog_corpus(shard, matches, stride, run_bundle, out)
        return

    source = ReplaySource(shard, limit=line + 1)
    match_id = source.match_ids()[line]
    events, att, real = run(source, match_id)

    _echo_table(
        f"fog agreement — {match_id}",
        {
            "agreement": f"{real.rate:.2%}",
            "false positive": f"{real.false_positive_rate:.2%}",
            "false negative": f"{real.false_negative_rate:.2%}",
            "compared": f"{real.compared:,}",
        },
    )
    typer.echo("")
    _echo_table("by region", {k: f"{v:.1%}" for k, v in real.region_rates().items()})

    if synthetic:
        from shadowcast.packets.synth import SyntheticSource

        synth = SyntheticSource(terrain)
        _, _, ref = run(synth, synth.match_ids()[0])
        typer.echo("")
        _echo_table(
            "the same code on synthetic packets",
            {
                "agreement": f"{ref.rate:.2%}",
                "false positive": f"{ref.false_positive_rate:.2%}",
                "false negative": f"{ref.false_negative_rate:.2%}",
            },
        )

    report = decompose_fog(events, att, terrain, table, stride=stride)
    for title, bands, note in (
        (
            "by the TARGET's position staleness — is a source missing?",
            report.by_target_age,
            "the last column is a floor: better trajectories cannot close it",
        ),
        (
            "by the nearest OBSERVER's staleness — is a source misplaced?",
            report.by_observer_age,
            "vision comes from someone else, so their position is the one that moves it",
        ),
    ):
        typer.echo("")
        typer.secho(title, bold=True)
        typer.echo(
            f"  {'since an anchor':>16}{'n':>8}{'agree':>9}{'visible→src':>13}"
            f"{'hidden→src':>12}{'no src in range':>17}"
        )
        for b in bands:
            flag = "" if b.informative else "   <- not informative"
            typer.echo(
                f"  {b.label:>16}{b.n:>8,}{b.rate:>8.1%}"
                f"{b.visible_source_distance:>12,.0f} u{b.hidden_source_distance:>10,.0f} u"
                f"{b.visible_without_source:>16.1%}{flag}"
            )
        typer.secho(f"  {note}", dim=True)

    typer.echo("")
    _echo_table(
        "model coverage",
        {
            "lane minions": f"{events.minion_waves.size:,}",
            "front-line contacts": f"{events.minion_contacts.size:,}",
            "orders attributed": f"{events.order_attribution_rate:.2%}",
        },
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
