# Shadowcast — working notes for Claude

A belief-state engine for League of Legends information asymmetry, built on packet-level decoded
replays. `README.md` explains what and why; this file is how to work in the repo.

## Commands

```bash
uv sync                          # dev tools included
uv run pytest                    # full suite, ~4s warm. Do not deselect -m slow.
uv run pytest tests/test_fov.py -x -q
uv run ruff check --fix . && uv run ruff format .
uv run shadowcast terrain build  # navgrid -> 512^2 channels + brush groups
uv run shadowcast fov build      # precompute the visibility table (~5s)
uv run shadowcast doctor         # versions, config hashes, stale artifacts
```

Everything runs through `uv`. Never invoke a bare `python`/`pytest`/`ruff` — the venv is
uv-managed and a bare call silently picks up whatever is on PATH.

## Layout

`src/shadowcast/` maps onto the L0–L4 pipeline in the README:

| Path | Layer | Contents |
|---|---|---|
| `packets/` | L0/L1 | `source.py` packet-source seam, `synth.py` synthetic generator, `conformance.py` |
| `terrain/` | — | `navgrid.py` `.aimesh_ngrid` parser, `terrain.py` three-channel build |
| `geom/` | — | `grid.py`, `bitset.py`, `path.py` — the numeric primitives |
| `fov/` | L2 | `shadowcast.py` kernel, `reference.py` brute-force oracle, `table.py`, `union.py` |
| `config.py` `constants.py` `sr.py` | — | hashed config, documented literals, map constants |

## Invariants that are load-bearing

These are not style preferences. Each one has a test that exists specifically to keep it true.

- **Radius monotonicity.** One FOV table serves every sight radius because
  `FOV(r) == FOV_max AND disc(r)`. Two implementation choices break it and are therefore banned
  in code: a **wall-lighting post-pass** and **flood-revealing the source's whole brush**. If you
  are ever tempted to add either to make a mask look nicer, you are deleting the property the
  160 MB table design rests on.
- **The table is a cache, not a data structure.** A miss falls back to a live FOV computation.
  Correctness must never depend on table coverage — that is what lets vision sources sit in
  non-walkable cells (wall-hop dashes, over-wall Farsight wards) with no special cases.
- **Terrain has three channels.** `blocks_move`, `blocks_vision`, `brush_id`. Riot stamps
  see-through cells along wall diagonals that block movement but transmit vision. Deriving
  vision from walkability reproduces a bug Riot patched after S5 Worlds.

## Measured numbers only

`docs/validation.md` is written by `shadowcast validate`, not by hand, and nothing appears in the
README that is not in `validation.md` first. When adding a claim about accuracy, coverage or
performance: either produce it from a command, or mark it `[pending]`. Do not write a plausible
figure into prose. This rule is the project's entire argument for being more than a visualisation.

Likewise the corpus-reality table in the README — every row is a count from parsing real packets.
Do not soften or extrapolate those without re-measuring.

## Numba conventions

House style is `@njit(cache=True)`, with `inline="always"` on small helpers and
`parallel=True` + `prange` only where the table build needs it. No `fastmath` anywhere — the
geometry needs exact float comparisons for shadow intervals.

Set `NUMBA_DISABLE_JIT=1` to run a kernel as plain Python when a typing error is unreadable or you
need a real traceback. Add a test rather than a print. See the `numba-kernel` skill.

## Testing

- Expensive fixtures (`terrain`, `fov_table`, `synth_clean`, `synth_dirty`) are **session-scoped**
  in `tests/conftest.py`. Use them; do not rebuild terrain inside a test.
- Terrain and FOV tests skip cleanly when `data/terrain/AIPath_SRX.aimesh_ngrid` is absent. CI
  fetches it, verifies its SHA-256, and asserts nothing skipped — a skipped test here is a test
  that quietly stopped running.
- The `slow` marker exists for the exhaustive oracles, but CI runs the whole suite anyway. Don't
  add `-m "not slow"` to a workflow.

## Style

- Short mathematical names (`dx`, `r2`, `si`, `wj`) are deliberate and ruff is configured to allow
  them; ambiguous single letters (`l`, `I`, `O`) stay banned via E741.
- Comments explain *why*, at length where the reasoning is not obvious — matching the existing
  files. A comment restating the code is worse than none.
- `data/` is gitignored and every artifact in it is reproducible from a documented command. Never
  `git add -f` anything under it.

## Library docs

Use **context7** (`resolve-library-id` then `query-docs`) before answering from memory about
numba, numpy, polars, duckdb, pyarrow or typer APIs. Numba in particular pins numpy and changes
its typing rules between releases; a remembered signature is likely to be a version behind.
