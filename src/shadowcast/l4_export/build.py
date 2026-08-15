"""Turning a reconstructed match into the arrays the artifact holds.

One pass. The vision masks and the belief both arrive as generators that can only be
consumed once, and neither is small enough to materialise — 472 MB of masks and 115 MB
of particles for a fifteen-minute match — so this tees the mask stream into the belief
filter and takes what it needs from both as they go past.

**The mixture fit is where the lossiness lives**, and it is the only lossy step in the
artifact that is not simple quantisation. Sixteen weighted k-means components stand in
for a thousand particles, so the export reports the divergence between the mixture
rasterised onto the display grid and the particle cloud rasterised onto the same grid.
A lossy encoding whose loss has never been measured is a claim rather than a format.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast.config import ExportSpec, FilterSpec
from shadowcast.geom.bitset import unpack_rows
from shadowcast.l1_events.schema import MatchEvents
from shadowcast.l3_infer.metrics import LatticeIndex, belief_summary
from shadowcast.l3_infer.pf import BeliefFilter
from shadowcast.l3_infer.policy import Observation, PublicInfo
from shadowcast.l4_export.encode import fit_mixture, quantise_positions, rasterise_mixture
from shadowcast.l4_export.spec import SCALAR_NAMES, ArtifactDims
from shadowcast.terrain.terrain import Terrain

__all__ = ["BuildResult", "build_arrays", "downsample_mask"]

MaskFactory = Callable[[], Iterator[tuple[int, np.ndarray, np.ndarray]]]


class BuildResult:
    """The arrays, their dims, and what the encoding cost."""

    def __init__(
        self,
        dims: ArtifactDims,
        arrays: dict[str, np.ndarray],
        stats: dict[str, Any],
    ) -> None:
        self.dims = dims
        self.arrays = arrays
        self.stats = stats


def downsample_mask(mask: np.ndarray, grid: int, out_grid: int) -> np.ndarray:
    """Packed 512² team mask to a packed `out_grid`² bitmap, LSB-first within bytes.

    A coarse cell counts as visible if *any* fine cell in it is. The alternative — a
    majority rule — would erase exactly the features the display exists to show: a ward's
    vision cone through a brush entrance is a few fine cells wide and would vanish, and
    the frontend would render a map with less vision than the engine computed.
    """
    bits = unpack_rows(mask, grid)
    block = grid // out_grid
    coarse = bits.reshape(out_grid, block, out_grid, block).any(axis=(1, 3))
    return np.packbits(coarse, axis=-1, bitorder="little").reshape(-1)


def build_arrays(
    events: MatchEvents,
    positions: np.ndarray,
    valid: np.ndarray,
    obs: Observation,
    public: PublicInfo,
    masks: MaskFactory,
    terrain: Terrain,
    filter_spec: FilterSpec | None = None,
    export_spec: ExportSpec | None = None,
    lattice: LatticeIndex | None = None,
    fit_iterations: int = 5,
    kl_samples: int = 128,
) -> BuildResult:
    """Run the belief filter over the match and collect everything the artifact ships."""
    filter_spec = filter_spec or FilterSpec()
    export_spec = export_spec or ExportSpec()
    lattice = lattice or LatticeIndex(terrain)

    n_ticks = int(obs.n_ticks)
    n_champs = int(events.n_heroes)
    grid = terrain.grid
    mask_stride = C.TICK_HZ // export_spec.mask_hz
    belief_stride = C.TICK_HZ // export_spec.belief_hz
    position_stride = C.TICK_HZ // export_spec.position_hz

    mask_ticks = len(range(0, n_ticks, mask_stride))
    belief_ticks = len(range(0, n_ticks, belief_stride))
    position_ticks = len(range(0, n_ticks, position_stride))
    dims = ArtifactDims(
        position_ticks=position_ticks,
        mask_ticks=mask_ticks,
        belief_ticks=belief_ticks,
        champions=n_champs,
    )

    mask_bytes = export_spec.mask_grid * export_spec.mask_grid // 8
    out_masks = np.zeros((mask_ticks, C.N_TEAMS, mask_bytes), dtype=np.uint8)

    def tee(sink) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        for tick, m0, m1 in masks():
            sink(tick, m0, m1)
            yield tick, m0, m1

    def record_mask(tick: int, m0: np.ndarray, m1: np.ndarray) -> None:
        if tick % mask_stride or tick // mask_stride >= mask_ticks:
            return
        row = tick // mask_stride
        out_masks[row, 0] = downsample_mask(m0, grid, export_spec.mask_grid)
        out_masks[row, 1] = downsample_mask(m1, grid, export_spec.mask_grid)

    k = export_spec.belief_components
    out_belief = np.zeros((belief_ticks, C.N_TEAMS, C.N_ENEMIES, k, 4), dtype=np.uint8)
    out_seen = np.zeros((belief_ticks, C.N_TEAMS, C.N_ENEMIES), dtype=np.uint8)
    out_scalars = np.zeros((belief_ticks, len(SCALAR_NAMES)), dtype=np.float32)

    centres = np.zeros((C.N_TEAMS, C.N_ENEMIES, k, 2))
    warm = np.zeros((C.N_TEAMS, C.N_ENEMIES), dtype=bool)
    fit = np.zeros((k, 4))
    kl_sum = 0.0
    kl_n = 0

    filt = BeliefFilter(filter_spec, terrain)
    for belief in filt.run(obs, public, tee(record_mask)):
        tick = belief.tick
        if tick % belief_stride or tick // belief_stride >= belief_ticks:
            continue
        row = tick // belief_stride

        for o in range(C.N_TEAMS):
            visible = 0
            for e in range(C.N_ENEMIES):
                seen = bool(belief.seen[o, e])
                out_seen[row, o, e] = 1 if seen else 0
                visible += seen

                cell = belief.cell[o, e]
                weights = _weights(belief.logw[o, e])
                entropy, area = belief_summary(lattice, cell, weights, filter_spec)
                out_scalars[row, o * C.N_ENEMIES + e] = entropy
                out_scalars[row, C.N_TEAMS * C.N_ENEMIES + o * C.N_ENEMIES + e] = area

                if seen:
                    # A point mass needs no mixture. Leaving the previous tick's bytes in
                    # place would encode as a delta of zero, but the frontend never reads
                    # them, so clarity wins over four bytes.
                    continue

                pts = _cell_points(cell, grid)
                fit_mixture(pts, weights, centres[o, e], bool(warm[o, e]), fit_iterations, fit)
                warm[o, e] = True
                out_belief[row, o, e] = _quantise_components(fit)
                if kl_n < kl_samples:
                    kl_sum += _mixture_divergence(pts, weights, fit)
                    kl_n += 1

            out_scalars[row, 2 * C.N_TEAMS * C.N_ENEMIES + 1 + o] = visible

        order_h = out_scalars[row, : C.N_ENEMIES].sum()
        chaos_h = out_scalars[row, C.N_ENEMIES : 2 * C.N_ENEMIES].sum()
        # Positive means Order knows more about Chaos than the reverse: Order's
        # uncertainty about its enemies is lower than Chaos's uncertainty about its own.
        out_scalars[row, 2 * C.N_TEAMS * C.N_ENEMIES] = chaos_h - order_h

    mask_row = np.minimum(np.arange(belief_ticks) * belief_stride // mask_stride, mask_ticks - 1)
    popcount = np.bitwise_count(out_masks).sum(axis=2) / (mask_bytes * 8)
    out_scalars[:, -2] = popcount[mask_row, 0]
    out_scalars[:, -1] = popcount[mask_row, 1]

    tick_index = np.arange(position_ticks) * position_stride
    out_positions = quantise_positions(np.nan_to_num(positions[tick_index, :n_champs]))
    out_alive = valid[tick_index, :n_champs].astype(np.uint8)

    stats = {
        "mixture_kl_mean": round(kl_sum / max(kl_n, 1), 5),
        "mixture_kl_samples": kl_n,
        "belief_fits": int((out_seen == 0).sum()),
        "visible_fraction": round(float(out_seen.mean()), 4),
        **filt.describe(),
    }
    return BuildResult(
        dims=dims,
        arrays={
            "positions": out_positions,
            "alive": out_alive,
            "masks": out_masks,
            "belief_seen": out_seen,
            "belief": out_belief,
            "scalars": out_scalars,
        },
        stats=stats,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _weights(logw: np.ndarray) -> np.ndarray:
    m = logw.max()
    if not np.isfinite(m):
        return np.full(logw.shape, 1.0 / logw.size)
    w = np.exp(logw - m)
    total = w.sum()
    return w / total if total > 0 else np.full(logw.shape, 1.0 / logw.size)


def _cell_points(cell: np.ndarray, grid: int) -> np.ndarray:
    """Flat cell indices to world-space cell centres."""
    j, i = np.divmod(cell.astype(np.int64), grid)
    pts = np.empty((cell.size, 2))
    pts[:, 0] = C.WORLD_MIN_X + (i + 0.5) * C.GRID_CELL_SIZE
    pts[:, 1] = C.WORLD_MIN_Z + (j + 0.5) * C.GRID_CELL_SIZE
    return pts


def _quantise_components(fit: np.ndarray) -> np.ndarray:
    """`(k, 4)` float components to bytes: x, z over the world span; weight and sigma.

    Sigma is quantised over 0..2000 units rather than the world span. A component's
    spread is a few hundred units at most — beyond that the cluster would have been split
    — so spending the byte on the range that occurs gives 7.8 units of resolution instead
    of 58.
    """
    out = np.empty(fit.shape, dtype=np.uint8)
    out[:, 0] = np.clip((fit[:, 0] - C.WORLD_MIN_X) / C.WORLD_SPAN * 255.0, 0, 255)
    out[:, 1] = np.clip((fit[:, 1] - C.WORLD_MIN_Z) / C.WORLD_SPAN * 255.0, 0, 255)
    out[:, 2] = np.clip(fit[:, 2] * 255.0, 0, 255)
    out[:, 3] = np.clip(fit[:, 3] / 2000.0 * 255.0, 0, 255)
    return out


def _mixture_divergence(pts: np.ndarray, weights: np.ndarray, fit: np.ndarray) -> float:
    """KL from the particle cloud to its mixture, **both rasterised the same way**.

    The comparison has to be like for like, and a first version was not. Binning the
    particles into a histogram and comparing it against smooth Gaussians measured mostly
    the spikiness of a thousand samples over a thousand bins — it reported 0.32 nats for
    a fit that was in fact close, because the reference was a discretisation artefact
    rather than the belief.

    So the particle cloud goes through the same kernel the mixture does: each particle is
    a component of weight `w` whose sigma falls to the same floor. What remains is the
    question actually being asked — did sixteen components capture the shape of a
    thousand particles — measured on the grid the frontend draws.
    """
    grid = C.DISPLAY_BELIEF_GRID
    reference = np.zeros((pts.shape[0], 4))
    reference[:, 0:2] = pts
    reference[:, 2] = weights
    p = rasterise_mixture(reference, grid).reshape(-1)
    q = rasterise_mixture(fit, grid).reshape(-1)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    eps = 1e-12
    q = (q + eps) / (q.sum() + eps * q.size)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / q[nz])).sum())
