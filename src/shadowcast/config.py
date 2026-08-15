"""Frozen configuration specs, and the staleness machinery built on them.

Every stage of the pipeline reads a file and writes a file. Each written file
carries a `StageHeader` recording the stage version, the hash of the config that
produced it, the hash of its inputs, and the git commit. Every consumer validates
the input hash before doing anything.

This exists because of one specific failure mode. The FOV table is a 160 MB
derived artifact keyed to the terrain it was built from. If the terrain changes
and the table does not, every visibility mask afterwards is subtly wrong — and
nothing crashes. Masks still look like masks, unions still union, the site still
renders, and the validation numbers move by a few percent in a way that reads as
a modelling issue. Content hashing turns that into a loud error at the boundary.

The specs are frozen dataclasses so a hash cannot drift from the values it
describes. Adding a field changes the hash, which invalidates every downstream
artifact, which is the correct and intended consequence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from shadowcast import constants as C

__all__ = [
    "ExportSpec",
    "FilterSpec",
    "GridSpec",
    "StageHeader",
    "TerrainSpec",
    "TickSpec",
    "content_hash",
    "file_hash",
    "git_sha",
]

_HASH_LEN = 16  # 64 bits of hex; ample for collision-free artifact keying


def content_hash(payload: Any) -> str:
    """Stable hash of a JSON-serialisable payload.

    `sort_keys` and a fixed separator make this stable across dict insertion
    order and Python versions. Floats go through `repr` via json, which is
    round-trip exact for IEEE doubles, so a hash never changes for a value that
    did not change.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_HASH_LEN]


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hash a file's bytes. Used for source data we did not generate."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:_HASH_LEN]


@lru_cache(maxsize=1)
def git_sha() -> str:
    """Current commit, with a `-dirty` suffix if the tree has uncommitted changes.

    Recorded in artifact headers so a surprising number can be traced to the code
    that produced it. The `-dirty` marker matters more than the sha: it says the
    artifact is not reproducible from any commit.
    """
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if sha.returncode != 0:
            return "unknown"
        out = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            out += "-dirty"
        return out
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class _Spec:
    """Mixin giving a frozen dataclass a content hash over its own fields."""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @property
    def content_hash(self) -> str:
        return content_hash({"spec": type(self).__name__, "fields": self.to_dict()})


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GridSpec(_Spec):
    """The internal simulation grid and the FOV table's radius bound."""

    grid: int = C.GRID
    world_min_x: float = C.WORLD_MIN_X
    world_min_z: float = C.WORLD_MIN_Z
    world_span: float = C.WORLD_SPAN
    rmax_units: float = C.RMAX_UNITS

    @property
    def cell_size(self) -> float:
        return self.world_span / self.grid

    @property
    def rmax_cells(self) -> int:
        import math

        return math.ceil(self.rmax_units / self.cell_size)

    @property
    def fov_window(self) -> int:
        return 2 * self.rmax_cells + 1

    @property
    def n_cells(self) -> int:
        return self.grid * self.grid


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TerrainSpec(_Spec):
    """How the navgrid becomes three boolean channels.

    `see_through_transmits_vision` is a modelling switch we expect to leave on
    forever, but it is expressed as a flag rather than baked in so the fog
    agreement rate can be measured both ways. If turning it off barely moves the
    number, that is worth knowing; if it moves it a lot, that is a result.
    """

    source: str = C.NAVGRID_PROVENANCE
    navgrid_hash: str = ""  # filled from the actual file at build time
    see_through_transmits_vision: bool = True
    brush_blocks_inward: bool = True
    expected_brush_patches: int = C.SR_BRUSH_PATCH_COUNT


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TickSpec(_Spec):
    tick_hz: int = C.TICK_HZ
    window_seconds: float = C.MATCH_WINDOW_SECONDS

    @property
    def dt(self) -> float:
        return 1.0 / self.tick_hz

    @property
    def n_ticks(self) -> int:
        return int(self.window_seconds * self.tick_hz) + 1


# ---------------------------------------------------------------------------
# Belief
# ---------------------------------------------------------------------------
MotionModel = Literal[
    "uniform",  # B0: resample uniformly over walkable every tick
    "disc",  # B1: uniform in a growing Euclidean disc
    "geodisc",  # B1': same, but geodesic — isolates the value of the navmesh
    "constant_velocity",  # B2: extrapolate, no terrain clamp
    "navmesh_diffusion",  # B3: random walk on the navmesh
    "navmesh_behavioural",  # Full: navmesh walk with an objective-aware prior
]

ObsModel = Literal[
    "none",  # B0 only
    "positive_only",  # collapse on sighting, ignore absence of sighting
    "positive_and_negative",  # Full: absence of sighting is evidence
]


@dataclass(frozen=True, slots=True)
class FilterSpec(_Spec):
    """A belief model. The six baselines are six instances of this.

    Making the baselines configurations rather than separate implementations is
    what makes the ablation trustworthy: B3 and Full run the identical code path
    and differ only in `obs`, so a difference between them cannot be an artefact
    of one being written more carefully than the other.
    """

    motion: MotionModel = "navmesh_behavioural"
    obs: ObsModel = "positive_and_negative"
    particles: int = C.PARTICLES
    pd_interior: float = C.PD_INTERIOR
    pd_edge: float = C.PD_EDGE
    pd_edge_ring_cells: int = C.PD_EDGE_RING_CELLS
    entropy_lattice: int = C.ENTROPY_LATTICE
    entropy_lattice_version: str = C.ENTROPY_LATTICE_VERSION
    credible_mass: float = C.CREDIBLE_MASS
    smoothing: float = C.SCORING_SMOOTHING
    seed: int = 0
    p_stay: float = C.PARTICLE_STAY_PROB
    persistence: float = C.HEADING_PERSISTENCE
    goal_beta: float = C.GOAL_BETA
    goal_arrive_cells: float = C.GOAL_ARRIVE_CELLS
    sub_steps: int = C.MOTION_SUB_STEPS
    v_max: float = C.V_MAX_UNITS_PER_SECOND
    ess_resample: float = C.ESS_RESAMPLE_FRACTION
    ess_depletion: float = C.ESS_DEPLETION_FRACTION

    def __post_init__(self) -> None:
        # Entropy in bits is only defined relative to a lattice, so a filter whose
        # maximum entropy exceeds log2(particles) would be reporting its own
        # particle budget. Refuse the configuration rather than emit the number.
        import math

        if not 0.0 <= self.p_stay < 1.0:
            # The walk normalises the stay weight against the move weights, so
            # p_stay == 1 divides by zero. A belief that never moves is not a
            # model of anything either.
            raise ValueError(f"p_stay must be in [0, 1), got {self.p_stay}")

        # No slack. An earlier version allowed two bits of headroom, which let the
        # shipped configuration through while the estimator was in fact pinned --
        # measured entropy 8.74 bits against a log2(400) = 8.64 ceiling.
        max_bits = 2 * math.log2(self.entropy_lattice)
        ceiling = math.log2(self.particles)
        if max_bits > ceiling:
            raise ValueError(
                f"entropy lattice {self.entropy_lattice}^2 admits up to {max_bits:.2f} bits "
                f"but {self.particles} particles cap the plug-in estimator at "
                f"{ceiling:.2f} bits. Entropy would measure the particle count, not the "
                "game. Coarsen the lattice or raise the particle budget."
            )

    @property
    def uses_negative_information(self) -> bool:
        return self.obs == "positive_and_negative"

    @property
    def effective_goal_beta(self) -> float:
        """Zero unless the model is the behavioural one.

        Goal-seeking is what makes `navmesh_behavioural` more than
        `navmesh_diffusion`, so letting a diffusion spec carry a non-zero beta
        would silently collapse the two into one model while the ablation table
        still printed two rows.
        """
        return self.goal_beta if self.motion == "navmesh_behavioural" else 0.0


#: The models the ablation sweep runs.
#:
#: Two adjacent pairs carry the argument, and they are adjacent on purpose --
#: each isolates exactly one thing:
#:
#:   diffusion -> behavioural   what the role-conditioned prior is worth
#:   behavioural -> full        what NEGATIVE INFORMATION is worth
#:
#: The second is the thesis. An earlier version of this table compared
#: `navmesh_diffusion + positive_only` directly against
#: `navmesh_behavioural + positive_and_negative`, which differ in two ways at
#: once -- so a win could have come entirely from the prior and the headline
#: claim would have been unsupported by its own ablation.
BASELINES: dict[str, FilterSpec] = {
    "uniform": FilterSpec(motion="uniform", obs="none"),
    "disc": FilterSpec(motion="disc", obs="positive_only"),
    "geodisc": FilterSpec(motion="geodisc", obs="positive_only"),
    "cv": FilterSpec(motion="constant_velocity", obs="positive_only"),
    "diffusion": FilterSpec(motion="navmesh_diffusion", obs="positive_only"),
    "behavioural": FilterSpec(motion="navmesh_behavioural", obs="positive_only"),
    "full": FilterSpec(motion="navmesh_behavioural", obs="positive_and_negative"),
}

#: The ablation pair the write-up stands on, named so no one has to infer it.
THESIS_PAIR = ("behavioural", "full")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExportSpec(_Spec):
    schema_version: int = C.ARTIFACT_SCHEMA_VERSION
    mask_grid: int = C.EXPORT_MASK_GRID
    mask_hz: int = C.EXPORT_MASK_HZ
    belief_components: int = C.BELIEF_COMPONENTS
    belief_hz: int = C.BELIEF_EXPORT_HZ
    belief_keyframe_seconds: float = C.BELIEF_KEYFRAME_SECONDS
    position_quant_bits: int = C.POSITION_QUANT_BITS
    position_hz: int = C.POSITION_EXPORT_HZ
    display_belief_grid: int = C.DISPLAY_BELIEF_GRID
    display_terrain_grid: int = C.DISPLAY_TERRAIN_GRID
    section_align: int = C.ARTIFACT_SECTION_ALIGN


# ---------------------------------------------------------------------------
# Stage headers
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StageHeader:
    """Provenance stamped into every derived file.

    `validate_against` is the whole point. A consumer calls it with the hash of
    what it actually loaded, and a mismatch raises instead of proceeding — which
    is the difference between "the numbers look slightly off" and a stack trace
    naming the stale file.
    """

    stage: str
    stage_version: int
    config_hash: str
    input_hash: str
    git_sha: str = field(default_factory=git_sha)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageHeader:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def validate_against(self, *, config_hash: str, input_hash: str) -> None:
        if self.config_hash != config_hash:
            raise StaleArtifactError(
                f"{self.stage!r} was built with config {self.config_hash} but the current "
                f"config hashes to {config_hash}. Rebuild it."
            )
        if self.input_hash != input_hash:
            raise StaleArtifactError(
                f"{self.stage!r} was built from input {self.input_hash} but the input now "
                f"hashes to {input_hash}. Rebuild it."
            )


class StaleArtifactError(RuntimeError):
    """A derived artifact does not match the inputs or config it is being used with."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def terrain_path(spec: TerrainSpec) -> Path:
    return data_dir() / "terrain" / f"terrain_{spec.content_hash}.npz"


def fov_table_dir(grid: GridSpec, terrain: TerrainSpec) -> Path:
    """The table is keyed by both specs, so a terrain change orphans it by name.

    Naming rather than in-file checking means a stale table is not merely
    detected, it is not found at all — and `fov build` writes the new one beside
    it instead of overwriting, so an interrupted rebuild cannot leave a
    half-written table where a valid one used to be.
    """
    return (
        data_dir()
        / "fov"
        / f"g{grid.grid}_r{grid.rmax_cells}_{grid.content_hash}_{terrain.content_hash}"
    )
