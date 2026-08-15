"""The artifact format, declared once.

This module is a **seam**, in the same sense `packets/source.py` is. Everything about
the on-disk format — which sections exist, their dtypes, their shapes, how each is
compressed — is stated here as data, and three things are generated from it: the Python
writer, the Python reader, and the TypeScript reader the site uses. A format change is
one edit in one table, and a change that reaches Python without reaching TypeScript
fails a test rather than producing a site that renders garbage.

That failure mode is the whole reason for the indirection. A section whose dtype changed
on one side of the language boundary does not throw — `new Float32Array` over bytes that
are now `Uint16Array` returns numbers, they are simply the wrong ones, and the first
symptom is a map where champions stand in walls.

## Why raw bytes and gzip rather than a framed format

Arrow IPC, protobuf and flatbuffers all lose here, and it is worth saying why rather than
appearing to have not considered them. Every section in this artifact is a **dense
numeric array of known shape**. That is the one case where a framed format's schema
machinery buys nothing — there are no optional fields, no varints worth having, no
polymorphism — while costing a dependency, a parse step, and a copy. Concatenated raw
bytes with a JSON directory means the entire TypeScript reader is `fetch` →
`arrayBuffer` → construct typed-array views at the recorded offsets, and the browser
inflates the gzip for free on the wire.

## The two rules that are not negotiable

**Sections are 8-byte aligned.** `new Float32Array(buffer, offset, n)` throws a
`RangeError` when `offset` is not a multiple of 4, and the message names neither the
section nor the writer that misaligned it. Padding every section to 8 bytes makes every
dtype we might ever add safe in advance.

**Deltas are modular.** A delta between two `u8` values ranges over [-255, 255] and does
not fit in a byte, so the encoder stores `(new - old) mod 256` and the decoder adds back
modulo 256. That is exact for every input rather than exact for small changes, which
matters because the one case that breaks a clamped scheme — a champion flashing across
the map — is precisely the case an analyst is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from shadowcast import constants as C

__all__ = ["CODECS", "DTYPES", "SECTIONS", "Section", "resolve_shape"]

Codec = Literal["raw", "delta", "xor"]

#: dtype name -> (numpy dtype, bytes, TypeScript array class). The TS name is part of
#: the format: it is what the generated reader constructs, and getting it wrong is the
#: silent-garbage failure this whole module exists to prevent.
#:
#: **Little-endian, explicitly.** JavaScript typed arrays read native order and every
#: platform that matters is little-endian, so `uint16` would work everywhere this will
#: ever run — but "works because nobody has a big-endian machine" is a latent bug rather
#: than a decision, and the `<` costs nothing.
DTYPES: dict[str, tuple[str, int, str]] = {
    "u8": ("uint8", 1, "Uint8Array"),
    "i8": ("int8", 1, "Int8Array"),
    "u16": ("<u2", 2, "Uint16Array"),
    "i16": ("<i2", 2, "Int16Array"),
    "u32": ("<u4", 4, "Uint32Array"),
    "f32": ("<f4", 4, "Float32Array"),
}

#: Dtypes a delta or XOR codec may be applied to.
#:
#: **Integers only, and this is a correctness constraint rather than a preference.** Both
#: codecs are defined on the integer representation, so applying either to a float
#: truncates it: encoding `[1.75, 2.5]` after `[1.5, 2.25]` and decoding gives
#: `[1.0, 2.0]`. Worse, it *appears* to work — a delta-coded `scalars` section compressed
#: twelve times better than raw, which is what destroying the fractional part will do.
#: A float section that needs to be small should be quantised to an integer first, where
#: the loss is a stated number rather than a silent one.
CODEC_DTYPES: frozenset[str] = frozenset({"u8", "i8", "u16", "i16", "u32"})

#: Codec -> what the decoder must do, in words. Kept next to the code that implements it
#: so a new codec cannot be added on one side of the boundary only.
CODECS: dict[str, str] = {
    "raw": "bytes as stored",
    "delta": (
        "row 0 and every keyframe are absolute; other rows add to the previous, modulo the dtype"
    ),
    "xor": "row 0 and every keyframe are absolute; other rows XOR with the previous",
}


@dataclass(frozen=True, slots=True)
class Section:
    """One dense array in the artifact.

    `shape` entries are either integers or the name of a dimension resolved from the
    artifact's `dims` block. Symbolic dimensions are what let the reader size its views
    without the writer having to hardcode a match length.
    """

    name: str
    dtype: str
    shape: tuple[int | str, ...]
    codec: Codec = "raw"
    #: Rows along axis 0 between absolute frames. Zero means only row 0 is absolute.
    #:
    #: Not just a compression knob — it is what makes the artifact **seekable**. A
    #: scrubber dropped at 11:42 must not have to decode from tick zero to draw a frame,
    #: and without keyframes a delta stream forces exactly that.
    keyframe: int = 0
    doc: str = ""

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"{self.name}: unknown dtype {self.dtype!r}")
        if self.codec not in CODECS:
            raise ValueError(f"{self.name}: unknown codec {self.codec!r}")
        if self.codec != "raw" and not self.shape:
            raise ValueError(f"{self.name}: {self.codec} needs an axis to run along")
        if self.codec != "raw" and self.dtype not in CODEC_DTYPES:
            raise ValueError(
                f"{self.name}: codec {self.codec!r} is defined on the integer "
                f"representation and would truncate {self.dtype}. Quantise to an "
                f"integer dtype first, so the loss is a number rather than a surprise."
            )

    @property
    def itemsize(self) -> int:
        return DTYPES[self.dtype][1]

    @property
    def numpy_dtype(self) -> str:
        return DTYPES[self.dtype][0]

    @property
    def ts_array(self) -> str:
        return DTYPES[self.dtype][2]


def resolve_shape(shape: tuple[int | str, ...], dims: dict[str, int]) -> tuple[int, ...]:
    """Substitute named dimensions, failing loudly on an unknown name."""
    out = []
    for axis in shape:
        if isinstance(axis, int):
            out.append(axis)
        elif axis in dims:
            out.append(int(dims[axis]))
        else:
            raise KeyError(f"artifact dims has no entry {axis!r}; known: {sorted(dims)}")
    return tuple(out)


# ---------------------------------------------------------------------------
# The format
# ---------------------------------------------------------------------------
SECTIONS: tuple[Section, ...] = (
    Section(
        name="positions",
        dtype="u16",
        shape=("position_ticks", "champions", 2),
        codec="delta",
        keyframe=C.POSITION_EXPORT_HZ * 8,
        doc=(
            "Champion positions, quantised to 12 bits over the world span (3.6 u/LSB). "
            "The quantisation and the delta width are one decision rather than two: at "
            "full u16 precision a 50-unit tick move is 220 LSB, while at 12 bits even a "
            "400-unit Flash is 110 — so deltas stay small and gzip stays effective "
            "without any clamping or escape mechanism."
        ),
    ),
    Section(
        name="alive",
        dtype="u8",
        shape=("position_ticks", "champions"),
        codec="raw",
        doc=(
            "1 while alive. MEASURED at 0.1 kB gzipped under every codec — ten booleans "
            "that change a handful of times a match compress to nothing whatever is done "
            "to them, so it stays raw and stays readable."
        ),
    ),
    Section(
        name="masks",
        dtype="u8",
        shape=("mask_ticks", "teams", "mask_bytes"),
        codec="xor",
        keyframe=C.EXPORT_MASK_HZ * 8,
        doc=(
            "Per-team visibility as a 128^2 bitmap, row-major, LSB-first within each "
            "byte. XOR against the previous frame because vision changes at its edges: "
            "a tick typically flips a few hundred of 16,384 bits, so the XOR is almost "
            "all zeros and gzip removes it."
        ),
    ),
    Section(
        name="belief_seen",
        dtype="u8",
        shape=("belief_ticks", "teams", "enemies"),
        codec="raw",
        doc=(
            "1 where the observer could see that enemy, in which case the belief is a "
            "point mass and the mixture for that tick is meaningless. The frontend draws "
            "a dot from `positions` instead of rasterising anything."
        ),
    ),
    Section(
        name="belief",
        dtype="u8",
        shape=("belief_ticks", "teams", "enemies", "components", 4),
        codec="delta",
        keyframe=int(C.BELIEF_EXPORT_HZ * C.BELIEF_KEYFRAME_SECONDS),
        doc=(
            "The belief as a 16-component mixture: (x, z, weight, sigma) per component, "
            "each a byte. Grids do not fit by any margin — 64^2 u8 at 8 Hz is 295 MB a "
            "match and 37 MB even at 1 Hz, and no compression closes that. Components "
            "are k-means centres warm-started from the previous tick's, which is worth "
            "6.5x on its own (19.9 kB against 129.5) because it keeps component identity "
            "stable across ticks. "
            "XOR rather than delta, MEASURED at 17.5 kB against 19.9: a component that "
            "jitters by one unit encodes as 0xFF under a modular delta, which is a "
            "high-entropy byte, and as 0x01 under XOR, which is not."
        ),
    ),
    Section(
        name="scalars",
        dtype="f32",
        shape=("belief_ticks", "scalars"),
        codec="raw",
        doc=(
            "Per-tick metrics: entropy and credible-region area for each of the ten "
            "(observer, enemy) pairs, the information advantage, and per-team visible "
            "counts. Small enough that delta-coding floats would trade real complexity "
            "for nothing."
        ),
    ),
)

SECTIONS_BY_NAME: dict[str, Section] = {s.name: s for s in SECTIONS}

#: Names of the per-tick scalars, in the order `scalars` stores them. Exported into the
#: TypeScript types so the frontend indexes by name rather than by a number that means
#: nothing at the call site.
SCALAR_NAMES: tuple[str, ...] = (
    *[f"entropy_{o}_{e}" for o in range(C.N_TEAMS) for e in range(C.N_ENEMIES)],
    *[f"area_{o}_{e}" for o in range(C.N_TEAMS) for e in range(C.N_ENEMIES)],
    "advantage",
    "visible_order",
    "visible_chaos",
    "mask_area_order",
    "mask_area_chaos",
)


@dataclass(frozen=True, slots=True)
class ArtifactDims:
    """The symbolic dimensions every section's shape is resolved against."""

    position_ticks: int
    mask_ticks: int
    belief_ticks: int
    champions: int = 2 * C.N_ENEMIES
    teams: int = C.N_TEAMS
    enemies: int = C.N_ENEMIES
    components: int = C.BELIEF_COMPONENTS
    mask_bytes: int = C.EXPORT_MASK_GRID * C.EXPORT_MASK_GRID // 8
    scalars: int = len(SCALAR_NAMES)

    def to_dict(self) -> dict[str, int]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True, slots=True)
class SectionEntry:
    """One row of the artifact's directory, as written into `meta.json`."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    codec: str
    keyframe: int
    offset: int
    length: int
    crc32: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "codec": self.codec,
            "keyframe": self.keyframe,
            "offset": self.offset,
            "length": self.length,
            "crc32": self.crc32,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SectionEntry:
        return cls(
            name=raw["name"],
            dtype=raw["dtype"],
            shape=tuple(raw["shape"]),
            codec=raw["codec"],
            keyframe=int(raw["keyframe"]),
            offset=int(raw["offset"]),
            length=int(raw["length"]),
            crc32=int(raw["crc32"]),
        )


#: What produced the match an artifact describes. Carried in `meta.json` and displayed,
#: because a viewer cannot tell from the pixels and the difference is enormous: fog
#: agreement is 98% on a generated match and 68% on a real one, so a synthetic artifact
#: shows the engine's geometry rather than its accuracy. Shipping one unlabelled would
#: misrepresent the project to exactly the audience the site exists for.
PROVENANCE_REAL = "real"
PROVENANCE_SYNTHETIC = "synthetic"


@dataclass(frozen=True, slots=True)
class ArtifactMeta:
    """`meta.json`: everything needed to read `data.bin` and nothing that belongs in it."""

    schema_version: int
    match_id: str
    duration: float
    tick_hz: int
    dims: dict[str, int]
    sections: list[SectionEntry]
    config: dict[str, str]
    heroes: list[dict[str, Any]] = field(default_factory=list)
    events: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    #: `real` or `synthetic`. Defaults to synthetic on an artifact that predates the
    #: field, which is the safe direction: mislabelling a real match as generated
    #: understates the work, while the reverse overstates it.
    provenance: str = PROVENANCE_SYNTHETIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "duration": self.duration,
            "tick_hz": self.tick_hz,
            "provenance": self.provenance,
            "dims": self.dims,
            "sections": [s.to_dict() for s in self.sections],
            "config": self.config,
            "heroes": self.heroes,
            "events": self.events,
            "stats": self.stats,
            "scalar_names": list(SCALAR_NAMES),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ArtifactMeta:
        if int(raw["schema_version"]) != C.ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact schema version {raw['schema_version']} but this build reads "
                f"{C.ARTIFACT_SCHEMA_VERSION}. Re-export rather than guessing."
            )
        return cls(
            schema_version=int(raw["schema_version"]),
            match_id=raw["match_id"],
            duration=float(raw["duration"]),
            tick_hz=int(raw["tick_hz"]),
            dims=dict(raw["dims"]),
            sections=[SectionEntry.from_dict(s) for s in raw["sections"]],
            config=dict(raw.get("config", {})),
            heroes=list(raw.get("heroes", [])),
            events=dict(raw.get("events", {})),
            stats=dict(raw.get("stats", {})),
            provenance=str(raw.get("provenance", PROVENANCE_SYNTHETIC)),
        )

    def section(self, name: str) -> SectionEntry:
        for s in self.sections:
            if s.name == name:
                return s
        raise KeyError(f"artifact has no section {name!r}")
