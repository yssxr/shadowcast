"""Writing and reading the artifact container.

Two files per match: `meta.json` and `data.bin.gz`. The JSON holds the section directory
and everything a human might want to read; the binary is the concatenated sections, each
8-byte aligned, each with its CRC32 recorded.

**The gzip is on disk, not in the reader.** Serving `data.bin.gz` with
`Content-Encoding: gzip` makes the browser inflate it during transfer, which means the
site ships zero decompression JavaScript and the bytes arrive already expanded. That is
the whole reason for a flat concatenation over a container format that compresses per
member: the transport already has a compressor and it is better than anything we would
bundle.

**Every section carries a CRC32**, and this is not belt-and-braces. The failure this
format is most exposed to is a section read at the wrong offset or the wrong dtype —
which produces numbers, not an error. A checksum per section turns "the map looks
strange" into a named assertion at load time, and it is what makes the cross-language
conformance test possible: `tests/test_export.py` has Node read the same file and compare
its checksums against Python's.
"""

from __future__ import annotations

import gzip
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shadowcast import constants as C
from shadowcast.config import ExportSpec
from shadowcast.l4_export.encode import apply_codec, invert_codec
from shadowcast.l4_export.spec import (
    SECTIONS,
    SECTIONS_BY_NAME,
    ArtifactDims,
    ArtifactMeta,
    SectionEntry,
    resolve_shape,
)

__all__ = ["Artifact", "read_artifact", "write_artifact"]

META_NAME = "meta.json"
DATA_NAME = "data.bin.gz"


@dataclass(frozen=True, slots=True)
class Artifact:
    """A decoded artifact: the metadata plus one array per section."""

    meta: ArtifactMeta
    arrays: dict[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    @property
    def dims(self) -> dict[str, int]:
        return self.meta.dims


def _pad(n: int, align: int) -> int:
    return (-n) % align


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with `null`, recursively.

    `json.dumps` emits bare `NaN` and `Infinity` by default. That is accepted by Python's
    own parser and by nothing else — `JSON.parse` throws on it, so a single unknown
    respawn time makes the whole artifact unreadable in a browser. The cross-language
    conformance test found this on its first run, which is the entire argument for
    having one.

    Paired with `allow_nan=False` below: this converts the values that legitimately have
    no number, and the flag turns anything missed into an exception at write time rather
    than a parse error in someone else's runtime.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def write_artifact(
    out_dir: Path | str,
    match_id: str,
    duration: float,
    dims: ArtifactDims,
    arrays: dict[str, np.ndarray],
    heroes: list[dict[str, Any]] | None = None,
    events: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
    config: dict[str, str] | None = None,
    spec: ExportSpec | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Encode, concatenate and write. Returns the directory and a size report.

    Every declared section must be present and must already have the declared dtype and
    shape. Coercing here instead would hide a real upstream mistake: a `belief` array
    that arrives as float64 means something built it wrong, and quietly casting it to u8
    turns that into a rendering artefact discovered weeks later.
    """
    spec = spec or ExportSpec()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = dims.to_dict()

    blob = bytearray()
    entries: list[SectionEntry] = []
    raw_sizes: dict[str, int] = {}

    for section in SECTIONS:
        if section.name not in arrays:
            raise KeyError(f"artifact is missing section {section.name!r}")
        arr = np.ascontiguousarray(arrays[section.name])
        want_shape = resolve_shape(section.shape, resolved)
        if arr.shape != want_shape:
            raise ValueError(
                f"section {section.name!r} has shape {arr.shape}, declared {want_shape}"
            )
        if arr.dtype != np.dtype(section.numpy_dtype):
            raise TypeError(
                f"section {section.name!r} has dtype {arr.dtype}, declared "
                f"{section.numpy_dtype}. The writer does not coerce, because a dtype "
                f"surprise here is an upstream bug and casting would hide it."
            )

        payload = apply_codec(arr, section.codec, section.keyframe).tobytes()
        offset = len(blob)
        blob += payload
        blob += b"\x00" * _pad(len(blob), spec.section_align)
        entries.append(
            SectionEntry(
                name=section.name,
                dtype=section.dtype,
                shape=want_shape,
                codec=section.codec,
                keyframe=section.keyframe,
                offset=offset,
                length=len(payload),
                crc32=zlib.crc32(payload) & 0xFFFFFFFF,
            )
        )
        raw_sizes[section.name] = len(payload)

    meta = ArtifactMeta(
        schema_version=C.ARTIFACT_SCHEMA_VERSION,
        match_id=match_id,
        duration=duration,
        tick_hz=C.TICK_HZ,
        dims=resolved,
        sections=entries,
        config=config or {"export": spec.content_hash},
        heroes=heroes or [],
        events=events or {},
        stats=stats or {},
    )

    data_path = out_dir / DATA_NAME
    # mtime=0 so the same input produces the same bytes; otherwise every export
    # invalidates a CDN cache and no two runs are comparable.
    with gzip.GzipFile(data_path, "wb", compresslevel=9, mtime=0) as fh:
        fh.write(bytes(blob))
    (out_dir / META_NAME).write_text(
        json.dumps(_json_safe(meta.to_dict()), indent=1, allow_nan=False)
    )

    gz_bytes = data_path.stat().st_size
    report = {
        "match_id": match_id,
        "sections": raw_sizes,
        "raw_bytes": len(blob),
        "gzipped_bytes": gz_bytes,
        "meta_bytes": (out_dir / META_NAME).stat().st_size,
        "total_bytes": gz_bytes + (out_dir / META_NAME).stat().st_size,
        "ratio": round(len(blob) / max(gz_bytes, 1), 2),
    }
    return out_dir, report


def read_artifact(path: Path | str, decode: bool = True) -> Artifact:
    """Read an artifact back. The reference the TypeScript reader is checked against."""
    path = Path(path)
    meta = ArtifactMeta.from_dict(json.loads((path / META_NAME).read_text()))
    with gzip.open(path / DATA_NAME, "rb") as fh:
        blob = fh.read()

    arrays: dict[str, np.ndarray] = {}
    for entry in meta.sections:
        section = SECTIONS_BY_NAME.get(entry.name)
        if section is None:
            continue  # a newer writer's section this build does not know about
        payload = blob[entry.offset : entry.offset + entry.length]
        actual = zlib.crc32(payload) & 0xFFFFFFFF
        if actual != entry.crc32:
            raise ValueError(
                f"section {entry.name!r} failed its checksum "
                f"({actual:#010x} != {entry.crc32:#010x}). The section was read at the "
                f"wrong offset or the file is damaged — either way the numbers would "
                f"have been plausible rather than absent."
            )
        arr = np.frombuffer(payload, dtype=np.dtype(section.numpy_dtype)).reshape(entry.shape)
        arrays[entry.name] = invert_codec(arr, entry.codec, entry.keyframe) if decode else arr
    return Artifact(meta=meta, arrays=arrays)
