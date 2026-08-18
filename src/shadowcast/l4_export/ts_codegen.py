"""Generating the TypeScript reader from the same table the Python writer uses.

The generated file is **committed**, and CI regenerates it and fails on a diff. That is
the mechanism the whole seam rests on: a section added, retyped or recoded in Python
cannot merge without its TypeScript counterpart, because the check is a byte comparison
rather than a promise.

Handwriting the reader instead would work right up until the first schema change, and
then it would keep working, returning numbers from the wrong offsets, silently, with a
map that renders and champions who stand in walls. There is no runtime error available
to catch that, so it has to be caught at build time or not at all.

The output is deliberately dependency-free and small enough to read in one sitting. If
this ever grows past a couple of hundred lines, the format has become too clever.
"""

from __future__ import annotations

import json
from pathlib import Path

from shadowcast import constants as C
from shadowcast.l4_export.spec import CODECS, DTYPES, SCALAR_NAMES, SECTIONS

__all__ = ["GENERATED_HEADER", "generate_typescript", "write_typescript"]

GENERATED_HEADER = """// GENERATED from src/shadowcast/l4_export/spec.py by `shadowcast export`.
// Do not edit. CI regenerates this file and fails on a diff, which is what stops a
// format change reaching Python without reaching TypeScript. A mismatch here does not
// throw, it returns numbers from the wrong offsets and renders a plausible wrong map.
"""


def _shape_comment(shape: tuple[int | str, ...]) -> str:
    return " x ".join(str(a) for a in shape)


def generate_typescript() -> str:
    """The whole reader, as a string."""
    lines: list[str] = [GENERATED_HEADER]

    lines.append(f"export const SCHEMA_VERSION = {C.ARTIFACT_SCHEMA_VERSION};\n")

    lines.append("export interface SectionEntry {")
    lines.append("  name: string;")
    lines.append("  dtype: string;")
    lines.append("  shape: number[];")
    lines.append("  codec: string;")
    lines.append("  keyframe: number;")
    lines.append("  offset: number;")
    lines.append("  length: number;")
    lines.append("  crc32: number;")
    lines.append("}\n")

    lines.append("export interface Dims {")
    for name in (
        "position_ticks",
        "mask_ticks",
        "belief_ticks",
        "champions",
        "teams",
        "enemies",
        "components",
        "mask_bytes",
        "scalars",
    ):
        lines.append(f"  {name}: number;")
    lines.append("}\n")

    lines.append("export interface Hero {")
    lines.append("  slot: number;")
    lines.append("  name: string;")
    lines.append("  champion: string;")
    lines.append("  team: number;")
    lines.append("  role: string;")
    lines.append("}\n")

    lines.append("export interface Meta {")
    lines.append("  schema_version: number;")
    lines.append("  match_id: string;")
    lines.append("  duration: number;")
    lines.append("  tick_hz: number;")
    lines.append('  /** "real" for decoded replay packets, "synthetic" for a generated match. */')
    lines.append('  provenance: "real" | "synthetic";')
    lines.append("  dims: Dims;")
    lines.append("  sections: SectionEntry[];")
    lines.append("  config: Record<string, string>;")
    lines.append("  heroes: Hero[];")
    lines.append("  events: Record<string, unknown>;")
    lines.append("  stats: Record<string, unknown>;")
    lines.append("  scalar_names: string[];")
    lines.append("}\n")

    lines.append("/** Decoded sections. Shapes are in the comments; the arrays are flat. */")
    lines.append("export interface Sections {")
    for section in SECTIONS:
        lines.append(f"  /** {_shape_comment(section.shape)}: {section.codec} */")
        lines.append(f"  {section.name}: {section.ts_array};")
    lines.append("}\n")

    lines.append("/** Index of each per-tick scalar within a `scalars` row. */")
    lines.append("export const SCALARS = {")
    for i, name in enumerate(SCALAR_NAMES):
        lines.append(f"  {name}: {i},")
    lines.append("} as const;\n")

    lines.append("export const SECTION_SHAPES: Record<string, (string | number)[]> = {")
    for section in SECTIONS:
        entries = ", ".join(json.dumps(a) if isinstance(a, str) else str(a) for a in section.shape)
        lines.append(f"  {section.name}: [{entries}],")
    lines.append("};\n")

    lines.append(_CRC_SOURCE)
    lines.append(_DECODE_SOURCE)
    lines.append(_READ_SOURCE)
    return "\n".join(lines) + "\n"


_CRC_SOURCE = """
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

/** CRC32 of a byte range. Matches Python `zlib.crc32`, which is what the test compares. */
export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
"""

_DECODE_SOURCE = """
function isKeyframe(row: number, keyframe: number): boolean {
  return row === 0 || (keyframe > 0 && row % keyframe === 0);
}

/**
 * Invert a section codec in place, along axis 0.
 *
 * `delta` adds each row to the one before it modulo the dtype width; typed arrays
 * already wrap on assignment, so the modulus is implicit and correct. `xor` is its own
 * inverse. Keyframe rows are absolute and are skipped, which is also what makes seeking
 * possible: a scrubber decodes from the nearest keyframe rather than from tick zero.
 */
function invertCodec(
  data: Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Float32Array,
  rows: number,
  stride: number,
  codec: string,
  keyframe: number,
): void {
  if (codec === "raw" || rows <= 1) return;
  for (let t = 1; t < rows; t++) {
    if (isKeyframe(t, keyframe)) continue;
    const here = t * stride;
    const prev = here - stride;
    if (codec === "delta") {
      for (let i = 0; i < stride; i++) data[here + i] = data[here + i] + data[prev + i];
    } else if (codec === "xor") {
      for (let i = 0; i < stride; i++) data[here + i] = data[here + i] ^ data[prev + i];
    } else {
      throw new Error(`unknown codec ${codec}`);
    }
  }
}
"""

_READ_SOURCE = """
const ARRAY_OF: Record<string, new (b: ArrayBuffer, o: number, n: number) => any> = {
  u8: Uint8Array,
  i8: Int8Array,
  u16: Uint16Array,
  i16: Int16Array,
  u32: Uint32Array,
  f32: Float32Array,
};

const BYTES_OF: Record<string, number> = { u8: 1, i8: 1, u16: 2, i16: 2, u32: 4, f32: 4 };

/**
 * Decode `data.bin` against its `meta.json`.
 *
 * Serve the payload as `data.bin.gz` with `Content-Encoding: gzip` and the browser
 * inflates it during transfer, so nothing here decompresses anything.
 *
 * Every section is checked against the CRC32 the writer recorded. That check is not
 * defensive padding: the failure this format is exposed to is a section read at the
 * wrong offset or the wrong dtype, which yields numbers rather than an error, and the
 * first symptom would be a rendered map that is subtly wrong.
 */
export function decode(meta: Meta, buffer: ArrayBuffer, verify = true): Sections {
  if (meta.schema_version !== SCHEMA_VERSION) {
    throw new Error(
      `artifact schema ${meta.schema_version} but this reader is ${SCHEMA_VERSION}`,
    );
  }
  const out: Record<string, unknown> = {};
  for (const entry of meta.sections) {
    const itemsize = BYTES_OF[entry.dtype];
    const ctor = ARRAY_OF[entry.dtype];
    if (!ctor) throw new Error(`unknown dtype ${entry.dtype} in section ${entry.name}`);
    if (verify) {
      const raw = new Uint8Array(buffer, entry.offset, entry.length);
      const actual = crc32(raw);
      if (actual !== entry.crc32 >>> 0) {
        throw new Error(
          `section ${entry.name} failed its checksum (${actual} != ${entry.crc32})`,
        );
      }
    }
    const count = entry.length / itemsize;
    const view = new ctor(buffer.slice(entry.offset, entry.offset + entry.length), 0, count);
    const rows = entry.shape[0];
    invertCodec(view, rows, count / rows, entry.codec, entry.keyframe);
    out[entry.name] = view;
  }
  return out as unknown as Sections;
}

/** Fetch and decode an artifact directory. */
export async function loadArtifact(baseUrl: string): Promise<{ meta: Meta; sections: Sections }> {
  const meta: Meta = await (await fetch(`${baseUrl}/meta.json`)).json();
  const buffer = await (await fetch(`${baseUrl}/data.bin.gz`)).arrayBuffer();
  return { meta, sections: decode(meta, buffer) };
}
"""


def write_typescript(path: Path | str) -> tuple[Path, bool]:
    """Write the reader. Returns `(path, changed)` so callers can report a drift."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = generate_typescript()
    changed = not path.exists() or path.read_text() != text
    if changed:
        path.write_text(text)
    return path, changed


def describe() -> dict[str, object]:
    return {
        "schema_version": C.ARTIFACT_SCHEMA_VERSION,
        "sections": len(SECTIONS),
        "dtypes": len(DTYPES),
        "codecs": len(CODECS),
        "scalars": len(SCALAR_NAMES),
    }
