// GENERATED from src/shadowcast/l4_export/spec.py by `shadowcast export`.
// Do not edit. CI regenerates this file and fails on a diff, which is what stops a
// format change reaching Python without reaching TypeScript — a mismatch here does not
// throw, it returns numbers from the wrong offsets and renders a plausible wrong map.

export const SCHEMA_VERSION = 1;

export interface SectionEntry {
  name: string;
  dtype: string;
  shape: number[];
  codec: string;
  keyframe: number;
  offset: number;
  length: number;
  crc32: number;
}

export interface Dims {
  position_ticks: number;
  mask_ticks: number;
  belief_ticks: number;
  champions: number;
  teams: number;
  enemies: number;
  components: number;
  mask_bytes: number;
  scalars: number;
}

export interface Hero {
  slot: number;
  name: string;
  champion: string;
  team: number;
  role: string;
}

export interface Meta {
  schema_version: number;
  match_id: string;
  duration: number;
  tick_hz: number;
  /** "real" for decoded replay packets, "synthetic" for a generated match. */
  provenance: "real" | "synthetic";
  dims: Dims;
  sections: SectionEntry[];
  config: Record<string, string>;
  heroes: Hero[];
  events: Record<string, unknown>;
  stats: Record<string, unknown>;
  scalar_names: string[];
}

/** Decoded sections. Shapes are in the comments; the arrays are flat. */
export interface Sections {
  /** position_ticks x champions x 2 — delta */
  positions: Uint16Array;
  /** position_ticks x champions — raw */
  alive: Uint8Array;
  /** mask_ticks x teams x mask_bytes — xor */
  masks: Uint8Array;
  /** belief_ticks x teams x enemies — raw */
  belief_seen: Uint8Array;
  /** belief_ticks x teams x enemies x components x 4 — delta */
  belief: Uint8Array;
  /** belief_ticks x scalars — raw */
  scalars: Float32Array;
}

/** Index of each per-tick scalar within a `scalars` row. */
export const SCALARS = {
  entropy_0_0: 0,
  entropy_0_1: 1,
  entropy_0_2: 2,
  entropy_0_3: 3,
  entropy_0_4: 4,
  entropy_1_0: 5,
  entropy_1_1: 6,
  entropy_1_2: 7,
  entropy_1_3: 8,
  entropy_1_4: 9,
  area_0_0: 10,
  area_0_1: 11,
  area_0_2: 12,
  area_0_3: 13,
  area_0_4: 14,
  area_1_0: 15,
  area_1_1: 16,
  area_1_2: 17,
  area_1_3: 18,
  area_1_4: 19,
  advantage: 20,
  visible_order: 21,
  visible_chaos: 22,
  mask_area_order: 23,
  mask_area_chaos: 24,
} as const;

export const SECTION_SHAPES: Record<string, (string | number)[]> = {
  positions: ["position_ticks", "champions", 2],
  alive: ["position_ticks", "champions"],
  masks: ["mask_ticks", "teams", "mask_bytes"],
  belief_seen: ["belief_ticks", "teams", "enemies"],
  belief: ["belief_ticks", "teams", "enemies", "components", 4],
  scalars: ["belief_ticks", "scalars"],
};


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

