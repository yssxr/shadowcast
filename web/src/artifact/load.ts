/**
 * Loading an artifact, and reading it without index arithmetic at every call site.
 *
 * The generated reader in `../generated/artifact.ts` returns flat typed arrays, which is
 * the right thing for it to do — it is generated from the format and knows nothing about
 * what the numbers mean. This wraps it in accessors so a renderer asks for "champion 3's
 * position at tick 900" rather than computing `(900 * 10 + 3) * 2`.
 *
 * That indirection is worth one file. Index arithmetic spread across four views is how a
 * frontend acquires an off-by-one that only shows up on one champion in one view.
 */

import {
  SCALARS,
  decode,
  type Meta,
  type Sections,
} from "../generated/artifact.ts";

/** World bounds, from the navgrid header. Mirrors `constants.py`. */
export const WORLD = {
  minX: -1.1048965,
  minZ: 32.7558,
  span: 14759.455,
} as const;

const POSITION_STEPS = 1 << 12;

/** Beyond this much movement in one tick, snap rather than interpolate. */
const SNAP_UNITS = 600;

export interface Hero {
  slot: number;
  name: string;
  champion: string;
  team: number;
  role: string;
}

export interface WardEvent {
  t0: number;
  t1: number;
  x: number;
  z: number;
  team: number;
  owner: number;
  sight: number;
}

export interface DeathEvent {
  t: number;
  victim: number;
  killer: number;
  respawn: number | null;
  confidence: number;
}

export class Artifact {
  readonly meta: Meta;
  readonly sections: Sections;
  readonly heroes: Hero[];
  readonly wards: WardEvent[];
  readonly deaths: DeathEvent[];

  /**
   * `enemyIndex[observer][slot]` -> the enemy index used by `belief` and `seen`, or -1.
   *
   * Precomputed because the alternative is a filter-and-sort, and the callers are the
   * per-frame ones: `visibleAt` runs for every champion and every trail tick, which is
   * around 140 lookups per map per frame. At 60 fps with two maps that was 17,000
   * array sorts a second to answer a question with ten possible answers.
   */
  private readonly enemyIndex: Int8Array[];
  private readonly enemySlots: number[][];

  constructor(meta: Meta, sections: Sections) {
    this.meta = meta;
    this.sections = sections;
    this.heroes = (meta.heroes as Hero[]) ?? [];
    const events = meta.events as { wards?: WardEvent[]; deaths?: DeathEvent[] };
    this.wards = events.wards ?? [];
    this.deaths = events.deaths ?? [];

    this.enemyIndex = [];
    this.enemySlots = [];
    for (let observer = 0; observer < meta.dims.teams; observer++) {
      const slots = this.heroes
        .filter((h) => h.team !== observer)
        .map((h) => h.slot)
        .sort((a, b) => a - b);
      const table = new Int8Array(Math.max(this.heroes.length, 1)).fill(-1);
      slots.forEach((slot, index) => (table[slot] = index));
      this.enemyIndex.push(table);
      this.enemySlots.push(slots);
    }
  }

  get duration(): number {
    return this.meta.duration;
  }

  get positionHz(): number {
    return this.meta.dims.position_ticks / this.meta.duration;
  }

  get beliefHz(): number {
    return this.meta.dims.belief_ticks / this.meta.duration;
  }

  get maskHz(): number {
    return this.meta.dims.mask_ticks / this.meta.duration;
  }

  /** Clamped, because a scrubber at exactly t=duration would index one past the end. */
  positionTick(t: number): number {
    const n = this.meta.dims.position_ticks;
    return Math.min(n - 1, Math.max(0, Math.round(t * this.positionHz)));
  }

  beliefTick(t: number): number {
    const n = this.meta.dims.belief_ticks;
    return Math.min(n - 1, Math.max(0, Math.round(t * this.beliefHz)));
  }

  maskTick(t: number): number {
    const n = this.meta.dims.mask_ticks;
    return Math.min(n - 1, Math.max(0, Math.round(t * this.maskHz)));
  }

  /** World position of a champion at a position tick. Allocates — see `positionInto`. */
  position(tick: number, slot: number): [number, number] {
    const { champions } = this.meta.dims;
    const base = (tick * champions + slot) * 2;
    const p = this.sections.positions;
    return [
      WORLD.minX + (p[base] / (POSITION_STEPS - 1)) * WORLD.span,
      WORLD.minZ + (p[base + 1] / (POSITION_STEPS - 1)) * WORLD.span,
    ];
  }

  /**
   * Write a position interpolated between two ticks, so champions move smoothly.
   *
   * Positions are exported at 8 Hz and the canvas draws at 60, so without this a champion
   * advances in eight visible steps a second — which reads as a stutter even though every
   * frame is on time. Linear interpolation between the bracketing ticks is what the game
   * client does with the same data.
   *
   * A jump beyond `SNAP_UNITS` is not interpolated: a Flash, a teleport or a respawn is a
   * discontinuity, and sliding a champion across the map over an eighth of a second would
   * invent a path they never walked.
   */
  positionLerpInto(t: number, slot: number, out: Float64Array, offset: number): void {
    const exact = Math.max(0, t * this.positionHz);
    const a = Math.min(this.meta.dims.position_ticks - 1, Math.floor(exact));
    const b = Math.min(this.meta.dims.position_ticks - 1, a + 1);
    const f = exact - a;
    this.positionInto(a, slot, out, offset);
    if (b === a || f <= 0) return;
    const ax = out[offset];
    const az = out[offset + 1];
    this.positionInto(b, slot, out, offset);
    const dx = out[offset] - ax;
    const dz = out[offset + 1] - az;
    if (dx * dx + dz * dz > SNAP_UNITS * SNAP_UNITS) return;
    out[offset] = ax + dx * f;
    out[offset + 1] = az + dz * f;
  }

  /**
   * Write a world position into `out` at `offset`, allocating nothing.
   *
   * The draw loop reads about 150 positions a frame per map — ten champions plus
   * fourteen trail ticks each — and a tuple per read is 18,000 short-lived arrays a
   * second. That does not show up as a slow frame, it shows up as a periodic one.
   */
  positionInto(tick: number, slot: number, out: Float64Array, offset: number): void {
    const base = (tick * this.meta.dims.champions + slot) * 2;
    const p = this.sections.positions;
    out[offset] = WORLD.minX + (p[base] / (POSITION_STEPS - 1)) * WORLD.span;
    out[offset + 1] = WORLD.minZ + (p[base + 1] / (POSITION_STEPS - 1)) * WORLD.span;
  }

  alive(tick: number, slot: number): boolean {
    return this.sections.alive[tick * this.meta.dims.champions + slot] !== 0;
  }

  /** Whether `observer` could see enemy index `enemy` at a belief tick. */
  seen(tick: number, observer: number, enemy: number): boolean {
    const { teams, enemies } = this.meta.dims;
    return this.sections.belief_seen[(tick * teams + observer) * enemies + enemy] !== 0;
  }

  /**
   * The 16 mixture components for one (observer, enemy) at a belief tick, in world units.
   *
   * Meaningless where `seen` is true — the belief is a point mass there and the exporter
   * writes zeros. Callers must check `seen` first, which every renderer does.
   */
  belief(
    tick: number,
    observer: number,
    enemy: number,
    out: Float64Array = new Float64Array(this.meta.dims.components * 4),
  ): Float64Array {
    const { teams, enemies, components } = this.meta.dims;
    const base = ((tick * teams + observer) * enemies + enemy) * components * 4;
    const b = this.sections.belief;
    for (let c = 0; c < components; c++) {
      const i = base + c * 4;
      const o = c * 4;
      out[o] = WORLD.minX + (b[i] / 255) * WORLD.span;
      out[o + 1] = WORLD.minZ + (b[i + 1] / 255) * WORLD.span;
      out[o + 2] = b[i + 2] / 255;
      out[o + 3] = (b[i + 3] / 255) * 2000;
    }
    return out;
  }

  scalar(tick: number, name: keyof typeof SCALARS): number {
    return this.sections.scalars[tick * this.meta.dims.scalars + SCALARS[name]];
  }

  /** Entropy in bits for one (observer, enemy). */
  entropy(tick: number, observer: number, enemy: number): number {
    const key = `entropy_${observer}_${enemy}` as keyof typeof SCALARS;
    return this.scalar(tick, key);
  }

  /** 90% credible-region area in ku² (a "ku" is a thousand game units). */
  area(tick: number, observer: number, enemy: number): number {
    const key = `area_${observer}_${enemy}` as keyof typeof SCALARS;
    return this.scalar(tick, key);
  }

  /**
   * Read one bit of a team's visibility bitmap.
   *
   * The bitmap is 128² row-major, LSB-first within each byte, and row 0 is z-minimum —
   * the same convention the terrain PNG is written in, so the two line up without a flip
   * anywhere in the renderer.
   */
  visible(tick: number, team: number, i: number, j: number): boolean {
    const { teams, mask_bytes } = this.meta.dims;
    const bit = j * 128 + i;
    const byte = (tick * teams + team) * mask_bytes + (bit >> 3);
    return (this.sections.masks[byte] & (1 << (bit & 7))) !== 0;
  }

  /** Enemy index within `observer`'s view back to a hero slot. */
  enemySlot(observer: number, enemy: number): number {
    return this.enemySlots[observer][enemy];
  }

  /** Hero slot to the enemy index `observer` files them under, or -1 for an ally. */
  enemyIndexOf(observer: number, slot: number): number {
    return this.enemyIndex[observer]?.[slot] ?? -1;
  }

  /**
   * Whether `observer` could see the champion in `slot` at belief tick `tick`.
   *
   * Allies are always visible to their own team, which is not a modelling shortcut —
   * it is the game rule that makes the fog oracle work at all: a fog event about a
   * champion can only have come from the opposing team's view.
   */
  seenSlot(tick: number, observer: number, slot: number): boolean {
    // An unresolved observer team is not an ally relationship, it is missing data, so
    // it answers "not visible" rather than throwing or claiming vision it cannot have.
    const index = this.enemyIndex[observer]?.[slot];
    if (index === undefined) return false;
    if (index < 0) return true;
    return this.seen(tick, observer, index);
  }
}

export async function loadArtifact(baseUrl: string): Promise<Artifact> {
  const [metaRes, dataRes] = await Promise.all([
    fetch(`${baseUrl}/meta.json`),
    fetch(`${baseUrl}/data.bin.gz`),
  ]);
  if (!metaRes.ok) throw new Error(`meta.json: ${metaRes.status} ${metaRes.statusText}`);
  if (!dataRes.ok) throw new Error(`data.bin.gz: ${dataRes.status} ${dataRes.statusText}`);

  const meta: Meta = await metaRes.json();
  let buffer = await dataRes.arrayBuffer();

  // In production the payload is served with `Content-Encoding: gzip` and the browser
  // has already inflated it. A dev server or a static host that does not set the header
  // hands over the compressed bytes instead, so this inflates them rather than failing a
  // checksum with a message that would send someone hunting through the writer.
  if (isGzip(buffer)) {
    buffer = await inflate(buffer);
  }
  return new Artifact(meta, decode(meta, buffer));
}

function isGzip(buffer: ArrayBuffer): boolean {
  const head = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength));
  return head[0] === 0x1f && head[1] === 0x8b;
}

async function inflate(buffer: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).arrayBuffer();
}
