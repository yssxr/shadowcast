/**
 * Rendering the belief.
 *
 * The artifact ships sixteen mixture components, not a grid, so the field is evaluated
 * here at draw time. That is what makes the artifact a megabyte instead of three hundred:
 * a 64² belief grid at 8 Hz is 295 MB a match and no compression closes that gap.
 *
 * **Clouds are drawn in the ENEMY's colour, not the observer's.** This diverges from the
 * mockup on purpose. A cloud is a belief *about* someone, and when it collapses it
 * collapses into a dot, so cloud and dot sharing a colour makes the collapse read as one
 * event rather than two unrelated marks.
 *
 * ## One rendering, not three
 *
 * There were three modes, cloud, contour, grid: and the toggle was a bad answer to the
 * question of how to draw a distribution. Putting all three on the same data settled it:
 *
 * *Grid* drew the raw display lattice: useful for checking what the model literally said,
 * useless for reading a game. A debug view with a place in the UI it had not earned.
 *
 * *Contour* drew iso-lines at fixed fractions of the peak density and degenerated into
 * two nested rectangles, because a belief concentrated in four cells puts every level on
 * nearly the same cells. It said less than grid did.
 *
 * *Cloud* read well at a glance but had no boundary. You could not tell whether the
 * faint edge held ten percent of the mass or one.
 *
 * So the shipped rendering is the cloud **with its 90% credible boundary drawn on it**.
 * The field gives the at-a-glance read; the outline gives a reader something to point at,
 * and it encloses exactly the area the search-area figure under each map reports. One
 * picture, no toggle, and strictly more than any of the three said alone.
 */

import { beliefColor, hexToRgb } from "../theme.ts";
import { WORLD } from "../artifact/load.ts";

/**
 * Resolution the mixture is evaluated at.
 *
 * The design's display lattice is 32², and an earlier version refused to go finer on the
 * grounds that a 461-unit cell has no sub-cell detail to recover. That is the right
 * argument about a histogram and the wrong one here: the belief is a continuous mixture
 * of Gaussians, so evaluating it at 128² reads the model rather than interpolating it.
 * 115 units a cell, finer than the boundary is meaningful to anyway.
 */
export const BELIEF_GRID = 128;

/** Mass enclosed by the drawn boundary. Matches `CREDIBLE_MASS` in constants.py. */
const CREDIBLE_MASS = 0.9;

/**
 * Resolution of the density histogram used to find the credible threshold.
 *
 * Sorting the field was the obvious implementation and it cost 36 fps: 16,384 floats
 * sorted per enemy per map per frame is six hundred sorts a second. A histogram finds the
 * same threshold in one linear pass, to within a 256th of the peak density, far finer
 * than a boundary drawn on a 128-cell lattice can express.
 */
const THRESHOLD_BINS = 256;

export interface BeliefScratch {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  img: ImageData;
  /** One field per enemy, reused. Five of them, so the cloud can be composited from
   *  the merged maximum and the outlines drawn on top afterwards. */
  fields: Float32Array[];
  /** The merged field across enemies, reused. */
  merged: Float32Array;
  /** Histogram bins for the credible threshold, reused. */
  bins: Float64Array;
}

/**
 * Allocate the scratch surfaces once, per map.
 *
 * Reused rather than created per draw. The first version allocated a canvas inside the
 * draw call, five per map per frame, six hundred GPU-backed surfaces a second across two
 * maps, and the symptom was not a slow frame but a periodic one, as the collector ran.
 */
export function createScratch(enemies = 5): BeliefScratch {
  const canvas = document.createElement("canvas");
  canvas.width = BELIEF_GRID;
  canvas.height = BELIEF_GRID;
  const ctx = canvas.getContext("2d")!;
  const cells = BELIEF_GRID * BELIEF_GRID;
  return {
    canvas,
    ctx,
    img: ctx.createImageData(BELIEF_GRID, BELIEF_GRID),
    fields: Array.from({ length: enemies }, () => new Float32Array(cells)),
    merged: new Float32Array(cells),
    bins: new Float64Array(THRESHOLD_BINS),
  };
}

/** Evaluate a mixture onto the render lattice. */
export function rasteriseMixture(components: Float64Array, out: Float32Array): Float32Array {
  out.fill(0);
  const cell = WORLD.span / BELIEF_GRID;
  const n = components.length / 4;

  for (let c = 0; c < n; c++) {
    const w = components[c * 4 + 2];
    if (w <= 0) continue;
    const cx = components[c * 4];
    const cz = components[c * 4 + 1];
    const sigma = Math.max(components[c * 4 + 3], cell * 0.5);
    const inv = 1 / (2 * sigma * sigma);
    const norm = w / (2 * Math.PI * sigma * sigma);

    // Three sigma covers 99.7% of the component's mass; past that the exponential is
    // below anything the display could show.
    const reach = Math.ceil((3 * sigma) / cell);
    const ci = Math.floor(((cx - WORLD.minX) / WORLD.span) * BELIEF_GRID);
    const cj = Math.floor(((cz - WORLD.minZ) / WORLD.span) * BELIEF_GRID);

    for (let j = Math.max(0, cj - reach); j <= Math.min(BELIEF_GRID - 1, cj + reach); j++) {
      const z = WORLD.minZ + (j + 0.5) * cell;
      for (let i = Math.max(0, ci - reach); i <= Math.min(BELIEF_GRID - 1, ci + reach); i++) {
        const x = WORLD.minX + (i + 0.5) * cell;
        const d2 = (x - cx) * (x - cx) + (z - cz) * (z - cz);
        out[j * BELIEF_GRID + i] += norm * Math.exp(-d2 * inv);
      }
    }
  }
  return out;
}

/**
 * Zero the field wherever no champion could stand.
 *
 * The filter already constrains its particles to the navmesh, but the mixture's Gaussians
 * are round and leak a little probability into walls, and a soft cloud edge overlapping
 * a wall reads as "he might be inside that wall". Removing it is cheap and it is what
 * makes the clouds look terrain-shaped rather than blobby.
 */
export function maskToWalkable(
  field: Float32Array,
  walkable: Uint8Array,
  terrainGrid: number,
): Float32Array {
  const block = terrainGrid / BELIEF_GRID;
  for (let j = 0; j < BELIEF_GRID; j++) {
    for (let i = 0; i < BELIEF_GRID; i++) {
      let any = false;
      for (let dj = 0; dj < block && !any; dj++) {
        for (let di = 0; di < block; di++) {
          if (walkable[(j * block + dj) * terrainGrid + (i * block + di)]) {
            any = true;
            break;
          }
        }
      }
      if (!any) field[j * BELIEF_GRID + i] = 0;
    }
  }
  return field;
}

/** Peak-relative, because absolute density has no display scale. */
export function normalise(field: Float32Array): Float32Array {
  let peak = 0;
  for (let k = 0; k < field.length; k++) if (field[k] > peak) peak = field[k];
  if (peak > 0) for (let k = 0; k < field.length; k++) field[k] /= peak;
  return field;
}

/**
 * Paint the merged field as a soft cloud.
 *
 * `alpha = v^0.75 * 205` and `screen` compositing are both from the mockup. The exponent
 * matters more than it looks: a linear ramp makes the tail of a diffuse belief invisible,
 * and the tail is where the interesting claim lives. A cloud carved by negative
 * information is mostly tail.
 */
export function drawCloud(
  ctx: CanvasRenderingContext2D,
  field: Float32Array,
  enemyTeam: number,
  size: number,
  scratch: BeliefScratch,
): void {
  const [r, g, b] = hexToRgb(beliefColor(enemyTeam));
  const { canvas, ctx: sctx, img } = scratch;

  for (let j = 0; j < BELIEF_GRID; j++) {
    const row = BELIEF_GRID - 1 - j; // the field ascends in z, the canvas descends
    for (let i = 0; i < BELIEF_GRID; i++) {
      const v = field[j * BELIEF_GRID + i];
      const k = (row * BELIEF_GRID + i) * 4;
      img.data[k] = r;
      img.data[k + 1] = g;
      img.data[k + 2] = b;
      img.data[k + 3] = v > 0 ? Math.min(255, Math.pow(v, 0.75) * 205) : 0;
    }
  }
  sctx.putImageData(img, 0, 0);

  ctx.save();
  ctx.globalCompositeOperation = "screen";
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(canvas, 0, 0, size, size);
  ctx.restore();
}

/**
 * Outline the region holding 90% of one enemy's probability mass.
 *
 * **Per enemy, never merged.** A credible region is a statement about one champion, and
 * the union of five 90% regions is not the 90% region of anything. The cloud behind it
 * merges because every enemy on a map is the same team and therefore the same colour; the
 * boundary cannot.
 *
 * The threshold comes from sorting densities descending and walking the cumulative sum,
 * which is the definition of a highest-density region. A fixed fraction of the peak was
 * the earlier version, and it degenerated into rectangles.
 */
export function drawCredibleBoundary(
  ctx: CanvasRenderingContext2D,
  field: Float32Array,
  enemyTeam: number,
  size: number,
  scratch: BeliefScratch,
): void {
  let total = 0;
  let peak = 0;
  for (let k = 0; k < field.length; k++) {
    total += field[k];
    if (field[k] > peak) peak = field[k];
  }
  if (total <= 0 || peak <= 0) return;

  // Highest-density region: bin the densities, then walk down from the top accumulating
  // mass until 90% of it is inside. Equivalent to sorting descending and walking the
  // cumulative sum, at a fraction of the cost.
  const bins = scratch.bins;
  bins.fill(0);
  const scale = (THRESHOLD_BINS - 1) / peak;
  for (let k = 0; k < field.length; k++) {
    if (field[k] > 0) bins[(field[k] * scale) | 0] += field[k];
  }
  let acc = 0;
  let bin = THRESHOLD_BINS - 1;
  for (; bin >= 0; bin--) {
    acc += bins[bin];
    if (acc >= CREDIBLE_MASS * total) break;
  }
  const threshold = Math.max(bin, 0) / scale;
  if (threshold <= 0) return;

  const [r, g, b] = hexToRgb(beliefColor(enemyTeam));
  const step = size / BELIEF_GRID;
  ctx.save();
  ctx.strokeStyle = `rgba(${r},${g},${b},.6)`;
  ctx.lineWidth = Math.max(1, size / 460);
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let j = 0; j < BELIEF_GRID; j++) {
    for (let i = 0; i < BELIEF_GRID; i++) {
      if (field[j * BELIEF_GRID + i] < threshold) continue;
      const x = i * step;
      const y = size - (j + 1) * step;
      // Only the edges where the region ends, so this is a boundary rather than a hatch
      // over every cell inside it.
      if (i === 0 || field[j * BELIEF_GRID + i - 1] < threshold) {
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + step);
      }
      if (i === BELIEF_GRID - 1 || field[j * BELIEF_GRID + i + 1] < threshold) {
        ctx.moveTo(x + step, y);
        ctx.lineTo(x + step, y + step);
      }
      if (j === 0 || field[(j - 1) * BELIEF_GRID + i] < threshold) {
        ctx.moveTo(x, y + step);
        ctx.lineTo(x + step, y + step);
      }
      if (j === BELIEF_GRID - 1 || field[(j + 1) * BELIEF_GRID + i] < threshold) {
        ctx.moveTo(x, y);
        ctx.lineTo(x + step, y);
      }
    }
  }
  ctx.stroke();
  ctx.restore();
}
