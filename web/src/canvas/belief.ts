/**
 * Rendering the belief cloud.
 *
 * The artifact ships sixteen mixture components, not a grid, so the field is rasterised
 * here at draw time. That is deliberate and it is what makes the artifact a megabyte
 * instead of three hundred: a 64² belief grid at 8 Hz is 295 MB a match and no
 * compression closes that gap.
 *
 * **Clouds are drawn in the ENEMY's colour, not the observer's.** This diverges from the
 * mockup on purpose. A cloud is a belief *about* someone, and when that belief collapses
 * it collapses into a dot — so cloud and dot sharing a colour makes the collapse legible
 * as one event rather than two unrelated marks. Colouring by observer instead means the
 * moment of discovery changes colour, which is exactly backwards.
 *
 * Three render modes, all from the same rasterisation:
 *
 *   cloud    blurred and screen-composited — the default, and the one that reads as
 *            uncertainty rather than as data
 *   contour  iso-lines at fixed density levels, for reading shape precisely
 *   grid     the raw display lattice, for seeing exactly what the model says
 */

import { beliefColor, hexToRgb } from "../theme.ts";
import { WORLD } from "../artifact/load.ts";

export type BeliefMode = "cloud" | "contour" | "grid";

/** The display lattice from the design. Matches `DISPLAY_BELIEF_GRID` in constants.py. */
export const DISPLAY_GRID = 32;

/**
 * Rasterise a mixture onto the display lattice.
 *
 * Mirrors `rasterise_mixture` in `l4_export/encode.py` — the same Gaussian sum with the
 * same half-cell sigma floor. They are checked against each other by the export's KL
 * measurement rather than by assertion, but they have to agree: if the site drew a
 * different field from the one the loss was measured on, the measurement would not be
 * about anything on screen.
 */
export function rasteriseMixture(
  components: Float64Array,
  out: Float32Array = new Float32Array(DISPLAY_GRID * DISPLAY_GRID),
): Float32Array {
  out.fill(0);
  const cell = WORLD.span / DISPLAY_GRID;
  const minSigma = cell * 0.5;
  const n = components.length / 4;

  for (let c = 0; c < n; c++) {
    const w = components[c * 4 + 2];
    if (w <= 0) continue;
    const cx = components[c * 4];
    const cz = components[c * 4 + 1];
    const sigma = Math.max(components[c * 4 + 3], minSigma);
    const inv = 1 / (2 * sigma * sigma);
    const norm = w / (2 * Math.PI * sigma * sigma);

    // Three sigma covers 99.7% of the component's mass; beyond that the exponential is
    // below the eighth of a bit the display can show anyway.
    const reach = Math.ceil((3 * sigma) / cell);
    const ci = Math.floor(((cx - WORLD.minX) / WORLD.span) * DISPLAY_GRID);
    const cj = Math.floor(((cz - WORLD.minZ) / WORLD.span) * DISPLAY_GRID);

    for (let j = Math.max(0, cj - reach); j <= Math.min(DISPLAY_GRID - 1, cj + reach); j++) {
      const z = WORLD.minZ + (j + 0.5) * cell;
      for (let i = Math.max(0, ci - reach); i <= Math.min(DISPLAY_GRID - 1, ci + reach); i++) {
        const x = WORLD.minX + (i + 0.5) * cell;
        const d2 = (x - cx) * (x - cx) + (z - cz) * (z - cz);
        out[j * DISPLAY_GRID + i] += norm * Math.exp(-d2 * inv);
      }
    }
  }
  return out;
}

/** Normalise to a 0..1 field. Peak-relative, because absolute density has no display scale. */
export function normalise(field: Float32Array): Float32Array {
  let peak = 0;
  for (let k = 0; k < field.length; k++) if (field[k] > peak) peak = field[k];
  if (peak <= 0) return field;
  for (let k = 0; k < field.length; k++) field[k] /= peak;
  return field;
}

/**
 * Multiply the field by the walkable mask, at display resolution.
 *
 * The belief filter already constrains particles to the navmesh, so this changes almost
 * nothing for the terrain-aware models — but the mixture's Gaussians are round and leak
 * a little probability into walls, and a cloud with a soft edge overlapping a wall reads
 * as "he might be inside that wall". Cheap to remove, and it is what makes the clouds
 * look terrain-shaped rather than blobby.
 */
export function maskToWalkable(
  field: Float32Array,
  walkable: Uint8Array,
  terrainGrid: number,
): Float32Array {
  const block = terrainGrid / DISPLAY_GRID;
  for (let j = 0; j < DISPLAY_GRID; j++) {
    for (let i = 0; i < DISPLAY_GRID; i++) {
      let any = false;
      for (let dj = 0; dj < block && !any; dj++) {
        for (let di = 0; di < block; di++) {
          if (walkable[(j * block + dj) * terrainGrid + (i * block + di)]) {
            any = true;
            break;
          }
        }
      }
      if (!any) field[j * DISPLAY_GRID + i] = 0;
    }
  }
  return field;
}

/** How far the cloud is upsampled before it is blurred. See `createScratch`. */
const SOFTEN_GRID = DISPLAY_GRID * 4;

export interface BeliefScratch {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  img: ImageData;
  /** Intermediate the blur is applied at, so it never runs at display resolution. */
  soft: HTMLCanvasElement;
  softCtx: CanvasRenderingContext2D;
}

/**
 * Allocate the scratch surfaces once, per map.
 *
 * Two of them, and the second is a performance fix with a measurement behind it.
 *
 * The first version created a canvas inside `drawBelief` — five allocations per map per
 * frame, six hundred a second across two maps, each a GPU-backed surface.
 *
 * The second version reused one surface but applied `ctx.filter = "blur(...)"` while
 * drawing it at full map size. MEASURED: 51.7 fps with an 83 ms worst frame, against
 * 107 fps for the unblurred contour mode. A canvas blur is a per-destination-pixel
 * operation, so blurring during an upscale to 800² costs 640,000 pixels of work per
 * cloud — and it was the single reason the page did not hold 60.
 *
 * So the blur now happens on a 128² intermediate, forty times smaller, and the smooth
 * upscale to map size does the rest of the softening for free in the sampler.
 */
export function createScratch(): BeliefScratch {
  const canvas = document.createElement("canvas");
  canvas.width = DISPLAY_GRID;
  canvas.height = DISPLAY_GRID;
  const ctx = canvas.getContext("2d")!;

  const soft = document.createElement("canvas");
  soft.width = SOFTEN_GRID;
  soft.height = SOFTEN_GRID;

  return {
    canvas,
    ctx,
    img: ctx.createImageData(DISPLAY_GRID, DISPLAY_GRID),
    soft,
    softCtx: soft.getContext("2d")!,
  };
}

/**
 * Paint one belief field onto the map canvas.
 *
 * `alpha = v^0.75 * 205` and `screen` compositing are both from the mockup. The exponent
 * matters more than it looks: a linear ramp makes the tail of a diffuse belief invisible,
 * and the tail is where the interesting claim lives — a cloud that has been carved by
 * negative information is mostly tail.
 */
export function drawBelief(
  ctx: CanvasRenderingContext2D,
  field: Float32Array,
  enemyTeam: number,
  size: number,
  mode: BeliefMode,
  scratch: BeliefScratch,
): void {
  const [r, g, b] = hexToRgb(beliefColor(enemyTeam));

  if (mode === "contour") {
    drawContours(ctx, field, `rgb(${r},${g},${b})`, size);
    return;
  }

  const { canvas, ctx: sctx, img, soft, softCtx } = scratch;

  for (let j = 0; j < DISPLAY_GRID; j++) {
    const row = DISPLAY_GRID - 1 - j; // field ascends in z, canvas descends
    for (let i = 0; i < DISPLAY_GRID; i++) {
      const v = field[j * DISPLAY_GRID + i];
      const k = (row * DISPLAY_GRID + i) * 4;
      img.data[k] = r;
      img.data[k + 1] = g;
      img.data[k + 2] = b;
      img.data[k + 3] = v > 0 ? Math.min(255, Math.pow(v, 0.75) * 205) : 0;
    }
  }
  sctx.putImageData(img, 0, 0);

  ctx.save();
  ctx.globalCompositeOperation = "screen";
  if (mode === "cloud") {
    // Upsample 4x, blur there, then let the smooth upscale to map size do the rest. The
    // blur costs 16,384 pixels instead of 640,000, and the result is visually the same
    // because a 25x magnification of a 32-cell field is already soft.
    softCtx.clearRect(0, 0, SOFTEN_GRID, SOFTEN_GRID);
    softCtx.imageSmoothingEnabled = true;
    softCtx.filter = "blur(2px)";
    softCtx.drawImage(canvas, 0, 0, SOFTEN_GRID, SOFTEN_GRID);
    softCtx.filter = "none";
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(soft, 0, 0, size, size);
  } else {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(canvas, 0, 0, size, size);
  }
  ctx.restore();
}

/**
 * Iso-density contours.
 *
 * Marching squares would give smoother lines, but the display lattice is 32² and its
 * cells are 461 world units — smoothing across a cell that wide invents detail the model
 * does not have. Cell-edge segments are honest about the resolution.
 */
function drawContours(
  ctx: CanvasRenderingContext2D,
  field: Float32Array,
  stroke: string,
  size: number,
): void {
  const step = size / DISPLAY_GRID;
  ctx.save();
  ctx.lineWidth = 1;
  for (const level of [0.15, 0.35, 0.6, 0.85]) {
    ctx.strokeStyle = stroke;
    ctx.globalAlpha = 0.25 + level * 0.5;
    ctx.beginPath();
    for (let j = 0; j < DISPLAY_GRID; j++) {
      for (let i = 0; i < DISPLAY_GRID; i++) {
        const here = field[j * DISPLAY_GRID + i] >= level;
        if (!here) continue;
        const x = i * step;
        const y = size - (j + 1) * step;
        if (i === 0 || field[j * DISPLAY_GRID + i - 1] < level) {
          ctx.moveTo(x, y);
          ctx.lineTo(x, y + step);
        }
        if (i === DISPLAY_GRID - 1 || field[j * DISPLAY_GRID + i + 1] < level) {
          ctx.moveTo(x + step, y);
          ctx.lineTo(x + step, y + step);
        }
        if (j === 0 || field[(j - 1) * DISPLAY_GRID + i] < level) {
          ctx.moveTo(x, y + step);
          ctx.lineTo(x + step, y + step);
        }
        if (j === DISPLAY_GRID - 1 || field[(j + 1) * DISPLAY_GRID + i] < level) {
          ctx.moveTo(x, y);
          ctx.lineTo(x + step, y);
        }
      }
    }
    ctx.stroke();
  }
  ctx.restore();
}
