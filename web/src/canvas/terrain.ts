/**
 * Terrain rendering: two cached offscreen canvases, drawn once.
 *
 * Terrain never changes, so redrawing 262,144 cells sixty times a second would be the
 * single most expensive thing on the page and would buy nothing. It is rasterised once
 * into a lit variant and an unlit variant, and every frame after that is two `drawImage`
 * calls with the fog mask deciding which pixels come from which.
 *
 * **Lit and unlit are separate palettes rather than one palette at two opacities.** Fog
 * here is not "the map, dimmer": brush stays legible in fog while open ground recedes,
 * which is what makes a brush-heavy region read as dangerous rather than merely dark. An
 * opacity ramp would flatten that distinction and the map would lose the one visual cue
 * that corresponds to the thing being measured.
 */

import { terrainPalette } from "../theme.ts";

export interface TerrainImage {
  grid: number;
  /** Pre-rasterised, in screen order: canvas row 0 is maximum z. */
  lit: HTMLCanvasElement;
  unlit: HTMLCanvasElement;

  /**
   * Channels indexed `[j * grid + i]` with **j ascending in z**, matching the engine and
   * the visibility bitmap.
   *
   * The PNG is stored flipped so it draws to a canvas without anyone having to remember
   * to flip it, and the flip is undone here, once: so every array in the app shares one
   * convention. Two conventions is how a map ends up correct in three views and upside
   * down in the fourth.
   */
  walkable: Uint8Array;
  brushId: Uint8Array;
  blocksVision: Uint8Array;
}

/**
 * Decode the three-channel terrain PNG into two rasters plus the raw channels.
 *
 * R = walkable, G = brush id, B = blocks vision. The third channel is not derivable from
 * the first: Riot stamps see-through cells along wall diagonals that block movement but
 * transmit vision, so a renderer that inferred one from the other would draw shadows
 * that were patched out after S5 Worlds.
 */
export async function loadTerrain(url: string): Promise<TerrainImage> {
  const image = new Image();
  image.src = url;
  await image.decode();

  const grid = image.naturalWidth;
  const scratch = document.createElement("canvas");
  scratch.width = grid;
  scratch.height = grid;
  const sctx = scratch.getContext("2d", { willReadFrequently: true })!;
  sctx.drawImage(image, 0, 0);
  const pixels = sctx.getImageData(0, 0, grid, grid).data;

  const n = grid * grid;
  const walkable = new Uint8Array(n);
  const brushId = new Uint8Array(n);
  const blocksVision = new Uint8Array(n);
  for (let row = 0; row < grid; row++) {
    const j = grid - 1 - row; // image row 0 is maximum z; our arrays ascend in z
    for (let i = 0; i < grid; i++) {
      const src = (row * grid + i) * 4;
      const k = j * grid + i;
      walkable[k] = pixels[src] > 127 ? 1 : 0;
      brushId[k] = pixels[src + 1];
      blocksVision[k] = pixels[src + 2] > 127 ? 1 : 0;
    }
  }

  return {
    grid,
    walkable,
    brushId,
    blocksVision,
    lit: rasterise(grid, walkable, brushId, terrainPalette.lit),
    unlit: rasterise(grid, walkable, brushId, terrainPalette.unlit),
  };
}

function rasterise(
  grid: number,
  walkable: Uint8Array,
  brushId: Uint8Array,
  palette: { ground: string; wall: string; brush: string },
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = grid;
  canvas.height = grid;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(grid, grid);

  const ground = rgb(palette.ground);
  const wall = rgb(palette.wall);
  const brush = rgb(palette.brush);

  for (let row = 0; row < grid; row++) {
    const j = grid - 1 - row; // back to screen order for the raster
    for (let i = 0; i < grid; i++) {
      const k = j * grid + i;
      const c = brushId[k] > 0 ? brush : walkable[k] ? ground : wall;
      const dst = (row * grid + i) * 4;
      img.data[dst] = c[0];
      img.data[dst + 1] = c[1];
      img.data[dst + 2] = c[2];
      img.data[dst + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

function rgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/**
 * Composite terrain for one team's view: lit inside the visible region, unlit outside.
 *
 * The mask is 128² against a 512² terrain, so each mask cell covers a 4×4 block. The
 * blocks are drawn as clip rectangles over the lit image rather than per-pixel, which
 * keeps this at a few thousand operations a frame instead of a quarter of a million.
 */
export function drawTerrain(
  ctx: CanvasRenderingContext2D,
  terrain: TerrainImage,
  size: number,
  visible: (i: number, j: number) => boolean,
  maskGrid = 128,
): void {
  ctx.imageSmoothingEnabled = false;
  // z is up on screen, and the PNG was written z-up, so it draws directly.
  ctx.drawImage(terrain.unlit, 0, 0, size, size);

  const step = size / maskGrid;
  ctx.save();
  ctx.beginPath();
  for (let j = 0; j < maskGrid; j++) {
    // Merge horizontal runs into one rectangle. A lit region is contiguous by
    // construction. It is a union of discs, so this typically turns 16,384 rectangles
    // into a few hundred.
    let runStart = -1;
    for (let i = 0; i <= maskGrid; i++) {
      const on = i < maskGrid && visible(i, j);
      if (on && runStart < 0) runStart = i;
      if (!on && runStart >= 0) {
        ctx.rect(runStart * step, size - (j + 1) * step, (i - runStart) * step, step);
        runStart = -1;
      }
    }
  }
  ctx.clip();
  ctx.drawImage(terrain.lit, 0, 0, size, size);
  ctx.restore();
}
