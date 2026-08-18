/**
 * Champions, wards, trails, everything drawn on top of the terrain.
 *
 * All of it follows the mockup: champions as filled dots with a dark halo and an inner
 * ring for junglers, wards as gold diamonds with a dashed radius ring, trails as
 * fourteen fading ticks.
 *
 * The one rule worth stating is about **what gets drawn at all**. In the observer's view,
 * an enemy is drawn only where the observer could actually see them. Drawing every
 * champion always and letting the fog dim them would be an ordinary replay viewer, and
 * would quietly contradict the whole point: the map is supposed to show what a team knew,
 * not what happened.
 */

import { color, rgba, teamColor } from "../theme.ts";
import { WORLD } from "../artifact/load.ts";

export interface Projection {
  size: number;
}

/** World (x, z) to canvas pixels, with z up the screen. */
export function project(x: number, z: number, size: number): [number, number] {
  return [
    ((x - WORLD.minX) / WORLD.span) * size,
    size - ((z - WORLD.minZ) / WORLD.span) * size,
  ];
}

export function unproject(px: number, py: number, size: number): [number, number] {
  return [
    WORLD.minX + (px / size) * WORLD.span,
    WORLD.minZ + ((size - py) / size) * WORLD.span,
  ];
}

export interface ChampionMark {
  x: number;
  z: number;
  team: number;
  jungler: boolean;
  dead: boolean;
  /** Draw a ring: this is the champion the view is focused on. */
  focused?: boolean;
}

export function drawChampion(
  ctx: CanvasRenderingContext2D,
  mark: ChampionMark,
  size: number,
): void {
  const [px, py] = project(mark.x, mark.z, size);
  const r = Math.max(3.5, size / 110);

  // The halo is what keeps a dot readable against both lit ground and a bright belief
  // cloud. Without it a blue champion inside a blue cloud disappears.
  ctx.beginPath();
  ctx.arc(px, py, r + 2.2, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(6,6,9,.78)";
  ctx.fill();

  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = mark.dead ? rgba(teamColor(mark.team), 0.28) : teamColor(mark.team);
  ctx.fill();

  if (mark.jungler) {
    ctx.beginPath();
    ctx.arc(px, py, r * 0.42, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(8,8,11,.85)";
    ctx.fill();
  }

  if (mark.focused) {
    ctx.beginPath();
    ctx.arc(px, py, r + 4.5, 0, Math.PI * 2);
    ctx.strokeStyle = rgba(color.accent, 0.85);
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }
}

/**
 * Fourteen ticks of history at .26 alpha, fading to nothing.
 *
 * `points` is a flat `[x, z, x, z, ...]` buffer with `count` pairs, reused across frames
 * rather than rebuilt. See `Artifact.positionInto`.
 */
export function drawTrail(
  ctx: CanvasRenderingContext2D,
  points: Float64Array,
  count: number,
  team: number,
  size: number,
): void {
  if (count < 2) return;
  ctx.save();
  ctx.lineWidth = Math.max(1, size / 400);
  ctx.lineCap = "round";
  for (let k = 1; k < count; k++) {
    ctx.strokeStyle = rgba(teamColor(team), (k / count) * 0.26);
    ctx.beginPath();
    const ax = ((points[(k - 1) * 2] - WORLD.minX) / WORLD.span) * size;
    const ay = size - ((points[(k - 1) * 2 + 1] - WORLD.minZ) / WORLD.span) * size;
    const bx = ((points[k * 2] - WORLD.minX) / WORLD.span) * size;
    const by = size - ((points[k * 2 + 1] - WORLD.minZ) / WORLD.span) * size;
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  ctx.restore();
}

export interface WardMark {
  x: number;
  z: number;
  sight: number;
  /** 0..1, how much of its lifetime remains. Drives the diamond's opacity. */
  remaining: number;
  showRadius: boolean;
}

export function drawWard(ctx: CanvasRenderingContext2D, ward: WardMark, size: number): void {
  const [px, py] = project(ward.x, ward.z, size);
  const s = Math.max(3, size / 150);

  if (ward.showRadius) {
    const r = (ward.sight / WORLD.span) * size;
    ctx.save();
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = rgba(color.accent, 0.22);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // A ward about to expire fades rather than vanishing, because the moment it goes is
  // the moment the entropy spikes and it should be possible to see it coming.
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = rgba(color.accent, 0.35 + ward.remaining * 0.55);
  ctx.fillRect(-s / 2, -s / 2, s, s);
  ctx.strokeStyle = "rgba(6,6,9,.7)";
  ctx.lineWidth = 1;
  ctx.strokeRect(-s / 2, -s / 2, s, s);
  ctx.restore();
}

/** A death, as a short-lived cross at the place it happened. */
export function drawDeath(
  ctx: CanvasRenderingContext2D,
  x: number,
  z: number,
  team: number,
  age: number,
  size: number,
): void {
  const fade = Math.max(0, 1 - age / 6);
  if (fade <= 0) return;
  const [px, py] = project(x, z, size);
  const s = Math.max(4, size / 90) * (1 + (1 - fade) * 0.6);
  ctx.save();
  ctx.strokeStyle = rgba(teamColor(team), fade * 0.9);
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(px - s, py - s);
  ctx.lineTo(px + s, py + s);
  ctx.moveTo(px + s, py - s);
  ctx.lineTo(px - s, py + s);
  ctx.stroke();
  ctx.restore();
}
