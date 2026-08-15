/**
 * One team's view of the map.
 *
 * Two of these side by side are the replay view, and the pairing is the whole argument:
 * the same instant rendered twice, each showing only what that team could see, with the
 * belief clouds standing where the other team actually was. Information asymmetry is
 * hard to describe and immediate to look at.
 *
 * ## The frame budget
 *
 * The component renders once and then never re-renders during playback. Everything after
 * mount happens inside a `requestAnimationFrame` callback writing to a canvas, which is
 * why its props are settings rather than state.
 *
 * Inside that callback the rule is **no allocation**. Every buffer — the belief field,
 * the mixture components, the trail points, the scratch surface — is allocated at mount
 * and reused. Five enemies times two maps times sixty frames is six hundred allocations
 * a second otherwise, and the symptom is not a slow frame but a periodic one: a sawtooth
 * as the collector runs, which reads as stutter rather than as slowness.
 *
 * The terrain composite is **cached against the mask tick**. Vision is exported at 4 Hz
 * and the canvas draws at 60, so the fog changes on one frame in fifteen; recomputing a
 * clip path of several hundred rectangles on the other fourteen is the single largest
 * saving available here.
 */

import { useEffect, useRef } from "react";
import type { Artifact } from "../artifact/load.ts";
import type { TerrainImage } from "../canvas/terrain.ts";
import { drawTerrain } from "../canvas/terrain.ts";
import {
  createScratch,
  drawCloud,
  drawCredibleBoundary,
  maskToWalkable,
  normalise,
  rasteriseMixture,
} from "../canvas/belief.ts";
import { drawChampion, drawDeath, drawTrail, drawWard, project } from "../canvas/entities.ts";
import type { PlaybackClock } from "../state/playback.ts";
import { color, font } from "../theme.ts";

const TRAIL_TICKS = 14;

export interface MapSettings {
  showBelief: boolean;
  /** The 90% credible outline on top of the cloud. */
  showBoundary: boolean;
  showVision: boolean;
  showTrails: boolean;
  showWards: boolean;
  showWardRadius: boolean;
  /** Slot to highlight, or -1 for all. */
  focusSlot: number;
}

export const defaultSettings: MapSettings = {
  showBelief: true,
  showBoundary: true,
  showVision: true,
  showTrails: true,
  showWards: true,
  showWardRadius: false,
  focusSlot: -1,
};

interface Props {
  artifact: Artifact;
  terrain: TerrainImage;
  clock: PlaybackClock;
  /** Whose knowledge is being shown. */
  observer: number;
  settings: MapSettings;
  size?: number;
  label?: string;
  onPickChampion?: (slot: number) => void;
}

export function MapCanvas({
  artifact,
  terrain,
  clock,
  observer,
  settings,
  size = 560,
  label,
  onPickChampion,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  // Settings live in a ref because the draw loop reads them every frame and must not
  // need a re-render — a stale closure here would silently freeze the controls.
  const live = useRef(settings);
  live.current = settings;
  const pick = useRef(onPickChampion);
  pick.current = onPickChampion;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || size <= 0) return;
    const ctx = canvas.getContext("2d", { alpha: false })!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    // Everything the draw loop needs, allocated once.
    const scratch = createScratch(artifact.meta.dims.enemies);
    const components = new Float64Array(artifact.meta.dims.components * 4);
    const trail = new Float64Array((TRAIL_TICKS + 1) * 2);
    const here = new Float64Array(2);

    // The fog composite is built at the TERRAIN's own resolution, not the display's.
    // Its source images are 512² and the map is drawn at roughly 800 CSS px on a 2x
    // display, so compositing at display size was doing 1600² of work — five million
    // pixels per rebuild — to upscale a 512² image. Building at 512² and letting the
    // per-frame blit do the magnification is the same picture for a tenth of the cost,
    // and it was most of the 26 ms worst frame.
    const fog = document.createElement("canvas");
    fog.width = terrain.grid;
    fog.height = terrain.grid;
    const fogCtx = fog.getContext("2d", { alpha: false })!;
    let fogTick = -1;
    let fogVision = live.current.showVision;

    // Terrain and belief are composited together at 512² and blitted once. Both are
    // inherently low-resolution — the terrain source IS 512² and the belief is a 32-cell
    // field — so compositing them at display size was doing 2.56 million pixels of
    // `screen` blending per map per frame to magnify images that had no detail to
    // magnify. MEASURED: the belief layer alone took the page from 110 fps to 67.
    // Entities are drawn afterwards at full display resolution, because a champion dot
    // is vector work and does want the pixels.
    const compose = document.createElement("canvas");
    compose.width = terrain.grid;
    compose.height = terrain.grid;
    const composeCtx = compose.getContext("2d", { alpha: false })!;

    // The composite is cached against the ticks that feed it. Vision is exported at 4 Hz
    // and the belief at 8, while the canvas draws at 60 — so rebuilding it every frame
    // recomputed an identical picture six or fourteen times over. Champions are drawn on
    // top every frame and interpolated between ticks, which is what actually needs 60.
    let composeKey = "";

    // Rebuilt only when the belief tick changes — see the compose cache below.
    function rebuildBelief(s: MapSettings, belTick: number, posTick: number) {
      if (!s.showBelief) return;
      // The CLOUD merges the five enemies; the BOUNDARY does not.
      //
      // Merging is right for the fill: every enemy on this map is on the same team and
      // therefore the same colour, so five separate `screen` composites produce a
      // picture that one composite of the per-enemy maximum already gives — and that
      // blend is the most expensive thing on the page.
      //
      // A credible region is the opposite case. It is a statement about ONE champion,
      // and the union of five 90% regions is not the 90% region of anything, so the
      // outlines are drawn per enemy and overlap where the beliefs do.
      let count = 0;
      scratch.merged.fill(0);
      for (let e = 0; e < artifact.meta.dims.enemies; e++) {
        if (artifact.seen(belTick, observer, e)) continue;
        const slot = artifact.enemySlot(observer, e);
        if (s.focusSlot >= 0 && slot !== s.focusSlot) continue;
        if (!artifact.alive(posTick, slot)) continue;

        const field = scratch.fields[count++];
        artifact.belief(belTick, observer, e, components);
        rasteriseMixture(components, field);
        maskToWalkable(field, terrain.walkable, terrain.grid);
        normalise(field);
        for (let k = 0; k < scratch.merged.length; k++) {
          if (field[k] > scratch.merged[k]) scratch.merged[k] = field[k];
        }
      }

      if (count > 0) {
        // Cloud first, outlines second. The cloud is a `screen` composite, so drawing
        // it over the outlines would wash them out exactly where the belief is
        // strongest — which is the one place the boundary needs to be legible.
        drawCloud(composeCtx, scratch.merged, 1 - observer, terrain.grid, scratch);
        if (s.showBoundary) {
          for (let k = 0; k < count; k++) {
            drawCredibleBoundary(
              composeCtx,
              scratch.fields[k],
              1 - observer,
              terrain.grid,
              scratch,
            );
          }
        }
      }
    }

    const draw = (t: number) => {
      const s = live.current;
      const maskTick = artifact.maskTick(t);
      const posTick = artifact.positionTick(t);
      const belTick = artifact.beliefTick(t);

      const key = `${maskTick}|${belTick}|${posTick}|${s.showVision}|${s.showBelief}|${s.showBoundary}|${s.focusSlot}`;
      if (key !== composeKey) {
        composeKey = key;

        if (maskTick !== fogTick || s.showVision !== fogVision) {
          fogTick = maskTick;
          fogVision = s.showVision;
          drawTerrain(
            fogCtx,
            terrain,
            terrain.grid,
            s.showVision ? (i, j) => artifact.visible(maskTick, observer, i, j) : () => true,
          );
        }
        composeCtx.imageSmoothingEnabled = false;
        composeCtx.drawImage(fog, 0, 0);
        rebuildBelief(s, belTick, posTick);
      }

      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(compose, 0, 0, size, size);

      if (s.showWards) {
        for (const ward of artifact.wards) {
          if (ward.team !== observer || t < ward.t0 || t > ward.t1) continue;
          drawWard(
            ctx,
            {
              x: ward.x,
              z: ward.z,
              sight: ward.sight,
              remaining: (ward.t1 - t) / Math.max(ward.t1 - ward.t0, 1),
              showRadius: s.showWardRadius,
            },
            size,
          );
        }
      }

      for (const death of artifact.deaths) {
        if (t < death.t || t - death.t > 6) continue;
        const victim = artifact.heroes[death.victim];
        if (!victim) continue;
        // A death is drawn only if this team could see it happen.
        if (!artifact.seenSlot(artifact.beliefTick(death.t), observer, death.victim)) continue;
        const [x, z] = artifact.position(artifact.positionTick(death.t), death.victim);
        drawDeath(ctx, x, z, victim.team, t - death.t, size);
      }

      for (const hero of artifact.heroes) {
        const own = hero.team === observer;
        if (!own && !artifact.seenSlot(belTick, observer, hero.slot)) continue;
        if (s.focusSlot >= 0 && hero.slot !== s.focusSlot && !own) continue;

        if (s.showTrails) {
          let count = 0;
          for (let k = TRAIL_TICKS; k >= 0; k--) {
            const tick = posTick - k;
            if (tick < 0) continue;
            // A trail is knowledge too: an enemy's path is drawn only for the ticks the
            // observer could actually see them, which is why it comes in dashes.
            if (!own) {
              const bt = artifact.beliefTick(tick / artifact.positionHz);
              if (!artifact.seenSlot(bt, observer, hero.slot)) continue;
            }
            artifact.positionInto(tick, hero.slot, trail, count * 2);
            count++;
          }
          drawTrail(ctx, trail, count, hero.team, size);
        }

        // Interpolated, so a champion glides at 60 fps from 8 Hz data.
        artifact.positionLerpInto(t, hero.slot, here, 0);
        drawChampion(
          ctx,
          {
            x: here[0],
            z: here[1],
            team: hero.team,
            jungler: hero.role === "jng",
            dead: !artifact.alive(posTick, hero.slot),
            focused: s.focusSlot === hero.slot,
          },
          size,
        );
      }

      if (label) {
        ctx.font = `500 10px ${font.mono}`;
        ctx.fillStyle = color.text[4];
        ctx.textBaseline = "top";
        ctx.fillText(label.toUpperCase(), 10, 10);
      }
    };

    draw(clock.t);
    return clock.onDraw(draw);
  }, [artifact, terrain, clock, observer, size, label]);

  return (
    <canvas
      ref={ref}
      onClick={(event) => {
        if (!pick.current) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const px = ((event.clientX - rect.left) / rect.width) * size;
        const py = ((event.clientY - rect.top) / rect.height) * size;
        const slot = nearestChampion(artifact, clock.t, observer, px, py, size);
        if (slot >= 0) pick.current(slot);
      }}
      style={{
        width: size,
        height: size,
        display: "block",
        borderRadius: 3,
        border: `1px solid ${color.borderSoft}`,
        background: color.page,
        cursor: onPickChampion ? "pointer" : "default",
        contain: "paint",
      }}
    />
  );
}

/** The champion nearest a click, within a generous radius, or -1. */
function nearestChampion(
  artifact: Artifact,
  t: number,
  observer: number,
  px: number,
  py: number,
  size: number,
): number {
  const posTick = artifact.positionTick(t);
  const belTick = artifact.beliefTick(t);
  let best = -1;
  let bestDistance = (size / 22) ** 2;
  for (const hero of artifact.heroes) {
    if (hero.team !== observer && !artifact.seenSlot(belTick, observer, hero.slot)) continue;
    const [x, z] = artifact.position(posTick, hero.slot);
    const [sx, sy] = project(x, z, size);
    const d = (sx - px) ** 2 + (sy - py) ** 2;
    if (d < bestDistance) {
      bestDistance = d;
      best = hero.slot;
    }
  }
  return best;
}
