/**
 * The advantage timeline: mirrored area fills about a midline, with events as ticks.
 *
 * Straight from the mockup — ward placements below the line, kills above, objectives as
 * dashed verticals, and pointer-capture drag anywhere on the strip to scrub.
 *
 * The quantity plotted is **information advantage**: the difference between the two
 * teams' total uncertainty about their enemies. Above the midline means Order knows more
 * about Chaos than Chaos knows about Order. Plotting it mirrored rather than as a single
 * signed line is what makes "who is ahead" pre-attentive — you see which side is filled
 * before reading anything.
 *
 * Pointer capture matters more than it sounds: without it a drag that leaves the strip
 * stops scrubbing mid-gesture, which feels broken in exactly the interaction people use
 * most.
 */

import { useEffect, useRef } from "react";
import type { Artifact } from "../artifact/load.ts";
import type { PlaybackClock } from "../state/playback.ts";
import { color, font, rgba } from "../theme.ts";

interface Props {
  artifact: Artifact;
  clock: PlaybackClock;
  width: number;
  height?: number;
}

export function Timeline({ artifact, clock, width, height = 96 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // One pixel of this strip covers about six belief ticks, so each pixel is the MEAN
    // over its range rather than a sample from it. Point-sampling was the first version
    // and it aliased badly: the series is noisy at tick resolution, so picking one tick
    // in six produced a dense fringe that looked like signal and was not.
    const ticks = artifact.meta.dims.belief_ticks;
    const samples = Math.min(width, ticks);
    const series = new Float32Array(samples);
    let peak = 1e-6;
    for (let k = 0; k < samples; k++) {
      const from = Math.floor((k / samples) * ticks);
      const to = Math.max(from + 1, Math.floor(((k + 1) / samples) * ticks));
      let sum = 0;
      for (let tick = from; tick < to; tick++) sum += artifact.scalar(tick, "advantage");
      series[k] = sum / (to - from);
      peak = Math.max(peak, Math.abs(series[k]));
    }

    const mid = height * 0.52;
    const scale = (height * 0.34) / peak;

    const background = (c: CanvasRenderingContext2D) => {
      c.clearRect(0, 0, width, height);

      c.strokeStyle = color.borderFaint;
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(0, mid + 0.5);
      c.lineTo(width, mid + 0.5);
      c.stroke();

      // Minute gridlines. The match window is fifteen minutes and a reader orients by
      // minute, not by second.
      c.strokeStyle = color.borderFaint;
      c.font = `400 9px ${font.mono}`;
      c.fillStyle = color.text[6];
      for (let m = 1; m * 60 < artifact.duration; m++) {
        const x = (m * 60 / artifact.duration) * width;
        c.beginPath();
        c.moveTo(x + 0.5, 4);
        c.lineTo(x + 0.5, height - 4);
        c.stroke();
        c.fillText(`${m}`, x + 3, height - 4);
      }

      for (const team of [0, 1]) {
        c.beginPath();
        c.moveTo(0, mid);
        for (let k = 0; k < samples; k++) {
          const x = (k / (samples - 1)) * width;
          const v = series[k];
          const signed = team === 0 ? Math.max(v, 0) : Math.max(-v, 0);
          c.lineTo(x, team === 0 ? mid - signed * scale : mid + signed * scale);
        }
        c.lineTo(width, mid);
        c.closePath();
        c.fillStyle = rgba(color.team[team], 0.3);
        c.fill();
        c.strokeStyle = rgba(color.team[team], 0.65);
        c.lineWidth = 1;
        c.stroke();
      }

      // Wards below the midline, kills above — the mockup's arrangement, and it keeps
      // the two event kinds from colliding on a dense strip.
      for (const ward of artifact.wards) {
        const x = (ward.t0 / artifact.duration) * width;
        c.strokeStyle = rgba(color.accent, 0.5);
        c.beginPath();
        c.moveTo(x + 0.5, height - 12);
        c.lineTo(x + 0.5, height - 6);
        c.stroke();
      }
      for (const death of artifact.deaths) {
        const victim = artifact.heroes[death.victim];
        if (!victim) continue;
        const x = (death.t / artifact.duration) * width;
        c.strokeStyle = rgba(color.team[victim.team], 0.85);
        c.lineWidth = 1.5;
        c.beginPath();
        c.moveTo(x + 0.5, 4);
        c.lineTo(x + 0.5, 12);
        c.stroke();
      }
    };

    // The series and the events never change, so they are drawn once into an offscreen
    // canvas and blitted each frame. Only the playhead moves, and redrawing a
    // 900-point series to move one vertical line would be the most expensive thing on
    // the page for no reason at all.
    const cache = document.createElement("canvas");
    cache.width = canvas.width;
    cache.height = canvas.height;
    const cctx = cache.getContext("2d")!;
    cctx.scale(dpr, dpr);
    background(cctx);

    const draw = (t: number) => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(cache, 0, 0, width, height);
      const x = (t / artifact.duration) * width;
      ctx.strokeStyle = color.text[1];
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
    };

    draw(clock.t);
    return clock.onDraw(draw);
  }, [artifact, clock, width, height]);

  const scrub = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    clock.seek(Math.min(1, Math.max(0, ratio)) * artifact.duration);
  };

  return (
    <canvas
      ref={ref}
      style={{ width, height, display: "block", cursor: "ew-resize", touchAction: "none" }}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        scrub(e);
      }}
      onPointerMove={(e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) scrub(e);
      }}
      onPointerUp={(e) => e.currentTarget.releasePointerCapture(e.pointerId)}
    />
  );
}
