/**
 * The playback clock.
 *
 * This is the mockup's architecture, kept deliberately: **a mutable `tNow` advanced by
 * `requestAnimationFrame`, with the canvas redrawn from it at 60 fps and React state
 * updated only about nine times a second for the sidebar.**
 *
 * It looks like a violation of how React is meant to work, and it is the right shape
 * anyway. Driving playback through `setState` re-renders the component tree sixty times
 * a second to change one number that only the canvas reads. The canvas does not care
 * about React's render cycle, and the sidebar does not need 60 Hz. Separating them means
 * the map animates smoothly while the text updates at a rate a human can read, and
 * neither is competing for the same frame budget.
 *
 * The consequence to remember: `clock.t` is a live mutable value, so a component reading
 * it during render gets whatever the last frame set. Anything that must trigger a render
 * subscribes instead.
 */

export type Listener = (t: number) => void;

export class PlaybackClock {
  /** Live playhead in seconds. Mutated every frame; never a React state value. */
  t = 0;
  duration: number;
  speed = 1;
  playing = false;

  private frame = 0;
  private last = 0;
  private drawers = new Set<Listener>();
  private throttled = new Set<Listener>();
  private lastThrottle = 0;
  private throttleInterval: number;

  constructor(duration: number, throttleHz = 9) {
    this.duration = duration;
    this.throttleInterval = 1000 / throttleHz;
  }

  /** Called every animation frame: the canvas renderers. */
  onDraw(fn: Listener): () => void {
    this.drawers.add(fn);
    return () => this.drawers.delete(fn);
  }

  /** Called about nine times a second: React state for the sidebar. */
  onTick(fn: Listener): () => void {
    this.throttled.add(fn);
    return () => this.throttled.delete(fn);
  }

  play(): void {
    if (this.playing) return;
    this.playing = true;
    this.last = performance.now();
    this.frame = requestAnimationFrame(this.step);
  }

  pause(): void {
    this.playing = false;
    cancelAnimationFrame(this.frame);
    this.emitThrottled(true);
  }

  toggle(): void {
    this.playing ? this.pause() : this.play();
  }

  /** Move the playhead. Redraws immediately so a drag feels attached to the pointer. */
  seek(t: number): void {
    this.t = Math.min(this.duration, Math.max(0, t));
    this.emitDraw();
    this.emitThrottled(true);
  }

  setSpeed(speed: number): void {
    this.speed = speed;
  }

  /** Force a redraw without moving time, for a settings change mid-pause. */
  refresh(): void {
    this.emitDraw();
  }

  dispose(): void {
    cancelAnimationFrame(this.frame);
    this.drawers.clear();
    this.throttled.clear();
  }

  private step = (now: number): void => {
    // Clamped, so a backgrounded tab that resumes after thirty seconds does not jump the
    // playhead half a match forward.
    const dt = Math.min((now - this.last) / 1000, 0.25);
    this.last = now;
    this.t += dt * this.speed;

    if (this.t >= this.duration) {
      this.t = this.duration;
      this.pause();
      this.emitDraw();
      return;
    }
    this.emitDraw();
    this.emitThrottled(false);
    this.frame = requestAnimationFrame(this.step);
  };

  private emitDraw(): void {
    for (const fn of this.drawers) fn(this.t);
  }

  private emitThrottled(force: boolean): void {
    const now = performance.now();
    if (!force && now - this.lastThrottle < this.throttleInterval) return;
    this.lastThrottle = now;
    for (const fn of this.throttled) fn(this.t);
  }
}

export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
