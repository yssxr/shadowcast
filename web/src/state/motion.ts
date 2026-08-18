/**
 * Motion helpers: measuring a box, and easing a number toward a target.
 *
 * `useAnimatedNumber` exists because of the 9 Hz sidebar. The canvas runs at sixty
 * frames a second and the React state driving the text updates about nine times a
 * second, which is the right split, nobody reads a number changing sixty times a second
 * and re-rendering the tree at that rate would compete with the maps for the same frame
 * budget. But a value that *jumps* nine times a second looks broken next to a map that
 * is moving smoothly.
 *
 * So the text keeps its 9 Hz updates and the displayed value eases toward them on its
 * own animation frame, writing straight to the DOM node. React re-renders nine times a
 * second; the digits move sixty. Neither has to know about the other.
 */

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Element size, tracked without polling. Used to size the maps to the actual viewport.
 *
 * A **callback ref**, not an object ref, and that is the whole subtlety. An object ref
 * with a mount-time effect observes whatever `ref.current` happened to be when the effect
 * ran, and the element it needs to watch does not exist during the loading state, so the
 * observer attaches to nothing and never retries. The measured width stays zero, the
 * maps never render, and the page is simply blank with no error anywhere. A callback ref
 * fires again every time the node changes, including the first time it appears.
 */
export function useMeasure<T extends HTMLElement>(): [
  (node: T | null) => void,
  { width: number; height: number },
] {
  const [box, setBox] = useState({ width: 0, height: 0 });
  const observer = useRef<ResizeObserver | null>(null);

  const ref = useCallback((node: T | null) => {
    observer.current?.disconnect();
    if (!node) return;
    observer.current = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      // Integer sizes only: a canvas at a fractional CSS width resamples every pixel and
      // the map goes soft for no reason anyone could name.
      setBox((prev) =>
        Math.round(prev.width) === Math.round(width) &&
        Math.round(prev.height) === Math.round(height)
          ? prev
          : { width: Math.round(width), height: Math.round(height) },
      );
    });
    observer.current.observe(node);
  }, []);

  useEffect(() => () => observer.current?.disconnect(), []);

  return [ref, box];
}

/**
 * A span whose text eases toward `value` at 60 fps.
 *
 * Written directly to `textContent` rather than through state: the point is to move the
 * digits without re-rendering, and a `setState` per frame would defeat the whole
 * arrangement.
 */
export function useEasedText(
  value: number,
  format: (v: number) => string,
  rate = 0.18,
): React.RefObject<HTMLSpanElement | null> {
  const ref = useRef<HTMLSpanElement>(null);
  const target = useRef(value);
  const current = useRef(value);
  target.current = value;

  useEffect(() => {
    let frame = 0;
    const step = () => {
      const delta = target.current - current.current;
      // Snap when close enough that easing would only produce visual noise, and snap
      // outright on a large jump. A scrub across the match should land, not glide.
      current.current =
        Math.abs(delta) < 0.005 || Math.abs(delta) > 1e4
          ? target.current
          : current.current + delta * rate;
      if (ref.current) ref.current.textContent = format(current.current);
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [format, rate]);

  return ref;
}
