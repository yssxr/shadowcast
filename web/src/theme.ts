/**
 * Design tokens, taken verbatim from the mockup.
 *
 * These are copied rather than re-derived. A palette that drifts by a few percent per
 * component is how a considered design becomes an approximately-considered one, and the
 * drift is invisible in any single view.
 *
 * The text ramp is nine steps because the mockup uses nine — headings, body, labels,
 * axis ticks, disabled state and four grades of de-emphasis. Collapsing it to three
 * would flatten exactly the hierarchy that makes a dense dashboard readable.
 */

export const color = {
  page: "#0A0A0C",
  panel: "#0D0D10",
  header: "#101013",
  control: "#191920",

  /** Brightest to faintest. `text[0]` is for headings only. */
  text: [
    "#FAF8F5",
    "#EEEBE6",
    "#A5A19A",
    "#8A8680",
    "#726E68",
    "#625E58",
    "#514D48",
    "#403C38",
    "#34302C",
  ],

  /** Team colours. Index by team id: 0 = Order (blue), 1 = Chaos (red). */
  team: ["#63A7E8", "#E86D50"],
  blue: "#63A7E8",
  red: "#E86D50",
  accent: "#D8A54E",

  border: "rgba(255,255,255,.07)",
  borderSoft: "rgba(255,255,255,.05)",
  borderFaint: "rgba(255,255,255,.04)",
} as const;

export const font = {
  sans: '"Geist", ui-sans-serif, system-ui, -apple-system, sans-serif',
  mono: '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace',
  serif: '"Instrument Serif", ui-serif, Georgia, serif',
} as const;

/**
 * Terrain colours for the map canvas.
 *
 * `lit` and `unlit` are separate palettes rather than one palette with an opacity,
 * because fog in this design is not "the map, dimmer" — brush stays legible in fog while
 * open ground recedes, which is what makes a brush-heavy region read as dangerous rather
 * than merely dark.
 *
 * The separation between them is deliberately large. A first pass used #1C1C22 against
 * #111116 — eleven levels apart out of 255 — and the result was a map where the fog of
 * war was invisible at a glance. Fog is the entire subject here; if a reader cannot see
 * the boundary without being told where to look, nothing else on the page matters.
 */
export const terrainPalette = {
  lit: { ground: "#2E2E38", wall: "#16161C", brush: "#2A3A26" },
  unlit: { ground: "#121217", wall: "#0A0A0D", brush: "#151D13" },
} as const;

/** The enemy's colour, not the observer's — see `BeliefLayer` for why. */
export function beliefColor(enemyTeam: number): string {
  return color.team[enemyTeam] ?? color.blue;
}

export function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function rgba(hex: string, alpha: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}
