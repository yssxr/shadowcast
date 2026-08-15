/**
 * Headless check: decode a real artifact and run every derived metric on it.
 *
 * Not a substitute for looking at the page, but it catches the class of bug that looking
 * at the page catches worst — an off-by-one in an index, a metric that silently returns
 * zero for everything, a coordinate convention flipped in one place. Those render as a
 * map that looks *plausible*, which is exactly the failure this project keeps running
 * into and keeps having to measure its way out of.
 *
 * Everything here runs in plain Node against the built artifact, so it needs no browser
 * and no test framework:
 *
 *     node --experimental-strip-types scripts/check.ts public/artifacts/<id>
 */

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { Artifact, WORLD } from "../src/artifact/load.ts";
import { decode, type Meta } from "../src/generated/artifact.ts";
import { scoreWards } from "../src/metrics/wards.ts";
import { analyseDeath } from "../src/metrics/autopsy.ts";

const dir = process.argv[2] ?? "public/artifacts/synth-0007-000";

const meta: Meta = JSON.parse(readFileSync(`${dir}/meta.json`, "utf8"));
const raw = gunzipSync(readFileSync(`${dir}/data.bin.gz`));
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
const artifact = new Artifact(meta, decode(meta, buffer));

const failures: string[] = [];
function check(name: string, ok: boolean, detail = ""): void {
  if (!ok) failures.push(`${name}${detail ? ` — ${detail}` : ""}`);
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}${detail ? `  ${detail}` : ""}`);
}

// --- structure --------------------------------------------------------------
check("heroes present", artifact.heroes.length === 10, `${artifact.heroes.length}`);
check(
  "teams are five and five",
  artifact.heroes.filter((h) => h.team === 0).length === 5 &&
    artifact.heroes.filter((h) => h.team === 1).length === 5,
);
check("roles resolved", artifact.heroes.every((h) => h.role.length > 0));

// --- positions --------------------------------------------------------------
let onMap = 0;
let total = 0;
for (let tick = 0; tick < meta.dims.position_ticks; tick += 17) {
  for (const hero of artifact.heroes) {
    const [x, z] = artifact.position(tick, hero.slot);
    total++;
    if (
      x >= WORLD.minX &&
      x <= WORLD.minX + WORLD.span &&
      z >= WORLD.minZ &&
      z <= WORLD.minZ + WORLD.span
    ) {
      onMap++;
    }
  }
}
check("every position is on the map", onMap === total, `${onMap}/${total}`);

// Champions must actually move. A decode that returns the same row every tick would pass
// every bounds check and render a map of ten stationary dots.
let moved = 0;
for (const hero of artifact.heroes) {
  const [x0, z0] = artifact.position(0, hero.slot);
  const [x1, z1] = artifact.position(meta.dims.position_ticks - 1, hero.slot);
  if (Math.hypot(x1 - x0, z1 - z0) > 500) moved++;
}
check("champions move over the match", moved >= 8, `${moved}/10 travelled >500u`);

// --- vision -----------------------------------------------------------------
let litCells = 0;
let maskCells = 0;
for (let tick = 0; tick < meta.dims.mask_ticks; tick += 37) {
  for (let j = 0; j < 128; j += 3) {
    for (let i = 0; i < 128; i += 3) {
      maskCells++;
      if (artifact.visible(tick, 0, i, j)) litCells++;
    }
  }
}
const litFraction = litCells / maskCells;
check(
  "vision covers a plausible share of the map",
  litFraction > 0.05 && litFraction < 0.6,
  `${(litFraction * 100).toFixed(1)}%`,
);

// The two teams must not have identical vision — if they do, the observer index is being
// ignored somewhere and every asymmetry claim on the site is empty.
let differing = 0;
for (let j = 0; j < 128; j += 2) {
  for (let i = 0; i < 128; i += 2) {
    if (artifact.visible(20, 0, i, j) !== artifact.visible(20, 1, i, j)) differing++;
  }
}
check("the two teams see different things", differing > 100, `${differing} cells differ`);

// --- belief -----------------------------------------------------------------
let mixtures = 0;
let massOk = 0;
let inBounds = 0;
for (let tick = 0; tick < meta.dims.belief_ticks; tick += 29) {
  for (let o = 0; o < 2; o++) {
    for (let e = 0; e < 5; e++) {
      if (artifact.seen(tick, o, e)) continue;
      const components = artifact.belief(tick, o, e);
      let mass = 0;
      let ok = true;
      for (let c = 0; c < components.length / 4; c++) {
        mass += components[c * 4 + 2];
        const x = components[c * 4];
        const z = components[c * 4 + 1];
        if (x < WORLD.minX || x > WORLD.minX + WORLD.span) ok = false;
        if (z < WORLD.minZ || z > WORLD.minZ + WORLD.span) ok = false;
      }
      mixtures++;
      if (mass > 0.5 && mass <= 1.35) massOk++;
      if (ok) inBounds++;
    }
  }
}
check("mixtures were sampled", mixtures > 50, `${mixtures}`);
check("mixture weight sums to about one", massOk === mixtures, `${massOk}/${mixtures}`);
check("mixture components are on the map", inBounds === mixtures, `${inBounds}/${mixtures}`);

// --- scalars ----------------------------------------------------------------
let entropySeen = 0;
let entropyRange = true;
for (let tick = 0; tick < meta.dims.belief_ticks; tick += 13) {
  for (let o = 0; o < 2; o++) {
    for (let e = 0; e < 5; e++) {
      const h = artifact.entropy(tick, o, e);
      if (h > 0) entropySeen++;
      if (h < 0 || h > 10.5) entropyRange = false;
    }
  }
}
check("entropy is populated", entropySeen > 100, `${entropySeen} non-zero samples`);
check("entropy is within the lattice ceiling", entropyRange);

// --- derived metrics --------------------------------------------------------
const wards = scoreWards(artifact);
check("wards scored", wards.length === artifact.wards.length, `${wards.length}`);
check(
  "ward exclusivity never exceeds coverage",
  wards.every((w) => w.exclusive <= w.covered),
);
const withYield = wards.filter((w) => w.exclusive > 0).length;
console.log(
  `      ${withYield}/${wards.length} wards had exclusive sightings; ` +
    `best ${Math.max(0, ...wards.map((w) => w.exclusive))} ticks`,
);

const verdicts = artifact.deaths.map((d) => analyseDeath(artifact, d));
check("deaths classified", verdicts.length === artifact.deaths.length, `${verdicts.length}`);
check(
  "every verdict has an explanation",
  verdicts.every((v) => v.explanation.length > 40 && Number.isFinite(v.visibleFraction)),
);
for (const [k, v] of verdicts.entries()) {
  const who = artifact.heroes[artifact.deaths[k].victim];
  console.log(
    `      ${who?.champion ?? "?"} at ${artifact.deaths[k].t.toFixed(0)}s: ${v.label} ` +
      `(${Math.round(v.visibleFraction * 100)}% visible, ${v.entropyAtDeath.toFixed(1)} bits)`,
  );
}

console.log("");
if (failures.length) {
  console.error(`${failures.length} check(s) failed:`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log(`all checks passed on ${meta.match_id}`);
