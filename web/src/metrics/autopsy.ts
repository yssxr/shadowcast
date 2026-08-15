/**
 * Classifying a death by what the victim's team could have known.
 *
 * Kept out of the view for the same reason as the ward metric: this is an interpretation
 * with thresholds in it, and an interpretation belongs somewhere it can be read and
 * argued with rather than buried in a component.
 *
 * The three readings are genuinely different situations, and conflating them is what
 * makes post-game analysis useless:
 *
 *   predictable   the killer was inside the victim team's vision for most of the
 *                 approach. The information existed and was not acted on — a decision
 *                 problem, not an information one.
 *   invisible     the killer was in fog and the belief was concentrated somewhere else.
 *                 Confident and wrong is the worst case, because there was no uncertainty
 *                 to act on.
 *   sudden        the belief was diffuse, so the danger was not ruled out but no specific
 *                 warning existed either.
 */

import type { Artifact, DeathEvent } from "../artifact/load.ts";

/** How long before the death counts as the approach. */
export const APPROACH_WINDOW = 20;

/** Above this fraction of the approach visible, the gank was there to be seen. */
const PREDICTABLE_VISIBLE = 0.5;
/** Below this, the killer was effectively never seen. */
const INVISIBLE_VISIBLE = 0.2;
/** Below this entropy the belief is a claim rather than a shrug — in bits. */
const CONFIDENT_BITS = 4;

export interface Verdict {
  label: "predictable" | "invisible" | "sudden";
  explanation: string;
  visibleFraction: number;
  entropyAtDeath: number;
  /** Distance from the killer's true position to the belief's centre of mass, or -1. */
  beliefError: number;
}

export function analyseDeath(artifact: Artifact, death: DeathEvent): Verdict {
  const victim = artifact.heroes[death.victim];
  const observer = victim?.team ?? 0;
  const killer = artifact.heroes[death.killer];
  const enemy = killer ? artifact.enemyIndexOf(observer, killer.slot) : -1;

  const start = Math.max(0, death.t - APPROACH_WINDOW);
  const from = artifact.beliefTick(start);
  const to = artifact.beliefTick(death.t);

  let visible = 0;
  let total = 0;
  for (let tick = from; tick <= to; tick++) {
    total++;
    if (enemy >= 0 && artifact.seen(tick, observer, enemy)) visible++;
  }
  const visibleFraction = total ? visible / total : 0;
  const entropyAtDeath = enemy >= 0 ? artifact.entropy(to, observer, enemy) : 0;
  const beliefError = enemy >= 0 ? errorAt(artifact, from, observer, enemy, killer!.slot) : -1;
  const side = observer === 0 ? "blue" : "red";

  if (visibleFraction > PREDICTABLE_VISIBLE) {
    return {
      label: "predictable",
      explanation:
        `The killer was inside ${side}'s vision for ${Math.round(visibleFraction * 100)}% ` +
        `of the approach. The information was available and the death happened anyway — ` +
        `that is a decision problem, not an information one.`,
      visibleFraction,
      entropyAtDeath,
      beliefError,
    };
  }

  if (visibleFraction < INVISIBLE_VISIBLE && entropyAtDeath < CONFIDENT_BITS) {
    return {
      label: "invisible",
      explanation:
        `The killer was in fog for almost the whole approach and the belief was ` +
        `concentrated — ${entropyAtDeath.toFixed(1)} bits` +
        (beliefError > 0 ? `, about ${Math.round(beliefError)} units from where they ` +
          `actually were` : "") +
        `. Confident and wrong is the worst case: there was no uncertainty to act on.`,
      visibleFraction,
      entropyAtDeath,
      beliefError,
    };
  }

  return {
    label: "sudden",
    explanation:
      `The killer was mostly unseen and the belief was diffuse at ` +
      `${entropyAtDeath.toFixed(1)} bits, so the danger was never ruled out but no ` +
      `specific warning existed either. This is what a map with no vision looks like ` +
      `from the inside.`,
    visibleFraction,
    entropyAtDeath,
    beliefError,
  };
}

/**
 * Distance between where the killer was and where the belief's mass sat.
 *
 * Only meaningful when the killer was unseen at that moment, which is checked here
 * rather than by the caller — a "belief error" measured while the enemy was on screen
 * would be zero by construction and would drag the average toward flattering nonsense.
 */
function errorAt(
  artifact: Artifact,
  tick: number,
  observer: number,
  enemy: number,
  killerSlot: number,
): number {
  if (artifact.seen(tick, observer, enemy)) return -1;
  const components = artifact.belief(tick, observer, enemy);
  let cx = 0;
  let cz = 0;
  let mass = 0;
  for (let c = 0; c < components.length / 4; c++) {
    const w = components[c * 4 + 2];
    cx += components[c * 4] * w;
    cz += components[c * 4 + 1] * w;
    mass += w;
  }
  if (mass <= 0) return -1;
  const [kx, kz] = artifact.position(artifact.positionTick(tick / artifact.beliefHz), killerSlot);
  return Math.hypot(kx - cx / mass, kz - cz / mass);
}
