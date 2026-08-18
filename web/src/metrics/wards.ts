/**
 * Ward information yield.
 *
 * Kept out of the view component on purpose: this is the analysis, and it needs to be
 * checkable without a browser. `scripts/check.ts` runs it against a real artifact and
 * asserts the numbers are sane, which a React component cannot be made to do.
 *
 * ## What is counted
 *
 * A ward is credited with a **sighting** when an enemy stands inside its sight radius at
 * a tick where that enemy was visible to the ward's team, and **no allied champion or
 * turret also covered them**. The exclusivity clause is the metric: without it a ward
 * beside a turret would be credited with everything the turret saw, and the wards that
 * scored highest would be the most redundant ones.
 *
 * ## What it is not
 *
 * Not causal. It says the ward was the only plausible source of vision for those
 * sightings, not that they would not have happened without it. A team without the ward
 * would have played differently, and no replay corpus can answer that.
 */

import type { Artifact } from "../artifact/load.ts";

/** Champion sight radius at patch 12.22. Turrets share it. */
export const CHAMPION_SIGHT = 1350;

export interface WardScore {
  index: number;
  t0: number;
  t1: number;
  team: number;
  owner: number;
  sight: number;
  lifetime: number;
  /** Ticks where this ward was the only plausible source of a sighting. */
  exclusive: number;
  /** Ticks where it saw an enemy at all, redundantly or not. */
  covered: number;
  /** Distinct enemies it exclusively revealed. */
  enemies: number;
}

interface Turret {
  x: number;
  z: number;
  team: number;
}

export function scoreWards(artifact: Artifact): WardScore[] {
  const turrets =
    (artifact.meta.events as { turrets?: Turret[] }).turrets ?? [];

  return artifact.wards.map((ward, index) => {
    const from = artifact.beliefTick(ward.t0);
    const to = artifact.beliefTick(ward.t1);
    const observer = ward.team;
    const allies = artifact.heroes.filter((h) => h.team === observer);
    const enemies = artifact.heroes.filter((h) => h.team !== observer);
    // Only turrets whose vision could possibly overlap this ward's are worth testing;
    // the rest cannot contest a sighting inside it.
    const nearbyTurrets = turrets.filter(
      (t) => t.team === observer && within(t.x, t.z, ward.x, ward.z, ward.sight + CHAMPION_SIGHT),
    );

    let exclusive = 0;
    let covered = 0;
    const revealed = new Set<number>();

    for (let tick = from; tick <= to; tick++) {
      const posTick = artifact.positionTick(tick / artifact.beliefHz);
      for (const enemy of enemies) {
        if (!artifact.seenSlot(tick, observer, enemy.slot)) continue;
        if (!artifact.alive(posTick, enemy.slot)) continue;
        const [ex, ez] = artifact.position(posTick, enemy.slot);
        if (!within(ex, ez, ward.x, ward.z, ward.sight)) continue;
        covered++;

        let alsoCovered = false;
        for (const ally of allies) {
          if (!artifact.alive(posTick, ally.slot)) continue;
          const [ax, az] = artifact.position(posTick, ally.slot);
          if (within(ex, ez, ax, az, CHAMPION_SIGHT)) {
            alsoCovered = true;
            break;
          }
        }
        if (!alsoCovered) {
          for (const turret of nearbyTurrets) {
            if (within(ex, ez, turret.x, turret.z, CHAMPION_SIGHT)) {
              alsoCovered = true;
              break;
            }
          }
        }
        if (!alsoCovered) {
          exclusive++;
          revealed.add(enemy.slot);
        }
      }
    }

    return {
      index,
      t0: ward.t0,
      t1: ward.t1,
      team: ward.team,
      owner: ward.owner,
      sight: ward.sight,
      lifetime: ward.t1 - ward.t0,
      exclusive,
      covered,
      enemies: revealed.size,
    };
  });
}

function within(ax: number, az: number, bx: number, bz: number, radius: number): boolean {
  const dx = ax - bx;
  const dz = az - bz;
  return dx * dx + dz * dz <= radius * radius;
}
