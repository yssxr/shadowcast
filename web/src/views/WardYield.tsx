/**
 * Ward information yield: which wards actually told their team something.
 *
 * Every League analytics tool counts wards. Counting is a proxy for a thing nobody
 * measures — whether the ward *revealed* anything — and this view measures that thing
 * directly, because the engine knows where every enemy was and what every ward could see.
 *
 * ## What is being counted, precisely
 *
 * A ward is credited with a **sighting** when an enemy is inside its sight radius at a
 * tick where that enemy was visible to the ward's team, and **no allied champion or
 * turret also covered them**. That exclusivity clause is the whole metric: without it a
 * ward next to a turret would be credited with everything the turret saw, and the most
 * valuable-looking wards would be the most redundant ones.
 *
 * ## What this is not
 *
 * It is not causal. It says "this ward was the only plausible source of vision for these
 * sightings", not "without this ward those sightings would not have happened" — a team
 * without the ward would have played differently. A counterfactual would need re-running
 * the game, which no replay corpus can offer.
 *
 * Riot's own `mVisionScore` is replicated in the packet stream, so this metric will
 * eventually be benchmarked head-to-head against it. That comparison is pending.
 */

import { useMemo, useState } from "react";
import type { Artifact } from "../artifact/load.ts";
import { formatClock } from "../state/playback.ts";
import { Bar, Heading, Note, Panel, Stat } from "../components/ui.tsx";
import { color, font, rgba, teamColor } from "../theme.ts";
import { scoreWards } from "../metrics/wards.ts";

interface Props {
  artifact: Artifact;
}

export function WardYield({ artifact }: Props) {
  const [sort, setSort] = useState<"exclusive" | "covered" | "time">("exclusive");
  const scores = useMemo(() => scoreWards(artifact), [artifact]);

  const ranked = useMemo(() => {
    const copy = [...scores];
    copy.sort((a, b) =>
      sort === "time" ? a.t0 - b.t0 : sort === "covered" ? b.covered - a.covered : b.exclusive - a.exclusive,
    );
    return copy;
  }, [scores, sort]);

  const best = Math.max(1, ...scores.map((s) => s.exclusive));
  const totalExclusive = scores.reduce((sum, s) => sum + s.exclusive, 0);
  const silent = scores.filter((s) => s.exclusive === 0).length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 1080, margin: "0 auto" }}>
      <Panel title="ward information yield">
        <Heading>Counting wards is not measuring vision.</Heading>
        <Note>
          A ward is credited with a sighting when an enemy stands inside its radius, that
          enemy was visible to its team, and <strong>no allied champion or turret also
          covered them</strong>. Everything hangs on that last clause. Drop it and a ward
          parked beside a turret gets credit for everything the turret saw, so the wards
          scoring highest are the ones that added least.
        </Note>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap", marginTop: 4 }}>
          <Stat label="wards placed" value={`${scores.length}`} />
          <Stat label="exclusive sightings" value={`${totalExclusive}`} unit="ticks" />
          <Stat
            label="revealed nothing"
            value={`${silent}`}
            unit={`of ${scores.length}`}
            tone={silent > scores.length / 2 ? color.red : undefined}
          />
        </div>
      </Panel>

      <Panel
        delay={60}
        title="leaderboard"
        right={
          <div style={{ display: "flex", gap: 10 }}>
            {(["exclusive", "covered", "time"] as const).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => setSort(key)}
                style={{
                  background: "none",
                  border: "none",
                  padding: 0,
                  color: sort === key ? color.accent : color.text[5],
                  font: `400 10px ${font.mono}`,
                  cursor: "pointer",
                }}
              >
                {key}
              </button>
            ))}
          </div>
        }
      >
        {ranked.length === 0 ? (
          <Note>No wards were placed in this match.</Note>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {ranked.map((ward) => {
              const owner = artifact.heroes[ward.owner];
              return (
                <div
                  key={ward.index}
                  className="sc-lift"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "52px 1fr 130px 74px",
                    alignItems: "center",
                    gap: 12,
                    padding: "7px 8px",
                    background: color.header,
                    border: `1px solid ${color.borderFaint}`,
                    borderRadius: 3,
                  }}
                >
                  <span
                    style={{
                      font: `400 11px ${font.mono}`,
                      color: color.text[4],
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {formatClock(ward.t0)}
                  </span>
                  <span style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span style={{ font: `400 12px ${font.sans}`, color: color.text[2] }}>
                      {owner?.champion ?? `slot ${ward.owner}`}
                      <span style={{ color: color.text[6], font: `400 10px ${font.mono}` }}>
                        {"  "}
                        {ward.sight >= 900 ? "totem" : "farsight"} · {ward.lifetime.toFixed(0)}s
                      </span>
                    </span>
                    <Bar
                      value={ward.exclusive}
                      max={best}
                      tint={rgba(teamColor(ward.team), 0.8)}
                    />
                  </span>
                  <span
                    style={{
                      font: `400 10px ${font.mono}`,
                      color: color.text[5],
                      textAlign: "right",
                    }}
                  >
                    {ward.covered > 0
                      ? `${ward.covered} covered · ${ward.enemies} enemies`
                      : "nothing in range"}
                  </span>
                  <span
                    style={{
                      font: `500 13px ${font.mono}`,
                      color: ward.exclusive > 0 ? color.text[1] : color.text[6],
                      textAlign: "right",
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {ward.exclusive}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </Panel>

      <Panel title="what this does not claim" delay={120}>
        <Note>
          This is an <strong>attribution</strong>, not a counterfactual. It says the ward
          was the only plausible source of vision for those sightings. It does not say the
          sightings would have been missed without it: a team without that ward would have
          played differently, and no replay corpus can answer what they would have done.
        </Note>
        <Note>
          Riot's own <code style={{ font: `400 11px ${font.mono}` }}>mVisionScore</code> is
          replicated in the packet stream, so this metric can be benchmarked head-to-head
          against it. That comparison has not been run yet.
        </Note>
      </Panel>
    </div>
  );
}
