/**
 * Gank autopsy: the twenty seconds before a death, from the victim's side.
 *
 * The question this view answers is the one every player asks after dying — *should I
 * have known?* — and it is answerable here in a way it is not from a replay, because the
 * belief state says what the victim's team could have concluded from what they could see.
 *
 * Three readings, and they are genuinely different:
 *
 *   predictable   the killer was inside the victim team's belief for most of the
 *                 approach: the information was there and was not acted on
 *   invisible     the killer was in fog and the belief had them elsewhere: nothing the
 *                 victim could have known
 *   sudden        the belief was diffuse — high entropy — so no specific warning
 *                 existed even though the danger was not ruled out
 *
 * The classification is stated as a rule below rather than hidden in a heuristic,
 * because it is an interpretation and the reader is entitled to disagree with it.
 */

import { useEffect, useMemo, useState } from "react";
import type { Artifact, DeathEvent } from "../artifact/load.ts";
import type { TerrainImage } from "../canvas/terrain.ts";
import type { PlaybackClock } from "../state/playback.ts";
import { formatClock } from "../state/playback.ts";
import { MapCanvas, type MapSettings } from "../components/MapCanvas.tsx";
import { Heading, Note, Panel, Stat, transition } from "../components/ui.tsx";
import { color, font, rgba } from "../theme.ts";
import { APPROACH_WINDOW as WINDOW, analyseDeath, type Verdict } from "../metrics/autopsy.ts";


interface Props {
  artifact: Artifact;
  terrain: TerrainImage;
  clock: PlaybackClock;
  settings: MapSettings;
  mapSize: number;
}

export function Autopsy({ artifact, terrain, clock, settings, mapSize }: Props) {
  const [index, setIndex] = useState(0);
  const death = artifact.deaths[index];

  const verdicts = useMemo(
    () => artifact.deaths.map((d) => analyseDeath(artifact, d)),
    [artifact],
  );

  useEffect(() => {
    if (death) clock.seek(Math.max(0, death.t - WINDOW));
  }, [clock, death]);

  if (!artifact.deaths.length) {
    return (
      <Panel title="gank autopsy">
        <Note>
          No deaths were recovered in this match. Deaths are inferred from health
          replication joined to the last damage event — the packet stream contains no death
          packet at all — so a match with no inferred deaths means the health series never
          reached zero within the window, not that nobody died.
        </Note>
      </Panel>
    );
  }

  const verdict = verdicts[index];
  const victim = artifact.heroes[death.victim];
  const observer = victim?.team ?? 0;

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <MapCanvas
          artifact={artifact}
          terrain={terrain}
          clock={clock}
          observer={observer}
          settings={{ ...settings, focusSlot: -1 }}
          size={mapSize}
          label={`what ${observer === 0 ? "blue" : "red"} knew`}
        />
        <Scrubber clock={clock} death={death} />
      </div>

      <aside style={{ display: "flex", flexDirection: "column", gap: 12, width: 320 }}>
        <Panel title="deaths">
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {artifact.deaths.map((d, k) => {
              const who = artifact.heroes[d.victim];
              const active = k === index;
              return (
                <button
                  key={k}
                  type="button"
                  onClick={() => setIndex(k)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "6px 8px",
                    background: active ? color.control : "transparent",
                    border: `1px solid ${active ? color.border : "transparent"}`,
                    borderRadius: 3,
                    cursor: "pointer",
                    textAlign: "left",
                    transition: `background ${transition}, border-color ${transition}`,
                  }}
                >
                  <span
                    style={{
                      font: `400 11px ${font.mono}`,
                      color: color.text[4],
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {formatClock(d.t)}
                  </span>
                  <span
                    style={{
                      font: `400 12px ${font.sans}`,
                      color: who ? color.team[who.team] : color.text[3],
                      flex: 1,
                    }}
                  >
                    {who?.champion ?? `slot ${d.victim}`}
                  </span>
                  <span
                    style={{
                      font: `400 10px ${font.mono}`,
                      color: verdictColor(verdicts[k].label),
                    }}
                  >
                    {verdicts[k].label}
                  </span>
                </button>
              );
            })}
          </div>
        </Panel>

        <Panel title="verdict">
          <Heading>{verdict.label}</Heading>
          <Note>{verdict.explanation}</Note>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
            <Stat
              label="killer visible"
              value={`${Math.round(verdict.visibleFraction * 100)}`}
              unit="% of approach"
            />
            <Stat label="entropy at death" value={verdict.entropyAtDeath.toFixed(1)} unit="bits" />
            <Stat
              label="belief error"
              value={verdict.beliefError >= 0 ? `${Math.round(verdict.beliefError)}` : "—"}
              unit="units"
            />
          </div>
        </Panel>

        <Panel title="how this is judged">
          <Note>
            Over the {WINDOW} seconds before the death, the killer is either inside the
            victim team's vision or not. <strong>Predictable</strong> means visible for more
            than half of it. <strong>Invisible</strong> means visible for under a fifth
            while the team's belief was concentrated somewhere else — they were confident
            and wrong. <strong>Sudden</strong> is the remainder: the belief was too diffuse
            to constitute a warning.
          </Note>
          <Note>
            Killer attribution is inferred, not recorded — the last unit to damage the
            victim before their health reached zero. This one is credited at{" "}
            {Math.round((death.confidence ?? 0) * 100)}% confidence, which is the share of
            damage in the final window it accounts for.
          </Note>
        </Panel>
      </aside>
    </div>
  );
}

function verdictColor(label: Verdict["label"]): string {
  if (label === "predictable") return color.accent;
  if (label === "invisible") return color.red;
  return color.text[4];
}

/** A local scrubber over the twenty seconds before the selected death. */
function Scrubber({ clock, death }: { clock: PlaybackClock; death: DeathEvent }) {
  const [t, setT] = useState(clock.t);
  useEffect(() => clock.onTick(setT), [clock]);
  const start = Math.max(0, death.t - WINDOW);
  const relative = Math.max(0, Math.min(WINDOW, t - start));

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "10px 14px",
        background: color.panel,
        border: `1px solid ${color.borderSoft}`,
        borderRadius: 4,
      }}
    >
      <button
        type="button"
        onClick={() => clock.toggle()}
        style={{
          width: 28,
          height: 28,
          background: color.control,
          border: `1px solid ${color.border}`,
          borderRadius: 3,
          color: color.text[1],
          cursor: "pointer",
        }}
      >
        {clock.playing ? "❚❚" : "▶"}
      </button>
      <span
        style={{
          font: `500 13px ${font.mono}`,
          color: relative >= WINDOW ? color.red : color.text[2],
          minWidth: 62,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        T−{(WINDOW - relative).toFixed(1)}s
      </span>
      <input
        type="range"
        min={start}
        max={death.t}
        step={0.05}
        value={Math.min(Math.max(t, start), death.t)}
        onChange={(event) => clock.seek(Number(event.target.value))}
        style={{ flex: 1, accentColor: color.red, cursor: "pointer" }}
      />
      <span
        style={{
          font: `400 10px ${font.mono}`,
          color: rgba(color.text[1], 0.6),
        }}
      >
        {formatClock(death.t)}
      </span>
    </div>
  );
}
