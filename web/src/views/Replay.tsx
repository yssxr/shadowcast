/**
 * The replay view: the same instant, twice, once per team's knowledge.
 *
 * This is the argument the whole project makes, rendered. On the left is everything Blue
 * could see and everything Blue believed about Red; on the right the mirror. The clouds
 * sit where the enemy might be, and the dots on the *other* map show where they actually
 * were — so the gap between belief and truth is a thing you look at rather than a number.
 *
 * The sidebar updates at about nine hertz while the maps run at sixty. That split is
 * deliberate: nobody can read a number changing sixty times a second, and re-rendering
 * React at that rate to change one label would compete with the canvas for the same
 * frame budget.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { Artifact } from "../artifact/load.ts";
import type { TerrainImage } from "../canvas/terrain.ts";
import type { PlaybackClock } from "../state/playback.ts";
import { formatClock } from "../state/playback.ts";
import { MapCanvas, type MapSettings } from "../components/MapCanvas.tsx";
import { Timeline } from "../components/Timeline.tsx";
import { Bar, Panel, SegmentedControl, Stat, Toggle, transition } from "../components/ui.tsx";
import { color, font, rgba } from "../theme.ts";

interface Props {
  artifact: Artifact;
  terrain: TerrainImage;
  clock: PlaybackClock;
  settings: MapSettings;
  onSettings: (next: MapSettings) => void;
  mapSize: number;
}

export function Replay({ artifact, terrain, clock, settings, onSettings, mapSize }: Props) {
  const [t, setT] = useState(clock.t);
  const [playing, setPlaying] = useState(clock.playing);

  useEffect(() => clock.onTick(setT), [clock]);

  const belTick = artifact.beliefTick(t);
  const posTick = artifact.positionTick(t);

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 12, flex: "0 0 auto" }}>
        <div style={{ display: "flex", gap: 12 }}>
          {[0, 1].map((observer) => (
            <div key={observer} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <MapCanvas
                artifact={artifact}
                terrain={terrain}
                clock={clock}
                observer={observer}
                settings={settings}
                size={mapSize}
                label={observer === 0 ? "what blue knows" : "what red knows"}
                onPickChampion={(slot) =>
                  onSettings({ ...settings, focusSlot: settings.focusSlot === slot ? -1 : slot })
                }
              />
              <TeamSummary artifact={artifact} observer={observer} tick={belTick} />
            </div>
          ))}
        </div>

        <Panel padding={0}>
          <Timeline artifact={artifact} clock={clock} width={mapSize * 2 + 12} />
        </Panel>

        <Transport
          t={t}
          duration={artifact.duration}
          playing={playing}
          onToggle={() => {
            clock.toggle();
            setPlaying(clock.playing);
          }}
          onSeek={(next) => clock.seek(next)}
          onSpeed={(speed) => clock.setSpeed(speed)}
        />
      </div>

      <aside style={{ display: "flex", flexDirection: "column", gap: 12, width: 250 }}>
        <Panel title="belief">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <SegmentedControl
              value={settings.beliefMode}
              onChange={(beliefMode) => onSettings({ ...settings, beliefMode })}
              options={[
                { value: "cloud", label: "Cloud" },
                { value: "contour", label: "Contour" },
                { value: "grid", label: "Grid" },
              ]}
            />
            <Toggle
              label="Belief clouds"
              on={settings.showBelief}
              onChange={(showBelief) => onSettings({ ...settings, showBelief })}
            />
            <Toggle
              label="Fog of war"
              on={settings.showVision}
              onChange={(showVision) => onSettings({ ...settings, showVision })}
            />
            <Toggle
              label="Trails"
              on={settings.showTrails}
              onChange={(showTrails) => onSettings({ ...settings, showTrails })}
            />
            <Toggle
              label="Wards"
              on={settings.showWards}
              onChange={(showWards) => onSettings({ ...settings, showWards })}
            />
            <Toggle
              label="Ward radius"
              on={settings.showWardRadius}
              onChange={(showWardRadius) => onSettings({ ...settings, showWardRadius })}
            />
          </div>
        </Panel>

        <Panel
          title="tracked by the enemy"
          right={
            settings.focusSlot >= 0 ? (
              <button
                type="button"
                onClick={() => onSettings({ ...settings, focusSlot: -1 })}
                style={{
                  background: "none",
                  border: "none",
                  color: color.accent,
                  font: `400 10px ${font.mono}`,
                  cursor: "pointer",
                }}
              >
                clear
              </button>
            ) : null
          }
        >
          <p
            style={{
              font: `300 11px/1.5 ${font.sans}`,
              color: color.text[5],
              margin: "0 0 8px",
            }}
          >
            Each champion as their <em>opponents</em> see them: “seen” means the enemy team
            has them on screen right now, and the number is how uncertain that team is
            about their position.
          </p>
          <ChampionList
            artifact={artifact}
            belTick={belTick}
            posTick={posTick}
            focusSlot={settings.focusSlot}
            onFocus={(slot) =>
              onSettings({ ...settings, focusSlot: settings.focusSlot === slot ? -1 : slot })
            }
          />
        </Panel>
      </aside>
    </div>
  );
}

/**
 * Per-team readout under each map.
 *
 * "Darkness" is the count of living enemies this team cannot currently see, and it is the
 * number this whole engine exists to make computable. Reading it next to the map is what
 * turns the clouds from decoration into a measurement.
 */
function TeamSummary({
  artifact,
  observer,
  tick,
}: {
  artifact: Artifact;
  observer: number;
  tick: number;
}) {
  let hidden = 0;
  let entropy = 0;
  let area = 0;
  for (let e = 0; e < artifact.meta.dims.enemies; e++) {
    if (!artifact.seen(tick, observer, e)) hidden++;
    entropy += artifact.entropy(tick, observer, e);
    area += artifact.area(tick, observer, e);
  }
  return (
    <div
      style={{
        display: "flex",
        gap: 18,
        padding: "9px 12px",
        background: color.panel,
        border: `1px solid ${color.borderSoft}`,
        borderRadius: 3,
      }}
    >
      <Stat label="in the dark" value={`${hidden}`} unit="/ 5" tone={color.team[1 - observer]} />
      <Stat label="total entropy" value={entropy.toFixed(1)} unit="bits" />
      <Stat label="search area" value={area.toFixed(1)} unit="ku²" />
    </div>
  );
}

function ChampionList({
  artifact,
  belTick,
  posTick,
  focusSlot,
  onFocus,
}: {
  artifact: Artifact;
  belTick: number;
  posTick: number;
  focusSlot: number;
  onFocus: (slot: number) => void;
}) {
  // The scale is fixed across the match rather than per-tick, so a bar that is short
  // means "this enemy is well located" instead of "this enemy is the best located right
  // now" — a relative scale would make every row look the same at every moment.
  const maxEntropy = 10;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      {artifact.heroes.map((hero) => {
        const observer = 1 - hero.team;
        const enemy = artifact.enemyIndexOf(observer, hero.slot);
        const seen = enemy >= 0 ? artifact.seen(belTick, observer, enemy) : true;
        const entropy = enemy >= 0 ? artifact.entropy(belTick, observer, enemy) : 0;
        const dead = !artifact.alive(posTick, hero.slot);
        const focused = focusSlot === hero.slot;
        return (
          <button
            key={hero.slot}
            type="button"
            onClick={() => onFocus(hero.slot)}
            style={{
              display: "grid",
              gridTemplateColumns: "8px 1fr auto",
              alignItems: "center",
              gap: 8,
              padding: "5px 6px",
              background: focused ? color.control : "transparent",
              border: `1px solid ${focused ? color.border : "transparent"}`,
              borderRadius: 3,
              cursor: "pointer",
              textAlign: "left",
              transition: `background ${transition}, border-color ${transition}`,
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: dead ? rgba(color.team[hero.team], 0.3) : color.team[hero.team],
                transition: `background ${transition}`,
              }}
            />
            <span style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
              <span
                style={{
                  font: `400 12px ${font.sans}`,
                  color: dead ? color.text[6] : color.text[2],
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {hero.champion || hero.name}
                <span style={{ color: color.text[6], font: `400 10px ${font.mono}` }}>
                  {" "}
                  {hero.role}
                </span>
              </span>
              <Bar
                value={dead ? 0 : entropy}
                max={maxEntropy}
                tint={rgba(color.team[hero.team], 0.75)}
              />
            </span>
            <span
              style={{
                font: `400 10px ${font.mono}`,
                color: dead ? color.text[6] : seen ? color.text[4] : color.accent,
                whiteSpace: "nowrap",
              }}
            >
              {dead ? "dead" : seen ? "seen" : `${entropy.toFixed(1)}b`}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function Transport({
  t,
  duration,
  playing,
  onToggle,
  onSeek,
  onSpeed,
}: {
  t: number;
  duration: number;
  playing: boolean;
  onToggle: () => void;
  onSeek: (t: number) => void;
  onSpeed: (speed: number) => void;
}) {
  const [speed, setSpeed] = useState(1);
  const shortcuts = useRef({ onToggle, onSeek, t });
  shortcuts.current = { onToggle, onSeek, t };

  // Space to play, arrows to step. Keyboard control is the difference between a demo and
  // something someone will actually scrub through for twenty minutes.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return;
      const { onToggle: toggle, onSeek: seek, t: now } = shortcuts.current;
      if (event.code === "Space") {
        event.preventDefault();
        toggle();
      } else if (event.code === "ArrowLeft") {
        seek(now - (event.shiftKey ? 30 : 5));
      } else if (event.code === "ArrowRight") {
        seek(now + (event.shiftKey ? 30 : 5));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const speeds = useMemo(() => [0.25, 0.5, 1, 2, 4], []);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 14,
        padding: "10px 14px",
        background: color.panel,
        border: `1px solid ${color.borderSoft}`,
        borderRadius: 4,
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-label={playing ? "Pause" : "Play"}
        style={{
          width: 30,
          height: 30,
          display: "grid",
          placeItems: "center",
          background: color.control,
          border: `1px solid ${color.border}`,
          borderRadius: 3,
          color: color.text[1],
          cursor: "pointer",
          transition: `background ${transition}`,
        }}
      >
        {playing ? "❚❚" : "▶"}
      </button>

      <span
        style={{
          font: `500 15px ${font.mono}`,
          color: color.text[1],
          minWidth: 58,
          // Tabular figures, so the clock does not shimmer as digits change width.
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {formatClock(t)}
      </span>

      <input
        type="range"
        min={0}
        max={duration}
        step={0.05}
        value={t}
        onChange={(event) => onSeek(Number(event.target.value))}
        style={{ flex: 1, accentColor: color.accent, cursor: "pointer" }}
      />

      <div style={{ display: "flex", gap: 2 }}>
        {speeds.map((value) => (
          <button
            key={value}
            type="button"
            onClick={() => {
              setSpeed(value);
              onSpeed(value);
            }}
            style={{
              padding: "4px 7px",
              background: speed === value ? color.control : "transparent",
              border: `1px solid ${speed === value ? color.border : "transparent"}`,
              borderRadius: 3,
              color: speed === value ? color.text[1] : color.text[5],
              font: `400 10px ${font.mono}`,
              cursor: "pointer",
              transition: `background ${transition}, color ${transition}`,
            }}
          >
            {value}×
          </button>
        ))}
      </div>
    </div>
  );
}
