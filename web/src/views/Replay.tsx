/**
 * The replay view: the same instant, twice, once per team's knowledge.
 *
 * This is the argument the whole project makes, rendered. On the left is everything Blue
 * could see and everything Blue believed about Red; on the right the mirror. The clouds
 * sit where the enemy might be, and the dots on the *other* map show where they actually
 * were, so the gap between belief and truth is a thing you look at rather than a number.
 *
 * **One scrubber, and it is the timeline.** An earlier version also had a range slider in
 * the transport bar, which meant two controls for one piece of state sitting eight pixels
 * apart. The timeline already carries the advantage series, the ward ticks and the kill
 * ticks, so dragging on it is dragging on the thing you are reading. A separate slider
 * is a second answer to a question that already had one.
 *
 * The sidebar updates at about nine hertz while the maps run at sixty. That split is
 * deliberate: nobody can read a number changing sixty times a second, and re-rendering
 * React at that rate to change one label would compete with the canvas for the same frame
 * budget. `useEasedText` then moves the digits at sixty without re-rendering anything, so
 * the split is invisible.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { Artifact } from "../artifact/load.ts";
import type { TerrainImage } from "../canvas/terrain.ts";
import type { PlaybackClock } from "../state/playback.ts";
import { formatClock } from "../state/playback.ts";
import { useEasedText } from "../state/motion.ts";
import { MapCanvas, type MapSettings } from "../components/MapCanvas.tsx";
import { Timeline } from "../components/Timeline.tsx";
import { Bar, Panel, Toggle, transition } from "../components/ui.tsx";
import { color, font, rgba, teamColor } from "../theme.ts";

interface Props {
  artifact: Artifact;
  terrain: TerrainImage;
  clock: PlaybackClock;
  settings: MapSettings;
  onSettings: (next: MapSettings) => void;
  mapSize: number;
  boardWidth: number;
}

export function Replay({
  artifact,
  terrain,
  clock,
  settings,
  onSettings,
  mapSize,
  boardWidth,
}: Props) {
  const [t, setT] = useState(clock.t);
  const [playing, setPlaying] = useState(clock.playing);

  useEffect(() => clock.onTick(setT), [clock]);

  const belTick = artifact.beliefTick(t);
  const posTick = artifact.positionTick(t);

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start", width: "100%" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 12, flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", gap: 12 }}>
          {[0, 1].map((observer) => (
            <TeamBoard
              key={observer}
              artifact={artifact}
              terrain={terrain}
              clock={clock}
              observer={observer}
              settings={settings}
              onSettings={onSettings}
              size={mapSize}
              tick={belTick}
            />
          ))}
        </div>

        <Panel padding={0}>
          <Timeline artifact={artifact} clock={clock} width={boardWidth} />
        </Panel>

        <Transport
          t={t}
          playing={playing}
          onToggle={() => {
            clock.toggle();
            setPlaying(clock.playing);
          }}
          onSeek={(next) => clock.seek(next)}
          onSpeed={(speed) => clock.setSpeed(speed)}
        />
      </div>

      <aside
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 12,
          width: 250,
          flex: "0 0 250px",
        }}
      >
        <Panel title="belief" delay={40}>
          <p
            style={{
              font: `300 11px/1.5 ${font.sans}`,
              color: color.text[5],
              margin: "0 0 8px",
            }}
          >
            The cloud is where the enemy might be. The outline encloses 90% of that
            probability, and is exactly the area reported under each map.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <Toggle
              label="Belief clouds"
              on={settings.showBelief}
              onChange={(showBelief) => onSettings({ ...settings, showBelief })}
            />
            <Toggle
              label="90% region outline"
              on={settings.showBoundary}
              onChange={(showBoundary) => onSettings({ ...settings, showBoundary })}
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
          delay={80}
          right={
            settings.focusSlot >= 0 ? (
              <button
                type="button"
                className="sc-press"
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
            Each champion as their <em>opponents</em> see them. "Seen" means the enemy team
            has them on screen right now; the number is how uncertain that team is about
            where they are.
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
 * One map with its own header and readout.
 *
 * The header is real markup rather than text drawn into the canvas. Canvas text does not
 * scale with the browser's font settings, cannot be selected, and is invisible to a
 * screen reader, and this particular label is the one thing a first-time reader needs in
 * order to understand what they are looking at.
 */
function TeamBoard({
  artifact,
  terrain,
  clock,
  observer,
  settings,
  onSettings,
  size,
  tick,
}: {
  artifact: Artifact;
  terrain: TerrainImage;
  clock: PlaybackClock;
  observer: number;
  settings: MapSettings;
  onSettings: (next: MapSettings) => void;
  size: number;
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

  const side = observer === 0 ? "Blue" : "Red";
  const tint = teamColor(observer);

  return (
    <section
      className="sc-rise"
      style={{
        display: "flex",
        flexDirection: "column",
        background: color.panel,
        border: `1px solid ${color.borderSoft}`,
        borderRadius: 4,
        overflow: "hidden",
        flex: 1,
        minWidth: 0,
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 9,
          padding: "10px 13px",
          background: color.header,
          borderBottom: `1px solid ${color.borderFaint}`,
        }}
      >
        <span
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: tint,
            boxShadow: `0 0 10px ${rgba(tint, 0.75)}`,
          }}
        />
        <span style={{ font: `400 14px ${font.sans}`, color: color.text[1] }}>
          {side} sees
        </span>
        <span style={{ flex: 1 }} />
        <span
          style={{
            font: `500 11px ${font.mono}`,
            color: hidden > 0 ? teamColor(1 - observer) : color.text[5],
            transition: `color ${transition}`,
          }}
        >
          {hidden} of 5 hidden
        </span>
      </header>

      <MapCanvas
        artifact={artifact}
        terrain={terrain}
        clock={clock}
        observer={observer}
        settings={settings}
        size={size}
        onPickChampion={(slot) =>
          onSettings({ ...settings, focusSlot: settings.focusSlot === slot ? -1 : slot })
        }
      />

      <footer
        style={{
          display: "flex",
          gap: 22,
          padding: "9px 13px",
          borderTop: `1px solid ${color.borderFaint}`,
        }}
      >
        <Readout label="in the dark" value={hidden} unit="/ 5" digits={0} tone={tint} />
        <Readout label="total entropy" value={entropy} unit="bits" digits={1} />
        <Readout label="search area" value={area} unit="ku²" digits={1} />
      </footer>
    </section>
  );
}

/** A number that eases toward its target at 60 fps while React updates it at nine. */
function Readout({
  label,
  value,
  unit,
  digits,
  tone,
}: {
  label: string;
  value: number;
  unit?: string;
  digits: number;
  tone?: string;
}) {
  const format = useMemo(() => (v: number) => v.toFixed(digits), [digits]);
  const ref = useEasedText(value, format);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <span
        style={{
          font: `400 9px/1 ${font.mono}`,
          letterSpacing: ".08em",
          color: color.text[5],
          textTransform: "uppercase",
        }}
      >
        {label}
      </span>
      <span
        style={{
          font: `400 17px/1 ${font.mono}`,
          color: tone ?? color.text[1],
          fontVariantNumeric: "tabular-nums",
        }}
      >
        <span ref={ref}>{format(value)}</span>
        {unit && (
          <span style={{ font: `400 10px ${font.mono}`, color: color.text[5] }}> {unit}</span>
        )}
      </span>
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
  // The scale is fixed across the match rather than per-tick, so a short bar means "this
  // enemy is well located" instead of "best located right now". A relative scale would
  // make every row look the same at every moment.
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
            className="sc-lift"
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
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: dead ? rgba(teamColor(hero.team), 0.3) : teamColor(hero.team),
                // A living, currently-visible champion breathes. It is the one piece of
                // ambient motion in the sidebar and it maps to the thing being measured:
                // the dot is lit exactly while the enemy has eyes on them.
                animation: !dead && seen ? "sc-pulse 2.4s ease-in-out infinite" : undefined,
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
                  transition: `color ${transition}`,
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
                tint={rgba(teamColor(hero.team), 0.75)}
              />
            </span>
            <span
              style={{
                font: `400 10px ${font.mono}`,
                color: dead ? color.text[6] : seen ? color.text[4] : color.accent,
                whiteSpace: "nowrap",
                transition: `color ${transition}`,
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

/**
 * Play, clock, speed. **No slider**: the timeline above is the scrubber.
 */
function Transport({
  t,
  playing,
  onToggle,
  onSeek,
  onSpeed,
}: {
  t: number;
  playing: boolean;
  onToggle: () => void;
  onSeek: (t: number) => void;
  onSpeed: (speed: number) => void;
}) {
  const [speed, setSpeed] = useState(1);
  const shortcuts = useRef({ onToggle, onSeek, t });
  shortcuts.current = { onToggle, onSeek, t };
  const clockRef = useEasedText(t, formatClock, 0.5);

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

  const speeds = [0.25, 0.5, 1, 2, 4];

  return (
    <div
      className="sc-rise"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 16,
        padding: "10px 14px",
        background: color.panel,
        border: `1px solid ${color.borderSoft}`,
        borderRadius: 4,
      }}
    >
      <button
        type="button"
        className="sc-press"
        onClick={onToggle}
        aria-label={playing ? "Pause" : "Play"}
        style={{
          width: 32,
          height: 32,
          display: "grid",
          placeItems: "center",
          background: playing ? color.control : color.accent,
          border: `1px solid ${playing ? color.border : "transparent"}`,
          borderRadius: 3,
          color: playing ? color.text[1] : "#14100A",
          font: "11px/1 system-ui",
          cursor: "pointer",
          transition: `background ${transition}, color ${transition}`,
        }}
      >
        {playing ? "❚❚" : "▶"}
      </button>

      <span
        ref={clockRef}
        style={{
          font: `500 16px ${font.mono}`,
          color: color.text[1],
          minWidth: 56,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {formatClock(t)}
      </span>

      <span style={{ font: `300 11px ${font.sans}`, color: color.text[5], flex: 1 }}>
        Drag the timeline to scrub · space to play · ← → to step
      </span>

      <div style={{ display: "flex", gap: 2 }}>
        {speeds.map((value) => (
          <button
            key={value}
            type="button"
            className="sc-press"
            onClick={() => {
              setSpeed(value);
              onSpeed(value);
            }}
            style={{
              padding: "4px 8px",
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
