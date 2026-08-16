/**
 * The shell: loading, routing, and the header.
 *
 * Routing is the URL hash, not a router library. Four views and no nested state does not
 * justify a dependency, and a hash means every view is deep-linkable — including the
 * playhead, so a specific moment in a specific match can be sent to someone as a link,
 * which is most of what makes an analysis tool useful to more than one person.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMeasure } from "./state/motion.ts";
import { Artifact, loadArtifact } from "./artifact/load.ts";
import { loadTerrain, type TerrainImage } from "./canvas/terrain.ts";
import { PlaybackClock } from "./state/playback.ts";
import { defaultSettings, type MapSettings } from "./components/MapCanvas.tsx";
import { Replay } from "./views/Replay.tsx";
import { Autopsy } from "./views/Autopsy.tsx";
import { WardYield } from "./views/WardYield.tsx";
import { Method } from "./views/Method.tsx";
import { color, font } from "./theme.ts";
import { transition } from "./components/ui.tsx";

const VIEWS = [
  { id: "replay", label: "Replay" },
  { id: "autopsy", label: "Gank autopsy" },
  { id: "wards", label: "Ward yield" },
  { id: "method", label: "Method" },
] as const;

type ViewId = (typeof VIEWS)[number]["id"];

/**
 * The artifact to load if `artifacts/index.json` is missing or empty.
 *
 * `shadowcast export --web` writes that index, newest first, so the site follows whatever
 * was last exported instead of naming a match in source. Hardcoding one meant a deploy
 * that exported anything else built a page that 404s on its own data — and nothing catches
 * it, because the bundle typechecks and compiles perfectly well while pointing at a file
 * nobody wrote.
 */
const FALLBACK_MATCH = "12_22-batch_001-0";

async function firstArtifact(): Promise<string> {
  try {
    const res = await fetch("artifacts/index.json");
    if (!res.ok) return FALLBACK_MATCH;
    const names: unknown = await res.json();
    if (Array.isArray(names) && typeof names[0] === "string") return names[0];
  } catch {
    // An index is a convenience, not a dependency. A dev server mid-export can 404 it.
  }
  return FALLBACK_MATCH;
}

/** Sidebar width plus the gap beside it. */
const SIDEBAR = 250 + 16;

export function App() {
  const [artifact, setArtifact] = useState<Artifact | null>(null);
  const [terrain, setTerrain] = useState<TerrainImage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ViewId>(() => readHash().view);
  const [settings, setSettings] = useState<MapSettings>(defaultSettings);
  const [board, boardBox] = useMeasure<HTMLDivElement>();

  const clockRef = useRef<PlaybackClock | null>(null);

  useEffect(() => {
    let cancelled = false;
    firstArtifact()
      .then((match) =>
        Promise.all([loadArtifact(`artifacts/${match}`), loadTerrain("terrain.png")]),
      )
      .then(([loaded, loadedTerrain]) => {
        if (cancelled) return;
        setArtifact(loaded);
        setTerrain(loadedTerrain);
      })
      .catch((cause) => !cancelled && setError(String(cause)));
    return () => {
      cancelled = true;
    };
  }, []);

  // The clock outlives every view, so switching tabs does not reset the playhead — which
  // matters because the autopsy view moves it deliberately and the replay view should
  // come back where you left it.
  const clock = useMemo(() => {
    if (!artifact) return null;
    const created = new PlaybackClock(artifact.duration);
    created.t = readHash().t;
    clockRef.current = created;
    return created;
  }, [artifact]);

  useEffect(() => () => clockRef.current?.dispose(), []);

  // Without this, the hash is write-only: a pasted deep link into an already-open tab
  // does nothing, and the browser's back button appears broken. Both are the ways
  // someone else arrives at a specific moment, which is the entire point of putting the
  // playhead in the URL.
  useEffect(() => {
    const onHashChange = () => {
      const next = readHash();
      setView(next.view);
      clockRef.current?.seek(next.t);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // The hash is written on a timer rather than every frame: `history.replaceState` at
  // 60 Hz is a documented way to make Safari stutter, and one update a second is plenty
  // for a link somebody copies.
  useEffect(() => {
    if (!clock) return;
    const id = window.setInterval(() => {
      const next = `#${view}&t=${clock.t.toFixed(1)}`;
      if (window.location.hash !== next) {
        window.history.replaceState(null, "", next);
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [clock, view]);

  const changeView = useCallback((next: ViewId) => {
    setView(next);
    window.history.replaceState(null, "", `#${next}`);
  }, []);

  if (error) {
    return (
      <Shell view={view} onView={changeView}>
        <p style={{ font: `400 13px ${font.mono}`, color: color.red, padding: 24 }}>
          {error}
          <br />
          <span style={{ color: color.text[4] }}>
            Run <code>uv run shadowcast export --web</code> to build the artifact.
          </span>
        </p>
      </Shell>
    );
  }

  if (!artifact || !terrain || !clock) {
    return (
      <Shell view={view} onView={changeView}>
        <p style={{ font: `400 12px ${font.mono}`, color: color.text[4], padding: 24 }}>
          decoding artifact…
        </p>
      </Shell>
    );
  }

  return (
    <Shell
      view={view}
      onView={changeView}
      matchId={artifact.meta.match_id}
      provenance={artifact.meta.provenance}
    >
      {/* Keyed on the view so React remounts on change and the entrance animation runs.
          A cross-fade between two mounted trees would mean two live canvases competing
          for the frame budget during the transition, which is the one moment it is most
          visible. */}
      {/* Outside the keyed wrapper: it must not be remounted when the view changes, or
          the width would drop to zero for a frame and every canvas would resize twice. */}
      <div ref={board} style={{ width: "100%", height: 0 }} aria-hidden />
      <div key={view} className="sc-fade" style={{ width: "100%" }}>
        {/* Nothing canvas-backed renders until the board has been measured. A canvas
            sized from a width of zero throws on the first `drawImage`, and sizing it from
            a guess means every map is drawn once at the wrong size and then again. */}
        {boardBox.width > 0 && view === "replay" && (
          <Replay
            artifact={artifact}
            terrain={terrain}
            clock={clock}
            settings={settings}
            onSettings={(next) => {
              setSettings(next);
              // Redraw immediately: a toggle that waits for the next animation frame
              // while paused looks like it did not register.
              requestAnimationFrame(() => clock.refresh());
            }}
            mapSize={fitMapSize(boardBox.width - SIDEBAR)}
            boardWidth={Math.max(0, boardBox.width - SIDEBAR)}
          />
        )}
        {boardBox.width > 0 && view === "autopsy" && (
          <Autopsy
            artifact={artifact}
            terrain={terrain}
            clock={clock}
            settings={settings}
            mapSize={Math.max(320, Math.min(boardBox.width - 360, 780))}
          />
        )}
        {view === "wards" && <WardYield artifact={artifact} />}
        {view === "method" && <Method artifact={artifact} />}
      </div>
    </Shell>
  );
}

function Shell({
  view,
  onView,
  matchId,
  provenance,
  children,
}: {
  view: ViewId;
  onView: (v: ViewId) => void;
  matchId?: string;
  provenance?: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ minHeight: "100vh", background: color.page }}>
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          display: "flex",
          alignItems: "center",
          gap: 22,
          padding: "0 20px",
          height: 50,
          background: color.header,
          borderBottom: `1px solid ${color.border}`,
          backdropFilter: "blur(6px)",
        }}
      >
        <span
          style={{
            font: `400 19px ${font.serif}`,
            color: color.text[0],
            letterSpacing: "-.01em",
          }}
        >
          Shadowcast
        </span>
        <nav style={{ display: "flex", gap: 2 }}>
          {VIEWS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onView(item.id)}
              style={{
                position: "relative",
                padding: "6px 11px",
                background: "transparent",
                border: "none",
                borderRadius: 3,
                color: view === item.id ? color.text[1] : color.text[4],
                font: `400 12px ${font.sans}`,
                cursor: "pointer",
                transition: `color ${transition}`,
              }}
            >
              {item.label}
              <span
                style={{
                  position: "absolute",
                  left: 11,
                  right: 11,
                  bottom: 0,
                  height: 1,
                  background: color.accent,
                  // Scaled rather than shown/hidden, so the underline slides instead of
                  // popping. Transform and opacity are the two properties that animate
                  // without touching layout.
                  transform: `scaleX(${view === item.id ? 1 : 0})`,
                  opacity: view === item.id ? 1 : 0,
                  transition: `transform ${transition}, opacity ${transition}`,
                }}
              />
            </button>
          ))}
        </nav>
        <span style={{ flex: 1 }} />
        {matchId && (
          <span style={{ font: `400 10px ${font.mono}`, color: color.text[6] }}>
            {matchId}
            {/* Never hardcoded. A viewer cannot tell a reconstructed real match from a
                generated one by looking, and the difference is the whole claim: fog
                agreement is 98% on synthetic and 68% on real. This label said
                "synthetic" unconditionally for as long as the site existed, which was
                true then and would have quietly become a lie the moment it was not. */}
            <span
              style={{
                color: provenance === "real" ? color.accent : color.text[7],
              }}
              title={
                provenance === "real"
                  ? "Decoded replay packets. Positions, vision and belief are all reconstructed."
                  : "A generated match with known ground truth, used to validate the engine."
              }
            >
              {" · "}
              {provenance === "real" ? "real packets" : "synthetic"}
            </span>
          </span>
        )}
      </header>
      <main
        style={{
          padding: "20px 20px 40px",
          display: "flex",
          justifyContent: "center",
        }}
      >
        {/* Capped, because two maps stretched across an ultrawide are unreadable — but
            capped generously, so a normal laptop fills its screen instead of leaving a
            third of it empty. */}
        <div style={{ width: "100%", maxWidth: 1680 }}>{children}</div>
      </main>
    </div>
  );
}

function readHash(): { view: ViewId; t: number } {
  const hash = window.location.hash.replace(/^#/, "");
  const [name, ...rest] = hash.split("&");
  const found = VIEWS.find((v) => v.id === name);
  const tParam = rest.find((part) => part.startsWith("t="));
  return {
    view: found ? found.id : "replay",
    t: tParam ? Number(tParam.slice(2)) || 0 : 0,
  };
}

/** Two maps side by side plus a sidebar, without ever inducing a horizontal scrollbar. */
/**
 * Two maps side by side inside the measured board width.
 *
 * Measured rather than derived from `window.innerWidth`: the sidebar, the padding and the
 * scrollbar all take width, and every attempt to predict their total is a guess that goes
 * wrong on somebody's machine. A `ResizeObserver` on the actual element is the answer the
 * browser already has.
 */
function fitMapSize(boardWidth: number): number {
  if (boardWidth <= 0) return 420;
  // 12 px gap between the two panels, 2 px of border each.
  return Math.max(240, Math.floor((boardWidth - 12) / 2) - 2);
}
