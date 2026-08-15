/**
 * The shell: loading, routing, and the header.
 *
 * Routing is the URL hash, not a router library. Four views and no nested state does not
 * justify a dependency, and a hash means every view is deep-linkable — including the
 * playhead, so a specific moment in a specific match can be sent to someone as a link,
 * which is most of what makes an analysis tool useful to more than one person.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

const MATCH = "synth-0007-000";

export function App() {
  const [artifact, setArtifact] = useState<Artifact | null>(null);
  const [terrain, setTerrain] = useState<TerrainImage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ViewId>(() => readHash().view);
  const [settings, setSettings] = useState<MapSettings>(defaultSettings);
  const [mapSize, setMapSize] = useState(() => fitMapSize());

  const clockRef = useRef<PlaybackClock | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([loadArtifact(`artifacts/${MATCH}`), loadTerrain("terrain.png")])
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

  useEffect(() => {
    const onResize = () => setMapSize(fitMapSize());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

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
    <Shell view={view} onView={changeView} matchId={artifact.meta.match_id}>
      {view === "replay" && (
        <Replay
          artifact={artifact}
          terrain={terrain}
          clock={clock}
          settings={settings}
          onSettings={(next) => {
            setSettings(next);
            // Redraw immediately: a toggle that waits for the next animation frame while
            // paused looks like it did not register.
            requestAnimationFrame(() => clock.refresh());
          }}
          mapSize={mapSize}
        />
      )}
      {view === "autopsy" && (
        <Autopsy
          artifact={artifact}
          terrain={terrain}
          clock={clock}
          settings={settings}
          mapSize={Math.min(mapSize * 1.4, 660)}
        />
      )}
      {view === "wards" && <WardYield artifact={artifact} />}
      {view === "method" && <Method artifact={artifact} />}
    </Shell>
  );
}

function Shell({
  view,
  onView,
  matchId,
  children,
}: {
  view: ViewId;
  onView: (v: ViewId) => void;
  matchId?: string;
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
            <span style={{ color: color.text[7] }}> · synthetic</span>
          </span>
        )}
      </header>
      <main style={{ padding: 20 }}>{children}</main>
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
function fitMapSize(): number {
  const available = window.innerWidth - 250 - 16 - 40 - 12;
  return Math.max(260, Math.min(560, Math.floor(available / 2)));
}
