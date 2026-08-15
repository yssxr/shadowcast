/**
 * Method: how this was built and how far it can be trusted.
 *
 * The mockup had a corpus view here, showing aggregates over thousands of ranked games.
 * That view cannot be built yet and will not be faked. The decoded corpus is about
 * 32,000 matches — not the 1.4 million its documentation claims — and carries no rank,
 * region, patch or match id at all, so "Diamond+" and a match code like `EUW1_6412887731`
 * are strings with nothing behind them. Every figure on this page comes from the artifact
 * being displayed or from a committed measurement, and anything not yet measured says so.
 *
 * That is not modesty for its own sake. The project's entire claim to being more than a
 * visualisation is that it has a ground-truth oracle and submits to it, and a view full
 * of plausible aggregates would spend that claim for nothing.
 */

import type { Artifact } from "../artifact/load.ts";
import { Heading, Note, Panel, Stat } from "../components/ui.tsx";
import { color, font } from "../theme.ts";

/**
 * Measured figures, from `docs/validation.md` — each produced by a command, none typed in
 * from memory. `pending` entries are ones that genuinely have not been measured.
 */
const MEASURED = [
  {
    group: "Vision reconstruction",
    rows: [
      ["Fog agreement, reconstructed positions", "96.90%", "vs. the stream's own fog events"],
      ["Fog agreement, true positions — the floor", "98.53%", "cell snapping and model limits"],
      ["Brush-adjacent cells", "93.98%", "the worst region, as predicted"],
      ["Field-of-view vs. ray-march reference", "0 disagreements", "947,984 cells"],
    ],
  },
  {
    group: "Belief",
    rows: [
      ["Negative information vs. the same model without it", "0.418 vs 0.822", "NLL, lower is better"],
      ["Particle filter vs exact 256-state Bayes", "TV 0.030", "falling as 1/√P"],
      ["90% credible region contains the truth", "84.0%", "target 90%"],
      ["Information-barrier leak detector", "bit-identical", "2,000+ perturbed positions"],
    ],
  },
  {
    group: "Entity resolution",
    rows: [
      ["Teams recovered", "100%", "turret names + a 5/5 constraint"],
      ["Roles recovered", "100%", "lane occupancy + ward share"],
      ["Harmful movement-order misattribution", "0.00–0.15%", "owners ≥300u apart"],
    ],
  },
] as const;

const PENDING = [
  "Fog agreement on the real corpus, by region",
  "Ward yield benchmarked against Riot's own mVisionScore",
  "Corpus aggregates — rank, region and patch are absent from the data",
];

export function Method({ artifact }: { artifact: Artifact }) {
  const stats = artifact.meta.stats as Record<string, unknown>;
  const config = artifact.meta.config as Record<string, string>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 1080, margin: "0 auto" }}>
      <Panel title="what you are looking at">
        <Heading level={1}>Every number here was produced by a command.</Heading>
        <Note>
          Nothing on this site is a placeholder, and nothing is a plausible figure typed in
          by hand. The measured values below are written by <code style={mono}>shadowcast
          pipeline</code> and <code style={mono}>shadowcast ablate</code> into a committed
          report; anything not yet measured is listed as pending rather than estimated.
        </Note>
        <Note>
          The match being displayed is <strong>synthetic</strong> — generated with known
          ground truth so the engine can be validated against an oracle that real replays
          cannot provide. Real-corpus figures will be worse, and they will be published
          whatever they are.
        </Note>
      </Panel>

      {MEASURED.map((section, index) => (
        <Panel key={section.group} title={section.group} delay={60 + index * 50}>
          <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
            {section.rows.map(([label, value, note]) => (
              <div
                key={label}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto 200px",
                  alignItems: "baseline",
                  gap: 16,
                  padding: "7px 0",
                  borderBottom: `1px solid ${color.borderFaint}`,
                }}
              >
                <span style={{ font: `300 13px ${font.sans}`, color: color.text[2] }}>
                  {label}
                </span>
                <span
                  style={{
                    font: `500 13px ${font.mono}`,
                    color: color.text[0],
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {value}
                </span>
                <span style={{ font: `300 11px ${font.sans}`, color: color.text[5] }}>{note}</span>
              </div>
            ))}
          </div>
        </Panel>
      ))}

      <Panel title="this artifact" delay={220}>
        <div style={{ display: "flex", gap: 26, flexWrap: "wrap" }}>
          <Stat label="match" value={artifact.meta.match_id} />
          <Stat label="duration" value={`${Math.round(artifact.duration / 60)}`} unit="min" />
          <Stat
            label="mixture loss"
            value={`${Number(stats.mixture_kl_mean ?? 0).toFixed(4)}`}
            unit="nats KL"
          />
          <Stat label="depletion events" value={`${stats.depletion_events ?? 0}`} />
          <Stat label="particles" value={`${stats.particles ?? "—"}`} unit="per filter" />
          <Stat label="motion model" value={String(stats.motion ?? "—")} />
        </div>
        <div style={{ marginTop: 14 }}>
          <Note>
            The belief is shipped as a 16-component mixture rather than a grid — a 64²
            grid at 8 Hz would be 295 MB a match — and the number above bounds what that
            costs: the KL divergence between the particle cloud and its mixture, both
            rasterised onto the grid you are looking at. A lossy encoding whose loss has
            never been measured is a claim, not a format.
          </Note>
          <span style={{ font: `400 10px ${font.mono}`, color: color.text[6] }}>
            config {Object.entries(config).map(([k, v]) => `${k}=${v}`).join("  ")}
          </span>
        </div>
      </Panel>

      <Panel title="not yet measured" delay={260}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {PENDING.map((item) => (
            <span key={item} style={{ font: `300 13px ${font.sans}`, color: color.text[4] }}>
              <span style={{ color: color.accent, font: `400 11px ${font.mono}` }}>pending</span>{" "}
              {item}
            </span>
          ))}
        </div>
      </Panel>

      <Panel title="the data" delay={300}>
        <Note>
          Built on Henry Zhu's decoded replay corpus, released under Apache 2.0. The
          published dataset is rougher than its documentation: movement orders carry no
          entity id, there is no death packet, and there is no team, role, rank, region or
          match id anywhere in the stream. Trajectories are recovered by data association;
          teams, roles and kills are inferred and measured.
        </Note>
        <Note>
          Shadowcast is not endorsed by Riot Games and does not reflect the views of Riot
          Games or anyone officially involved in producing or managing League of Legends.
        </Note>
      </Panel>
    </div>
  );
}

const mono = { font: `400 11px ${font.mono}`, color: color.text[2] } as const;
