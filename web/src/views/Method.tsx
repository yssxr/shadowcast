/**
 * Method: how this was built, and how far it can be trusted.
 *
 * The mockup had a corpus view here, aggregates over thousands of ranked games. It cannot
 * be built and will not be faked. The decoded corpus holds about 32,000 matches, not the
 * 1.4 million its documentation claims, and carries no rank, region, patch or match id
 * anywhere, so "Diamond+" and a match code like `EUW1_6412887731` are strings with nothing
 * behind them. Every figure here comes from the artifact on screen or from a committed
 * measurement, and whatever has not been measured says so.
 *
 * Not modesty for its own sake. This project's only claim to being more than a
 * visualisation is that it has a ground-truth oracle and submits to it, and a page of
 * plausible aggregates would spend that claim for nothing.
 */

import type { Artifact } from "../artifact/load.ts";
import { Heading, Note, Panel, Stat } from "../components/ui.tsx";
import { color, font } from "../theme.ts";

/**
 * Measured figures, from `docs/validation.md`. Each produced by a command, none typed in
 * from memory. `pending` entries are ones that genuinely have not been measured.
 */
const MEASURED = [
  {
    group: "On real packets: all 23 matches in one shard",
    rows: [
      ["Fog agreement", "68.26%", "median; 61.4-73.3%, sd 2.8"],
      ["False negative / false positive", "20.2% / 12.2%", "darkness we invent / vision we invent"],
      ["Worst region, best region", "river 51.5% / lane 73.4%", "jungle 53.0%"],
      ["Movement orders attributed", "91.9%", "median across matches"],
      ["Teams recovered", "8 / 8", "100.0% of hero damage across the split"],
      ["Negative information is worth", "+0.148 nats", "vs. the same model without it"],
      ["Full model vs. a plain geodesic disc", "loses, 4.372 / 4.168", "NLL, lower is better"],
      ["90% credible region contains the truth", "30.2%", "target 90%, open defect"],
    ],
  },
  {
    group: "On synthetic matches, where truth is known",
    rows: [
      ["Fog agreement, reconstructed positions", "98.17%", "vs. the stream's own fog events"],
      ["Fog agreement, true positions. The floor", "98.84%", "cell snapping and model limits"],
      ["Brush-adjacent cells", "90.81%", "the worst region, as predicted"],
      ["Field-of-view vs. ray-march reference", "0 disagreements", "947,984 cells"],
      ["Negative information is worth", "+0.243 nats", "one field of one spec apart"],
      ["Particle filter vs. exact 256-state Bayes", "TV 0.030", "falling as 1/√P"],
      ["90% credible region contains the truth", "43.4%", "target 90%, open defect"],
      ["Information-barrier leak detector", "bit-identical", "2,000+ perturbed positions"],
      ["Harmful movement-order misattribution", "0.00-0.15%", "owners ≥300u apart"],
    ],
  },
] as const;

const PENDING = [
  "Ward yield benchmarked against Riot's own mVisionScore",
  "Anything beyond one shard, 23 matches of roughly 32,000, sorted by duration",
  "Corpus aggregates: rank, region and patch are absent from the data",
  "The flicker: we emit 2-3× more visibility transitions than the game does",
];

export function Method({ artifact }: { artifact: Artifact }) {
  const stats = artifact.meta.stats as Record<string, unknown>;
  const config = artifact.meta.config as Record<string, string>;
  const real = artifact.meta.provenance === "real";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 1080, margin: "0 auto" }}>
      <Panel title="what you are looking at">
        <Heading level={1}>Every number here came out of a command.</Heading>
        <Note>
          No placeholders, and nothing typed in from memory because it sounded about right.
          The values below get written by <code style={mono}>shadowcast realfog</code>,{" "}
          <code style={mono}>pipeline</code> and <code style={mono}>ablate</code> into a
          committed report. What hasn't been measured is listed as pending instead of
          estimated.
        </Note>
        {real ? (
          <Note>
            This match is <strong>real</strong>. Decoded replay packets, with positions,
            vision and belief all reconstructed. The corpus carries no ground truth, so
            nothing here is checked against what actually happened. It's checked against the
            fog transitions the stream publishes about itself, which the reconstruction
            agrees with 68% of the time.
          </Note>
        ) : (
          <Note>
            This match is <strong>synthetic</strong>, generated with known ground truth so
            the engine can be held to an oracle real replays can't provide. Real matches
            score far worse. Those numbers are published too.
          </Note>
        )}
        <Note>
          The two tables below stay apart on purpose. Synthetic figures answer whether the
          engine is correct; real ones answer whether it works. Averaging them would bury a
          thirty-point gap between two different questions.
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
          <Stat label="particles" value={`${stats.particles ?? ": "}`} unit="per filter" />
          <Stat label="motion model" value={String(stats.motion ?? ": ")} />
        </div>
        <div style={{ marginTop: 14 }}>
          <Note>
            The belief ships as a 16-component mixture rather than a grid, because a 64²
            grid at 8 Hz would be 295 MB a match. The number above bounds what that costs:
            the KL divergence between the particle cloud and its mixture, both rasterised
            onto the grid you are looking at. A lossy encoding whose loss has never been
            measured is a claim, not a format.
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
          published dataset is rougher than its documentation. Movement orders carry no entity
          id. There's no death packet. No team, no role, no rank, region or match id anywhere
          in the stream. Trajectories get recovered by data association; teams, roles and kills
          are inferred, then measured against whatever can be checked.
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
