# Shadowcast

**Reconstructing what League of Legends teams could actually see.**

A belief-state engine for MOBA information asymmetry, built on packet-level decoded replays.

> Status: in development, engine complete through artifact export; the frontend is next. Numbers marked `[pending]`
> are not yet measured, and this file carries no figure that was not produced by `shadowcast
> pipeline` or `shadowcast ablate`. See [`docs/validation.md`](docs/validation.md) for the full
> report, including the caveats that matter for reading the belief numbers.

---

## What this is

Every League analytics tool measures vision by counting wards. That is a proxy, and a bad one.

Shadowcast reconstructs the actual *information state* of both teams at every moment of a game:
not just where everyone was, but where each team could plausibly have believed the enemy was, and
how uncertain that belief was. From that it derives metrics that do not currently exist —
positional entropy, information advantage over time, per-ward information yield, gank
predictability.

The interesting part is the **negative information**. A particle sitting inside a team's visible
region without a corresponding sighting is provably falsified: if Blue can see the whole river
and does not see Red's jungler, he is not in the river. That is what makes the belief
distributions terrain-shaped and strange-looking instead of round circles growing from a last
known position, and it is what separates this from every "last seen here" overlay.

## Why it is possible at all

Riot's public API gives player positions once per minute — a jungler crosses half the map in that
window, so spatial analysis on the official API is impossible. `.rofl` replay files are encrypted
with per-patch obfuscation, which is why the entire commercial ecosystem stops at ward counts.

Henry Zhu ([maknee](https://maknee.github.io/blog/2025/League-Data-Scraping/)) reverse-engineered
the format with an instruction emulator and trampoline hooks into the game binary, decoded a
large corpus of games, published it under Apache 2.0, and then
[got busy](https://maknee.github.io/blog/2025/League-Data-Scraping/). That dataset is the
foundation here, and this project is downstream of that work.

## What the dataset actually contains

The published corpus is rougher than its documentation suggests. These are measured, not
inferred — the numbers come from range-fetching shards and parsing 965,768 real packets:

| Claim | Reality |
|---|---|
| "1TB+ (700k+ replays)" / "over 1.4M league replays" | **≈ 32,000 matches.** `12_22/batch_001.jsonl.gz` is 76 MB gzipped → 2.01 GB of JSON → 23 matches. Extrapolated over 108.47 GB of shards. |
| Patch splits `12_22, 12_23, 13_01, 13_02, 13_03` | Actual directories are `12_22`, `12_23`, `13_1`, `13_2`. There is no `13_3`. |
| Complete games | **Truncated prefixes**, 12–21 minutes, always ending on an exact 30-second chunk boundary. Shards are duration-sorted, so one shard is not a random sample. |
| `WaypointGroup.waypoints` is `Dict[net_id, List[Position]]` | The dict key is the **list length**, not a net_id. 100.0000% of 41,129 pairs checked. Movement orders carry no entity attribution. |
| `HeroDie` exists | **Never fires.** Zero occurrences; no hero net_id ever appears as a death target. No kills, deaths or assists in the stream. |
| `CreateHero` identifies a player | It gives `net_id`, summoner name and champion — but **no team, no role, no position**. |
| Match metadata | **None.** No match ID, region, patch, rank, win/loss or duration. |

Consequences for anyone else considering this data: the official
[`…-gym` loader](https://github.com/Maknee/league-of-legends-decoded-replay-packets-gym) is not
usable — `parse_waypoints` treats the length key as a net_id, so all of its position tracking
(including its demo GIF) is wrong, and `get_heroes_by_team` reads a `team` field that does not
exist in the data. Reading the JSONL with `gzip` + `json` is about fifteen lines and strictly
more reliable.

What *is* there, and better than expected:

- **Fog transitions for all ten champions.** A team always sees its own members, so a fog event
  about champion C can only come from the opposing team's view — which makes the observer team
  derivable per event, and gives a ground-truth visibility oracle for **both** sides.
- **Wards, completely.** They arrive as `SpawnMinion` (`SightWard`/`YellowTrinket`,
  `JammerDevice`, `VisionWard`, …), with exact placement in `position1`, the owner's hero net_id
  in `targetable_on_client`, and expiry via a `WardCorpse` unit. Placement, owner and lifetime
  are all directly observed.
- **`mVisionScore`** is replicated, so our ward metric can be benchmarked head-to-head against
  Riot's own.

## Architecture

```
L0   acquisition     HuggingFace shards -> local
L1   normalisation   packets -> typed event tables
L1.5 resolution      entity <-> team <-> role; movement-order attribution
L2   reconstruction  trajectories + vision sources -> per-team visibility masks
L3   inference       masks -> belief distributions -> metrics -> validation
L4   presentation    precomputed artifacts -> static site
```

Everything is precomputed. There is no backend, no API key, no rate limit, and no ongoing cost —
which means it works identically in three years with zero maintenance.

Three design decisions worth knowing about:

**One visibility table serves every sight radius.** For each source cell we store the
shadowcast field of view at the *maximum* radius; visibility at any smaller radius `r` is exactly
`FOV_max AND disc(r)`. This holds because shadowcasting decides a cell using only shadow
intervals cast by strictly nearer occluders, so an occluder outside `disc(r)` cannot affect
anything inside it. Verified with zero mismatches across 11,034 trials. It reduces a naively
8.6 TB all-pairs table to about 160 MB. Two implementation choices break the property — a
wall-lighting post-pass (68% of cases) and flood-revealing the source's whole brush (1.2%) — so
both are banned in code, with a test that keeps them banned.

**The table is a cache, not a data structure.** A miss falls back to a live field-of-view
computation, so coverage is a performance knob and correctness never depends on it. That is what
lets sources exist in non-walkable cells (wall-hop dashes, over-wall Farsight wards) without
special cases.

**Terrain has three channels, not two.** `blocks_move`, `blocks_vision`, and `brush_id` — because
Riot stamps *see-through* cells along wall diagonals (1,819 of them on Summoner's Rift) that
block movement but transmit vision. They were added after S5 Worlds specifically to fix
line-of-sight artefacts, so deriving vision from walkability reproduces a bug they patched.

## Getting the data

```bash
# Terrain: the Summoner's Rift navgrid (9.4 MB). Exact wall, brush and see-through masks.
mkdir -p data/terrain && curl -L -o data/terrain/AIPath_SRX.aimesh_ngrid \
  "https://raw.githubusercontent.com/FrankTheBoxMonster/LoL-NGRID-converter/master/test%20files/SummonersRiftSeason10/AIPath_SRX.aimesh_ngrid"

# Replays: one shard first (76 MB, 23 matches), then a full patch (~15 GB, ~4,400 matches).
uv tool install "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download maknee/league-of-legends-decoded-replay-packets --repo-type dataset \
   --include "12_22/batch_001.jsonl.gz" --local-dir data/raw
```

## Running it

```bash
uv sync
uv run shadowcast terrain build      # navgrid -> 512^2 channels + brush groups
uv run shadowcast fov build          # precompute the visibility table (~5 s)
uv run shadowcast pipeline           # synthetic match end to end + fog agreement
uv run shadowcast ablate             # seven belief models, one table, the thesis
uv run shadowcast diagnose           # HOW the belief is wrong: drift or collapse
uv run shadowcast export --web       # the artifact the site reads, ~1 MB per match
uv run shadowcast doctor             # versions, config hashes, stale artifacts

cd web && npm install && npm run dev  # the site, at localhost:5173
```

## The site

Four views, all rendered from the artifact above — no backend, no API key, no rate limit.

**Replay** puts the same instant on screen twice, once per team's knowledge. The left map
is everything Blue could see and everything Blue believed about Red; the right is the
mirror. Belief clouds are drawn in the **enemy's** colour, so a cloud and the dot it
collapses into share one, and the moment of discovery reads as a single event.

**Gank autopsy** takes the twenty seconds before a death and asks whether the victim's
team could have known. *Predictable* means the killer was visible for most of the
approach. *Invisible* means they were in fog while the belief sat somewhere else —
confident and wrong. *Sudden* means the belief was too diffuse to be a warning.

**Ward yield** credits a ward with a sighting only when no allied champion or turret also
covered the enemy. That exclusivity clause is the metric: without it the wards that score
best are the most redundant ones. On the sample match, six of ten wards revealed nothing
at all.

**Method** is every measured number with its provenance, and an explicit list of what has
*not* been measured. The corpus view from the original design is not there, because rank,
region and patch do not exist in the data and a plausible aggregate would cost more than
it is worth.

The belief renders one way: a soft cloud with the **90% credible region** outlined on it —
the field to read at a glance, the outline to point at, enclosing exactly the area the
search-area figure reports.

The maps hold **96 fps** with both boards live at 2x scale (`npm run perf`), and the belief
layer is free — the same frame rate as with it switched off. Nothing allocates in the draw
loop, the terrain-and-belief composite is cached against its 4 Hz and 8 Hz source ticks
rather than rebuilt at 60, and champion positions are interpolated between ticks so they
glide instead of stepping. React state for the sidebar is throttled to about 9 Hz; the
digits are eased back up to 60 by writing straight to the DOM node.

## Development

```bash
uv sync                      # includes dev tools
uv run pytest                # full suite, ~115s warm (fog validation + belief ablation)
uv run ruff check --fix .     # lint
uv run ruff format .          # format
uv run pre-commit install    # optional: lint + format on commit
uv run shadowcast doctor     # versions, config hashes, stale artifacts
```

CI runs lint, format-check and the full test suite. It fetches the navgrid and verifies its
SHA-256, because without that file the terrain and FOV tests skip cleanly — which would mean CI
passing while never exercising the radius-monotonicity check the whole table design rests on.

## Validation

The point of having a ground-truth oracle is to be held to it. Every number below is produced by
`shadowcast validate` and written to [`docs/validation.md`](docs/validation.md), not typed in by
hand.

| | |
|---|---|
| Fog agreement, reconstructed positions | **98.17%** (synthetic) |
| Fog agreement, true positions substituted — the floor | **98.84%** (synthetic) |
| — brush-adjacent cells specifically | 90.81% (worst category, as predicted) |
| Movement-order attribution, harmful misattribution rate | **0.00–0.15%** |
| Team / role recovery | **100% / 100%** (synthetic) |
| Belief calibration — does the 90% region contain the truth 90% of the time? | **43.4%** — open defect |
| **Log-likelihood vs. the same model without negative information** | **3.887 vs 4.132 nats** |
| Particle filter vs an exact 256-state Bayes forward pass | **TV 0.030**, falling as 1/√P |
| Information-barrier leak detector | **bit-identical** |
| Artifact size per match | **1.24 MB** gzipped (budget: 2 MB) |
| Python writer vs TypeScript reader | **identical** across all 6 sections |

The fourth row is the one that matters: if the full model did not beat the same model without
negative information, negative information would be doing nothing and the central claim would
be empty. It holds, on a comparison between two `FilterSpec`s that differ in exactly one field,
so nothing else can explain it.

**The third row is an open defect, and it is stated here rather than buried.** The belief is
overconfident: its 90% region contains the truth 39% of the time, and a plain geodesic disc —
enormously vague, and better calibrated — now beats the full model on likelihood over a whole
match. This surfaced when two vision bugs were fixed: enemies had been visible 84.5% of the time
instead of a realistic 45.9%, darkness episodes were short, and short episodes hid it. `shadowcast diagnose` classifies it: the truth sits a median of 182 units from the nearest
particle but 1,862 from the cloud's centre of mass, so the cloud covers the right ground and
puts its mass elsewhere. That is **drift**, a motion-model error, not a filter defect — which
also means it cannot honestly be tuned away against a synthetic generator.

Every figure above is measured on **synthetic** matches, where ground truth is known.
Real-corpus numbers are still pending and will be worse. The fog agreement is deliberately
reported as a pair — substituting true positions separates the irreducible floor (cell
snapping, shadowcasting's permissiveness, ward and minion models) from what the
reconstruction itself costs, and a single percentage cannot tell a modelling limit from a
bug.

Enemies are visible **45.9%** of the time here, against 25–40% in a real game. It read 84.5%
until two vision bugs were found: the fog-attack reveal fired on attacks that had no target, so
every champion revealed itself roughly once a second wherever it stood; and minion waves marched
the entire lane and parked in the enemy fountain, giving each team three permanent floodlights
inside the other team's spawn.

## Limitations, stated plainly

- **This is a historical corpus.** Patches 12.22–13.2, late 2022 to early 2023. Fog-of-war
  mechanics and map geometry are unchanged, and the object of study is information dynamics
  rather than champion balance — but nothing here is current-meta advice.
- **Terrain provenance.** The navgrid is the Season 10 Summoner's Rift dump. Patch 12.22
  references `AIPath_SRX_2.aimesh_ngrid`; SR did not change structurally in between, but that is
  an argument rather than a verification.
- **Trajectories are reconstructed, not recorded.** Movement orders are unattributed, so
  champion paths come from data association anchored on position-tagged spell and attack packets.
  The residual distribution is published; mobility-heavy champions are expected to be worst.
- **Kills are inferred**, from health replication joined to the last damage event, because the
  stream contains no death packet.
- **Wards are observed but ward expiry is partly modelled** — placement and destruction are in
  the data, but a timed ward that simply runs out needs its duration computed from the average
  champion level at placement.

## Credits and licence

The decoded replay corpus is by **Henry Zhu (maknee)**, released under Apache 2.0:
[dataset](https://huggingface.co/datasets/maknee/league-of-legends-decoded-replay-packets) ·
[write-up](https://maknee.github.io/blog/2025/League-Data-Scraping/). None of this exists without
that work.

Terrain parsing follows the `.aimesh_ngrid` format documented by
[FrankTheBoxMonster/LoL-NGRID-converter](https://github.com/FrankTheBoxMonster/LoL-NGRID-converter)
and [TheKillerey/MapgeoAddon](https://github.com/TheKillerey/MapgeoAddon).

Shadowcast is licensed under Apache 2.0. See [LICENSE](LICENSE).

Shadowcast isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or
anyone officially involved in producing or managing League of Legends. League of Legends and Riot
Games are trademarks or registered trademarks of Riot Games, Inc.
