# Shadowcast

**Reconstructing what League of Legends teams could actually see.**

A belief-state engine for MOBA information asymmetry, built on packet-level decoded replays.

> Status: in development. The engine runs end to end on real packets, and the site renders a real
> match. What is left is accuracy, not coverage. No figure in this file was typed in by hand; each
> one comes out of a command, and [`docs/validation.md`](docs/validation.md) has the full report
> with the caveats attached.

---

## What this is

Every League analytics tool measures vision by counting wards. That is a proxy, and a bad one.

Shadowcast reconstructs the *information state* of both teams at every moment of a game. Not just
where everyone was, but where each team could plausibly have believed the enemy was, and how
uncertain that belief was. Out of that fall metrics nothing else computes: positional entropy,
information advantage over time, per-ward information yield, gank predictability.

The interesting part is the negative information. A particle sitting inside a team's visible
region without a matching sighting has been disproved. If Blue can see the whole river and does
not see Red's jungler, he is not in the river. That single idea is why the belief distributions
come out terrain-shaped and strange rather than as circles growing from a last known position,
and it is the whole difference between this and a "last seen here" overlay.

## Why it is possible at all

Riot's public API reports player positions once a minute. A jungler crosses half the map in that
window, so spatial analysis on the official API simply cannot be done. `.rofl` replay files are
encrypted with per-patch obfuscation, which is why the commercial ecosystem stops at ward counts.

Henry Zhu ([maknee](https://maknee.github.io/blog/2025/League-Data-Scraping/)) reverse-engineered
the format with an instruction emulator and trampoline hooks into the game binary, decoded a large
corpus of games, published it under Apache 2.0, and then
[got busy](https://maknee.github.io/blog/2025/League-Data-Scraping/). That dataset is the
foundation here. This project is downstream of that work.

## What the dataset actually contains

The published corpus is rougher than its documentation suggests. Everything below was measured by
range-fetching shards and parsing 965,768 real packets, not inferred from the docs:

| Claim | Reality |
|---|---|
| "1TB+ (700k+ replays)" / "over 1.4M league replays" | **≈ 32,000 matches.** `12_22/batch_001.jsonl.gz` is 76 MB gzipped, 2.01 GB of JSON, 23 matches. Extrapolated over 108.47 GB of shards. |
| Patch splits `12_22, 12_23, 13_01, 13_02, 13_03` | The directories are `12_22`, `12_23`, `13_1`, `13_2`. There is no `13_3`. |
| Complete games | **Truncated prefixes**, 10–25 minutes, always ending on an exact 30-second chunk boundary. Shards are sorted by duration, so one shard is not a random sample. |
| `WaypointGroup.waypoints` is `Dict[net_id, List[Position]]` | The dict key is the **list length**, not a net_id. True in 100.0000% of 41,129 pairs checked. Movement orders carry no entity attribution at all. |
| `HeroDie` exists | **Never fires.** Zero occurrences, and no hero net_id ever appears as a death target. There are no kills, deaths or assists in the stream. |
| `CreateHero` identifies a player | It gives a `net_id`, a summoner name and a champion. **No team, no role, no position.** |
| Match metadata | **None.** No match ID, region, patch, rank, win/loss or duration. |

If you are considering this data yourself: the official
[`…-gym` loader](https://github.com/Maknee/league-of-legends-decoded-replay-packets-gym) is not
usable. `parse_waypoints` treats the length key as a net_id, so all of its position tracking is
wrong, including its demo GIF, and `get_heroes_by_team` reads a `team` field that does not exist.
Reading the JSONL with `gzip` and `json` takes about fifteen lines and is strictly more reliable.

Three things are there, and better than expected:

- **Fog transitions for all ten champions.** A team always sees its own members, so a fog event
  about champion C can only have come from the opposing team's view. That makes the observing team
  derivable per event, which turns the corpus into a ground-truth visibility oracle for both sides
  at once. Everything downstream rests on it, and it was tested on real packets before it was
  trusted.
- **Wards, completely.** They arrive as `SpawnMinion` rows with exact placement in `position1`, the
  owner's hero net_id in `targetable_on_client`, and destruction via a `WardCorpse` unit.
- **`mVisionScore` is replicated**, so the ward metric here can eventually be benchmarked
  head-to-head against Riot's own.

## Architecture

```
L0   acquisition     HuggingFace shards -> local
L1   normalisation   packets -> typed event tables
L1.5 resolution      entity <-> team <-> role; movement-order attribution
L2   reconstruction  trajectories + vision sources -> per-team visibility masks
L3   inference       masks -> belief distributions -> metrics -> validation
L4   presentation    precomputed artifacts -> static site
```

Everything is precomputed. No backend, no API key, no rate limit, no running cost. It will work
identically in three years with nobody maintaining it.

Three decisions worth knowing about.

**One visibility table serves every sight radius.** For each source cell the table stores the
shadowcast field of view at the maximum radius, and visibility at any smaller radius `r` is
exactly `FOV_max AND disc(r)`. This works because shadowcasting decides a cell using only shadow
intervals from strictly nearer occluders, so an occluder outside `disc(r)` cannot reach anything
inside it. Checked across 11,034 trials with zero mismatches. It takes a naive 8.6 TB all-pairs
table down to about 160 MB. Two tempting implementation choices break the property: a
wall-lighting post-pass, which breaks 68% of cases, and flood-revealing the source's whole brush,
which breaks 1.2%. Both are banned in code, with a test that keeps them banned.

**The table is a cache, not a data structure.** A miss falls back to a live field-of-view
computation, so coverage is a performance knob and correctness never depends on it. That is what
lets vision sources sit in non-walkable cells, like wall-hop dashes and over-wall Farsight wards,
with no special cases anywhere.

**Terrain has three channels, not two:** `blocks_move`, `blocks_vision`, `brush_id`. Riot stamps
see-through cells along wall diagonals, 1,819 of them on Summoner's Rift, that block movement but
transmit vision. They were added after S5 Worlds to fix line-of-sight artefacts, so deriving
vision from walkability reproduces a bug Riot already patched.

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
uv run shadowcast inspect <shard>    # test the fog oracle against real packets
uv run shadowcast realfog --matches 23      # real fog agreement across a whole shard
uv run shadowcast ablate --shard <shard>    # seven belief models, and the thesis
uv run shadowcast diagnose --shard <shard>  # how the belief is wrong, not just how much
uv run shadowcast doctor             # versions, config hashes, stale artifacts

# The artifact the site reads, about 1 MB either way.
uv run shadowcast export --web --shard data/raw/12_22/batch_001.jsonl.gz

cd web && npm install && npm run dev  # the site, at localhost:5173
```

Artifacts are derived, so they are gitignored. A fresh clone has to run the export before
`npm run dev` has anything to read, and `App.tsx` names the one it loads.

## The site

Four views, all drawn from that one artifact.

**Replay** puts the same instant on screen twice, once per team's knowledge. The left map is
everything Blue could see and everything Blue believed about Red. The right is the mirror. Belief
clouds are drawn in the enemy's colour so that a cloud and the dot it collapses into share one,
which makes the moment of discovery read as a single event rather than two.

**Gank autopsy** takes the twenty seconds before a death and asks whether the victim's team could
have known. *Predictable* means the killer was visible for most of the approach. *Invisible* means
they were in fog while the belief sat somewhere else, confident and wrong. *Sudden* means the
belief was too diffuse to count as a warning.

**Ward yield** credits a ward with a sighting only when no allied champion or turret also covered
that enemy. The exclusivity clause is the metric. Without it, the wards that score best are the
most redundant ones.

**Method** lists every measured number with its provenance, and an explicit list of what has not
been measured. There is no corpus view, because rank, region and patch do not exist in this data
and a plausible-looking aggregate would cost more credibility than it could buy.

The belief renders one way: a soft cloud with the 90% credible region outlined on it. The field is
what you read at a glance, the outline is what you point at, and it encloses exactly the area the
search-area figure reports.

Both boards hold 96 fps live at 2× scale (`npm run perf`), and the belief layer is free — the same
frame rate as with it switched off. Nothing allocates in the draw loop. The terrain-and-belief
composite is cached against its 4 Hz and 8 Hz source ticks instead of being rebuilt at 60, and
champion positions are interpolated between ticks so they glide rather than step. Sidebar React
state is throttled to about 9 Hz, and the digits are eased back up to 60 by writing straight to
the DOM node.

## Development

```bash
uv sync                      # includes dev tools
uv run pytest                # full suite, ~115s warm (fog validation + belief ablation)
uv run ruff check --fix .    # lint
uv run ruff format .         # format
uv run pre-commit install    # optional: lint + format on commit
```

CI runs lint, format-check and the full suite. It fetches the navgrid and verifies its SHA-256,
because without that file the terrain and FOV tests skip cleanly — which would mean CI passing
green while never once exercising the radius-monotonicity property the whole table design rests
on.

## Validation

The point of having a ground-truth oracle is to be held to it. Every number below comes out of a
command and is written into [`docs/validation.md`](docs/validation.md).

### On real packets

All 23 matches in one shard, none skipped.

| | |
|---|---|
| **Fog agreement** | **68.26% median** (61.4–73.3%, sd 2.8) |
| — false negative / false positive | 20.2% / 12.2% |
| — by region | lane 73.4%, base 60.3%, brush 59.4%, jungle 53.0%, river 51.5% |
| Movement orders attributed | 91.9% median |
| Teams recovered | **8 / 8**, with 100.0% of hero damage across the split |
| Conformance errors from the real packet source | **0** |
| Negative information is worth | **+0.148 nats** |
| Full model vs. a plain geodesic disc | **loses**, 4.372 against 4.168 |
| 90% credible region contains the truth | **30.2%** |
| Visibility transitions we emit that the game did not | **2–3× too many** |

### On synthetic matches, where truth is known

| | |
|---|---|
| Fog agreement, reconstructed positions | 98.17% |
| Fog agreement, true positions substituted — the floor | **98.84%** |
| — brush-adjacent cells, the worst category | 90.81% |
| Harmful movement-order misattribution | **0.00–0.15%** |
| Negative information is worth | **+0.243 nats** |
| 90% credible region contains the truth | **43.4%** |
| Particle filter vs. an exact 256-state Bayes forward pass | **TV 0.030**, falling as 1/√P |
| Information-barrier leak detector | **bit-identical** |
| Python writer vs. TypeScript reader | **identical** across all 6 sections |
| Artifact size per match | **1.06 MB** gzipped, against a 2 MB budget |

Three of those rows deserve reading together, because they are the honest state of the project.

**Negative information works.** It is worth +0.243 nats on synthetic data and +0.148 on real, and
in both cases the comparison is between two `FilterSpec`s differing in exactly one field, so
nothing else can account for the gap. That is the central claim and it survives contact with real
packets.

**The full model still loses to a plain geodesic disc on real data**, 4.372 against 4.168, having
beaten it on synthetic. The full model concentrates belief into 5.8 km² and gets punished for
being confident and wrong; the disc spreads over 57 km² and hedges. This project has seen that
inversion once before, caused by a broken minion model, and fixing the vision layer reversed it.
The same reading applies now: 68% fog agreement is not good enough for a confident filter to pay
off, and the bottleneck is vision, not belief.

**Calibration is an open defect.** The 90% region contains the truth 43.4% of the time on
synthetic data and 30.2% on real. On real data there is no ground truth to check against either,
so "truth" there means this pipeline's own reconstruction, and the number measures tracking rather
than accuracy.

## Limitations, stated up front

- **This is a historical corpus.** Patches 12.22–13.2, late 2022 to early 2023. Fog-of-war
  mechanics and map geometry have not changed since, and the object of study is information
  dynamics rather than champion balance, but nothing here is current-meta advice.
- **Everything real rests on one shard.** 23 matches out of roughly 32,000, from a file whose
  contents are sorted by duration. Between-match spread on fog agreement is 2.8 points, so any
  claimed improvement smaller than about three points cannot be demonstrated on a single match.
- **Terrain provenance.** The navgrid is the Season 10 Summoner's Rift dump. Patch 12.22 references
  `AIPath_SRX_2.aimesh_ngrid`. SR did not change structurally in between, but that is an argument,
  not a verification.
- **Trajectories are reconstructed, not recorded.** Movement orders carry no entity id, so champion
  paths come from data association anchored on position-tagged spell and attack packets. The
  residual distribution is published. Mobility-heavy champions are expected to be worst.
- **The reconstruction flickers.** It produces two to three times more visibility transitions than
  the game does, with a median visible interval of 0.9 seconds against the game's 4.9. The cause is
  a position error of 75–220 units meeting a boolean test on a 28.8-unit cell. It is diagnosed and
  not yet fixed, and it will not be fixed by smoothing the mask, because the mask is what the
  belief filter consumes.
- **Kills are inferred** from health replication joined to the last damage event, because the
  stream contains no death packet.
- **Ward expiry is partly modelled.** Placement and destruction are in the data, but a ward that
  simply times out needs its duration computed from the average champion level when it was placed.

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
