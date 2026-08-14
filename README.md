# Shadowcast

**Reconstructing what League of Legends teams could actually see.**

A belief-state engine for MOBA information asymmetry, built on packet-level decoded replays.

> Status: in development. Numbers marked `[pending]` below are not yet measured, and this file
> will not carry a figure that has not been produced by `shadowcast validate`. See
> [`docs/validation.md`](docs/validation.md) for whatever has been measured so far.

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
uv run shadowcast synth make --seed 7
uv run shadowcast pipeline data/synth/m0001
cd web && npm install && npm run dev
```

## Development

```bash
uv sync                      # includes dev tools
uv run pytest                # full suite, ~4s warm (slow oracles included)
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
| Visibility agreement vs. fog events | `[pending]` |
| — brush-adjacent cells specifically | `[pending]` (expected to be the worst category) |
| Movement-order attribution residual, p99 | `[pending]` |
| Belief calibration (does the 90% region contain the truth 90% of the time?) | `[pending]` |
| Log-likelihood vs. navmesh diffusion without negative information | `[pending]` |

The last row is the one that matters: if the full model does not beat navmesh-constrained
diffusion, negative information is not doing anything and the central claim is empty.

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
