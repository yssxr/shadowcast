"""Every magic number in Shadowcast, with its provenance.

Nothing else in the package may hardcode a world bound, a sight radius, a tick
rate, or a grid size. If a number needs justifying to a reader, it belongs here
with the citation attached.

Two classes of number live here and they must not be confused:

*Measured* values were read out of a file or a packet stream and are facts about
the data. They are annotated MEASURED with what was measured and when.

*Chosen* values are our modelling decisions. They are annotated CHOSEN with the
reasoning, because a reviewer's first question about any of them is "why that?"
"""

from __future__ import annotations

import math
from typing import Final

# ---------------------------------------------------------------------------
# Summoner's Rift world geometry
# ---------------------------------------------------------------------------
# MEASURED: parsed from the header of AIPath_SRX.aimesh_ngrid (format version
# 7.1), the Season 10 Summoner's Rift navgrid shipped in
# FrankTheBoxMonster/LoL-NGRID-converter under `test files/SummonersRiftSeason10/`.
#
# Riot uses Y-up, so the horizontal plane is X/Z and the header's `y` component
# is the height range (-71.24 .. 184.52), which we discard entirely.
#
# Note these are neither of the two numbers folklore repeats. The widely-copied
# [-120, 14870] x [-120, 14980] extent is Riot-API space, a padded superset used
# so screenshot-derived minimaps line up; 0..15000 has no source at all. The
# navgrid header is the engine's own answer and it is what the packet coordinates
# actually live in.
NAVGRID_MIN_X: Final = -1.1048965
NAVGRID_MIN_Z: Final = 32.755800
NAVGRID_MAX_X: Final = 14718.4004
NAVGRID_MAX_Z: Final = 14792.2109
NAVGRID_CELL_SIZE: Final = 50.0
NAVGRID_CELLS_X: Final = 295
NAVGRID_CELLS_Z: Final = 296

NAVGRID_SPAN_X: Final = NAVGRID_MAX_X - NAVGRID_MIN_X  # 14719.505
NAVGRID_SPAN_Z: Final = NAVGRID_MAX_Z - NAVGRID_MIN_Z  # 14759.455

# Navgrid vision/pathing flag bits, from the version-7.1 format documented by
# TheKillerey/MapgeoAddon's navgrid.py and Pupix's 010 template.
#
# SEE_THROUGH is the one that matters and the one a naive implementation misses.
# It marks cells that block movement but NOT vision, and Riot stamped them along
# wall diagonals after S5 Worlds "where some line of sight oddities with regards
# to these diagonals surfaced", then later around structures. Deriving vision
# from walkability reproduces exactly the bug they patched.
# MEASURED on the S10 SR grid: 32,365 NOT_PASSABLE, 2,129 HAS_GRASS,
# 1,819 SEE_THROUGH, out of 87,320 cells.
NGRID_HAS_GRASS: Final = 0x01  # brush
NGRID_NOT_PASSABLE: Final = 0x02  # wall
NGRID_BUSY: Final = 0x04
NGRID_TARGETED: Final = 0x08
NGRID_MARKED: Final = 0x10
NGRID_PATHED_ON: Final = 0x20
NGRID_SEE_THROUGH: Final = 0x40  # blocks movement, transmits vision
NGRID_UNKNOWN_VISION: Final = 0x80  # undocumented; set on many walkable jungle cells
NGRID_ALWAYS_VISIBLE: Final = 0x100  # unused on SR

# The LoL Wiki documents 39 brush patches on Summoner's Rift. Connected-component
# labelling of the S10 navgrid's HAS_GRASS cells gives 40, identically under
# 4- and 8-connectivity (so no two patches are even diagonally adjacent, and the
# connectivity choice is moot). Component sizes run 20-124 cells.
#
# The discrepancy is one patch and we have not chased it down; the wiki may count
# a split brush as one, or SR may have gained one between S10 and S12. It does not
# matter for correctness, because what the assertion needs to catch is a raster
# that FUSES patches (which would give far fewer) or SHATTERS one (far more).
# Asserting an exact 39 would fail on correct output, so the test takes a band.
SR_BRUSH_PATCHES_DOCUMENTED: Final = 39
SR_BRUSH_PATCHES_MEASURED: Final = 40
SR_BRUSH_PATCH_COUNT: Final = SR_BRUSH_PATCHES_MEASURED

# MEASURED on the same navgrid: 54,955 of 87,320 cells are walkable (62.9%), and
# every one of them is reachable from the map centre -- zero orphaned pockets,
# which is strong evidence the flag block was read at the right offset.
#
# Note this is well above the 25-40% the FOV table was budgeted against, so a full
# table is ~165k rows and ~285 MB rather than ~160 MB. Still mmap-friendly and
# under 2% of a 16 GB machine, but the estimate was wrong and the real number
# belongs here.
SR_WALKABLE_FRACTION_MEASURED: Final = 0.62935

# ---------------------------------------------------------------------------
# Internal simulation grid
# ---------------------------------------------------------------------------
# CHOSEN: 512 x 512. Cells must be square for a circular radius mask to be
# correct, so the grid covers a square span equal to the larger navgrid axis,
# anchored at the navgrid origin. X therefore overruns the navgrid's own maximum
# by 40 u (1.4 cells); those cells fall outside the source data and are walls.
#
# Why not coarser: brush entrances are 100-250 world units wide. At 256^2
# (57.7 u/cell) a 100 u gap rasterises to 1 cell or 0, and a gap that closes is a
# topology change -- the brush seals, vision stops leaking through a real
# corridor, and the error is silent and systematic. At 512^2 a 100 u gap is 3.5
# cells even under adverse alignment.
#
# Why not finer: at 1024^2 the FOV table is 1.4-2.0 GB, which thrashes a 16 GB
# machine once masks, particles and export buffers are also resident -- and the
# source navgrid is 50 u/cell, so 14.5 u cells would be inventing precision the
# data does not have. 512^2 already resamples finer than the source; see
# GRID_CELL_SIZE below.
GRID: Final = 512
WORLD_MIN_X: Final = NAVGRID_MIN_X
WORLD_MIN_Z: Final = NAVGRID_MIN_Z
WORLD_SPAN: Final = max(NAVGRID_SPAN_X, NAVGRID_SPAN_Z)  # 14759.455, square
GRID_CELL_SIZE: Final = WORLD_SPAN / GRID  # 28.827 u

# The effective resolution of the terrain is the navgrid's 50 u, not our 28.8 u.
# The finer grid buys rounder radius discs and less source-snapping error, not
# more terrain detail. Say so in the write-up rather than implying otherwise.
TERRAIN_SOURCE_RESOLUTION: Final = NAVGRID_CELL_SIZE

# ---------------------------------------------------------------------------
# Sight radii, patch 12.22
# ---------------------------------------------------------------------------
# MEASURED: LoL Wiki "Sight" and "Ward", revision 3480276 dated 2022-11-04,
# retrieved through the MediaWiki API. The revision matters: the LIVE page is
# wrong for this patch. V13.22 cut the fog-attack reveal from 400/4.5s to
# 300/2.0s, so building against the current wiki would be 33% off on radius and
# 125% off on duration.
#
# Radii are measured centre-to-centre: a unit becomes visible when its centre
# enters the observer's radius.
SIGHT_CHAMPION: Final = 1350.0  # also pets, super minions, and turrets
SIGHT_TURRET: Final = 1350.0
SIGHT_MINION: Final = 1200.0  # melee, caster and siege alike
SIGHT_WARD_TOTEM: Final = 900.0  # yellow trinket
SIGHT_WARD_CONTROL: Final = 900.0
SIGHT_WARD_ZOMBIE: Final = 900.0
SIGHT_WARD_FARSIGHT: Final = 500.0  # blue trinket
SIGHT_GHOST_PORO: Final = 450.0

# MEASURED: no champion has a non-standard BASE sight radius at 12.22 -- neither
# the 2022 revision nor the current page lists an exception. Differences come
# from abilities (Quinn's Heightened Senses), which is a v2 concern.
#
# MEASURED: neutral monsters grant NO team vision. The wiki is explicit that only
# "non-neutral units" innately award sight; monsters have aggro and leash ranges,
# not vision radii. Modelling jungle camps as vision sources would invent vision
# that does not exist -- exactly the kind of error that inflates a fog-agreement
# number in the wrong direction.
SIGHT_NEUTRAL_MONSTER: Final = 0.0

# Reveal-on-attack-from-fog. Attacking anything (including a ward) from your own
# team's fog reveals a disc centred on you when the attack completes.
FOG_ATTACK_REVEAL_RADIUS: Final = 400.0
FOG_ATTACK_REVEAL_DURATION: Final = 4.5

# Ward lifetimes. Totem duration scales 90-120s with the AVERAGE of all ten
# champions' levels (changed from owner level in V8.23), so it is a function of
# game state, not of the placer.
WARD_TOTEM_DURATION_MIN: Final = 90.0
WARD_TOTEM_DURATION_MAX: Final = 120.0
WARD_TOTEM_MAX_PLACED: Final = 3
WARD_CONTROL_MAX_PLACED: Final = 1  # indefinite duration, 4 HP
WARD_FARSIGHT_MAX_PLACED: Final = 0  # 0 == unlimited

# CHOSEN: 1500 u, above every radius above, with headroom for ability-granted
# sight. The FOV table is built at this radius and every smaller radius is served
# by intersecting it with a circular mask; anything larger takes the live-compute
# path, so this is a performance boundary and never a correctness one.
RMAX_UNITS: Final = 1500.0
RMAX_CELLS: Final = math.ceil(RMAX_UNITS / GRID_CELL_SIZE)  # 53
FOV_WINDOW: Final = 2 * RMAX_CELLS + 1  # 107

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
# CHOSEN: 8 Hz internal simulation. At a champion's ~400 u/s that is 50 u or 1.7
# cells per tick -- fine enough that a vision transition is never missed by more
# than a cell, coarse enough that a full match reconstruct stays in seconds.
# Trajectory validation against fog-transition timestamps uses the exact segment
# table instead, because those arrive at ~30 Hz and comparing them against an
# 8 Hz resample would bake in a systematic +-62 ms.
TICK_HZ: Final = 8
TICK_DT: Final = 1.0 / TICK_HZ

# CHOSEN: 0..15:00. The design commits to it, and it is also forced by the data:
# decoding stops at the last cleanly-decoded 30-second chunk, so matches run
# 12-21 minutes rather than to completion. Requiring 900 s keeps every corpus
# aggregate over an identical window, which cross-match statistics need.
# MEASURED: roughly 60% of matches decode past 900 s.
MATCH_WINDOW_SECONDS: Final = 900.0
MATCH_TICKS: Final = int(MATCH_WINDOW_SECONDS * TICK_HZ) + 1  # 7201

# MEASURED: packet timestamps are floats in seconds since match start, with
# ~30 ms granularity, consistent with a 30 Hz server tick. maknee documents no
# tick rate, so this is inference from observed timestamp spacing, not a
# guarantee. Used only to set validation tolerances.
PACKET_TIME_GRANULARITY: Final = 1.0 / 30.0

# ---------------------------------------------------------------------------
# Belief and entropy
# ---------------------------------------------------------------------------
# CHOSEN: 1024 particles per filter, 10 filters (2 observing teams x 5 enemies).
#
# The number is tied to the entropy lattice and must not drift from it. A
# 32^2 lattice restricted to walkable bins has 890 of them, so its maximum
# entropy is log2(890) = 9.80 bits, while the plug-in estimator over P
# particles saturates at log2(P). At 1024 that ceiling is exactly 10.0 bits and
# the lattice fits underneath it.
#
# CORRECTED: this was 400, on an estimate of "~358 walkable bins, max 8.49
# bits". MEASURED, the bin count is 890 -- the estimate assumed roughly a third
# of bins would be unwalkable, but a 32^2 bin is 461 units on a side and almost
# every 461-unit square of Summoner's Rift contains some walkable ground. At 400
# particles the measured entropy of a uniform belief was 8.74 bits against a
# log2(400) = 8.64 ceiling, i.e. the estimator was pinned and H was reporting
# the particle budget rather than the game. That is precisely the failure the
# lattice choice was written to prevent, and the arithmetic was simply wrong.
PARTICLES: Final = 1024
N_TEAMS: Final = 2
N_ENEMIES: Final = 5

# CHOSEN, and this is the subtle one. Plug-in Shannon entropy of a P-particle
# cloud saturates at log2(P). A lattice whose maximum entropy exceeds that
# ceiling makes H a measurement of the particle budget rather than of the game.
#
# MEASURED: 32^2 gives 890 walkable bins (461 u per side), so max 9.80 bits,
# which is why PARTICLES is 1024 and not 400. The two constants are one
# decision: 2*log2(32) = 10.0 = log2(1024), so the lattice ceiling and the
# estimator ceiling coincide exactly and neither is the binding constraint.
#
# Why not coarser: at 16^2 a bin is 922 units and the full model's 90% credible
# region is 0.53 ku^2, which would be two and a half bins -- too coarse to
# measure the quantity being reported. Why not finer: 64^2 needs 3,180 bins and
# 11.63 bits, so P would have to exceed 3,200.
#
# The lattice is frozen and hashed into every artifact header. Entropy in bits is
# only defined relative to a choice of reference measure, so a number computed
# against a different lattice is not comparable and must never be mixed in.
# Changing this means bumping to V2 and re-exporting everything.
ENTROPY_LATTICE: Final = 32
ENTROPY_LATTICE_VERSION: Final = "ENTROPY_LATTICE_V1"

# CHOSEN: the credible region reported alongside entropy. Area in km^2 is
# resolution-robust, particle-count-robust, and unit-ful -- "the enemy could be
# anywhere in 4.1 km^2" is a claim a reader can check against the map, which
# "6.2 bits" is not. Bits stay as the on-screen number because the design uses
# them; area is the primary number in the validation report.
CREDIBLE_MASS: Final = 0.90

# CHOSEN: one-bin smoothing applied to the scoring histogram, and part of the
# metric's definition rather than a tweak.
#
# A particle set cannot resolve a distribution below about one particle per bin,
# so a truth landing in a bin that happens to hold no particles gets probability
# exactly zero -- and every credible region, at every level, excludes it. That
# is a statement about the sample, not the belief, and it made calibration
# unreadable: coverage at the 25% level came out at 0.0% for every propagated
# model. Spreading a fraction of each bin's mass to its eight neighbours is the
# lattice form of a kernel density estimate, with the bandwidth pinned to one
# bin so there is no knob to tune.
#
# The ranking must not depend on it. `test_belief.py` re-runs the comparison
# unsmoothed and asserts the models finish in the same order, because a result
# that only appears under smoothing is an artefact of the smoothing.
SCORING_SMOOTHING: Final = 0.25

# CHOSEN: detection probability for the negative update. A particle sitting in
# the observer's visible region without a corresponding observation is falsified,
# but not with certainty -- radius quantisation (+-14 u at this grid), the ~3%
# asymmetry inherent to recursive shadowcasting, and trajectory timing error all
# make the mask slightly wrong at its edges. Weighting by log1p(-p_d) instead of
# killing outright absorbs that instead of converting it into catastrophic
# particle death. p_d = 1.0 is retained as an ablation to show the difference.
PD_INTERIOR: Final = 0.98
PD_EDGE: Final = 0.75
PD_EDGE_RING_CELLS: Final = 2

# CHOSEN: fallback respawn time, used only when a death has no observable
# respawn. `resolve/deaths.py` recovers respawn from the victim's next anchor,
# so this applies to the tail of a match where no anchor follows. MEASURED
# (patch 12.22 base respawn table): 16 s at level 6 rising to 40 s at level 13,
# which is the level range reached inside a fifteen-minute window; 25 s sits mid
# range. The time-increase factor starts at 15:00 and so never applies here.
# Only affects how long a dead enemy's position counts as known.
RESPAWN_FALLBACK_SECONDS: Final = 25.0

# CHOSEN: coarse lattice for geodesic reachability. A reachability set answers
# "which cells could he have walked to by now", and at 115 u resolution that
# answer is already finer than the question -- while Dijkstra over 16k cells is
# ~60x cheaper than over 262k, which matters because the geodesic baseline wants
# a fresh field at every sighting.
REACH_LATTICE: Final = 128

# CHOSEN: effective travel speed for reachability and the disc baselines. Base
# movement speed is ~335 u/s and boots put a laner near 400; junglers with a
# trail buff exceed it. MEASURED across synthetic matches, the 99th percentile
# of reconstructed speed is 415 u/s. Using a value slightly above the plausible
# maximum is the conservative direction for a reachability set: too small
# excludes the truth and breaks calibration outright, too large only costs
# sharpness.
V_MAX_UNITS_PER_SECOND: Final = 450.0

# ---------------------------------------------------------------------------
# Motion model
# ---------------------------------------------------------------------------
# The belief's random walk is a RANDOM-WAYPOINT model, not a diffusion, and the
# four constants below were FITTED rather than chosen -- an unbiased walk cannot
# reproduce how far champions travel at any setting of its parameters.
#
# MEASURED, median champion displacement against synthetic ground truth:
#
#     horizon      2 s      5 s     10 s     20 s
#     truth      268.1u   565.2u   976.5u  1394.9u
#     fitted     257.8u   531.5u   979.7u  1782.9u
#
# Mean absolute log error 0.087 over a grid search on real Summoner's Rift
# terrain, reproducible across seeds (a second seed gives 248/539/979/1786).
# The best diffusive walk reached only 900 u at twenty seconds against a truth
# of 1,395 -- a random walk is recurrent and wanders back over itself, while a
# champion crossing the map does not -- and raising heading persistence far
# enough to fix that broke the short horizon, since a straight-line walk at
# champion speed covers 1,600 u in the two seconds where the truth is 268.
#
# KNOWN BIAS: the fit overshoots at twenty seconds by 28%, because real
# champions reverse course (recall, then walk back) and the model does not.
# That direction is the safe one -- the belief is slightly too spread rather
# than too confident -- but it is a real limit and is stated in the validation
# report rather than smoothed over.
#
# These are fitted against SYNTHETIC truth, whose champions move by A* between
# waypoints. That is the same family as the model, so the fit is well specified
# and correspondingly unimpressive as evidence. Refitting on the real corpus is
# an explicit item for M9.
MOTION_SUB_STEPS: Final = 2
PARTICLE_STAY_PROB: Final = 0.10
HEADING_PERSISTENCE: Final = 1.0

# CHOSEN/FITTED: strength of the pull toward a particle's current destination.
# exp(beta * cos t) at 0.45 makes a step toward the goal about 2.5x as likely as
# one directly away -- directed enough to travel, weak enough that walls and
# accumulated negative information still dominate. Zero reduces the walk to pure
# diffusion, which is exactly the `navmesh_diffusion` baseline.
GOAL_BETA: Final = 0.45

# FITTED: how close counts as arrived, in cells (461 units). Larger than it
# sounds because a goal is a landmark, not a coordinate -- "go mid" is satisfied
# by arriving anywhere in mid.
GOAL_ARRIVE_CELLS: Final = 16.0

# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
# CHOSEN: resample when the effective sample size falls below half. The standard
# rule of thumb, and the cost of being wrong is small in both directions --
# resampling too eagerly loses diversity, too rarely wastes particles.
ESS_RESAMPLE_FRACTION: Final = 0.5

# CHOSEN: below this, the cloud is no longer a sample of anything and is rebuilt
# from the geodesic reachability set. Deliberately low: reinitialisation throws
# away accumulated negative information, so it should be a last resort rather
# than a routine step. `depletion_events` is a QA signal, not just a counter --
# frequent depletion means the vision masks, the trajectories or p_d are wrong.
ESS_DEPLETION_FRACTION: Final = 0.05

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
# CHOSEN: 128^2 vision masks -- 115 u/cell, recognisable SR shapes, and XOR
# against the previous tick compresses to ~200 changed bits of 16,384.
EXPORT_MASK_GRID: Final = 128
EXPORT_MASK_HZ: Final = 4

# CHOSEN: 32^2 belief display grid, matching the mockup. The artifact does not
# ship grids at all (see below); the frontend rasterises a mixture into this.
DISPLAY_BELIEF_GRID: Final = 32
DISPLAY_TERRAIN_GRID: Final = 64  # the mockup's chunky terrain render

# CHOSEN: belief as a 16-component mixture, not a grid. Grids do not fit by any
# margin -- 64^2 u8 at 8 Hz is 295 MB per match and 37 MB even at 1 Hz, and no
# compression closes that. 16 components at (x, z, w, sigma) as u8 is 64 bytes
# per observer-enemy-tick, delta-coded to ~0.7 MB per match, and the frontend
# rasterises it to DISPLAY_BELIEF_GRID at draw time so the visual is unchanged.
BELIEF_COMPONENTS: Final = 16
BELIEF_EXPORT_HZ: Final = 8
BELIEF_KEYFRAME_SECONDS: Final = 8.0

# CHOSEN: 12-bit fixed point for positions. The quantisation and the delta width
# are one decision, not two: at full u16 precision (0.23 u/LSB) a 50 u tick move
# is 220 LSB and overflows an int8 delta, while at 12-bit (3.6 u/LSB) even a
# 400 u Flash fits in 110.
POSITION_QUANT_BITS: Final = 12
POSITION_EXPORT_HZ: Final = 8

ARTIFACT_SCHEMA_VERSION: Final = 1
# Sections are 8-byte aligned because `new Float32Array(buf, offset, n)` throws a
# RangeError on a misaligned offset, and the message does not point at the writer.
ARTIFACT_SECTION_ALIGN: Final = 8

# ---------------------------------------------------------------------------
# Packet stream quirks
# ---------------------------------------------------------------------------
# MEASURED over 965,768 real packets from 12_22/batch_001.jsonl.gz.
#
# Waypoint coordinates are map-centred (x in [-7306, 7278]) while every other
# position field is world-framed (x in [0, 14484]). The offset is near 7500, but
# +7500 puts the observed maximum at 14778, overshooting the navgrid's 14718 by
# 60 u -- so it is a starting point for calibration, not a constant. Calibrate by
# maximising the fraction of waypoints landing on walkable cells.
WAYPOINT_OFFSET_GUESS: Final = 7500.0
WAYPOINT_OFFSET_SEARCH: Final = 200.0

# Hero net_ids were contiguous 0x4000001E..0x40000027 in every match sampled.
# Convenient, not guaranteed -- resolve heroes from CreateHero, and use this only
# as a sanity check.
HERO_NETID_HINT_LO: Final = 1073741854
HERO_NETID_HINT_HI: Final = 1073741863
N_HEROES: Final = 10

# LeaveFog is 65-70% of all packets and maknee documents "20+ repeats sometimes"
# at an identical timestamp. Dedupe on (time, net_id) before anything else.
FOG_DEDUPE_REQUIRED: Final = True

# Replication attributes. MEASURED: 57% of real entries have an EMPTY `name`, leaving
# only the (primary_index, secondary_index) pair, so a reader that keys on the name
# alone silently discards the majority. These pairs were confirmed against real data.
ATTR_MOVE_SPEED: Final = "mMoveSpeed"
ATTR_HP: Final = "mHP"
REPL_INDEX_NAMES: Final = {
    (32, 0): "mHP",
    (32, 1): "mMaxHP",
    (32, 10): "mExp",
    (32, 14): "mPAR",
    (32, 18): "mSAR",
    (32, 24): ATTR_MOVE_SPEED,
    (32, 30): "mIsTargetableToTeamFlags",
}

# Champion base movement speed spans roughly 325-345 before items, and boots plus
# haste can push it far higher. Used only as a sanity band on recovered values.
MOVE_SPEED_MIN: Final = 200.0
MOVE_SPEED_MAX: Final = 1200.0
MOVE_SPEED_DEFAULT: Final = 335.0

# SpawnMinion.time is denormal-float garbage (observed max 1.96e-39) and
# BarrackSpawnUnit.minion_type/minion_level are garbage too (u64-max values,
# levels of 113 and 202). Classify minions by SpawnMinion.name/skin_name and take
# timing from the surrounding packets' clock.
MIN_VALID_PACKET_TIME: Final = 1e-9

# Ward units, keyed by (SpawnMinion.name, skin_name). Wards are not their own
# packet type -- they arrive as SpawnMinion, with position1 giving the exact
# placement, targetable_on_client apparently holding the owner hero's net_id, and
# WardCorpse marking the death. That makes placement, owner and lifetime all
# directly observable, which is better than the strategy document assumed.
WARD_UNITS: Final = {
    ("SightWard", "YellowTrinket"): "totem",
    ("SightWard", "BlueTrinket"): "farsight",
    ("VisionWard", "SightWard"): "control",
    ("JammerDevice", "JammerDevice"): "control",
    ("Ward", "PerksZombieWard"): "zombie",
    ("PlantVision", "SRU_Plant_Vision"): "scryer",
    ("FakeCrab", "Sru_CrabWard"): "scuttle",
}
WARD_CORPSE_UNIT: Final = ("WardCorpse", "S5Test_WardCorpse")

# Lane minions also arrive as SpawnMinion. Matched on a name substring rather than an
# exact table because the real internal names are not confirmed — the dataset research
# enumerated ward and plant units but not the regular minion ones. A substring match is
# robust to whatever they turn out to be, and confirming them is a recon item.
MINION_NAME_TOKENS: Final = ("Minion", "Melee", "Ranged", "Siege", "Super")
MINION_SIGHT: Final = SIGHT_MINION

WARD_SIGHT_BY_KIND: Final = {
    "totem": SIGHT_WARD_TOTEM,
    "control": SIGHT_WARD_CONTROL,
    "farsight": SIGHT_WARD_FARSIGHT,
    "zombie": SIGHT_WARD_ZOMBIE,
    "scryer": SIGHT_WARD_TOTEM,
    "scuttle": SIGHT_CHAMPION,
}

# Turret internal names encode team: T1 is ORDER (blue), T2 is CHAOS (red).
# CreateTurret carries no position, so positions come from a static SR table and
# the name is the join key -- which makes turrets both our static vision sources
# and the anchor for team resolution, since CreateHero has no team field.
TEAM_ORDER: Final = 0  # blue
TEAM_CHAOS: Final = 1  # red
TURRET_TEAM_TOKENS: Final = {"T1": TEAM_ORDER, "T2": TEAM_CHAOS}
TURRET_SHRINE_TOKENS: Final = {
    "OrderTurretShrine": TEAM_ORDER,
    "ChaosTurretShrine": TEAM_CHAOS,
}

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
HF_REPO: Final = "maknee/league-of-legends-decoded-replay-packets"
# The typo is real and required.
HF_REPO_S12: Final = "maknee/leaague-of-legends-decoded-replay-packets-s12-unorganized"
# MEASURED: 108.47 GB across 1,348 shards in four patch directories. The dataset
# card declares splits 13_01 and 13_03 that do not exist.
HF_PATCH_DIRS: Final = ("12_22", "12_23", "13_1", "13_2")
DEFAULT_PATCH: Final = "12_22"
# MEASURED: batch_001 is 76 MB gzipped, 2.01 GB of JSON, 23 matches. Extrapolated
# over the repo that is ~32,000 matches -- not the 700k or 1.4M the card and blog
# claim. Never repeat those figures.
MEASURED_MATCHES_PER_SHARD: Final = 23

NAVGRID_URL: Final = (
    "https://raw.githubusercontent.com/FrankTheBoxMonster/LoL-NGRID-converter/"
    "master/test%20files/SummonersRiftSeason10/AIPath_SRX.aimesh_ngrid"
)
# The S10 dump. Patch 12.22's map11.bin references AIPath_SRX_2.aimesh_ngrid;
# SR terrain did not change structurally between S10 and S12, but that is an
# argument, not a verification. Disclose the provenance and revisit only if the
# fog-agreement numbers suggest terrain is the culprit.
NAVGRID_PROVENANCE: Final = "SummonersRiftSeason10/AIPath_SRX.aimesh_ngrid (v7.1)"
