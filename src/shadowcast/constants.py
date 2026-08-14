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

# MEASURED: the LoL Wiki documents 39 brush patches on Summoner's Rift. Brush
# component labelling asserts against this, which catches both a raster that
# fuses two patches and a grouping that splits one.
SR_BRUSH_PATCH_COUNT: Final = 39

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
# CHOSEN: 400 particles per filter, 10 filters (2 observing teams x 5 enemies).
PARTICLES: Final = 400
N_TEAMS: Final = 2
N_ENEMIES: Final = 5

# CHOSEN, and this is the subtle one. Plug-in Shannon entropy of a P-particle
# cloud saturates at log2(P) = 8.64 bits at P=400. A lattice whose maximum
# entropy exceeds that ceiling makes H a measurement of the particle budget
# rather than of the game. 32^2 over the map gives ~358 walkable bins and a
# maximum of ~8.49 bits, which sits just under the ceiling.
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
KDE_BANDWIDTH_UNITS: Final = 300.0

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
