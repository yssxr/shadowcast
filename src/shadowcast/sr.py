"""Summoner's Rift landmarks in world coordinates: lanes, jungle routes, turrets, wards.

**Provenance, and it matters.** These are *schematic* positions, not extracted game
data. They come from the normalised layout in the project's own design mockup,
converted into world coordinates and snapped onto walkable ground. They are accurate
enough to drive a synthetic match, champions walk down lanes, junglers visit camps,
wards sit in plausible spots, and they are not accurate enough to be used as ground
truth about the real map.

Nothing in the real-data path should depend on this module, and it does not need to,
because **the real positions are recoverable from the packet stream itself**:

- Turrets: `CreateTurret` gives `net_id` -> internal name but no position. However
  `BasicAttackPos` carries `source_net_id` alongside `source_position`, and turrets
  shoot minions all game. The mode of a turret net_id's attack positions is its
  location, to within a unit. So the coordinates below exist only so the synthetic
  generator has somewhere to put turrets.
- Jungle camps: `CreateNeutral.position1` is already exact, and `name` embeds the
  camp identity (`SRU_Red4.1.1`, `SRU_Murkwolf8.1.1`, `SRU_Dragon_Earth`).
- Wards: `SpawnMinion.position1` is exact.

The design mockup uses screen coordinates with y increasing downward and blue base at
the lower left. World z increases the other way, hence the flip in `_world`.
"""

from __future__ import annotations

import numpy as np

from shadowcast.constants import TEAM_CHAOS, TEAM_ORDER, WORLD_MIN_X, WORLD_MIN_Z, WORLD_SPAN

__all__ = [
    "FOUNTAINS",
    "JUNGLE_ROUTES",
    "LANES",
    "TURRETS",
    "WARD_SPOTS",
    "lerp_polyline",
    "polyline_length",
    "snap_polyline",
]


def _world(pts: list[tuple[float, float]]) -> np.ndarray:
    """Design-mockup normalised coordinates -> world (x, z), flipping the z axis."""
    out = np.empty((len(pts), 2), dtype=np.float64)
    for n, (x, y) in enumerate(pts):
        out[n, 0] = WORLD_MIN_X + x * WORLD_SPAN
        out[n, 1] = WORLD_MIN_Z + (1.0 - y) * WORLD_SPAN
    return out


LANES: dict[str, np.ndarray] = {
    "top": _world(
        [(0.105, 0.795), (0.075, 0.44), (0.10, 0.155), (0.165, 0.095), (0.60, 0.078), (0.80, 0.10)]
    ),
    "mid": _world([(0.175, 0.83), (0.35, 0.655), (0.5, 0.5), (0.655, 0.35), (0.83, 0.175)]),
    "bot": _world(
        [(0.20, 0.895), (0.60, 0.922), (0.80, 0.905), (0.905, 0.835), (0.925, 0.56), (0.90, 0.20)]
    ),
}

#: Jungle clear routes, roughly camp to camp, ending back where they began.
JUNGLE_ROUTES: dict[int, np.ndarray] = {
    TEAM_ORDER: _world(
        [
            (0.10, 0.62),
            (0.204, 0.516),
            (0.308, 0.412),
            (0.332, 0.332),
            (0.44, 0.44),
            (0.484, 0.796),
            (0.588, 0.692),
            (0.412, 0.63),
            (0.24, 0.72),
        ]
    ),
    TEAM_CHAOS: _world(
        [
            (0.90, 0.38),
            (0.796, 0.484),
            (0.692, 0.588),
            (0.668, 0.668),
            (0.56, 0.56),
            (0.516, 0.204),
            (0.412, 0.308),
            (0.60, 0.37),
            (0.76, 0.28),
        ]
    ),
}

FOUNTAINS: dict[int, np.ndarray] = {
    TEAM_ORDER: _world([(0.075, 0.925)])[0],
    TEAM_CHAOS: _world([(0.925, 0.075)])[0],
}


def polyline_length(pts: np.ndarray) -> float:
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def lerp_polyline(pts: np.ndarray, s: float) -> np.ndarray:
    """Point at arclength fraction `s` along a polyline. `s` is clamped to [0, 1]."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    total = seg.sum()
    if total <= 0:
        return pts[0].copy()
    want = float(np.clip(s, 0.0, 1.0)) * total
    for n, d in enumerate(seg):
        if want <= d or n == len(seg) - 1:
            f = want / d if d > 0 else 0.0
            return pts[n] + (pts[n + 1] - pts[n]) * f
        want -= d
    return pts[-1].copy()


def _turrets() -> list[tuple[str, int, np.ndarray]]:
    """Turrets at plausible lane fractions, named in the engine's convention.

    Real names look like `Turret_T1_C_05_A`, where T1 is ORDER and T2 is CHAOS, and
    the letter is the lane (L top, C mid, R bot). Team resolution on real data reads
    exactly that token, so keeping the convention here means the synthetic stream
    exercises the same parser.
    """
    out: list[tuple[str, int, np.ndarray]] = []
    lane_letter = {"top": "L", "mid": "C", "bot": "R"}
    # Fractions from each team's end: outer, inner, inhibitor.
    for lane, fracs in (
        ("mid", (0.14, 0.30, 0.46)),
        ("top", (0.16, 0.34, 0.46)),
        ("bot", (0.16, 0.34, 0.46)),
    ):
        for idx, f in enumerate(fracs):
            tier = ("01", "02", "03")[idx]
            out.append(
                (
                    f"Turret_T1_{lane_letter[lane]}_{tier}_A",
                    TEAM_ORDER,
                    lerp_polyline(LANES[lane], f),
                )
            )
            out.append(
                (
                    f"Turret_T2_{lane_letter[lane]}_{tier}_A",
                    TEAM_CHAOS,
                    lerp_polyline(LANES[lane], 1.0 - f),
                )
            )
    # Nexus pair plus the shrine, which is the fountain turret.
    for team, token, sign in ((TEAM_ORDER, "T1", 1), (TEAM_CHAOS, "T2", -1)):
        base = FOUNTAINS[team]
        toward = np.array([1.0, 1.0]) * sign * 900.0
        out.append((f"Turret_{token}_C_04_A", team, base + toward))
        out.append((f"Turret_{token}_C_05_A", team, base + toward * 1.4))
        shrine = "OrderTurretShrine" if team == TEAM_ORDER else "ChaosTurretShrine"
        out.append((f"Turret_{shrine}_A", team, base.copy()))
    return out


TURRETS: list[tuple[str, int, np.ndarray]] = _turrets()

#: Ward spots with human labels, for the synthetic scenario and the ward-yield view.
#: Labels are the mockup's own, which is what the design renders.
WARD_SPOTS: list[tuple[str, np.ndarray]] = [
    ("enemy blue-side entrance", _world([(0.204, 0.516)])[0]),
    ("river crossing, top", _world([(0.332, 0.332)])[0]),
    ("deep top jungle", _world([(0.516, 0.204)])[0]),
    ("bot river mouth", _world([(0.588, 0.692)])[0]),
    ("drake pit approach", _world([(0.668, 0.668)])[0]),
    ("mid-river pinch", _world([(0.44, 0.44)])[0]),
    ("blue top jungle", _world([(0.412, 0.308)])[0]),
    ("centre river", _world([(0.56, 0.56)])[0]),
    ("blue base entrance", _world([(0.24, 0.72)])[0]),
    ("bot lane brush", _world([(0.484, 0.796)])[0]),
]


def snap_polyline(terrain, pts: np.ndarray, max_radius: int = 24) -> tuple[np.ndarray, float]:
    """Snap world points onto walkable ground.

    Returns the snapped points and the largest displacement in world units. Callers
    should assert that displacement is small: a landmark that moved hundreds of units
    was in the wrong place to begin with, and silently accepting it would put a
    vision source somewhere the scenario did not intend.
    """
    from shadowcast.geom.grid import cell_to_world, world_to_cell
    from shadowcast.geom.path import nearest_walkable

    out = np.empty_like(pts, dtype=np.float64)
    worst = 0.0
    for n, (x, z) in enumerate(np.atleast_2d(pts)):
        i, j = world_to_cell(float(x), float(z))
        i = int(np.clip(i, 0, terrain.grid - 1))
        j = int(np.clip(j, 0, terrain.grid - 1))
        sj, si = nearest_walkable(terrain.walkable, j, i, max_radius=max_radius)
        wx, wz = cell_to_world(si, sj)
        out[n] = (wx, wz)
        worst = max(worst, float(np.hypot(wx - x, wz - z)))
    return out, worst


# ---------------------------------------------------------------------------
# Minion waves
# ---------------------------------------------------------------------------
#: Game knowledge, not map data: waves spawn on a fixed cadence and walk their lane.
#: Lane vision in League is mostly minion vision, so omitting waves would badly
#: overstate fog: but tracking individual minions is not possible, because movement
#: orders carry no entity id and minions have none of the labelled position packets that
#: make champion attribution work.
#:
#: So a wave is modelled as ONE clump at the wave centre. That is a real simplification:
#: an actual wave is six units spread over roughly 400 units, and a single 1,200-unit
#: sight source at the centre approximates the union of theirs rather than reproducing it.
#: What makes it defensible is that both the ground-truth oracle and the reconstruction
#: use this same model, so minion vision is a shared constant and the fog-agreement figure
#: measures champion trajectories, ward lifetimes and field-of-view geometry instead.
#: On real data the model's own error becomes part of the measured disagreement, and that
#: has to be said when the number is published.
MINION_WAVE_INTERVAL = 30.0
FIRST_WAVE_SPAWN = 65.0
MINION_SPEED = 325.0
#: How long a clump survives before the opposing wave clears it.
#:
#: MEASURED from the geometry: a wave reaches the meeting point 18.7 s after spawn on mid
#: and 27.6 s on top and bot, and the next wave arrives one interval, 30 s: after that.
#: So a clump lives roughly until its replacement shows up, which is 49-58 s depending on
#: the lane. 55 s is one constant across all three.
#:
#: The old value of 62 s was justified as "the average that puts the meeting point near
#: the lane midpoint", which stopped being the mechanism when the meeting point became
#: explicit: and was never really the mechanism, since the clump used to march past the
#: midpoint and out the far end of the lane regardless.
MINION_CLUMP_LIFETIME = 55.0
#: Arclength fraction along the lane where each team's minions enter.
MINION_SPAWN_S = {TEAM_ORDER: 0.055, TEAM_CHAOS: 0.945}
#: Arclength fraction where the two waves meet and stop advancing. Both teams spawn
#: simultaneously and move at the same speed, so it is the midpoint. The real equilibrium
#: shifts as turrets fall; turret destruction is absent from the corpus and is not
#: modelled, so this is the early-game answer.
MEETING_S = 0.5


def minion_wave_schedule(duration: float) -> list[tuple[float, str, int]]:
    """Every (spawn time, lane, team) wave within a match window."""
    out: list[tuple[float, str, int]] = []
    t = FIRST_WAVE_SPAWN
    while t < duration:
        for lane in LANES:
            for team in (TEAM_ORDER, TEAM_CHAOS):
                out.append((t, lane, team))
        t += MINION_WAVE_INTERVAL
    return out


def minion_spawn_point(lane: str, team: int) -> np.ndarray:
    return lerp_polyline(LANES[lane], MINION_SPAWN_S[team])


def minion_clump_position(
    lane: str,
    team: int,
    spawn_t: float,
    t: float,
    death_t: float | None = None,
    front_s: float = MEETING_S,
) -> np.ndarray | None:
    """Where a wave's clump is at time `t`, or None if it has not spawned or has died.

    Walks its lane from its own end toward the enemy's at `MINION_SPEED`. Everything it
    needs: the lane, the side, the spawn time and the death time, is observable from
    `SpawnMinion` and `NPCDieMapView`, so the reconstruction needs no minion tracking.
    """
    if t < spawn_t:
        return None
    end = death_t if death_t is not None else spawn_t + MINION_CLUMP_LIFETIME
    if t > end:
        return None
    length = polyline_length(LANES[lane])
    if length <= 0:
        return None
    travelled = MINION_SPEED * (t - spawn_t) / length
    sign = 1.0 if team == TEAM_ORDER else -1.0
    s = MINION_SPAWN_S[team] + sign * travelled

    # **A wave stops where it meets the opposing wave.** Clipping to [0, 1] instead let
    # it march the entire lane and park in the enemy fountain: at 325 u/s a 62-second
    # clump covers 20,150 units on a lane about 16,000 long, so by five minutes every
    # wave was sitting in the enemy base granting 1,200 units of vision there, three
    # permanent floodlights per team inside the other team's spawn, which showed up as
    # unexplained circles on the map.
    #
    # Both teams spawn together and move at the same speed, so the midpoint is where they
    # meet *on average*, and only on average. `front_s` carries the measured meeting
    # point when there is evidence for one (see `l2_reconstruct.front`), because the real
    # front sits a median 1,442 units from the midpoint on top and 1,640 on bot, which is
    # further than a minion can see. It defaults to the midpoint so a caller with no
    # evidence, including the synthetic oracle, gets the early-game answer unchanged.
    spawn_s = MINION_SPAWN_S[team]
    lo, hi = (spawn_s, front_s) if spawn_s < front_s else (front_s, spawn_s)
    return lerp_polyline(LANES[lane], float(np.clip(s, lo, hi)))


def arclength_fraction(lane: str, point: np.ndarray) -> float:
    """Where along a lane a point sits, as an arclength fraction in [0, 1]."""
    pts = LANES[lane]
    samples = np.linspace(0.0, 1.0, 400)
    ref = np.stack([lerp_polyline(pts, s) for s in samples])
    return float(samples[int(np.argmin(np.hypot(*(ref - point).T)))])


def nearest_lane(point: np.ndarray) -> tuple[str, float]:
    """The lane whose centre line passes closest to a point, and that distance."""
    best, best_d = next(iter(LANES)), np.inf
    for lane, pts in LANES.items():
        samples = np.linspace(0.0, 1.0, 240)
        ref = np.stack([lerp_polyline(pts, s) for s in samples])
        d = float(np.hypot(*(ref - point).T).min())
        if d < best_d:
            best_d, best = d, lane
    return best, best_d
