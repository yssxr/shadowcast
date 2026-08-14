"""Summoner's Rift landmarks in world coordinates: lanes, jungle routes, turrets, wards.

**Provenance, and it matters.** These are *schematic* positions, not extracted game
data. They come from the normalised layout in the project's own design mockup,
converted into world coordinates and snapped onto walkable ground. They are accurate
enough to drive a synthetic match — champions walk down lanes, junglers visit camps,
wards sit in plausible spots — and they are not accurate enough to be used as ground
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
