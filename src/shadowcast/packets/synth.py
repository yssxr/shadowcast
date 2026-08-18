"""Synthetic replay generator: a scripted match with known ground truth.

Exists because the two hardest components in the engine cannot be validated against
real data at all. Movement-order attribution has no labels in the corpus, and the
belief filter's correctness is not something a real replay can adjudicate. Both need a
match where the answer is known, and this produces one, on the *real* terrain, so
real geometry (brush entrances, jungle walls, base chokes) is exercised throughout.

**It is adversarial on purpose.** A generator that emitted clean, labelled,
well-ordered packets would validate nothing, because the real stream is none of those
things. Every defect measured in the real corpus is reproduced here and individually
toggleable, so a failing test can be bisected to the specific pathology that broke it:

    anonymous_orders     movement orders carry no entity id (the defining defect)
    quantise_time        timestamps rounded to the ~30 Hz the real stream shows
    drop_speed_replicas  a fraction of movement-speed updates never arrive
    orders_mid_path      a new order supersedes one still in progress
    order_start_jitter   waypoints[0] disagrees slightly with the true position
    reorder_window       packets arrive out of order within a short window
    silent_ward_expiry   one ward vanishes with no WardCorpse
    ghost_caster         a net_id casts spells without ever being created
    corrupt_minion_time  SpawnMinion timestamps are denormal garbage
    keyframe_creates     CreateHero is re-emitted every 60 s as a resync
    duplicate_fog_max    each fog transition repeated up to N times at one timestamp

With every pathology off, downstream reconstruction should recover truth exactly,
that pins the algebra. With all on, it should stay within stated tolerances. That
pins the robustness. Between them the two settings cover what L1 and L2 can get wrong.

**Motion is defined by the orders, not the other way round.** Each champion picks a
goal, paths to it, and the emitted order is a walkable simplification of that path;
the champion then walks the emitted polyline exactly. So integrating the published
orders reproduces the truth by construction, which is the property the trajectory
reconstructor is being asked to invert. Deriving orders from a separately-computed
truth would have left an inconsistency no amount of downstream cleverness could close.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shadowcast import constants as C
from shadowcast import sr
from shadowcast.geom.grid import cell_to_world, world_to_cell
from shadowcast.geom.path import astar, nearest_walkable, simplify_path
from shadowcast.packets.source import (
    BARRACK_SPAWN,
    BASIC_ATTACK,
    CAST_SPELL,
    CREATE_HERO,
    CREATE_NEUTRAL,
    CREATE_TURRET,
    DAMAGE,
    NPC_DIE,
    REPLICATION,
    SPAWN_MINION,
    USE_ITEM,
    WAYPOINT,
    WAYPOINT_XZ,
    MatchMeta,
    PacketBundle,
)
from shadowcast.terrain.terrain import Terrain

__all__ = ["ROSTER", "Pathologies", "ScenarioSpec", "SyntheticSource", "Truth"]

# Hero net_ids mirror the real corpus, where all ten were contiguous from 0x4000001E.
_HERO_NETID_BASE = C.HERO_NETID_HINT_LO
_TURRET_NETID_BASE = 0x40000004
_WARD_NETID_BASE = 0x40001000
_NEUTRAL_NETID_BASE = 0x40002000

# CHOSEN: how close an enemy champion has to be for an attack to name them as its
# target. 700 units sits between a melee champion's ~175 and a marksman's ~600 plus
# projectile travel; the exact figure only decides how often a synthetic fight produces
# a fog-attack reveal, and any value in that band produces the same qualitative result.
_ATTACK_RANGE = 700.0
_MINION_NETID_BASE = 0x40003000
#: Barracks, one per lane per side. Not turrets and not in `CreateTurret`, matching the
#: real stream, where the six barrack net_ids appear in no create packet at all.
_BARRACK_NETID_BASE = 0x40004000
#: Damage exchanges emitted per wave. The labelling needs only a handful; this is enough
#: to clear `_BARRACK_MIN_VOTES` several times over without inflating the damage table.
_BARRACK_DAMAGE_EXCHANGES = 4
#: How close a champion must stand to a wave to be counted as farming it. Wider than
#: `_ATTACK_RANGE` because the evidence is "was in the fight", not "landed an auto", and
#: the front estimator wants the laner rather than only the last-hitter.
_FARM_RANGE = 900.0
_GHOST_NETID = 0x40009999

ROLES = ("top", "jungle", "mid", "bot", "support")

#: (champion, summoner name) per team, in ROLES order. Lowercase internal champion ids
#: like the real `CreateHero.champion` field.
ROSTER: tuple[tuple[str, str], ...] = (
    ("fiora", "anodyne"),
    ("nunu", "kestrel"),
    ("syndra", "lull"),
    ("caitlyn", "faraday"),
    ("sona", "pallor"),
    ("chogath", "umber"),
    ("viego", "sable"),
    ("katarina", "tenet"),
    ("varus", "vervain"),
    ("blitzcrank", "wren"),
)

_LANE_FOR_ROLE = {"top": "top", "mid": "mid", "bot": "bot", "support": "bot"}

#: How far ahead a champion aims when issuing a movement order, in seconds. At ~335
#: units/second this puts the destination roughly 1,300 units away, which is the
#: distance a player actually clicks.
_GOAL_LOOKAHEAD = 4.0


@dataclass(frozen=True, slots=True)
class Pathologies:
    """Real-stream defects, each independently toggleable."""

    quantise_time: bool = True
    drop_speed_replicas: float = 0.02
    orders_mid_path: bool = True
    order_start_jitter: float = 12.0  # world units
    reorder_window: float = 0.1  # seconds
    silent_ward_expiry: bool = True
    ghost_caster: bool = True
    corrupt_minion_time: bool = True
    keyframe_creates: bool = True
    #: Repeat each fog transition this many times at an identical timestamp. The real
    #: stream's single largest defect: LeaveFog is 65-70% of ALL packets and maknee
    #: documents "20+ repeats sometimes". Without this the dedupe path is never
    #: exercised, and dedupe is the first thing any consumer of the corpus must do.
    duplicate_fog_max: int = 12

    @classmethod
    def none(cls) -> Pathologies:
        """A clean stream. Downstream reconstruction must recover truth exactly."""
        return cls(
            quantise_time=False,
            drop_speed_replicas=0.0,
            orders_mid_path=False,
            order_start_jitter=0.0,
            reorder_window=0.0,
            silent_ward_expiry=False,
            ghost_caster=False,
            corrupt_minion_time=False,
            keyframe_creates=False,
            duplicate_fog_max=1,
        )

    @classmethod
    def all(cls) -> Pathologies:
        return cls()


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    seed: int = 7
    duration: float = C.MATCH_WINDOW_SECONDS
    tick_hz: int = C.TICK_HZ
    pathologies: Pathologies = field(default_factory=Pathologies)
    #: Offset from the map-centred waypoint frame to world coordinates. Deliberately
    #: NOT the 7500 that a calibrator would guess first, so the calibration step is
    #: actually tested rather than trivially satisfied.
    #: Per-axis map-centred offset the synthetic stream emits its waypoints in.
    #:
    #: Deliberately NOT the true navgrid midpoint: perturbing it is what gives the frame
    #: calibration something to recover, and a synthetic stream that already sat at the
    #: right answer would let a broken calibration pass. The perturbation is under a cell
    #: on each axis, which is the resolution the fit can actually resolve.
    waypoint_offset_x: float = C.WAYPOINT_OFFSET_X - 6.0
    waypoint_offset_z: float = C.WAYPOINT_OFFSET_Z + 4.0

    @property
    def n_ticks(self) -> int:
        return int(self.duration * self.tick_hz) + 1

    @property
    def dt(self) -> float:
        return 1.0 / self.tick_hz

    def content_hash(self) -> str:
        from shadowcast.config import content_hash

        return content_hash(dataclasses.asdict(self))


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Truth:
    """Everything the generator knows and the packet stream does not say.

    Imported only by tests. A production module reaching for this would be reading the
    answer key, and the information-barrier test in L3 exists to catch exactly that.
    """

    spec: ScenarioSpec
    net_ids: np.ndarray  # u4[10]
    team: np.ndarray  # u1[10]
    role: tuple[str, ...]
    champion: tuple[str, ...]
    pos: np.ndarray  # f8[n_ticks, 10, 2] world coords
    alive: np.ndarray  # u1[n_ticks, 10]
    speed: np.ndarray  # f8[n_ticks, 10]
    brush: np.ndarray  # i2[n_ticks, 10]
    visible: np.ndarray  # u1[n_ticks, 2, 10] from the independent oracle
    order_owner: np.ndarray  # i2[n_orders] which champion issued each order
    order_tick: np.ndarray  # i4[n_orders]
    wards: np.ndarray  # structured, see WARD_TRUTH
    kills: np.ndarray  # structured, see KILL_TRUTH

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            net_ids=self.net_ids,
            team=self.team,
            pos=self.pos,
            alive=self.alive,
            speed=self.speed,
            brush=self.brush,
            visible=self.visible,
            order_owner=self.order_owner,
            order_tick=self.order_tick,
            wards=self.wards,
            kills=self.kills,
            meta=np.frombuffer(
                json.dumps(
                    {
                        "spec": dataclasses.asdict(self.spec),
                        "role": list(self.role),
                        "champion": list(self.champion),
                    }
                ).encode("utf-8"),
                dtype=np.uint8,
            ),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Truth:
        with np.load(path) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            spec_d = dict(meta["spec"])
            spec_d["pathologies"] = Pathologies(**spec_d["pathologies"])
            return cls(
                spec=ScenarioSpec(**spec_d),
                net_ids=z["net_ids"],
                team=z["team"],
                role=tuple(meta["role"]),
                champion=tuple(meta["champion"]),
                pos=z["pos"],
                alive=z["alive"],
                speed=z["speed"],
                brush=z["brush"],
                visible=z["visible"],
                order_owner=z["order_owner"],
                order_tick=z["order_tick"],
                wards=z["wards"],
                kills=z["kills"],
            )


WARD_TRUTH = np.dtype(
    [
        ("net_id", "u4"),
        ("owner_net_id", "u4"),
        ("team", "u1"),
        ("kind", "U10"),
        ("x", "f8"),
        ("z", "f8"),
        ("t0", "f8"),
        ("t1", "f8"),
        ("spot", "U32"),
        ("silent_expiry", "u1"),
    ]
)

KILL_TRUTH = np.dtype([("t", "f8"), ("killer", "i2"), ("victim", "i2"), ("respawn_t", "f8")])


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------
class _Mover:
    """Walks a polyline at a given speed. Position is a pure function of progress."""

    __slots__ = ("poly", "prog", "seg", "speed", "total")

    def __init__(self, start: np.ndarray, speed: float) -> None:
        self.poly = np.array([start, start], dtype=np.float64)
        self.seg = np.zeros(1)
        self.total = 0.0
        self.prog = 0.0
        self.speed = speed

    def set_order(self, poly: np.ndarray) -> None:
        self.poly = np.asarray(poly, dtype=np.float64)
        self.seg = np.hypot(*np.diff(self.poly, axis=0).T)
        self.total = float(self.seg.sum())
        self.prog = 0.0

    @property
    def done(self) -> bool:
        return self.prog >= self.total

    @property
    def pos(self) -> np.ndarray:
        want = min(self.prog, self.total)
        for n, d in enumerate(self.seg):
            if want <= d:
                f = want / d if d > 0 else 0.0
                return self.poly[n] + (self.poly[n + 1] - self.poly[n]) * f
            want -= d
        return self.poly[-1].copy()

    def step(self, dt: float) -> None:
        self.prog = min(self.total, self.prog + self.speed * dt)


def _dense_route(terrain: Terrain, pts: np.ndarray) -> np.ndarray:
    """A*-connect snapped polyline vertices into one dense, navmesh-legal route."""
    snapped, _ = sr.snap_polyline(terrain, pts)
    cells: list[int] = []
    for n in range(len(snapped) - 1):
        i0, j0 = world_to_cell(*snapped[n])
        i1, j1 = world_to_cell(*snapped[n + 1])
        leg = astar(terrain.walkable, j0 * terrain.grid + i0, j1 * terrain.grid + i1)
        if leg.size == 0:
            raise RuntimeError(f"no route between landmark {n} and {n + 1}")
        cells.extend(leg[1:].tolist() if cells else leg.tolist())
    arr = np.array(cells, dtype=np.int64)
    j, i = np.divmod(arr, terrain.grid)
    x, z = cell_to_world(i, j)
    return np.stack([x, z], axis=1)


def _route_lut(route: np.ndarray, n: int = 2048) -> np.ndarray:
    """Resample a polyline to `n` points evenly spaced by arclength.

    The dense A*-built routes have hundreds of vertices, so a Python-level arclength
    lerp inside the per-tick loop would walk all of them every call, hundreds of
    millions of operations over a match. A lookup makes it an array index.
    """
    out = np.empty((n, 2))
    for k in range(n):
        out[k] = sr.lerp_polyline(route, k / (n - 1))
    return out


def _lut_point(lut: np.ndarray, s: float) -> np.ndarray:
    k = int(np.clip(s, 0.0, 1.0) * (lut.shape[0] - 1))
    return lut[k]


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------
class SyntheticSource:
    """A `PacketSource` producing scripted matches on real terrain."""

    def __init__(self, terrain: Terrain, spec: ScenarioSpec | None = None, n_matches: int = 1):
        self.terrain = terrain
        self.spec = spec or ScenarioSpec()
        self.n_matches = n_matches
        self._routes: dict[str, np.ndarray] = {}
        self._truth: dict[str, Truth] = {}

    # -- PacketSource -------------------------------------------------
    def match_ids(self) -> list[str]:
        return [f"synth-{self.spec.seed:04d}-{n:03d}" for n in range(self.n_matches)]

    def read(self, match_id: str) -> PacketBundle:
        bundle, truth = self.generate(match_id)
        self._truth[match_id] = truth
        return bundle

    def truth(self, match_id: str) -> Truth:
        if match_id not in self._truth:
            self.read(match_id)
        return self._truth[match_id]

    # -- routes -------------------------------------------------------
    def routes(self) -> dict[str, np.ndarray]:
        """Dense, navmesh-legal routes, arclength-resampled for O(1) lookup."""
        if not self._routes:
            for lane, pts in sr.LANES.items():
                self._routes[f"lane_{lane}"] = _route_lut(_dense_route(self.terrain, pts))
            for team, pts in sr.JUNGLE_ROUTES.items():
                self._routes[f"jungle_{team}"] = _route_lut(_dense_route(self.terrain, pts))
        return self._routes

    # -- generation ---------------------------------------------------
    def generate(self, match_id: str) -> tuple[PacketBundle, Truth]:
        spec = self.spec
        idx = int(match_id.rsplit("-", 1)[1])
        rng = np.random.default_rng(spec.seed * 1000 + idx)
        terrain = self.terrain
        routes = self.routes()
        n_ticks, dt = spec.n_ticks, spec.dt
        n_champs = 10

        net_ids = np.arange(_HERO_NETID_BASE, _HERO_NETID_BASE + n_champs, dtype=np.uint32)
        team = np.array([0] * 5 + [1] * 5, dtype=np.uint8)
        role = tuple(ROLES) * 2
        champion = tuple(c for c, _ in ROSTER)
        summoner = tuple(s for _, s in ROSTER)

        phase = rng.uniform(0, 2 * np.pi, size=n_champs)
        base_speed = np.full(n_champs, 335.0)

        movers: list[_Mover] = []
        for c in range(n_champs):
            movers.append(_Mover(sr.FOUNTAINS[int(team[c])].copy(), base_speed[c]))

        # Scripted kills. The killer's goal becomes the victim's position for the
        # preceding few seconds, which produces a genuine approach-and-gank rather
        # than two champions teleporting together.
        kill_script = [
            (150.0, 6, 2),
            (300.0, 1, 8),
            (455.0, 6, 0),
            (610.0, 1, 7),
            (760.0, 9, 4),
        ]
        kill_rows = np.empty(len(kill_script), dtype=KILL_TRUTH)
        respawn_lookup: dict[float, float] = {}
        for n, (kt, killer, victim) in enumerate(kill_script):
            respawn_t = kt + 20.0 + 0.5 * (kt / 60.0)
            kill_rows[n] = (kt, killer, victim, respawn_t)
            respawn_lookup[kt] = respawn_t

        # Scripted wards.
        ward_script: list[tuple[float, int, int, str, float]] = []
        for n, _spot in enumerate(sr.WARD_SPOTS):
            owner_team = 0 if n < 6 else 1
            kind = "control" if n % 3 == 2 else "totem"
            place_t = 90.0 + 62.0 * n
            life = 150.0 if kind == "control" else 105.0
            ward_script.append((place_t, owner_team, n, kind, life))

        # --- state ---
        pos = np.zeros((n_ticks, n_champs, 2))
        alive = np.ones((n_ticks, n_champs), dtype=np.uint8)
        speed_t = np.zeros((n_ticks, n_champs))
        respawn_at = np.zeros(n_champs)
        next_order_t = np.zeros(n_champs)

        rows_waypoint: list[tuple[float, int, int, int]] = []
        xz: list[tuple[float, float]] = []
        order_owner: list[int] = []
        order_tick: list[int] = []
        rows_repl: list[tuple] = []
        rows_cast: list[tuple] = []
        rows_attack: list[tuple] = []
        rows_damage: list[tuple] = []

        # Initial speed replication for everyone.
        for c in range(n_champs):
            rows_repl.append((0.0, int(net_ids[c]), 32, 24, "mMoveSpeed", base_speed[c], 1))

        # Lane offsets are in WORLD UNITS, not arclength fractions.
        #
        # Fractions were the first attempt and they are badly wrong: the top and bottom
        # lanes are ~20,200 units long, so a plausible-looking +-0.085 offset puts
        # opposing laners 3,430 units apart when champion sight is 1,350. Three of the
        # ten champions were then never seen by the enemy for an entire match, which
        # made the fog oracle look broken when the scenario was.
        LANE_SEPARATION = 420.0  # each laner sits this far from the lane midpoint
        SUPPORT_OFFSET = 180.0  # supports hover just behind their carry
        # Three superposed oscillations, chosen so implied speed is realistic rather
        # than so the amplitudes look plausible. A single slow swing (900 units over
        # 165 s) moves the goal ~10 units/second, less than half a cell per order
        # interval, champions then stand still, no order is ever long enough to emit,
        # and median implied speed comes out at zero. Peak contribution of a term is
        # amplitude * 2*pi / period, so the short-period terms are what create motion.
        LANE_WAVES = ((900.0, 90.0), (320.0, 23.0), (140.0, 7.0))
        RECALL_PERIOD = 190.0
        RECALL_DURATION = 26.0

        lane_len = {k: float(np.hypot(*np.diff(v, axis=0).T).sum()) for k, v in routes.items()}

        def goal_for(c: int, t: float, current: np.ndarray) -> np.ndarray:
            r = role[c]
            tm = int(team[c])
            # A gank overrides everything: head for the victim. This makes the kill an
            # actual approach the belief filter can be asked about, rather than two
            # champions materialising next to each other.
            for kt, killer, victim in kill_script:
                if c == killer and kt - 9.0 <= t < kt:
                    return current[victim].copy()
            if r == "jungle":
                route = routes[f"jungle_{tm}"]
                cyc = 118.0
                u = ((t + phase[c] * 12.0) % (2 * cyc)) / cyc
                s = u if u <= 1.0 else 2.0 - u
                return _lut_point(route, s)

            key = f"lane_{_LANE_FOR_ROLE[r]}"
            lane = routes[key]
            length = lane_len[key]
            if t < 70.0:
                # Leave the fountain and walk to the lane.
                return _lut_point(lane, 0.10 if tm == 0 else 0.90)
            if r == "support" and ((250.0 < t < 330.0) or (560.0 < t < 640.0)):
                # Roam: head for a river ward spot on the contested half.
                return sr.WARD_SPOTS[5 if tm == 0 else 7][1].copy()

            # Periodic recall. A long walk back to base and out again, which is what
            # exercises trajectory reconstruction over distance rather than over the
            # few hundred units of lane shuffling.
            cycle = (t + phase[c] * 30.0) % RECALL_PERIOD
            if cycle < RECALL_DURATION:
                return sr.FOUNTAINS[tm].copy()

            sign = -1.0 if tm == 0 else 1.0
            offset = sign * LANE_SEPARATION
            for amp, period in LANE_WAVES:
                offset += amp * np.sin(2.0 * np.pi * t / period + phase[c])
            if r == "support":
                offset += sign * SUPPORT_OFFSET
            s = 0.5 + offset / length
            return _lut_point(lane, float(np.clip(s, 0.06, 0.94)))

        path_cache: dict[tuple[int, int], np.ndarray] = {}

        def order_polyline(start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
            i0, j0 = world_to_cell(*start)
            i1, j1 = world_to_cell(*goal)
            j0, i0 = nearest_walkable(terrain.walkable, j0, i0)
            j1, i1 = nearest_walkable(terrain.walkable, j1, i1)
            key = (j0 * terrain.grid + i0, j1 * terrain.grid + i1)
            if key not in path_cache:
                cells = astar(terrain.walkable, key[0], key[1])
                if cells.size == 0:
                    path_cache[key] = np.empty((0, 2))
                else:
                    keep = simplify_path(terrain.walkable, cells, max_points=7)
                    jj, ii = np.divmod(keep, terrain.grid)
                    x, z = cell_to_world(ii, jj)
                    path_cache[key] = np.stack([x, z], axis=1)
            poly = path_cache[key]
            if poly.shape[0] == 0:
                return None
            # The champion's continuous position is PREPENDED, not substituted for the
            # first path vertex. Substituting looked equivalent and was not: the chord
            # from an off-centre point to the *second* vertex is a different segment
            # from the centre-to-centre one the simplifier verified, and it can clip a
            # third cell. Prepending keeps the extra segment inside a single cell,
            # cells are convex, so it is walkable by construction, and leaves every
            # verified chord intact.
            return np.vstack([np.asarray(start, dtype=np.float64)[None, :], poly])

        # Static entities, resolved before the tick loop because turret attacks
        # emitted during it must use the same snapped positions the vision oracle
        # does, otherwise a recovered turret position would not match the source
        # that produced the visibility it is meant to explain.
        rows_turret: list[tuple] = []
        turret_pos: list[tuple[float, float, int]] = []
        for n, (name, tteam, tpos) in enumerate(sr.TURRETS):
            snapped, _ = sr.snap_polyline(terrain, np.asarray(tpos)[None, :])
            nid = _TURRET_NETID_BASE + n
            rows_turret.append((0.0, nid, nid, name))
            turret_pos.append((float(snapped[0, 0]), float(snapped[0, 1]), int(tteam)))

        pth = spec.pathologies
        for tick in range(n_ticks):
            t = tick * dt
            current = np.array([m.pos for m in movers])

            for c in range(n_champs):
                if alive[max(0, tick - 1), c] == 0 and t < respawn_at[c]:
                    alive[tick, c] = 0
                    pos[tick, c] = sr.FOUNTAINS[int(team[c])]
                    speed_t[tick, c] = 0.0
                    continue
                if alive[max(0, tick - 1), c] == 0 and t >= respawn_at[c]:
                    # Respawn is a discontinuity, not a walk: the champion reappears
                    # at the fountain. Trajectory reconstruction must not interpolate
                    # across it.
                    movers[c] = _Mover(sr.FOUNTAINS[int(team[c])].copy(), base_speed[c])
                    next_order_t[c] = t

                # Orders are issued on a timer, not on completion of the previous one.
                # Completion-triggered issuance was the first attempt and it produced
                # ~950 orders per jungler against ~40 per laner: the goal moves a few
                # units per tick, so the path to it is one cell, the order finishes
                # immediately, and another is issued next tick. A real player clicks
                # somewhere a second or two ahead, which is what the lookahead models.
                needs = t >= next_order_t[c]
                if pth.orders_mid_path:
                    needs = needs or (rng.random() < 0.01)
                if needs:
                    goal = goal_for(c, t + _GOAL_LOOKAHEAD, current)
                    poly = order_polyline(movers[c].pos, goal)
                    if poly is not None and poly.shape[0] >= 2:
                        # The champion walks the UNJITTERED path; only the published
                        # order is perturbed. Jittering the walked path too would make
                        # truth agree with the corrupted waypoint, which is the
                        # opposite of the pathology being modelled. The point is that
                        # waypoints[0] disagrees with where the unit actually is, and
                        # that disagreement is what the order residual measures.
                        movers[c].set_order(poly)
                        emitted = poly
                        if pth.order_start_jitter > 0:
                            emitted = poly.copy()
                            emitted[0] = emitted[0] + rng.normal(
                                0.0, pth.order_start_jitter / 2.0, size=2
                            )
                        off = len(xz)
                        for p in emitted:
                            xz.append((float(p[0]), float(p[1])))
                        rows_waypoint.append(
                            (t, off, len(emitted), 1 if rng.random() < 0.02 else 0)
                        )
                        order_owner.append(c)
                        order_tick.append(tick)
                    next_order_t[c] = t + float(rng.uniform(0.6, 2.5))

                # Recorded BEFORE stepping, so `pos[tick]` is the position at time
                # `tick * dt` rather than at `(tick + 1) * dt`. Stepping first put
                # every truth sample one tick ahead of its own timestamp. An error
                # that would have propagated silently into the fog oracle and every
                # metric derived from it.
                pos[tick, c] = movers[c].pos
                speed_t[tick, c] = movers[c].speed
                movers[c].step(dt)

            # Boots at eight minutes: a speed change mid-game that the reconstructor
            # must pick up from replication rather than assume.
            if abs(t - 480.0) < dt / 2:
                for c in range(n_champs):
                    movers[c].speed = 380.0
                    if rng.random() >= pth.drop_speed_replicas:
                        rows_repl.append((t, int(net_ids[c]), 32, 24, "mMoveSpeed", 380.0, 1))

            # Labelled position anchors. These are the only packets that tie a
            # position to a net_id, so they are what makes anonymous movement orders
            # attributable at all, roughly one per champion per 1.5 s, matching the
            # 546-1,085 per champion per match measured in the real corpus.
            for c in range(n_champs):
                if alive[tick, c] == 0:
                    continue
                if rng.random() < dt / 1.5:
                    x, z = pos[tick, c]
                    # The TARGET matters, and a first version left it at zero for every
                    # attack. The fog-attack reveal is conditioned on it. The rule is
                    # "attacking an enemy (including wards) from their team's fog of
                    # war", so an attack with no target reveals nobody, and one on an
                    # enemy reveals the attacker.
                    #
                    # With every target zero, the oracle and the reconstruction agreed
                    # (both revealed on every attack) and both were wrong: champions
                    # revealed themselves roughly once a second wherever they stood,
                    # including in their own fountain at 0:00. Both teams lit each
                    # other's spawn before anyone had moved, and match-wide visibility
                    # came out at 84% against a real 25-40%.
                    #
                    # So an attack names an enemy champion only when one is actually
                    # within reach. Everything else: farming a wave, clearing a camp,
                    # is a target this model does not track, and leaves at zero.
                    target = 0
                    for other in range(n_champs):
                        if team[other] == team[c] or alive[tick, other] == 0:
                            continue
                        ox, oz = pos[tick, other]
                        if (ox - x) ** 2 + (oz - z) ** 2 <= _ATTACK_RANGE**2:
                            target = int(net_ids[other])
                            break
                    tx, tz = (x + 200.0, z + 200.0) if target == 0 else pos[tick, other]
                    rows_attack.append((t, int(net_ids[c]), target, x, z, tx, tz))
                if rng.random() < dt / 3.0:
                    x, z = pos[tick, c]
                    rows_cast.append(
                        (t, int(net_ids[c]), f"{champion[c]}Q", 0x1234 + c, x, z, x, z, 0)
                    )

            # Turrets shoot too, and that is the only way their positions can be
            # recovered: `CreateTurret` carries a name but no coordinates, while
            # `BasicAttackPos` pairs `source_net_id` with `source_position`. Since a
            # turret never moves, the mode of its attack positions IS its location,
            # which makes turret team (from the name) and turret position (from the
            # attacks) jointly recoverable, and turrets are the anchor for resolving
            # champion teams.
            if tick % 8 == 0:
                for n, (tx, tz, _tteam) in enumerate(turret_pos):
                    if rng.random() < 0.35:
                        rows_attack.append(
                            (
                                t,
                                _TURRET_NETID_BASE + n,
                                _NEUTRAL_NETID_BASE,
                                tx,
                                tz,
                                tx + 300.0,
                                tz,
                            )
                        )

            # Kills: damage in the second before, then health to zero.
            for kt, killer, victim in kill_script:
                if abs(t - kt) < dt / 2:
                    # Several damagers, not one. A real kill is contested, and the
                    # killer inference reports its confidence as the last damager's
                    # share of the window. A figure that stays pinned at 1.0 and
                    # therefore untested if only one champion ever deals damage.
                    assist = (killer + 1) % n_champs
                    if team[assist] == team[victim]:
                        assist = (killer + 2) % n_champs
                    for n in range(6):
                        source = killer if n % 2 == 0 else assist
                        rows_damage.append(
                            (
                                kt - 1.0 + n * 0.15,
                                int(net_ids[source]),
                                int(net_ids[victim]),
                                180.0 if source == killer else 90.0,
                            )
                        )
                    # The killing blow is the killer's, immediately before death.
                    rows_damage.append(
                        (kt - 0.02, int(net_ids[killer]), int(net_ids[victim]), 220.0)
                    )
                    rows_repl.append((kt, int(net_ids[victim]), 32, 0, "mHP", 0.0, 1))
                    alive[tick, victim] = 0
                    respawn_at[victim] = respawn_lookup[kt]

        # --- wards ---
        ward_rows: list[tuple] = []
        rows_minion: list[tuple] = []
        rows_barrack: list[tuple] = []
        rows_item: list[tuple] = []
        for n, (place_t, owner_team, spot_idx, kind, life) in enumerate(ward_script):
            if place_t >= spec.duration:
                continue  # a short scenario simply gets fewer wards
            spot_name, spot_pos = sr.WARD_SPOTS[spot_idx]
            snapped, _ = sr.snap_polyline(terrain, spot_pos[None, :])
            wx, wz = snapped[0]
            # The nearest living champion of the owning team is credited. Falling back
            # to the whole team matters: with everyone dead there is no living placer,
            # and dropping the ward silently would make ward counts depend on the
            # kill script in a way no test would notice.
            tick = min(n_ticks - 1, round(place_t / dt))
            cand = [c for c in range(n_champs) if team[c] == owner_team and alive[tick, c]]
            if not cand:
                cand = [c for c in range(n_champs) if team[c] == owner_team]
            owner = min(cand, key=lambda c: float(np.hypot(*(pos[tick, c] - (wx, wz)))))
            net_id = _WARD_NETID_BASE + n
            silent = bool(pth.silent_ward_expiry and n == 3)
            ward_rows.append(
                (
                    net_id,
                    int(net_ids[owner]),
                    owner_team,
                    kind,
                    wx,
                    wz,
                    place_t,
                    place_t + life,
                    spot_name,
                    1 if silent else 0,
                )
            )
            name, skin = (
                ("VisionWard", "SightWard") if kind == "control" else ("SightWard", "YellowTrinket")
            )
            rows_minion.append((place_t, net_id, wx, wz, name, skin, int(net_ids[owner]), 1))
            rows_item.append((place_t, int(net_ids[owner]), 7 if kind == "control" else 6))
            if not silent:
                rows_minion.append(
                    (
                        place_t + life,
                        _WARD_NETID_BASE + 500 + n,
                        wx,
                        wz,
                        "WardCorpse",
                        "S5Test_WardCorpse",
                        int(net_ids[owner]),
                        1,
                    )
                )
        wards = np.array(ward_rows, dtype=WARD_TRUTH)

        rows_neutral: list[tuple] = []
        rows_death: list[tuple] = []

        for n, (_label, p) in enumerate(sr.WARD_SPOTS[:6]):
            snapped, _ = sr.snap_polyline(terrain, p[None, :])
            rows_neutral.append(
                (
                    0.0,
                    _NEUTRAL_NETID_BASE + n,
                    float(snapped[0, 0]),
                    float(snapped[0, 1]),
                    f"SRU_Camp{n}.1.1",
                    n + 1,
                    67,
                )
            )

        # Lane minion waves. Emitted through `BarrackSpawnUnit`, which is where the real
        # stream puts them and, verified on a real match. The ONLY place it puts them:
        # not one lane minion appears in `SpawnMinion`, whose contents are wards, plants,
        # camps and ability summons. A generator that spawned them with a position and a
        # readable name would be handing downstream code two facts the corpus withholds,
        # and the missing-minion hole would only surface on real data.
        #
        # So the barrack carries no lane, no team and no coordinates, and the only route
        # to any of them is the damage it exchanges with the enemy turret defending that
        # lane. Those exchanges are emitted here for the same reason: without them the
        # labelling has nothing to read, on real data or synthetic.
        barracks: dict[tuple[str, int], int] = {}
        barrack_turret: dict[tuple[str, int], int] = {}
        for n, lane in enumerate(sr.LANES):
            for wave_team in (C.TEAM_ORDER, C.TEAM_CHAOS):
                key = (lane, wave_team)
                barracks[key] = _BARRACK_NETID_BASE + 2 * n + wave_team
                front = sr.lerp_polyline(sr.LANES[lane], sr.MEETING_S)
                best, best_d = 0, float("inf")
                for k, (_, tteam, tpos) in enumerate(sr.TURRETS):
                    if tteam != 1 - wave_team:
                        continue
                    d = float(np.hypot(tpos[0] - front[0], tpos[1] - front[1]))
                    if d < best_d:
                        best, best_d = _TURRET_NETID_BASE + k, d
                barrack_turret[key] = best

        wave_specs: list[tuple[float, str, int, int]] = []
        for n, (wave_t, lane, wave_team) in enumerate(sr.minion_wave_schedule(spec.duration)):
            net_id = _MINION_NETID_BASE + n
            key = (lane, wave_team)
            rows_barrack.append((wave_t, net_id, barracks[key], n, 0))
            death_t = wave_t + sr.MINION_CLUMP_LIFETIME
            if death_t < spec.duration:
                rows_death.append((death_t, 0, net_id, 0))
                # The wave trades with the turret it is walking into. Both directions,
                # because the labelling reads either.
                turret = barrack_turret[key]
                for k in range(_BARRACK_DAMAGE_EXCHANGES):
                    when = wave_t + (death_t - wave_t) * (k + 1) / (_BARRACK_DAMAGE_EXCHANGES + 1)
                    rows_damage.append((when, net_id, turret, 20.0))
                    rows_damage.append((when, turret, net_id, 60.0))

                    # Champions farming the wave. This is the *only* evidence for where a
                    # lane's front line actually sits. The reconstruction has no minion
                    # positions, so it infers the front from where champions stand when
                    # they hit minions. Emitting it means the front estimator is exercised
                    # against a generator whose waves really do meet at `MEETING_S`, so a
                    # broken estimator shows up as a fog disagreement rather than passing
                    # unnoticed until real data.
                    clump = sr.minion_clump_position(lane, wave_team, wave_t, when)
                    if clump is None:
                        continue
                    tick_at = min(n_ticks - 1, round(when / dt))
                    for c in range(n_champs):
                        if alive[tick_at, c] == 0 or team[c] == wave_team:
                            continue
                        if float(np.hypot(*(pos[tick_at, c] - clump))) <= _FARM_RANGE:
                            rows_damage.append((when, int(net_ids[c]), net_id, 40.0))
            wave_specs.append((wave_t, lane, wave_team, net_id))

        # Camp clears. Death packets exist in the real stream but NEVER name a
        # champion as the victim, verified across 45,851 real rows, so the generator
        # emits them only for neutrals. Anything downstream that hoped to read
        # champion deaths from here will find nothing, which is the point.
        for n in range(6):
            killer_team = n % 2
            jungler = 1 if killer_team == 0 else 6
            clear_t = 105.0 + 92.0 * n
            if clear_t < spec.duration:
                rows_death.append((clear_t, int(net_ids[jungler]), _NEUTRAL_NETID_BASE + n, 0))

        # --- brush membership and the fog oracle ---
        brush = np.full((n_ticks, n_champs), -1, dtype=np.int16)
        for tick in range(n_ticks):
            for c in range(n_champs):
                i, j = world_to_cell(*pos[tick, c])
                if 0 <= i < terrain.grid and 0 <= j < terrain.grid:
                    brush[tick, c] = terrain.brush_id[j, i]

        # Reveal-on-attack needs two passes. Base visibility decides which attacks came
        # from fog; those attacks then become temporary 400-unit vision sources for the
        # opposing team and visibility is recomputed. Two passes rather than one because
        # the gate depends on the answer, and a reveal must never be able to trigger
        # another reveal.
        base = self._run_oracle(
            pos, brush, alive, team, wards, turret_pos, wave_specs, spec, reveals=[]
        )
        net_set = set(net_ids.tolist())
        slot_of = {int(n): k for k, n in enumerate(net_ids)}
        reveals: list[tuple[float, float, int, float, float]] = []
        for row in rows_attack:
            at_t, attacker, target = float(row[0]), int(row[1]), int(row[2])
            if attacker not in net_set:
                continue  # a turret, which is never in fog
            slot = slot_of[attacker]
            # The rule needs an ENEMY target. The oracle applies it exactly as the
            # reconstruction does, so a disagreement between them is a reconstruction
            # error rather than a difference of opinion about the rule.
            if target == 0 or team[slot_of.get(target, slot)] == team[slot]:
                continue
            obs = 1 - int(team[slot])
            tick = min(n_ticks - 1, max(0, round(at_t / dt)))
            if base[tick, obs, slot]:
                continue  # not attacking from fog
            reveals.append(
                (at_t, at_t + C.FOG_ATTACK_REVEAL_DURATION, obs, float(row[3]), float(row[4]))
            )
        visible = self._run_oracle(
            pos, brush, alive, team, wards, turret_pos, wave_specs, spec, reveals=reveals
        )

        fog_rows = _fog_rows(visible, team, net_ids, dt)
        if pth.duplicate_fog_max > 1 and fog_rows.size:
            reps = rng.integers(1, pth.duplicate_fog_max + 1, size=fog_rows.size)
            fog_rows = np.repeat(fog_rows, reps)

        # --- assemble ---
        heroes: list[tuple] = []
        keyframes = [0.0]
        if pth.keyframe_creates:
            keyframes = list(np.arange(0.0, spec.duration, 60.0))
        for kf in keyframes:
            for c in range(n_champs):
                heroes.append((kf, int(net_ids[c]), summoner[c], champion[c]))

        if pth.ghost_caster:
            # A pet or clone: casts spells, never created. The resolver must ignore it
            # rather than crash or invent an eleventh champion.
            rows_cast.append((45.0, _GHOST_NETID, "ghost", 0x9999, 5000.0, 5000.0, 0.0, 0.0, 0))

        def mk(rows, dtype):
            """Build a packet array from row tuples, leaving `seq` for later.

            Row tuples omit `seq` because stream position is only knowable once every
            kind has been built and interleaved.
            """
            fields = [f for f in dtype.names if f != "seq"]
            out = np.empty(len(rows), dtype=dtype)
            for n, r in enumerate(rows):
                for f, v in zip(fields, r):
                    out[n][f] = v
            out["seq"] = -1
            return out

        wp = mk(rows_waypoint, WAYPOINT)
        wp_xz = np.empty(len(xz), dtype=WAYPOINT_XZ)
        for n, (x, z) in enumerate(xz):
            wp_xz[n] = (x - spec.waypoint_offset_x, z - spec.waypoint_offset_z)

        minions = mk(rows_minion, SPAWN_MINION)
        minions.sort(order="t", kind="stable")
        true_minion_times = minions["t"].copy()
        if pth.corrupt_minion_time and minions.size:
            # Real SpawnMinion.time is denormal-float noise. Consumers must take
            # timing from the surrounding stream clock instead.
            minions["t"] = np.linspace(1e-40, 5e-39, minions.size)
            minions["t_valid"] = 0

        bundle_arrays = {
            "heroes": mk(heroes, CREATE_HERO),
            "waypoints": wp,
            "waypoint_xz": wp_xz,
            "fog": fog_rows,
            "replication": mk(rows_repl, REPLICATION),
            "turrets": mk(rows_turret, CREATE_TURRET),
            "minions": minions,
            "neutrals": mk(rows_neutral, CREATE_NEUTRAL),
            "casts": mk(rows_cast, CAST_SPELL),
            "attacks": mk(rows_attack, BASIC_ATTACK),
            "damage": mk(rows_damage, DAMAGE),
            "deaths": mk(rows_death, NPC_DIE),
            "items": mk(rows_item, USE_ITEM),
            "barracks": mk(rows_barrack, BARRACK_SPAWN),
        }

        order_owner_arr = np.array(order_owner, dtype=np.int16)
        order_tick_arr = np.array(order_tick, dtype=np.int32)

        for name in list(bundle_arrays):
            arr = bundle_arrays[name]
            if name in ("waypoint_xz", "minions") or arr.size == 0:
                continue
            if pth.quantise_time:
                arr["t"] = np.round(arr["t"] * 30.0) / 30.0
            perm = _arrival_permutation(arr["t"], pth.reorder_window, rng)
            bundle_arrays[name] = arr[perm]
            if name == "waypoints":
                # Truth's per-order attribution must follow the rows it describes.
                # It did not, and the reorder pathology exposed it: shuffling
                # `waypoints` left `order_owner[n]` pointing at a different order, so
                # a test comparing them saw map-scale disagreement and looked like a
                # generator bug rather than a bookkeeping one.
                order_owner_arr = order_owner_arr[perm]
                order_tick_arr = order_tick_arr[perm]

        # Interleave every kind into one stream and stamp each row with its position.
        #
        # `minions` is merged on its TRUE times even though the published `t` may be
        # denormal garbage, because that is exactly the situation on real data: the
        # timestamps are unusable but the stream position is not, and a ward's real
        # placement time is recoverable from the packets around it. Losing that would
        # make ward lifetimes. The project's headline metric, unrecoverable.
        merge_t: list[np.ndarray] = []
        merge_kind: list[np.ndarray] = []
        merge_idx: list[np.ndarray] = []
        for kind_id, name in enumerate(bundle_arrays):
            if name == "waypoint_xz":
                continue
            arr = bundle_arrays[name]
            if arr.size == 0:
                continue
            times = true_minion_times if name == "minions" else arr["t"]
            merge_t.append(np.asarray(times, dtype=np.float64))
            merge_kind.append(np.full(arr.size, kind_id, dtype=np.int64))
            merge_idx.append(np.arange(arr.size, dtype=np.int64))
        if merge_t:
            all_t = np.concatenate(merge_t)
            all_kind = np.concatenate(merge_kind)
            all_idx = np.concatenate(merge_idx)
            order = np.lexsort((all_idx, all_kind, all_t))
            for pos_in_stream, k in enumerate(order):
                name = list(bundle_arrays)[all_kind[k]]
                bundle_arrays[name]["seq"][all_idx[k]] = pos_in_stream

        meta = MatchMeta(
            match_id=match_id,
            source=f"synthetic/{spec.content_hash()}",
            duration=spec.duration,
            n_packets=sum(a.size for k, a in bundle_arrays.items() if k != "waypoint_xz"),
            patch="synthetic",
            extra={
                "seed": spec.seed,
                "waypoint_offset_x": spec.waypoint_offset_x,
                "waypoint_offset_z": spec.waypoint_offset_z,
            },
        )
        bundle = PacketBundle(meta=meta, **bundle_arrays)
        truth = Truth(
            spec=spec,
            net_ids=net_ids,
            team=team,
            role=role,
            champion=champion,
            pos=pos,
            alive=alive,
            speed=speed_t,
            brush=brush,
            visible=visible,
            order_owner=order_owner_arr,
            order_tick=order_tick_arr,
            wards=wards,
            kills=kill_rows,
        )
        return bundle, truth

    # -- oracle -------------------------------------------------------
    def _run_oracle(self, pos, brush, alive, team, wards, turret_pos, wave_specs, spec, reveals=()):
        """Build the per-tick vision-source lists and run the independent oracle."""
        from shadowcast.packets.synth_fog import compute_visibility

        terrain = self.terrain
        cs = terrain.grid_spec.cell_size
        n_ticks, n_champs, _ = pos.shape

        def to_cells(x, z):
            return (x - C.WORLD_MIN_X) / cs, (z - C.WORLD_MIN_Z) / cs

        def brush_at(x, z):
            i, j = world_to_cell(x, z)
            if 0 <= i < terrain.grid and 0 <= j < terrain.grid:
                return int(terrain.brush_id[j, i])
            return -1

        src_off = np.zeros(n_ticks * 2, dtype=np.int64)
        src_n = np.zeros(n_ticks * 2, dtype=np.int32)
        sx: list[float] = []
        sz: list[float] = []
        srad: list[float] = []
        sb: list[int] = []

        # Static per-team sources, computed once.
        static: dict[int, list[tuple[float, float, float, int]]] = {0: [], 1: []}
        for x, z, tteam in turret_pos:
            cx, cz = to_cells(x, z)
            static[tteam].append((cx, cz, C.SIGHT_TURRET / cs, brush_at(x, z)))

        dt = spec.dt
        for tick in range(n_ticks):
            t = tick * dt
            for obs in (0, 1):
                slot = tick * 2 + obs
                src_off[slot] = len(sx)
                start = len(sx)

                for cx, cz, r, b in static[obs]:
                    sx.append(cx)
                    sz.append(cz)
                    srad.append(r)
                    sb.append(b)

                for c in range(n_champs):
                    if team[c] != obs or alive[tick, c] == 0:
                        continue
                    cx, cz = to_cells(*pos[tick, c])
                    sx.append(cx)
                    sz.append(cz)
                    srad.append(C.SIGHT_CHAMPION / cs)
                    sb.append(int(brush[tick, c]))

                for w in wards:
                    if int(w["team"]) != obs or not (w["t0"] <= t <= w["t1"]):
                        continue
                    cx, cz = to_cells(float(w["x"]), float(w["z"]))
                    sx.append(cx)
                    sz.append(cz)
                    srad.append(C.WARD_SIGHT_BY_KIND[str(w["kind"])] / cs)
                    sb.append(brush_at(float(w["x"]), float(w["z"])))

                # Minion waves, from the same shared model the reconstruction uses.
                # Sharing it is deliberate: minion vision then becomes a constant on both
                # sides, so the fog-agreement figure measures champion trajectories, ward
                # lifetimes and field-of-view geometry rather than minion modelling.
                for r_t0, r_t1, r_team, r_x, r_z in reveals:
                    if r_team != obs or not (r_t0 <= t <= r_t1):
                        continue
                    cx, cz = to_cells(r_x, r_z)
                    sx.append(cx)
                    sz.append(cz)
                    srad.append(C.FOG_ATTACK_REVEAL_RADIUS / cs)
                    sb.append(brush_at(r_x, r_z))

                for wave_t, lane, wave_team, _net_id in wave_specs:
                    if wave_team != obs:
                        continue
                    p = sr.minion_clump_position(lane, wave_team, wave_t, t)
                    if p is None:
                        continue
                    cx, cz = to_cells(float(p[0]), float(p[1]))
                    sx.append(cx)
                    sz.append(cz)
                    srad.append(C.SIGHT_MINION / cs)
                    sb.append(brush_at(float(p[0]), float(p[1])))

                src_n[slot] = len(sx) - start

        champ_cx = np.empty((n_ticks, n_champs))
        champ_cz = np.empty((n_ticks, n_champs))
        for tick in range(n_ticks):
            for c in range(n_champs):
                champ_cx[tick, c], champ_cz[tick, c] = to_cells(*pos[tick, c])

        return compute_visibility(
            terrain.blocks_vision,
            terrain.brush_id,
            champ_cx,
            champ_cz,
            brush,
            alive,
            team,
            src_off,
            src_n,
            np.asarray(sx, dtype=np.float64),
            np.asarray(sz, dtype=np.float64),
            np.asarray(srad, dtype=np.float64),
            np.asarray(sb, dtype=np.int16),
        )


def _fog_rows(visible, team, net_ids, dt) -> np.ndarray:
    from shadowcast.packets.synth_fog import transitions_from_visibility

    return transitions_from_visibility(visible, team, net_ids, dt)


def _arrival_permutation(t: np.ndarray, window: float, rng) -> np.ndarray:
    """Index permutation putting rows in arrival order: sorted, then locally shuffled.

    Returns a permutation rather than mutating in place so callers can apply the same
    reordering to parallel bookkeeping. Real packets do not arrive perfectly ordered,
    and anything that assumes monotone arrival. A running clock, a state accumulator,
    has to tolerate it, so the generator produces it rather than leaving the
    assumption untested.
    """
    perm = np.argsort(t, kind="stable")
    if window <= 0 or perm.size < 2:
        return perm
    ts = t[perm]
    start = 0
    while start < perm.size:
        end = int(np.searchsorted(ts, ts[start] + window, side="right"))
        if end - start > 1:
            block = perm[start:end].copy()
            rng.shuffle(block)
            perm[start:end] = block
        start = max(end, start + 1)
    return perm
