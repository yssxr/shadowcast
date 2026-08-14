"""Tests for pathfinding and geodesic distance fields.

Both are checked against octile closed forms rather than against each other, then
cross-checked: A*'s path cost must equal the Dijkstra field's value at the goal. Two
independent routes to the same number.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.geom.path import (
    DIAG_COST,
    STEP_COST,
    UNREACHABLE,
    astar,
    field_to_units,
    geodesic_field,
    nearest_walkable,
)

G = 64


def _open():
    return np.ones((G, G), dtype=bool)


def _cell(j, i):
    return j * G + i


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------
def test_geodesic_field_matches_octile_distance_exactly():
    field = geodesic_field(_open(), np.array([_cell(10, 10)], dtype=np.int32))
    assert field[_cell(10, 10)] == 0
    assert field[_cell(10, 20)] == 10 * STEP_COST  # pure horizontal
    assert field[_cell(20, 10)] == 10 * STEP_COST  # pure vertical
    assert field[_cell(20, 20)] == 10 * DIAG_COST  # pure diagonal
    assert field[_cell(20, 15)] == 5 * DIAG_COST + 5 * STEP_COST  # mixed


def test_octile_weights_approximate_the_euclidean_metric():
    """99/70 must be sqrt 2 to well under a percent, or geodesic balls are the wrong
    shape and the belief filter's reachability set is systematically skewed."""
    assert abs(DIAG_COST / STEP_COST - np.sqrt(2)) < 1e-4


def test_astar_path_lengths_are_minimal():
    walkable = _open()
    assert len(astar(walkable, _cell(10, 10), _cell(10, 20))) == 11
    assert len(astar(walkable, _cell(10, 10), _cell(20, 20))) == 11
    assert len(astar(walkable, _cell(10, 10), _cell(10, 10))) == 1


def test_astar_cost_equals_the_dijkstra_field():
    """Two independent computations of the same distance."""
    rng = np.random.default_rng(2)
    walkable = _open()
    walkable[20, 5:60] = False  # a wall with a gap
    walkable[20, 32] = True
    start = _cell(5, 5)
    field = geodesic_field(walkable, np.array([start], dtype=np.int32))

    for _ in range(20):
        j, i = int(rng.integers(25, 60)), int(rng.integers(0, G))
        if not walkable[j, i]:
            continue
        goal = _cell(j, i)
        path = astar(walkable, start, goal)
        assert len(path) > 0
        pj, pi = np.divmod(path, G)
        steps = np.maximum(np.abs(np.diff(pj)), np.abs(np.diff(pi)))
        diag = (np.abs(np.diff(pj)) == 1) & (np.abs(np.diff(pi)) == 1)
        cost = int(diag.sum() * DIAG_COST + (len(path) - 1 - diag.sum()) * STEP_COST)
        assert steps.max() <= 1
        assert cost == field[goal], f"A* cost {cost} != field {field[goal]}"


# ---------------------------------------------------------------------------
# Corner cutting
# ---------------------------------------------------------------------------
def test_diagonal_between_two_walls_is_forbidden():
    """A unit must not slip through the join of two wall corners.

    On Summoner's Rift that join is where two jungle walls meet. Allowing the
    shortcut would put routes in the ground truth that the belief filter's
    navmesh-constrained motion can never reproduce, making the filter look wrong
    when it is right.
    """
    walkable = _open()
    walkable[10, 11] = False
    walkable[11, 10] = False
    field = geodesic_field(walkable, np.array([_cell(10, 10)], dtype=np.int32))
    # The direct diagonal is refused, so the cheapest route is three diagonals
    # swinging around the blocked pair: (10,10) -> (11,9) -> (12,10) -> (11,11).
    assert field[_cell(11, 11)] == 3 * DIAG_COST


def test_astar_respects_the_same_rule():
    walkable = _open()
    walkable[10, 11] = False
    walkable[11, 10] = False
    path = astar(walkable, _cell(10, 10), _cell(11, 11))
    assert len(path) == 4  # not 2


def test_diagonal_rule_is_the_permissive_variant():
    """A diagonal needs only ONE of its two orthogonal neighbours open.

    The stricter rule (both open) is arguably closer to League, where a champion's
    ~65-unit collision radius exceeds our 28.8-unit cell and so could not squeeze
    past a single open neighbour. It is not adopted because it would also seal
    legitimately narrow map corridors that champions do use, and the difference only
    affects the synthetic generator's realism — the belief filter's motion model
    reads this same function, so truth and model stay consistent either way.
    """
    walkable = _open()
    walkable[10, 11] = False  # one neighbour blocked, one open
    field = geodesic_field(walkable, np.array([_cell(10, 10)], dtype=np.int32))
    assert field[_cell(11, 11)] == DIAG_COST


# ---------------------------------------------------------------------------
# Unreachable
# ---------------------------------------------------------------------------
def test_unreachable_cells_carry_the_sentinel():
    walkable = _open()
    walkable[30, :] = False  # full-width wall, no gap
    field = geodesic_field(walkable, np.array([_cell(5, 5)], dtype=np.int32))
    assert field[_cell(10, 10)] < UNREACHABLE
    assert field[_cell(40, 40)] == UNREACHABLE
    # The sentinel must compare correctly against a budget without a separate mask.
    assert not (field[_cell(40, 40)] < 10_000_000)


def test_astar_returns_empty_when_no_path_exists():
    walkable = _open()
    walkable[30, :] = False
    assert astar(walkable, _cell(5, 5), _cell(40, 40)).size == 0


def test_astar_rejects_endpoints_in_walls():
    walkable = _open()
    walkable[10, 10] = False
    with pytest.raises(ValueError, match="start"):
        astar(walkable, _cell(10, 10), _cell(20, 20))
    with pytest.raises(ValueError, match="goal"):
        astar(walkable, _cell(20, 20), _cell(10, 10))
    with pytest.raises(ValueError, match="out of range"):
        astar(walkable, -1, _cell(20, 20))


def test_multiple_seeds_give_distance_to_the_nearest():
    walkable = _open()
    seeds = np.array([_cell(5, 5), _cell(50, 50)], dtype=np.int32)
    field = geodesic_field(walkable, seeds)
    assert field[_cell(5, 5)] == 0
    assert field[_cell(50, 50)] == 0
    assert field[_cell(48, 48)] == 2 * DIAG_COST  # nearer the second seed


def test_field_to_units_converts_and_marks_unreachable():
    walkable = _open()
    walkable[30, :] = False
    field = geodesic_field(walkable, np.array([_cell(5, 5)], dtype=np.int32))
    units = field_to_units(field, C.GRID_CELL_SIZE)
    assert units[_cell(5, 15)] == pytest.approx(10 * C.GRID_CELL_SIZE)
    assert np.isnan(units[_cell(40, 40)])


# ---------------------------------------------------------------------------
# Against real terrain
# ---------------------------------------------------------------------------
def test_all_of_summoners_rift_is_mutually_reachable(terrain):
    """Every walkable cell reachable from any one of them.

    A third independent confirmation that the navgrid was parsed correctly: the
    connected-component test used 8-connected flood fill on the navgrid, this uses
    octile Dijkstra on the resampled grid with diagonal corner-cutting forbidden. A
    parse error would leave pockets.
    """
    field = geodesic_field(terrain.walkable, terrain.walkable_cells()[:1])
    reachable = field < UNREACHABLE
    assert int(reachable.sum()) == terrain.n_walkable


def test_map_diameter_matches_the_documented_nexus_distance(terrain):
    """Independent corroboration of the whole coordinate pipeline.

    The LoL Wiki puts nexus obelisk to nexus obelisk at "about 20500" units. The
    longest geodesic path across our reconstructed terrain measures 20,786 — which
    it has no business matching unless the navgrid bounds, the cell size, the
    resample and the octile metric are all right. Nothing else in the suite tests
    those four things jointly against an external number.
    """
    field = geodesic_field(terrain.walkable, terrain.walkable_cells()[:1])
    diameter = np.nanmax(field_to_units(field, C.GRID_CELL_SIZE))
    assert 19_000 < diameter < 22_500, f"map diameter {diameter:,.0f} units looks wrong"


def test_astar_on_real_terrain_produces_a_legal_path(terrain):
    cells = terrain.walkable_cells()
    path = astar(terrain.walkable, int(cells[0]), int(cells[-1]))
    assert path.size > 0
    pj, pi = np.divmod(path, terrain.grid)
    assert terrain.walkable[pj, pi].all()
    assert (np.maximum(np.abs(np.diff(pj)), np.abs(np.diff(pi))) == 1).all()


def test_nearest_walkable_snaps_and_refuses_the_impossible(terrain):
    j, i = np.unravel_index(
        int(np.flatnonzero(~terrain.walkable.ravel())[len(terrain.walkable_cells()) // 3]),
        terrain.walkable.shape,
    )
    sj, si = nearest_walkable(terrain.walkable, int(j), int(i))
    assert terrain.walkable[sj, si]
    assert max(abs(sj - int(j)), abs(si - int(i))) <= 24

    # A landmark that cannot be snapped is a data error, not something to paper over:
    # leaving it put would place a vision source inside terrain.
    walls = np.zeros((G, G), dtype=bool)
    with pytest.raises(ValueError, match="no walkable cell"):
        nearest_walkable(walls, 32, 32, max_radius=4)
