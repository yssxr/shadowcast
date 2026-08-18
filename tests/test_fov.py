"""Tests for recursive shadowcasting.

The centrepiece is `test_radius_monotonicity_is_exact`. That property is what lets
one precomputed table serve every sight radius in the game, turning a naively
8.6 TB all-pairs problem into 286 MB. It is a permanent test rather than a one-off
verification because the two changes that break it. A wall-lighting post-pass and
flood-revealing the source's brush, are both things a well-meaning contributor
would plausibly add.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.config import GridSpec, StageHeader, TerrainSpec
from shadowcast.fov.reference import boundary_band, fov_reference
from shadowcast.fov.shadowcast import fov_bool, fov_into, new_scratch
from shadowcast.geom.grid import disc_mask, radius_cells_sq
from shadowcast.terrain.terrain import NO_BRUSH, Terrain

HALF = 20
WINDOW = 2 * HALF + 1


def make_terrain(grid: int = 96, walls=None, brush=None) -> Terrain:
    """A synthetic Terrain for fixtures with analytically known answers."""
    blocks_vision = np.zeros((grid, grid), dtype=bool)
    brush_id = np.full((grid, grid), NO_BRUSH, dtype=np.int16)
    if walls is not None:
        blocks_vision |= walls
    n_groups = 0
    if brush is not None:
        for gid, mask in enumerate(brush):
            brush_id[mask] = gid
            n_groups += 1
    return Terrain(
        grid=grid,
        walkable=~blocks_vision,
        blocks_vision=blocks_vision,
        brush_id=brush_id,
        n_brush_groups=n_groups,
        header=StageHeader(stage="test", stage_version=1, config_hash="t", input_hash="t"),
        spec=TerrainSpec(navgrid_hash="synthetic"),
        grid_spec=GridSpec(grid=grid),
    )


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("radius", [100.0, 200.0, 500.0, 900.0])
def test_empty_grid_gives_exactly_the_integer_disc(radius):
    """With nothing to occlude, field of view IS the radius disc. No tolerance.

    This also catches the octant-transform class of bug: indexing the output window
    with scan-space offsets instead of grid offsets collapses all eight octants onto
    one, and the resulting mask is a small fraction of the disc.
    """
    terrain = make_terrain()
    got = fov_bool(terrain, 48, 48, radius, half=HALF)
    want = disc_mask(radius, window=WINDOW)
    np.testing.assert_array_equal(got, want)


def test_a_full_length_wall_casts_an_analytic_half_plane_shadow():
    """A wall spanning the window blocks everything beyond it, exactly.

    Wall cells themselves stay visible. A wall face is visible from outside, and
    lighting occluders inside the scan (rather than in a post-pass) is what keeps
    radius separability intact.
    """
    grid = 96
    walls = np.zeros((grid, grid), dtype=bool)
    walls[:, 51] = True  # vertical wall 3 cells right of the source
    terrain = make_terrain(grid, walls=walls)

    got = fov_bool(terrain, 48, 48, 900.0, half=HALF)
    disc = disc_mask(900.0, window=WINDOW)

    di = np.arange(-HALF, HALF + 1)[None, :].repeat(WINDOW, axis=0)
    near_side = disc & (di <= 3)
    far_side = disc & (di > 3)

    np.testing.assert_array_equal(got & near_side, near_side)
    assert not (got & far_side).any(), "vision leaked past a full-length wall"


def test_off_grid_is_opaque():
    """The map border must block, or vision wraps around the edge of the world."""
    terrain = make_terrain(grid=96)
    got = fov_bool(terrain, 2, 2, 900.0, half=HALF)
    # Nothing outside the grid can be visible, so the window's far corner, which
    # maps to negative grid coordinates, must be dark.
    assert not got[: HALF - 2, : HALF - 2].any()


# ---------------------------------------------------------------------------
# Brush semantics. The four cases
# ---------------------------------------------------------------------------
def _brush_fixture():
    """Two separate brush patches, both within sight of the origin cell.

    grid layout around the source at (48, 48):
        brush A: columns 50-53, rows 44-52   (to the right)
        brush B: columns 56-59, rows 44-52   (further right, separate)
    """
    grid = 96
    a = np.zeros((grid, grid), dtype=bool)
    a[44:53, 50:54] = True
    b = np.zeros((grid, grid), dtype=bool)
    b[44:53, 56:60] = True
    return make_terrain(grid, brush=[a, b]), a, b


def test_brush_is_opaque_from_outside():
    """Case 1: standing outside a brush, you cannot see into it.

    The single most consequential rule in the module. Marking brush visible from
    outside would mean an enemy hiding in it counted as seen, inverting the central
    mechanic of jungle and river play and quietly inflating every fog-agreement
    number.
    """
    terrain, a, _ = _brush_fixture()
    got = fov_bool(terrain, 48, 48, 900.0, half=HALF)
    aj, ai = np.nonzero(a)
    assert not got[aj - 48 + HALF, ai - 48 + HALF].any()


def test_own_brush_is_visible_from_inside():
    """Case 2: standing inside a brush, you see it."""
    terrain, a, _ = _brush_fixture()
    got = fov_bool(terrain, 51, 48, 900.0, half=HALF)  # inside brush A
    aj, ai = np.nonzero(a)
    inside = got[aj - 48 + HALF, ai - 51 + HALF]
    assert inside.mean() > 0.9, "should see essentially all of the brush standing in it"


def test_a_different_brush_is_still_opaque_from_inside_one():
    """Case 3: brush A does not reveal brush B."""
    terrain, _, b = _brush_fixture()
    got = fov_bool(terrain, 51, 48, 1350.0, half=HALF)  # inside brush A
    bj, bi = np.nonzero(b)
    assert not got[bj - 48 + HALF, bi - 51 + HALF].any()


def test_you_can_see_out_of_your_own_brush():
    """Case 4: brush is transparent outward. No vision penalty for standing in it.

    Checked in the direction away from the other brush, so brush B's opacity cannot
    account for the result.
    """
    terrain, _, _ = _brush_fixture()
    inside = fov_bool(terrain, 51, 48, 900.0, half=HALF)
    open_ground = make_terrain(96)
    outside = fov_bool(open_ground, 51, 48, 900.0, half=HALF)
    left = np.s_[:, : HALF - 4]  # away from both brushes
    np.testing.assert_array_equal(inside[left], outside[left])


def test_source_brush_can_be_overridden():
    """Brush membership comes from the continuous position, not the snapped cell.

    At 28.8 units per cell a champion 10 units inside a brush can snap to a cell
    classified as non-brush. Brush transparency is a discrete switch, so that error
    is not small. It flips a large part of the field of view. The caller must
    therefore be able to pass the brush id it determined from the true position.
    """
    terrain, a, _ = _brush_fixture()
    at_edge = (49, 48)  # just outside brush A
    as_outside = fov_bool(terrain, *at_edge, 900.0, half=HALF, src_brush=NO_BRUSH)
    as_inside = fov_bool(terrain, *at_edge, 900.0, half=HALF, src_brush=0)
    assert as_inside.sum() > as_outside.sum()
    aj, ai = np.nonzero(a)
    assert not as_outside[aj - 48 + HALF, ai - 49 + HALF].any()
    assert as_inside[aj - 48 + HALF, ai - 49 + HALF].any()


# ---------------------------------------------------------------------------
# The load-bearing property
# ---------------------------------------------------------------------------
GAME_RADII = [
    C.FOG_ATTACK_REVEAL_RADIUS,
    C.SIGHT_WARD_FARSIGHT,
    C.SIGHT_WARD_TOTEM,
    C.SIGHT_MINION,
    C.SIGHT_CHAMPION,
]


def test_radius_monotonicity_is_exact(terrain):
    """`fov(r) == fov(RMAX) & disc(r)`, exactly, on real Summoner's Rift terrain.

    This is the property the entire table design rests on. It holds because
    shadowcasting decides a cell from shadow intervals cast by strictly *nearer*
    occluders, so an occluder outside `disc(r)` cannot influence anything inside it.

    If this fails, someone has added one of the following, and the table must be
    abandoned or the change reverted:
      - a wall-lighting post-pass (breaks ~68% of cases)
      - flood-revealing the source's whole brush (breaks ~1.2%)
      - fractional or anti-aliased visibility
      - a permissive-FOV variant that widens shadows with range
    """
    rng = np.random.default_rng(20240814)
    cells = terrain.walkable_cells()
    picks = rng.choice(cells, size=400, replace=False)

    half = C.RMAX_CELLS
    window = 2 * half + 1
    at_max = np.zeros((window, window), dtype=bool)
    at_r = np.zeros((window, window), dtype=bool)
    scratch = new_scratch()
    discs = {r: disc_mask(r, window=window) for r in GAME_RADII}

    mismatches = 0
    for k in picks:
        j, i = divmod(int(k), terrain.grid)
        src_brush = int(terrain.brush_id[j, i])

        at_max.fill(False)
        fov_into(
            at_max,
            terrain.blocks_vision,
            terrain.brush_id,
            i,
            j,
            src_brush,
            radius_cells_sq(C.RMAX_UNITS),
            half,
            scratch,
        )
        for r in GAME_RADII:
            at_r.fill(False)
            fov_into(
                at_r,
                terrain.blocks_vision,
                terrain.brush_id,
                i,
                j,
                src_brush,
                radius_cells_sq(r),
                half,
                scratch,
            )
            if not np.array_equal(at_r, at_max & discs[r]):
                mismatches += 1

    assert mismatches == 0, (
        f"{mismatches} of {len(picks) * len(GAME_RADII)} (source, radius) pairs violate "
        "radius separability. The single-table FOV design is invalid"
    )


def test_radius_monotonicity_holds_through_brush(terrain):
    """Specifically for sources standing in brush.

    Brush is where separability is most fragile: flood-revealing the source's brush
    breaks it exactly when the brush extends past `r`, which is the long mid and
    river patches. Sampling uniformly would mostly miss brush cells, so they get
    their own pass.
    """
    rng = np.random.default_rng(5)
    brush_cells = np.flatnonzero(terrain.brush.ravel())
    picks = rng.choice(brush_cells, size=120, replace=False)

    half = C.RMAX_CELLS
    window = 2 * half + 1
    at_max = np.zeros((window, window), dtype=bool)
    at_r = np.zeros((window, window), dtype=bool)
    scratch = new_scratch()

    for k in picks:
        j, i = divmod(int(k), terrain.grid)
        src_brush = int(terrain.brush_id[j, i])
        assert src_brush >= 0
        at_max.fill(False)
        fov_into(
            at_max,
            terrain.blocks_vision,
            terrain.brush_id,
            i,
            j,
            src_brush,
            radius_cells_sq(C.RMAX_UNITS),
            half,
            scratch,
        )
        for r in (C.SIGHT_WARD_TOTEM, C.SIGHT_CHAMPION):
            at_r.fill(False)
            fov_into(
                at_r,
                terrain.blocks_vision,
                terrain.brush_id,
                i,
                j,
                src_brush,
                radius_cells_sq(r),
                half,
                scratch,
            )
            np.testing.assert_array_equal(at_r, at_max & disc_mask(r, window=window))


def test_scan_stack_headroom(terrain):
    """Measure the real high-water mark rather than arguing about worst cases.

        A pathological checkerboard could in principle push O(radius^2) frames. Real
        Summoner's Rift peaks around 52 of 8,192, so the capacity is not a live concern
    but it is measured, and a terrain change that altered this by two orders of
        magnitude would be caught here rather than by a truncated mask in production.
    """
    rng = np.random.default_rng(9)
    cells = terrain.walkable_cells()
    picks = rng.choice(cells, size=2000, replace=False)

    half = C.RMAX_CELLS
    out = np.zeros((2 * half + 1,) * 2, dtype=bool)
    scratch = new_scratch()
    r2 = radius_cells_sq(C.RMAX_UNITS)

    worst = 0
    for k in picks:
        j, i = divmod(int(k), terrain.grid)
        out.fill(False)
        worst = max(
            worst,
            fov_into(
                out,
                terrain.blocks_vision,
                terrain.brush_id,
                i,
                j,
                int(terrain.brush_id[j, i]),
                r2,
                half,
                scratch,
            ),
        )
    assert worst < scratch.shape[0] // 4, f"stack high-water mark {worst} is close to capacity"


# ---------------------------------------------------------------------------
# Against the independent oracle
# ---------------------------------------------------------------------------
def _compare(terrain, i, j, radius, half=None, band_width=1):
    """Compare shadowcasting to the reference at one source.

    Returns (considered, permissive, restrictive, excluded_fraction) where
    `permissive` counts cells shadowcasting lights and the reference does not, and
    `restrictive` the reverse.
    """
    half = C.RMAX_CELLS if half is None else half
    got = fov_bool(terrain, i, j, radius, half=half)
    want = fov_reference(terrain, i, j, radius, half=half)
    band = boundary_band(got, band_width) | boundary_band(want, band_width)
    disc = disc_mask(radius, window=2 * half + 1)
    keep = disc & ~band
    return (
        int(keep.sum()),
        int((got & ~want & keep).sum()),
        int((~got & want & keep).sum()),
        int((disc & band).sum()),
    )


def test_shadowcast_is_never_more_restrictive_than_ray_marching(terrain):
    """The strong invariant, and a far better check than an agreement percentage.

    Shadowcasting tracks a slope *interval* and lights any cell whose extremities
    overlap it, so it is inherently more permissive than requiring a clear
    centre-to-centre segment. Disagreements are therefore expected in exactly one
    direction, and MEASURED to be: 4,286 permissive against 0 restrictive over three
    million cells of adversarial geometry.

    Asserting the direction rather than a rate is what makes this test load-bearing.
    A rate absorbs almost any error if it is small enough; one-directionality is an
    algebraic property, and a transposed octant, an off-by-one in a slope, or an
    inverted brush comparison would all break it immediately.
    """
    rng = np.random.default_rng(31)
    picks = rng.choice(terrain.walkable_cells(), size=40, replace=False)

    considered = permissive = restrictive = excluded = 0
    for k in picks:
        j, i = divmod(int(k), terrain.grid)
        c, p, r, e = _compare(terrain, i, j, C.SIGHT_CHAMPION)
        considered += c
        permissive += p
        restrictive += r
        excluded += e

    excl_frac = excluded / (considered + excluded)
    assert excl_frac < 0.15, f"excluded {excl_frac:.1%} as boundary, too much to conclude from"
    assert restrictive == 0, (
        f"{restrictive} cells the reference can see but shadowcasting cannot. "
        "Shadowcasting should only ever over-report; a restrictive disagreement means "
        "vision is being lost, not merely quantised."
    )
    rate = (considered - permissive) / considered
    assert rate >= 0.999, f"agreement {rate:.4%} below the 99.9% gate"


def test_reference_and_shadowcast_agree_exactly_on_an_empty_grid():
    """With no occluders the two algorithms must be identical, boundary band or not."""
    terrain = make_terrain()
    a = fov_bool(terrain, 48, 48, 500.0, half=HALF)
    b = fov_reference(terrain, 48, 48, 500.0, half=HALF)
    np.testing.assert_array_equal(a, b)


def _adversarial_map():
    """A small map built to contain the geometry that generates FOV artefacts.

    Random sampling on the big map can miss systematic errors, because an octant
    transposition or a slope off-by-one only shows up in particular configurations.
    This one has a long wall with a one-cell door, a perpendicular wall, a diagonal
    (the classic artefact generator), a pillar, and two adjacent brushes.
    """
    grid = 64
    walls = np.zeros((grid, grid), dtype=bool)
    walls[20, 10:50] = True
    walls[20, 30] = False  # the door
    walls[10:50, 44] = True
    for d in range(20):
        walls[30 + d, 12 + d] = True
    walls[40:43, 20:23] = True  # pillar
    a = np.zeros((grid, grid), dtype=bool)
    a[24:28, 24:28] = True
    b = np.zeros((grid, grid), dtype=bool)
    b[24:28, 30:34] = True
    walls[a | b] = False
    return make_terrain(grid, walls=walls, brush=[a, b])


@pytest.mark.slow
def test_exhaustive_small_map_oracle():
    """Every source cell of the adversarial map, checked against ray marching.

    MEASURED: 4,286 permissive and 0 restrictive disagreements over 3,064,927
    compared cells, all of them within two cells of a shadow boundary, 4,241 at
    distance 1 and 45 at distance 2, none beyond. That distribution is the whole
    point of the test: quantisation clusters at edges, whereas a genuine bug in the
    slope arithmetic or the octant transforms would scatter disagreements through
    the interior. A two-cell band is used here because the geometry is deliberately
    worse than Summoner's Rift.
    """
    terrain = _adversarial_map()
    half = 16

    considered = permissive = restrictive = 0
    for j in range(terrain.grid):
        for i in range(terrain.grid):
            if terrain.blocks_vision[j, i]:
                continue
            c, p, r, _ = _compare(terrain, i, j, 500.0, half=half, band_width=2)
            considered += c
            permissive += p
            restrictive += r

    assert restrictive == 0, f"{restrictive} cells lost vision relative to the reference"
    rate = (considered - permissive) / considered
    assert rate >= 0.999, f"exhaustive agreement {rate:.4%} over {considered} cells"


@pytest.mark.slow
def test_disagreements_are_confined_to_shadow_boundaries():
    """The diagnostic that separates quantisation from a bug.

    A headline agreement rate cannot distinguish "the two algorithms round shadow
    edges differently" from "one of them is wrong in the open". This asserts the
    former directly: every disagreement must lie within a few cells of a visibility
    boundary. If that ever fails, the rate is irrelevant and there is a real defect.
    """
    terrain = _adversarial_map()
    half = 16
    disc = disc_mask(500.0, window=2 * half + 1)

    worst_distance = 0
    total = 0
    for j in range(0, terrain.grid, 3):
        for i in range(0, terrain.grid, 3):
            if terrain.blocks_vision[j, i]:
                continue
            got = fov_bool(terrain, i, j, 500.0, half=half)
            want = fov_reference(terrain, i, j, 500.0, half=half)
            band = boundary_band(got) | boundary_band(want)
            disagree = (got != want) & disc & ~band
            if not disagree.any():
                continue
            total += int(disagree.sum())
            bj, bi = np.nonzero(band)
            dj, di = np.nonzero(disagree)
            l1 = np.abs(dj[:, None] - bj[None, :]) + np.abs(di[:, None] - bi[None, :])
            worst_distance = max(worst_distance, int(l1.min(axis=1).max()))

    assert total > 0, "expected some boundary quantisation on adversarial geometry"
    assert worst_distance <= 2, (
        f"a disagreement sits {worst_distance} cells from any visibility boundary. "
        "Boundary quantisation cannot explain that. This is a real FOV bug."
    )
