"""Shared fixtures.

The navgrid is a downloaded artifact rather than a committed one (9.4 MB, and it
is Riot's data, not ours), so terrain tests skip cleanly when it is absent instead
of failing. The skip message says how to get it.
"""

from __future__ import annotations

import pytest

from shadowcast import constants as C
from shadowcast.terrain.navgrid import default_navgrid_path


def _navgrid_or_skip():
    path = default_navgrid_path()
    if not path.exists():
        pytest.skip(
            f"navgrid not present at {path}. Fetch it with:\n"
            f'  mkdir -p {path.parent} && curl -L -o {path} "{C.NAVGRID_URL}"'
        )
    return path


@pytest.fixture(scope="session")
def navgrid_path():
    return _navgrid_or_skip()


@pytest.fixture(scope="session")
def navgrid(navgrid_path):
    from shadowcast.terrain.navgrid import read_navgrid

    return read_navgrid(navgrid_path)


@pytest.fixture(scope="session")
def terrain(navgrid_path):
    """The real Summoner's Rift terrain on the simulation grid.

    Session-scoped: the brush component labelling is a Python-level flood fill over
    6,397 cells and there is no reason to repeat it per test.
    """
    from shadowcast.terrain.terrain import build_terrain

    return build_terrain(navgrid_path=navgrid_path)


@pytest.fixture(scope="session")
def fov_table(terrain, tmp_path_factory):
    """The precomputed visibility table. Session-scoped — building it is ~4 s."""
    from shadowcast.fov.table import build_table

    return build_table(terrain, out_dir=tmp_path_factory.mktemp("fov_session"))


@pytest.fixture(scope="session")
def synth_clean(terrain):
    """A full synthetic match with no pathologies.

    Session-scoped because generation is ~2.5 s and includes the fog oracle. The
    clean variant is the one that pins the algebra: with nothing injected,
    downstream reconstruction must recover truth exactly.
    """
    from shadowcast.packets.synth import Pathologies, ScenarioSpec, SyntheticSource

    src = SyntheticSource(terrain, ScenarioSpec(seed=7, pathologies=Pathologies.none()))
    mid = src.match_ids()[0]
    return src.read(mid), src.truth(mid)


@pytest.fixture(scope="session")
def synth_dirty(terrain):
    """A full synthetic match with every pathology enabled."""
    from shadowcast.packets.synth import Pathologies, ScenarioSpec, SyntheticSource

    src = SyntheticSource(terrain, ScenarioSpec(seed=7, pathologies=Pathologies.all()))
    mid = src.match_ids()[0]
    return src.read(mid), src.truth(mid)
