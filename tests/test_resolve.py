"""Tests for team, role and death resolution.

Everything here is an inference the corpus does not state, so each test measures an
accuracy rather than asserting a fact. Where the synthetic scenario makes something
easier than reality will be, the test says so rather than banking the easy number.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from shadowcast import constants as C
from shadowcast.l1_events.normalise import normalise
from shadowcast.l1_events.resolve import attribute
from shadowcast.l1_events.resolve.deaths import resolve_deaths, with_deaths
from shadowcast.l1_events.resolve.roles import ROLES, resolve_all, resolve_roles, with_roles
from shadowcast.l1_events.resolve.teams import resolve_teams, with_teams
from shadowcast.l1_events.schema import UNKNOWN


@pytest.fixture(scope="module")
def resolved(synth_dirty, terrain):
    """A fully resolved adversarial match, plus its ground truth."""
    bundle, truth = synth_dirty
    events = normalise(bundle, terrain)
    at = attribute(events)
    events, info = resolve_all(events, at.pos, at.valid)
    return events, truth, at, info


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
def test_teams_are_recovered_exactly(resolved):
    """CreateHero states no team, so this is entirely inferred from turret names.

    Turret internal names carry T1/T2, turret positions are recovered from their attack
    packets, and champions start the match among their own structures.
    """
    events, truth, _, _ = resolved
    np.testing.assert_array_equal(events.heroes["team"], truth.team)


def test_teams_split_five_and_five(resolved):
    events, _, _, info = resolved
    for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
        assert int((events.heroes["team"] == team).sum()) == 5
    assert info["teams"]["balanced"]
    assert info["teams"]["method"] == "turret_proximity_with_5v5_constraint"


def test_team_resolution_reports_its_margin(resolved):
    """The margin is the evidence, and on this scenario it is very large.

    Worth noting rather than celebrating: champions here sit at their fountain until the
    first anchor, so the nearest-shrine signal is unusually clean. Real matches will have
    champions that move first, and the 5/5 constraint is what carries those.
    """
    _, _, _, info = resolved
    assert info["teams"]["min_margin"] > 1000.0


def test_the_five_five_constraint_rescues_an_ambiguous_champion(synth_dirty, terrain):
    """Rank-and-split must beat per-champion thresholding.

    One champion's evidence is deliberately destroyed by moving its early anchors to the
    map centre, equidistant from both bases. A threshold would have to guess; ranking the
    other nine correctly leaves only one place for it to go.
    """
    bundle, truth = synth_dirty
    events = normalise(bundle, terrain)

    victim_slot = 3
    anchors = events.anchors.copy()
    early = (anchors["slot"] == victim_slot) & (anchors["t"] < 30.0)
    anchors["x"][early] = C.WORLD_MIN_X + C.WORLD_SPAN / 2
    anchors["z"][early] = C.WORLD_MIN_Z + C.WORLD_SPAN / 2
    nobbled = dataclasses.replace(events, anchors=anchors)

    res = resolve_teams(nobbled)
    assert res.resolved
    assert int((res.team == C.TEAM_ORDER).sum()) == 5
    assert res.team[victim_slot] == truth.team[victim_slot]
    # Its own evidence really was the weakest of the ten.
    assert abs(res.lean[victim_slot]) == pytest.approx(np.abs(res.lean).min())


def test_teams_are_unresolved_without_turrets(synth_clean, terrain):
    """No turrets means no entity states a side, and the honest answer is UNKNOWN.

    Inventing a split from champion positions alone would be guessing which corner is
    which, and every downstream number would silently inherit that guess.
    """
    bundle, _ = synth_clean
    events = normalise(bundle, terrain)
    stripped = dataclasses.replace(events, turret_sites=events.turret_sites[:0])
    res = resolve_teams(stripped)
    assert not res.resolved
    assert (res.team == UNKNOWN).all()
    assert res.method == "unresolved"


def test_fog_observer_teams_are_derived_from_champion_teams(resolved):
    """The step that recovers a per-team oracle from packets with no observer field.

    A team never loses sight of its own members, so a fog event naming champion C can
    only come from C's opponents. Without this the fog stream says only "someone stopped
    seeing this champion", which no per-team metric can use.
    """
    events, truth, _, _ = resolved
    assert (events.fog["observer_team"] != UNKNOWN).all()
    for row in events.fog:
        assert row["observer_team"] == 1 - truth.team[row["slot"]]


def test_observer_teams_stay_unknown_when_teams_are(synth_clean, terrain):
    bundle, _ = synth_clean
    events = normalise(bundle, terrain)
    stripped = dataclasses.replace(events, turret_sites=events.turret_sites[:0])
    out = with_teams(stripped, resolve_teams(stripped))
    assert (out.fog["observer_team"] == UNKNOWN).all()


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
def test_roles_are_recovered_exactly(resolved):
    events, truth, _, _ = resolved
    np.testing.assert_array_equal(events.heroes["role"], np.array(truth.role))


def test_each_team_fields_one_of_each_role(resolved):
    events, _, _, _ = resolved
    for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
        roles = events.heroes["role"][events.heroes["team"] == team]
        assert sorted(roles.tolist()) == sorted(ROLES)


def test_junglers_are_identified_by_being_off_lane(resolved):
    """The clean part of role inference: a jungler is near none of the three lanes."""
    events, truth, at, _ = resolved
    res = resolve_roles(events, at.pos, at.valid)
    junglers = [s for s in range(C.N_HEROES) if truth.role[s] == "jungle"]
    others = [s for s in range(C.N_HEROES) if truth.role[s] != "jungle"]
    assert res.jungle_time[junglers].min() > res.jungle_time[others].max()


def test_supports_are_separated_from_carries_by_observed_ward_ownership(resolved):
    """The part that positions cannot do, and why it uses a measurement instead.

    A support and its carry stand together by design, so lane occupancy cannot tell them
    apart. Ward ownership is directly observed — `targetable_on_client` names the owner —
    so this is a reading rather than a behavioural guess.
    """
    events, truth, at, _ = resolved
    res = resolve_roles(events, at.pos, at.valid)
    for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
        members = [s for s in range(C.N_HEROES) if truth.team[s] == team]
        support = next(s for s in members if truth.role[s] == "support")
        carry = next(s for s in members if truth.role[s] == "bot")
        assert res.ward_share[support] > res.ward_share[carry], (
            f"team {team}: support ward share {res.ward_share[support]:.2f} "
            f"vs carry {res.ward_share[carry]:.2f}"
        )


def test_bot_and_support_share_the_bottom_lane(resolved):
    events, truth, at, _ = resolved
    res = resolve_roles(events, at.pos, at.valid)
    bot_index = 2  # top, mid, bot
    for team in (C.TEAM_ORDER, C.TEAM_CHAOS):
        members = [s for s in range(C.N_HEROES) if truth.team[s] == team]
        pair = [s for s in members if truth.role[s] in ("bot", "support")]
        assert min(res.lane_time[s, bot_index] for s in pair) > 0.3


def test_roles_stay_unresolved_when_teams_are(synth_clean, terrain):
    """Roles are constrained per team, so without teams there is nothing to constrain."""
    bundle, _ = synth_clean
    events = normalise(bundle, terrain)
    at = attribute(events)
    res = resolve_roles(events, at.pos, at.valid)
    assert (res.role == "").all()
    assert res.stats["teams_without_five_members"] == 2


def test_with_roles_leaves_the_original_untouched(resolved):
    events, _, at, _ = resolved
    res = resolve_roles(events, at.pos, at.valid)
    before = events.heroes["role"].copy()
    out = with_roles(events, res)
    np.testing.assert_array_equal(events.heroes["role"], before)
    np.testing.assert_array_equal(out.heroes["role"], res.role)


# ---------------------------------------------------------------------------
# Deaths
# ---------------------------------------------------------------------------
def test_deaths_are_inferred_from_health_reaching_zero(resolved):
    """There is no death packet. HeroDie fires zero times in 965,768 real packets."""
    events, truth, _, _ = resolved
    assert events.deaths.size == truth.kills.size
    for row in events.deaths:
        match = truth.kills[int(np.argmin(np.abs(truth.kills["t"] - row["t"])))]
        assert row["t"] == pytest.approx(match["t"], abs=0.5)
        assert row["victim"] == match["victim"]


def test_killers_are_inferred_from_the_last_damage(resolved):
    events, truth, _, _ = resolved
    for row in events.deaths:
        match = truth.kills[int(np.argmin(np.abs(truth.kills["t"] - row["t"])))]
        assert row["killer"] == match["killer"]


def test_killer_confidence_reflects_contested_damage(resolved):
    """Confidence is the killer's share of the damage window, and must not be pinned.

    A figure that is always 1.0 tells a consumer nothing. The generator gives every kill
    an assisting damager for exactly this reason, so a chaotic teamfight scores lower
    than a clean solo kill — which is the right shape, because that is when the
    attribution is genuinely least trustworthy.
    """
    events, _, _, info = resolved
    conf = events.deaths["killer_confidence"]
    assert (conf > 0.0).all()
    assert (conf < 1.0).all(), "confidence pinned at 1.0 means it is not being tested"
    assert 0.5 < float(info["deaths"]["mean_killer_confidence"]) < 0.95


def test_respawn_is_observed_rather_than_computed(resolved):
    """The next labelled observation, not a level-and-clock formula.

    The stream already answers the question, and a formula could be wrong for the patch.
    """
    events, truth, _, info = resolved
    assert info["deaths"]["respawn_unobserved"] == 0
    for row in events.deaths:
        match = truth.kills[int(np.argmin(np.abs(truth.kills["t"] - row["t"])))]
        assert row["respawn_t"] >= match["respawn_t"] - 0.5
        assert row["respawn_t"] < match["respawn_t"] + 5.0


def test_repeated_health_zeros_are_one_death(resolved):
    """Health can be replicated at zero more than once for a single death."""
    events, truth, _, _ = resolved
    hp = events.hp.copy()
    duplicated = np.concatenate([hp, hp[hp["value"] == 0.0]])
    duplicated.sort(order="t", kind="stable")
    res = resolve_deaths(dataclasses.replace(events, hp=duplicated))
    assert res.n_deaths == truth.kills.size
    assert res.stats["duplicate_zeros_dropped"] == truth.kills.size


def test_a_death_with_no_champion_damage_has_no_killer(resolved):
    """Execution by a minion or turret leaves no champion to blame, and says so."""
    events, _, _, _ = resolved
    res = resolve_deaths(dataclasses.replace(events, damage=events.damage[:0]))
    assert res.n_deaths > 0
    assert res.killers_identified == 0
    assert (res.deaths["killer"] == UNKNOWN).all()
    assert res.stats["killer_unattributed"] == res.n_deaths


def test_deaths_are_empty_without_health_replication(resolved):
    events, _, _, _ = resolved
    res = resolve_deaths(dataclasses.replace(events, hp=events.hp[:0]))
    assert res.n_deaths == 0
    assert "reason" in res.stats


def test_with_deaths_writes_them_back(resolved):
    events, _, _, _ = resolved
    res = resolve_deaths(events)
    out = with_deaths(dataclasses.replace(events, deaths=events.deaths[:0]), res)
    assert out.deaths_resolved
    np.testing.assert_array_equal(out.deaths, res.deaths)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def test_resolve_all_fills_everything_in(resolved):
    events, _, _, info = resolved
    assert events.teams_resolved
    assert events.roles_resolved
    assert events.deaths_resolved
    assert set(info) == {"teams", "deaths", "roles"}


def test_resolve_all_is_deterministic(synth_clean, terrain):
    bundle, _ = synth_clean
    events = normalise(bundle, terrain)
    at = attribute(events)
    a, _ = resolve_all(events, at.pos, at.valid)
    b, _ = resolve_all(events, at.pos, at.valid)
    np.testing.assert_array_equal(a.heroes, b.heroes)
    np.testing.assert_array_equal(a.deaths, b.deaths)
