"""Tests for the real-shard reconnaissance.

These skip when no shard is present — it is 86 MB of someone else's data and is not
committed — but when one is there they are the only tests in the project that touch a
real packet stream, and they check the single claim everything downstream is built on.

The `oracle_holds` criterion is worth reading closely, because a weaker version of it
would pass on data that meant something else entirely. It is a *contrast*: hiding moves a
champion much further from enemies than it does from allies. A test that only asked
"are hidden champions far from enemies" would also pass on camera-based interest culling,
where hidden champions are far from everyone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SHARD = Path("data/raw/12_22/batch_001.jsonl.gz")


def _shard_or_skip() -> Path:
    if not SHARD.exists():
        pytest.skip(
            f"no decoded-replay shard at {SHARD}. Fetch one with:\n"
            '  uv run python -c "from huggingface_hub import hf_hub_download as d; '
            "d(repo_id='maknee/league-of-legends-decoded-replay-packets', "
            "repo_type='dataset', filename='12_22/batch_001.jsonl.gz', local_dir='data/raw')\""
        )
    return SHARD


@pytest.fixture(scope="module")
def reports():
    """Recon over several real matches. Module-scoped: each one parses 380,000 packets."""
    from shadowcast.packets.inspect import inspect_fog, read_matches

    _shard_or_skip()
    return [inspect_fog(m) for m in read_matches(SHARD, limit=6)]


def test_every_match_has_ten_heroes(reports):
    assert all(r.n_heroes == 10 for r in reports)


def test_fog_transitions_alternate_after_dedup(reports):
    """The 6:1 raw EnterFog:LeaveFog ratio is duplication, not a semantic asymmetry.

    `LeaveFog` is 65–70% of every packet in the corpus and maknee documents 20+ repeats.
    Deduping exact `(time, kind)` pairs and then collapsing consecutive same-kind runs
    leaves a sequence that alternates perfectly — which is what a visibility signal must
    do, and what rules out the events meaning something other than a state change.
    """
    assert all(r.alternates for r in reports)
    assert np.median([r.raw_ratio for r in reports]) > 4.0, "expected the raw ratio to be lopsided"


def test_position_packets_arrive_while_visible(reports):
    """Which settles the polarity of the two names.

    A packet carrying a unit's coordinates can only reach a client that can see it, so if
    `LeaveFog` means "became visible" then labelled positions land in `LeaveFog` intervals.
    MEASURED: 84% do. The remainder is boundary jitter and late packets.
    """
    assert np.median([r.position_packets_while_visible for r in reports]) > 0.7


def test_teams_are_recovered_exactly_from_damage_alone(reports):
    """Five and five, with essentially all hero damage across the split.

    Champions damage enemies and not allies, so the true split is the maximum cut — solved
    exactly here, since ten champions admit only 126 balanced splits. MEASURED: 23 of 23
    matches recover 5/5, and the median match puts 100.00% of hero-to-hero damage across
    the cut.

    This is an independent check on the resolver, which derives teams from turret names
    and a 5/5 constraint instead. Two unrelated methods agreeing is worth more than either
    one passing its own test.
    """
    assert all(r.bipartite for r in reports)
    assert np.median([r.cut_fraction for r in reports]) > 0.98


def test_the_fog_oracle_holds_on_real_packets(reports):
    """**The claim the whole project rests on.**

    A fog event naming champion C reflects whether C's OPPONENTS can see C — which is what
    makes `observer_team = 1 - subject_team` correct, and what turns the corpus into a
    ground-truth visibility oracle for both sides at once.

    The discriminating prediction is a contrast. Hiding should move a champion much
    further from enemies while leaving its distance to allies alone. MEASURED across 23
    matches: enemy distance ratio 2.39x, ally distance ratio 0.94x. Camera-based interest
    culling would move both together; a stream carrying one team's view would not toggle
    that team's own members at all. Neither is what the data does.
    """
    for r in reports:
        assert r.enemy_ratio > 1.5, r.describe()
        assert r.enemy_ratio > 1.3 * r.ally_ratio, r.describe()
        assert r.oracle_holds, r.describe()


def test_ally_distance_is_indifferent_to_fog(reports):
    """Stated separately because it is the half that rules out the alternatives.

    If the fog signal were about proximity to the action, or to a spectator camera, a
    hidden champion would be further from *everyone*. It is not: the ally ratio sits at
    0.94, meaning a hidden champion is if anything marginally CLOSER to its own team.
    """
    assert 0.6 < np.median([r.ally_ratio for r in reports]) < 1.5


def test_replication_carries_the_documented_index_pairs():
    """R3: the attribute indices, confirmed against real packets.

    `Replication` arrives as a dict of `net_id -> {primary_index, secondary_index, name,
    data}`, not as flat rows — a shape the synthetic source models differently, which is
    exactly the kind of difference the packet-source seam exists to absorb. 38% of entries
    carry a name at all, so the index pair is the only reliable key.
    """
    import gzip
    import json

    _shard_or_skip()
    with gzip.open(SHARD, "rt") as fh:
        events = json.loads(fh.readline())["events"]

    pairs: dict[str, tuple[int, int]] = {}
    for event in events:
        if next(iter(event)) != "Replication":
            continue
        for data in event["Replication"]["net_id_to_replication_datas"].values():
            if data.get("name"):
                pairs[data["name"]] = (data["primary_index"], data["secondary_index"])

    assert pairs["mHP"] == (32, 0)
    assert pairs["mMaxHP"] == (32, 1)
    assert pairs["mPAR"] == (32, 14)
    assert pairs["mMoveSpeed"] == (32, 24)


def test_the_resolver_recovers_the_same_teams_as_the_recon():
    """Two unrelated methods, one answer — on real packets.

    `resolve_teams` and `packets.inspect` both recover teams, and they share nothing:
    one runs over normalised slot-indexed damage after attribution, the other over raw
    net_ids straight from the shard. Agreement is therefore evidence, not tautology.

    It also records a real regression that only real data could surface. The turret-
    proximity resolver scored 100% on synthetic matches and was wrong on 2-4 champions in
    7 of 8 real ones, because the synthetic scenario holds champions at their fountain
    until their first anchor while real champions leave base immediately. The damage graph
    is now tried first and agrees with the recon in 8 of 8.
    """
    from shadowcast.l1_events.normalise import normalise
    from shadowcast.l1_events.resolve import attribute, resolve_all
    from shadowcast.packets.inspect import read_matches, teams_from_damage
    from shadowcast.packets.replay import ReplaySource
    from shadowcast.terrain.terrain import build_terrain

    _shard_or_skip()
    terrain = build_terrain()
    source = ReplaySource(SHARD, limit=3)
    raws = list(read_matches(SHARD, limit=3))

    for index, match_id in enumerate(source.match_ids()):
        events = normalise(source.read(match_id), terrain)
        at = attribute(events)
        events, info = resolve_all(events, at)

        assert info["teams"]["method"] == "damage_max_cut", info["teams"]
        assert info["teams"]["balanced"]
        assert info["teams"]["damage_across_the_split"] > 0.98

        recon = teams_from_damage(raws[index])
        recon.pop("_cut_fraction", None)
        mine = {int(h["net_id"]): int(h["team"]) for h in events.heroes}
        same = sum(1 for net_id, team in mine.items() if recon.get(net_id) == team)
        # Identical or exactly complementary: the two label the sides independently.
        assert same in (0, len(mine)), f"{match_id}: {same} of {len(mine)} agree"


def test_a_real_match_runs_the_whole_pipeline():
    """Read, normalise, attribute, resolve — on real packets, end to end.

    The point of the `packets/source.py` seam. Everything upstream was built against a
    synthetic generator; this asserts the real reader satisfies the same contract well
    enough for every layer above it to run.
    """
    from shadowcast.l1_events.normalise import normalise
    from shadowcast.l1_events.resolve import attribute, resolve_all
    from shadowcast.packets.replay import ReplaySource
    from shadowcast.terrain.terrain import build_terrain

    _shard_or_skip()
    terrain = build_terrain()
    source = ReplaySource(SHARD, limit=1)
    events = normalise(source.read(source.match_ids()[0]), terrain)
    at = attribute(events)
    events, info = resolve_all(events, at)

    assert events.teams_resolved
    assert events.roles_resolved
    assert events.n_heroes == 10
    # The frame is the navgrid midpoint; see R2 in docs/validation.md.
    assert events.frame.well_determined
    assert events.frame.walkable_fraction > 0.95
    # Attribution on real trajectories is far looser than on synthetic ones, which is the
    # honest number rather than a target: 93% attributed against 99.9%.
    assert at.attributed_fraction > 0.85
    assert info["deaths"]["deaths"] > 0


def test_the_real_reader_passes_the_same_conformance_suite():
    """The seam's whole purpose: one invariant suite, two sources.

    Warnings are expected — 17 ms timestamp jitter is one 30 Hz tick, entities referenced
    before their create packet were made before the recording started, and the fog
    duplication is documented. ERRORS are not.
    """
    from shadowcast.packets.conformance import validate_source
    from shadowcast.packets.replay import ReplaySource

    _shard_or_skip()
    for report in validate_source(ReplaySource(SHARD, limit=2)):
        assert not report.errors, (report.match_id, report.errors)
        assert report.stats["distinct_turrets"] == 24
        assert report.stats["labelled_anchors"] > 1000
