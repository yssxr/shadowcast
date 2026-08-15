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
