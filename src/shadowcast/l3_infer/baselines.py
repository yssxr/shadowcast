"""The ablation: seven models, one code path, one table.

The point of this module is to make a specific claim falsifiable. The claim is that
reconstructing *negative* information — the region a team is actively looking at, and
therefore where the enemy provably is not — produces a materially better position
estimate than the alternatives. The alternatives are not strawmen: `geodisc` is a
geodesic reachability ball, which is already better than anything shipping today, and
`behavioural` is a navmesh random walk with a role-conditioned prior.

Two adjacent rows carry the argument:

    diffusion  -> behavioural    what the behavioural prior is worth
    behavioural -> full          what negative information is worth

They are adjacent because each differs from its neighbour in exactly one field of one
frozen spec. If `full` does not beat `behavioural`, negative information is contributing
nothing and the central claim is empty — and that result would be worth publishing too,
which is why the comparison is set up to be capable of producing it.

Every model sees identical observations and identical vision masks. Only the motion model
and the observation model differ, so nothing else can explain a gap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from shadowcast.config import BASELINES, THESIS_PAIR, FilterSpec
from shadowcast.l3_infer.metrics import BeliefScore, LatticeIndex, evaluate
from shadowcast.l3_infer.pf import BeliefFilter
from shadowcast.l3_infer.policy import Observation, PublicInfo, TruthTable
from shadowcast.l3_infer.reachability import ReachabilityIndex
from shadowcast.terrain.terrain import Terrain

__all__ = ["Ablation", "ablate", "run_model"]

#: A mask stream is consumed once, so every model needs a fresh one. The caller supplies
#: a factory rather than a stream for that reason.
MaskFactory = Callable[[], Iterator[tuple[int, np.ndarray, np.ndarray]]]


def run_model(
    name: str,
    spec: FilterSpec,
    terrain: Terrain,
    obs: Observation,
    public: PublicInfo,
    truth: TruthTable,
    masks: MaskFactory,
    lattice: LatticeIndex,
    reach: ReachabilityIndex | None = None,
    stride: int = 1,
) -> BeliefScore:
    """Run one model and score it.

    `truth` goes to `evaluate` and never to the filter — the two arguments sit side by
    side in this signature and are handed to different functions, which is as close as
    Python gets to making the barrier visible at the call site.
    """
    filt = BeliefFilter(spec, terrain, reach=reach)
    score = evaluate(
        name,
        spec,
        filt.run(obs, public, masks()),
        truth,
        lattice,
        stride=stride,
    )
    # Depletion and resample counts only exist once the stream has been consumed, which
    # happens inside `evaluate`, so they are folded in afterwards rather than passed
    # ahead of time.
    return replace(
        score,
        depletion_events=int(filt.state.depletions.sum()),
        stats={**score.stats, **filt.describe()},
    )


@dataclass(frozen=True, slots=True)
class Ablation:
    """The table, plus the one comparison it exists to make."""

    scores: dict[str, BeliefScore]

    @property
    def thesis_delta(self) -> float:
        """`behavioural` NLL minus `full` NLL. Positive means negative information helps."""
        a, b = THESIS_PAIR
        return self.scores[a].nll - self.scores[b].nll

    @property
    def thesis_holds(self) -> bool:
        return self.thesis_delta > 0.0

    def table(self) -> list[dict[str, Any]]:
        return [s.describe() for s in self.scores.values()]

    def describe(self) -> dict[str, Any]:
        return {
            "models": self.table(),
            "thesis_pair": list(THESIS_PAIR),
            "thesis_delta_nll": round(self.thesis_delta, 4),
            "thesis_holds": self.thesis_holds,
        }


def ablate(
    terrain: Terrain,
    obs: Observation,
    public: PublicInfo,
    truth: TruthTable,
    masks: MaskFactory,
    models: dict[str, FilterSpec] | None = None,
    lattice: LatticeIndex | None = None,
    stride: int = 1,
) -> Ablation:
    """Run every model over the same match.

    The reachability index is shared across models on purpose: it is a property of the
    terrain, not of a belief, and rebuilding it per model would multiply the Dijkstra
    count by seven for identical answers.
    """
    models = models if models is not None else BASELINES
    lattice = lattice if lattice is not None else LatticeIndex(terrain)
    reach = ReachabilityIndex(terrain)
    scores = {
        name: run_model(
            name, spec, terrain, obs, public, truth, masks, lattice, reach=reach, stride=stride
        )
        for name, spec in models.items()
    }
    return Ablation(scores=scores)
