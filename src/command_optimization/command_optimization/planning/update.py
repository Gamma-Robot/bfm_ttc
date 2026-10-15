"""One Planning Update over a shared candidate evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sampling_optimizers import IterationLimit, IterationRecord, SamplingOptimizer
from bfm_adapters import MotionCommand
from command_optimization.planning.evaluation import EvaluationLedger, RolloutEvaluator


@dataclass(frozen=True, eq=False)
class PlanningUpdateDiagnostics:
    """Immutable evidence for a selected optimizer result."""

    iterations: tuple[IterationRecord, ...]
    evaluations: tuple[EvaluationLedger, ...]
    best_observed_candidate: np.ndarray
    best_observed_cost: float
    proposal_candidate: np.ndarray
    proposal_cost: float | None
    selected_source: Literal["best_observed", "proposal"]

    def __post_init__(self) -> None:
        scenario_identities = {
            (record.scenario_count, record.scenario_digest)
            for record in self.evaluations
        }
        if len(scenario_identities) != 1:
            raise ValueError(
                "one Planning Update must use one scenario count and digest"
            )
        object.__setattr__(
            self,
            "best_observed_candidate",
            _owned(self.best_observed_candidate),
        )
        object.__setattr__(self, "proposal_candidate", _owned(self.proposal_candidate))

    @property
    def candidate_evaluations(self) -> int:
        return sum(record.candidate_evaluations for record in self.evaluations)

    @property
    def logical_rollouts(self) -> int:
        return sum(record.logical_rollouts for record in self.evaluations)

    @property
    def physical_rollouts(self) -> int:
        return sum(record.physical_rollouts for record in self.evaluations)

    @property
    def penalized_candidate_evaluations(self) -> int:
        return sum(
            record.penalized_candidate_evaluations for record in self.evaluations
        )

    @property
    def invalid_scenario_rollouts(self) -> int:
        return sum(
            record.invalid_scenario_rollouts for record in self.evaluations
        )

    @property
    def scenario_count(self) -> int:
        return self.evaluations[0].scenario_count

    @property
    def scenario_digest(self) -> str | None:
        return self.evaluations[0].scenario_digest


@dataclass(frozen=True, eq=False)
class PlanningUpdateResult:
    """Selected latent, MotionCommand Plan, and cost from one Planning Update."""

    selected_latent: np.ndarray
    selected_command: MotionCommand
    selected_cost: float
    diagnostics: PlanningUpdateDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_latent", _owned(self.selected_latent))


def run_planning_update(
    optimizer: SamplingOptimizer,
    evaluator: RolloutEvaluator,
    *,
    seed: int,
    iterations: IterationLimit,
    initial_mean: np.ndarray | None = None,
    evaluate_proposal: bool = True,
) -> PlanningUpdateResult:
    """Initialize and optimize exactly once, then select an evaluated result.

    The selected candidate is the lower-cost of the best sampled candidate
    and the final proposal only when proposal evaluation is explicitly
    enabled. This is a configured choice and makes no global-optimality
    claim.
    """
    if not isinstance(evaluate_proposal, bool):
        raise TypeError("evaluate_proposal must be a bool")
    evaluations: list[EvaluationLedger] = []
    evaluated_candidates: list[tuple[np.ndarray, np.ndarray]] = []

    def evaluate_candidates(candidates: np.ndarray) -> np.ndarray:
        batch = evaluator.evaluate(candidates)
        evaluations.append(batch.ledger)
        evaluated_candidates.append((_owned(candidates), batch.costs))
        return batch.costs

    params = optimizer.init_params(seed=seed, initial_mean=initial_mean)
    params, report = optimizer.optimize(
        params,
        evaluate_candidates,
        iterations=iterations,
    )

    best_candidate = report.best_observed_candidate
    best_cost = float(report.best_observed_cost)
    proposal = np.asarray(optimizer.proposal(params))
    expected = (optimizer.num_knots, optimizer.dim)
    if proposal.dtype != np.float64 or proposal.shape != expected:
        raise ValueError(
            "optimizer proposal must be a float64 array with shape "
            f"{expected}, got {proposal.dtype} {proposal.shape}"
        )
    if not np.isfinite(proposal).all():
        raise ValueError("optimizer proposal must be finite")

    proposal_cost: float | None = None
    source: Literal["best_observed", "proposal"] = "best_observed"
    selected_latent = best_candidate
    selected_cost = best_cost
    if evaluate_proposal:
        if np.array_equal(proposal, best_candidate):
            proposal_cost = best_cost
        else:
            proposal_cost = _known_cost(proposal, evaluated_candidates)
            if proposal_cost is None:
                proposal_batch = evaluator.evaluate(proposal[None])
                evaluations.append(proposal_batch.ledger)
                proposal_cost = float(proposal_batch.costs[0])
        if proposal_cost < best_cost:
            source = "proposal"
            selected_latent = proposal
            selected_cost = proposal_cost

    iterations_copy = tuple(
        IterationRecord(
            iteration=record.iteration,
            best_observed_cost=record.best_observed_cost,
            best_observed_candidate=_owned(record.best_observed_candidate),
            mean_cost=record.mean_cost,
        )
        for record in report.iterations
    )
    diagnostics = PlanningUpdateDiagnostics(
        iterations=iterations_copy,
        evaluations=tuple(evaluations),
        best_observed_candidate=best_candidate,
        best_observed_cost=best_cost,
        proposal_candidate=proposal,
        proposal_cost=proposal_cost,
        selected_source=source,
    )
    return PlanningUpdateResult(
        selected_latent=selected_latent,
        selected_command=evaluator.decode_motion_command(selected_latent),
        selected_cost=selected_cost,
        diagnostics=diagnostics,
    )


def _owned(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.flags.writeable = False
    return result


def _known_cost(
    candidate: np.ndarray,
    evaluated_candidates: list[tuple[np.ndarray, np.ndarray]],
) -> float | None:
    for candidates, costs in reversed(evaluated_candidates):
        matches = np.all(candidates == candidate, axis=(1, 2))
        indices = np.flatnonzero(matches)
        if indices.size:
            return float(costs[int(indices[0])])
    return None
