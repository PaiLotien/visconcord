"""Memory-aware S5/S6 orchestration for the accuracy-repair revision."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from .candidate_planner import plan_visualization_candidates
from .config import SkillVISV2Config
from .contracts import (
    AnalyticalTaskResult,
    CandidatePlanningResult,
    FieldResolutionResult,
    SchemaProfile,
    VisualizationCandidate,
    VisualizationIntentIR,
)
from .memory import ExactMemoryLookup, MemoryRankingResult, TaskConditionedMemory
from .validator_formal import (
    CandidateSetContext,
    FormalCandidateDecision,
    FormalValidationSetResult,
    candidate_set_signature,
    validate_candidate_set,
)


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class MemoryPipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    schema_version: str = "skillvis-v2-memory-pipeline-v1.0.0"
    source: str
    exact_lookup: ExactMemoryLookup
    planning: CandidatePlanningResult | None
    candidates: tuple[VisualizationCandidate, ...]
    validation: FormalValidationSetResult | None
    memory_ranking: MemoryRankingResult | None
    planner_invoked: bool
    validator_invoked: bool
    safety_bypassed: bool = False
    candidate_mutation_count: int = 0
    deterministic_signature: str


PlannerCallable = Callable[..., CandidatePlanningResult]
ValidatorCallable = Callable[[CandidateSetContext], FormalValidationSetResult]


def _validate(
    *,
    candidates: tuple[VisualizationCandidate, ...],
    intent: VisualizationIntentIR,
    schema: SchemaProfile,
    source: str,
    validator: ValidatorCallable,
) -> FormalValidationSetResult:
    context = CandidateSetContext(
        context_id=f"memory:{intent.deterministic_id[:20]}",
        candidate_set_id=(
            f"memory:{source}:{candidate_set_signature(candidates)[:20]}"
        ),
        source_id=source,
        schema=schema,
        intent=intent,
        capability=intent.capability_assessment,
        candidates=candidates,
        candidate_set_signature=candidate_set_signature(candidates),
    )
    return validator(context)


def plan_with_memory(
    *,
    user_id: str,
    memory: TaskConditionedMemory,
    intent: VisualizationIntentIR,
    schema: SchemaProfile,
    field_resolution: FieldResolutionResult,
    task_result: AnalyticalTaskResult,
    config: SkillVISV2Config,
    planner: PlannerCallable = plan_visualization_candidates,
    validator: ValidatorCallable = validate_candidate_set,
) -> MemoryPipelineResult:
    """Reuse exact plans safely or run S5 Planner, S6 Validator, then memory rank.

    Safety is always evaluated upstream.  Every exact memory hit is revalidated
    before use, and a rejected cached plan falls back to the current Planner.
    """

    exact = memory.lookup_exact(user_id=user_id, intent=intent, schema=schema)
    if exact.hit and exact.candidate is not None:
        cached_candidates = (exact.candidate,)
        validation = _validate(
            candidates=cached_candidates,
            intent=intent,
            schema=schema,
            source="task_conditioned_memory_exact_hit",
            validator=validator,
        )
        cached_result = validation.candidate_results[0]
        if cached_result.decision != FormalCandidateDecision.REJECT:
            ranking = memory.rank(
                user_id=user_id,
                intent=intent,
                schema=schema,
                candidates=cached_candidates,
                validation=validation,
            )
            payload = {
                "source": "exact_memory",
                "entry_id": exact.entry_id,
                "candidates": [item.deterministic_signature for item in cached_candidates],
                "validation": validation.deterministic_signature,
                "ranking": ranking.deterministic_signature,
                "planner_invoked": False,
                "validator_invoked": True,
                "safety_bypassed": False,
                "candidate_mutation_count": 0,
            }
            return MemoryPipelineResult(
                source="exact_memory",
                exact_lookup=exact,
                planning=None,
                candidates=cached_candidates,
                validation=validation,
                memory_ranking=ranking,
                planner_invoked=False,
                validator_invoked=True,
                deterministic_signature=_hash(payload),
            )
        exact = ExactMemoryLookup(
            hit=False,
            reason="cached_plan_rejected_by_current_validator",
            entry_id=exact.entry_id,
            validator_signature=cached_result.deterministic_signature,
        )

    planning = planner(
        intent=intent,
        capability=intent.capability_assessment,
        schema=schema,
        field_resolution=field_resolution,
        task_result=task_result,
        config=config,
    )
    if not planning.candidates:
        payload = {
            "source": "planner_no_candidates",
            "lookup_reason": exact.reason,
            "planning": planning.deterministic_signature,
            "planner_invoked": True,
            "validator_invoked": False,
            "safety_bypassed": False,
            "candidate_mutation_count": 0,
        }
        return MemoryPipelineResult(
            source="planner_no_candidates",
            exact_lookup=exact,
            planning=planning,
            candidates=(),
            validation=None,
            memory_ranking=None,
            planner_invoked=True,
            validator_invoked=False,
            deterministic_signature=_hash(payload),
        )

    validation = _validate(
        candidates=planning.candidates,
        intent=intent,
        schema=schema,
        source="accuracy_repair_planner",
        validator=validator,
    )
    ranking = memory.rank(
        user_id=user_id,
        intent=intent,
        schema=schema,
        candidates=planning.candidates,
        validation=validation,
    )
    payload = {
        "source": "planner_then_memory_rank",
        "lookup_reason": exact.reason,
        "planning": planning.deterministic_signature,
        "validation": validation.deterministic_signature,
        "ranking": ranking.deterministic_signature,
        "planner_invoked": True,
        "validator_invoked": True,
        "safety_bypassed": False,
        "candidate_mutation_count": 0,
    }
    return MemoryPipelineResult(
        source="planner_then_memory_rank",
        exact_lookup=exact,
        planning=planning,
        candidates=planning.candidates,
        validation=validation,
        memory_ranking=ranking,
        planner_invoked=True,
        validator_invoked=True,
        deterministic_signature=_hash(payload),
    )
