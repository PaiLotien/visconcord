"""Isolated M1-M4 upstream pipeline with intent-obligation closure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from skillvis_v2_accuracy.contracts import (
    AbstentionState,
    AmbiguityStateType,
    CapabilityAssessment,
    M1M3PipelineResult,
    TypedAmbiguityState,
    VisualizationIntentIR,
)
from skillvis_v2_accuracy.pipeline import run_m1_m3_pipeline as run_frozen_v3_pipeline
from skillvis_v2_accuracy.rules import RuleRegistry

from .config import SkillVISV2Config, load_config
from .intent_compiler import compile_intent_obligations
from .semantic_safety import run_semantic_safety


def _upstream_abstention(result: M1M3PipelineResult) -> AbstentionState:
    if result.normalization.failure_state:
        return AbstentionState(abstained=True, stage="query_normalization", reason=result.normalization.failure_state)
    if result.schema_profile.failure_state:
        return AbstentionState(abstained=True, stage="schema_profiling", reason=result.schema_profile.failure_state)
    if result.field_resolution.abstention_reason:
        return AbstentionState(abstained=True, stage="field_resolution", reason=result.field_resolution.abstention_reason)
    if result.task_profile.primary_task == "unresolved":
        return AbstentionState(abstained=True, stage="task_profiling", reason=result.task_profile.failure_state)
    return AbstentionState(abstained=False)


def _identity(intent: VisualizationIntentIR) -> VisualizationIntentIR:
    payload = intent.model_dump(mode="json")
    payload.pop("deterministic_id", None)
    deterministic_id = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return VisualizationIntentIR.validate_serialized(
        intent.model_copy(update={"deterministic_id": deterministic_id}).model_dump(mode="json")
    )


def run_m1_m3_pipeline(
    *,
    query: str,
    dataset_id: str,
    data: pd.DataFrame | Mapping[str, Sequence[object]],
    aliases: Mapping[str, Sequence[str]] | None = None,
    config: SkillVISV2Config | None = None,
) -> M1M3PipelineResult:
    """Reuse frozen M1-M3, then replace only the isolated closure/safety boundary."""

    config = config or load_config()
    registry = RuleRegistry.from_config(config)
    frozen = run_frozen_v3_pipeline(
        query=query,
        dataset_id=dataset_id,
        data=data,
        aliases=aliases,
        config=config,
    )
    base_provenance = tuple(
        item for item in frozen.intent_ir.provenance if item.stage != "semantic_safety"
    )
    reset = frozen.intent_ir.model_copy(
        update={
            "ambiguity_state": TypedAmbiguityState(
                state=AmbiguityStateType.NONE,
                evidence_spans=(),
                triggering_rules=(),
                confidence=1.0,
                affected_fields=(),
                affected_tasks=(),
                resolution_strategy="proceed",
                explanation="Pending isolated M3.5 obligation-aware safety assessment.",
            ),
            "unsupported_scope": (),
            "capability_assessment": CapabilityAssessment(
                supported_operations=(),
                unsupported_operations=(),
                risk_level="unknown",
                abstention_required=False,
                explanation="Pending isolated M3.5 obligation-aware safety assessment.",
                provenance=(),
            ),
            "safety_warnings": (),
            "abstention_state": _upstream_abstention(frozen),
            "abstention_reason": _upstream_abstention(frozen).reason,
            "provenance": base_provenance,
        }
    )
    compiled = compile_intent_obligations(reset, frozen.task_profile, registry)
    safety = run_semantic_safety(compiled.intent, registry)
    extra_rule_ids = tuple(sorted(set(compiled.applied_rule_ids) | set(safety.applied_rule_ids)))
    extra_provenance = tuple(
        registry.trace(
            "semantic_safety" if rule_id.startswith("R-SA-") else "intent_assembly",
            rule_id,
            inputs=("query", f"dataset:{dataset_id}"),
            outputs=("semantic_safety" if rule_id.startswith("R-SA-") else "intent_obligations",),
        )
        for rule_id in extra_rule_ids
    )
    completed = compiled.intent.model_copy(
        update={
            "ambiguity_state": safety.ambiguity_state,
            "unsupported_scope": safety.unsupported_scope,
            "capability_assessment": safety.capability_assessment,
            "safety_warnings": safety.safety_warnings,
            "abstention_state": safety.abstention_state,
            "abstention_reason": safety.abstention_reason,
            "provenance": tuple(sorted((*base_provenance, *extra_provenance), key=lambda item: (item.stage, item.rule_id))),
        }
    )
    completed = _identity(completed)
    return frozen.model_copy(update={"task_profile": compiled.task_result, "intent_ir": completed})


def run_m1_m3_5_pipeline(**kwargs) -> M1M3PipelineResult:
    return run_m1_m3_pipeline(**kwargs)


__all__ = ["run_m1_m3_5_pipeline", "run_m1_m3_pipeline"]
