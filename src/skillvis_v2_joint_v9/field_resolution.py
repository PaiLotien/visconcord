"""Typed operation spans own weak lexical evidence at the field boundary."""

from __future__ import annotations

from skillvis_v2_accuracy.contracts import AnalyticalTaskResult, FieldResolutionResult
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_joint_v6.evidence_strength import EvidenceStrength, classify_evidence


_INDEPENDENT = {EvidenceStrength.EXPLICIT_GROUNDING, EvidenceStrength.SEMANTIC_GROUNDING}


def _covered(span, owner) -> bool:
    return owner.start <= span.start and span.end <= owner.end


def resolve_operation_span_ownership(
    *,
    fields: FieldResolutionResult,
    task: AnalyticalTaskResult,
    registry: RuleRegistry,
) -> tuple[FieldResolutionResult, tuple[str, ...]]:
    operation_spans = tuple(span for operation in task.transforms for span in operation.evidence_spans)
    if not operation_spans: return fields, ()
    required = set(fields.required_field_candidates); suppressions: dict[str, tuple[str, ...]] = {}
    for candidate in fields.candidate_fields:
        if candidate.field_name not in required: continue
        grounded = tuple(item for item in candidate.evidence if item.query_span is not None)
        if not grounded or any(classify_evidence(item.evidence_type) in _INDEPENDENT for item in grounded): continue
        if all(any(_covered(item.query_span, owner) for owner in operation_spans) for item in grounded):
            suppressions[candidate.field_name] = tuple(sorted({item.query_span.text for item in grounded}))
    if not suppressions: return fields, ()
    surviving = tuple(name for name in fields.required_field_candidates if name not in suppressions)
    ambiguity = []
    for item in fields.ambiguity:
        if item.dimension != "field": ambiguity.append(item); continue
        alternatives = tuple(name for name in item.alternatives if name in surviving)
        if len(alternatives) > 1: ambiguity.append(item.model_copy(update={"alternatives": alternatives}))
    updated = FieldResolutionResult.model_validate(
        fields.model_copy(update={"required_field_candidates": surviving, "ambiguity": tuple(ambiguity), "applied_rule_ids": registry.validate_ids((*fields.applied_rule_ids, "R-FR-090")), "abstention_reason": None if surviving else "no_field_candidate_after_operation_ownership"}).model_dump(mode="json")
    )
    traces = tuple(f"R-FR-090 operation-span ownership suppressed:{name}:weak={','.join(values)}" for name, values in sorted(suppressions.items()))
    return updated, traces


__all__ = ["resolve_operation_span_ownership"]
