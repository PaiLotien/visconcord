"""M3.5 capability gate with schema-grounded operation/field precedence."""

from __future__ import annotations

from collections.abc import Iterable

from skillvis_v2_accuracy.contracts import (
    AbstentionState,
    AmbiguityStateType,
    CapabilityAssessment,
    EvidenceSpan,
    SemanticSafetyResult,
    UnsupportedScopeRecord,
    VisualizationIntentIR,
)
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_accuracy.semantic_safety import (
    _UNSUPPORTED_PATTERNS,
    _dedupe_spans,
    _span,
    _supported_operations,
    _warning,
    classify_ambiguity,
)


_STRONG_GROUNDING = {
    "exact", "normalized_lexical", "alias", "semantic_concept",
    "acronym_expansion",
}


def _grounded_spans(intent: VisualizationIntentIR) -> tuple[tuple[str, str, EvidenceSpan], ...]:
    rows = []
    for ref in (
        *intent.measures,
        *intent.dimensions,
        *intent.temporal_fields,
        *intent.grouping_fields,
    ):
        for evidence in ref.evidence:
            if evidence.query_span is not None and evidence.evidence_type in _STRONG_GROUNDING:
                rows.append((ref.field_name, evidence.evidence_type, evidence.query_span))
    return tuple(rows)


def _covered_by_field(
    start: int,
    end: int,
    grounded: Iterable[tuple[str, str, EvidenceSpan]],
) -> bool:
    return any(span.start <= start and end <= span.end for _, _, span in grounded)


def assess_capabilities(
    intent: VisualizationIntentIR,
    registry: RuleRegistry,
) -> tuple[CapabilityAssessment, tuple[UnsupportedScopeRecord, ...], tuple[str, ...]]:
    """Classify operation lexemes only after checking grounded field spans."""

    query = intent.query
    grounded = _grounded_spans(intent)
    detected: dict[str, list[EvidenceSpan]] = {}
    suppressed_any = False
    for operation, patterns in _UNSUPPORTED_PATTERNS:
        matches: list[EvidenceSpan] = []
        for pattern in patterns:
            for match in pattern.finditer(query):
                if _covered_by_field(match.start(), match.end(), grounded):
                    suppressed_any = True
                    continue
                matches.append(
                    _span(
                        query,
                        match.start(),
                        match.end(),
                        kind=f"unsupported_{operation}",
                        rule_id="R-SA-003",
                    )
                )
        if matches:
            detected[operation] = list(_dedupe_spans(matches))

    if "forecast" in detected and "prediction" in detected:
        detected.pop("prediction")
    supported = _supported_operations(intent)
    precedence_rules = {"R-SA-006"} if suppressed_any else set()
    if detected:
        applied = registry.validate_ids({"R-SA-003", "R-SA-004", *precedence_rules})
        records = tuple(
            UnsupportedScopeRecord(
                operation=operation,
                evidence_spans=tuple(detected[operation]),
                triggering_rules=("R-SA-003",),
                explanation=(
                    f"{operation} is outside the versioned deterministic "
                    "visualization-planning capability set."
                ),
            )
            for operation in sorted(detected)
        )
        operations = tuple(item.operation for item in records)
        risk = "critical" if {"causal_inference", "optimization"} & set(operations) else "high"
        return (
            CapabilityAssessment(
                supported_operations=supported,
                unsupported_operations=operations,
                risk_level=risk,
                abstention_required=True,
                explanation=(
                    "Unsupported governing operation detected after schema-field "
                    "span precedence; supported fallback tasks may not reach planning."
                ),
                provenance=applied,
            ),
            records,
            applied,
        )

    applied = registry.validate_ids({"R-SA-005", *precedence_rules})
    explanation = (
        "Capability lexemes were contained by grounded schema-field spans; no "
        "separate unsupported governing operation was detected."
        if suppressed_any
        else "No unsupported governing operation was detected."
    )
    return (
        CapabilityAssessment(
            supported_operations=supported,
            unsupported_operations=(),
            risk_level="low",
            abstention_required=False,
            explanation=explanation,
            provenance=applied,
        ),
        (),
        applied,
    )


def run_semantic_safety(
    intent: VisualizationIntentIR,
    registry: RuleRegistry,
) -> SemanticSafetyResult:
    capability, unsupported, capability_rules = assess_capabilities(intent, registry)
    ambiguity, ambiguity_rules = classify_ambiguity(intent, unsupported, registry)
    applied = registry.validate_ids((*capability_rules, *ambiguity_rules))
    abstention = intent.abstention_state
    reason = intent.abstention_reason or abstention.reason
    if capability.abstention_required:
        operations = ",".join(capability.unsupported_operations)
        reason = f"unsupported_scope:{operations}"
        abstention = AbstentionState(abstained=True, stage="semantic_safety", reason=reason)
    elif ambiguity.state in {
        AmbiguityStateType.MATERIAL_AMBIGUITY,
        AmbiguityStateType.MISSING_REQUIRED_ROLE,
    }:
        reason = f"semantic_safety:{ambiguity.state.value.casefold()}"
        abstention = AbstentionState(abstained=True, stage="semantic_safety", reason=reason)
    return SemanticSafetyResult(
        ambiguity_state=ambiguity,
        unsupported_scope=unsupported,
        capability_assessment=capability,
        safety_warnings=_warning(ambiguity),
        abstention_state=abstention,
        abstention_reason=reason,
        applied_rule_ids=applied,
    )


__all__ = ["assess_capabilities", "run_semantic_safety"]
