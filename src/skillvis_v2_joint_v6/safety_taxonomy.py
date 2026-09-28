"""Canonical unsupported-operation taxonomy with lexical provenance."""

from __future__ import annotations

import re

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
from skillvis_v2_accuracy.semantic_safety import _warning, classify_ambiguity
from skillvis_v2_obligation.semantic_safety import assess_capabilities as assess_parent


_PREDICTION = re.compile(r"\b(?:predict(?:s|ed|ing)?|prediction|predictive)\b", re.I)
_FORECAST = re.compile(r"\bforecast(?:s|ed|ing)?\b", re.I)
_PROJECT_FUTURE = re.compile(
    r"\bproject(?:s|ed|ing|ion)?\b(?=.{0,30}\b(?:future|next)\b)", re.I
)
_EXPLICIT_GROUNDING = {"exact", "normalized_lexical", "alias", "acronym_expansion", "value"}


def _field_spans(intent: VisualizationIntentIR) -> tuple[EvidenceSpan, ...]:
    return tuple(
        evidence.query_span
        for ref in (*intent.measures, *intent.dimensions, *intent.temporal_fields, *intent.grouping_fields)
        for evidence in ref.evidence
        if evidence.query_span is not None and evidence.evidence_type in _EXPLICIT_GROUNDING
    )


def _covered(start: int, end: int, spans: tuple[EvidenceSpan, ...]) -> bool:
    return any(span.start <= start and end <= span.end for span in spans)


def _canonical_predictive_record(intent: VisualizationIntentIR) -> UnsupportedScopeRecord | None:
    spans = _field_spans(intent)
    for operation, family, pattern in (
        ("forecast", "predictive_modeling", _FORECAST),
        ("prediction", "predictive_modeling", _PREDICTION),
        ("forecast", "predictive_modeling", _PROJECT_FUTURE),
    ):
        match = next(
            (
                item
                for item in pattern.finditer(intent.query)
                if not _covered(item.start(), item.end(), spans)
            ),
            None,
        )
        if match is None:
            continue
        span = EvidenceSpan(
            start=match.start(),
            end=match.end(),
            text=intent.query[match.start():match.end()],
            normalized=intent.query[match.start():match.end()].casefold(),
            kind=f"unsupported_{operation}",
            rule_ids=("R-SA-020",),
        )
        return UnsupportedScopeRecord(
            operation=operation,
            evidence_spans=(span,),
            triggering_rules=("R-SA-020",),
            explanation=(
                f"canonical={operation}; family={family}; "
                f"lexical_form={span.normalized}; unsupported by the versioned deterministic planner"
            ),
        )
    return None


def assess_capabilities_v15(
    intent: VisualizationIntentIR,
    registry: RuleRegistry,
) -> tuple[CapabilityAssessment, tuple[UnsupportedScopeRecord, ...], tuple[str, ...]]:
    parent, records, parent_rules = assess_parent(intent, registry)
    canonical = _canonical_predictive_record(intent)
    if canonical is None:
        return parent, records, parent_rules
    retained = tuple(
        item for item in records if item.operation not in {"forecast", "prediction"}
    )
    records = tuple(sorted((*retained, canonical), key=lambda item: item.operation))
    operations = tuple(item.operation for item in records)
    rules = registry.validate_ids((*parent_rules, "R-SA-020"))
    risk = "critical" if {"causal_inference", "optimization"} & set(operations) else "high"
    return (
        CapabilityAssessment(
            supported_operations=parent.supported_operations,
            unsupported_operations=operations,
            risk_level=risk,
            abstention_required=True,
            explanation=(
                "Unsupported governing operation classified by the Joint v1.5 "
                "canonical taxonomy; lexical provenance is retained in each record."
            ),
            provenance=rules,
        ),
        records,
        rules,
    )


def run_semantic_safety_v15(
    intent: VisualizationIntentIR,
    registry: RuleRegistry,
) -> SemanticSafetyResult:
    capability, unsupported, capability_rules = assess_capabilities_v15(intent, registry)
    ambiguity, ambiguity_rules = classify_ambiguity(intent, unsupported, registry)
    applied = registry.validate_ids((*capability_rules, *ambiguity_rules))
    abstention = intent.abstention_state
    reason = intent.abstention_reason or abstention.reason
    if capability.abstention_required:
        reason = f"unsupported_scope:{','.join(capability.unsupported_operations)}"
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


__all__ = ["assess_capabilities_v15", "run_semantic_safety_v15"]
