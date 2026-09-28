"""M3.5 deterministic semantic-safety classification and capability gate."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .contracts import (
    AbstentionState,
    AmbiguityStateType,
    CapabilityAssessment,
    EvidenceSpan,
    SafetyWarning,
    SemanticSafetyResult,
    TypedAmbiguityState,
    UnsupportedScopeRecord,
    VisualizationIntentIR,
)
from .rules import RuleRegistry


_LEXICAL_RESIDUE = {
    "across",
    "against",
    "among",
    "but",
    "could",
    "dashboard",
    "give",
    "just",
    "kindly",
    "list",
    "overview",
    "per",
    "please",
    "quick",
    "quickly",
    "reference",
    "report",
    "simply",
    "where",
}

_MATERIAL_PATTERNS = (
    re.compile(r"\bimportant\b", re.IGNORECASE),
    re.compile(r"\bsignificant(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\binteresting\b", re.IGNORECASE),
    re.compile(r"\bmeaningful(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bkey\s+drivers?\b", re.IGNORECASE),
    re.compile(r"\bunusual(?:ly)?\b", re.IGNORECASE),
    re.compile(r"\bnormal\b", re.IGNORECASE),
)

# These phrases carry an analytical constraint that M1-M3 does not yet encode
# as a typed filter/time-unit transform.  They must not be dismissed merely
# because the remaining task and field roles are complete.
_UNRESOLVED_CONSTRAINT_PATTERNS = (
    (
        {"before", "after", "since", "until"},
        re.compile(r"\b(?:before|after|since|until)\b", re.IGNORECASE),
    ),
    (
        {"greater_than", "less_than"},
        re.compile(
            r"\b(?:above|below|greater|less|bigger|shorter)(?:\s+than)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        {"greater_than_or_equal", "less_than_or_equal"},
        re.compile(
            r"\b(?:at\s+least|at\s+most|no\s+less\s+than|no\s+more\s+than)\b",
            re.IGNORECASE,
        ),
    ),
    (
        {"contains"},
        re.compile(
            r"\b(?:include|includes|including|contain|contains|containing)\b",
            re.IGNORECASE,
        ),
    ),
    (
        {"on_or_after"},
        re.compile(
            r"\bfrom\s+(?:19|20)\d{2}\s+(?:onward|onwards)\b",
            re.IGNORECASE,
        ),
    ),
    (
        {"time_unit"},
        re.compile(
            r"\bbin(?:s|ned|ning)?\b.+\b(?:day|week|month|quarter|year|weekday)\b",
            re.IGNORECASE,
        ),
    ),
)

_UNSUPPORTED_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "forecast",
        (
            re.compile(r"\bforecast(?:s|ed|ing)?\b", re.IGNORECASE),
            re.compile(
                r"\bproject(?:ed|ion|ing)?\s+(?:future|next)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bpredict(?:s|ed|ing)?\b(?=.{0,30}\b(?:future|next\s+year|next\s+month)\b)",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "prediction",
        (
            re.compile(r"\bpredict(?:s|ed|ing)?\b", re.IGNORECASE),
            re.compile(r"\bprediction\b", re.IGNORECASE),
            re.compile(r"\bpredictive\b", re.IGNORECASE),
        ),
    ),
    (
        "causal_inference",
        (
            re.compile(r"\bcausal\b", re.IGNORECASE),
            re.compile(r"\bcause(?:s|d|ing)?\b", re.IGNORECASE),
            re.compile(r"\beffect\s+of\b", re.IGNORECASE),
        ),
    ),
    (
        "optimization",
        (
            re.compile(r"\boptimi[sz](?:e|es|ed|ing|ation)\b", re.IGNORECASE),
            re.compile(r"\boptimal\b", re.IGNORECASE),
            re.compile(r"\bmaximi[sz](?:e|es|ed|ing)\b", re.IGNORECASE),
            re.compile(r"\bminimi[sz](?:e|es|ed|ing)\b", re.IGNORECASE),
        ),
    ),
    (
        "simulation",
        (
            re.compile(r"\bsimulat(?:e|es|ed|ing|ion)\b", re.IGNORECASE),
            re.compile(r"\bwhat[- ]if\b", re.IGNORECASE),
            re.compile(r"\bscenario\b", re.IGNORECASE),
        ),
    ),
    (
        "network_analysis",
        (
            re.compile(r"\bnetwork(?:s)?\b", re.IGNORECASE),
            re.compile(r"\bcentrality\b", re.IGNORECASE),
            re.compile(r"\bdependencies\b", re.IGNORECASE),
        ),
    ),
    (
        "unsupported_join",
        (
            re.compile(r"\bjoin(?:s|ed|ing)?\b", re.IGNORECASE),
            re.compile(r"\bmerge(?:s|d|ing)?\b", re.IGNORECASE),
            re.compile(r"\bcombine\s+(?:the\s+)?tables?\b", re.IGNORECASE),
        ),
    ),
)


def _span(
    query: str,
    start: int,
    end: int,
    *,
    kind: str,
    rule_id: str,
) -> EvidenceSpan:
    return EvidenceSpan(
        start=start,
        end=end,
        text=query[start:end],
        normalized=query[start:end].casefold(),
        kind=kind,
        rule_ids=(rule_id,),
    )


def _full_query_span(intent: VisualizationIntentIR, kind: str) -> EvidenceSpan:
    return _span(
        intent.query,
        0,
        len(intent.query),
        kind=kind,
        rule_id="R-SA-002",
    )


def _dedupe_spans(spans: Iterable[EvidenceSpan]) -> tuple[EvidenceSpan, ...]:
    rows = {
        (
            span.start,
            span.end,
            span.kind,
            span.normalized,
            span.rule_ids,
        ): span
        for span in spans
    }
    return tuple(rows[key] for key in sorted(rows))


def _supported_operations(intent: VisualizationIntentIR) -> tuple[str, ...]:
    operations = {
        candidate.task
        for candidate in intent.task_candidates
        if candidate.task != "unresolved"
    }
    if intent.aggregation:
        operations.add(f"aggregate:{intent.aggregation.operation}")
    operations.update(f"filter:{item.operation}" for item in intent.filters)
    if intent.sort:
        operations.add(f"sort:{intent.sort.operation}")
    if intent.limit:
        operations.add("limit")
    return tuple(sorted(operations))


def assess_capabilities(
    intent: VisualizationIntentIR,
    registry: RuleRegistry,
) -> tuple[CapabilityAssessment, tuple[UnsupportedScopeRecord, ...], tuple[str, ...]]:
    """Assess the complete IR after task profiling; never changes task candidates."""

    query = intent.query
    detected: dict[str, list[EvidenceSpan]] = {}
    for operation, patterns in _UNSUPPORTED_PATTERNS:
        matches: list[EvidenceSpan] = []
        for pattern in patterns:
            for match in pattern.finditer(query):
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

    # A predictive verb with explicit future scope is forecasting, not two
    # separate capability violations.
    if "forecast" in detected and "prediction" in detected:
        detected.pop("prediction")

    supported = _supported_operations(intent)
    if detected:
        applied = registry.validate_ids({"R-SA-003", "R-SA-004"})
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
        operations = tuple(record.operation for record in records)
        critical = {"causal_inference", "optimization"}
        risk = "critical" if critical & set(operations) else "high"
        assessment = CapabilityAssessment(
            supported_operations=supported,
            unsupported_operations=operations,
            risk_level=risk,
            abstention_required=True,
            explanation=(
                "Unsupported governing operation detected; upstream supported "
                "task candidates are diagnostic only and may not reach planning."
            ),
            provenance=applied,
        )
        return assessment, records, applied

    applied = registry.validate_ids({"R-SA-005"})
    assessment = CapabilityAssessment(
        supported_operations=supported,
        unsupported_operations=(),
        risk_level="low",
        abstention_required=False,
        explanation=(
            "No unsupported governing operation was detected; upstream "
            "descriptive tasks are preserved unchanged."
        ),
        provenance=applied,
    )
    return assessment, (), applied


def _affected_fields(intent: VisualizationIntentIR) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                ref.field_name
                for ref in (
                    *intent.measures,
                    *intent.dimensions,
                    *intent.temporal_fields,
                    *intent.grouping_fields,
                )
            }
        )
    )


def _affected_tasks(intent: VisualizationIntentIR) -> tuple[str, ...]:
    return tuple(sorted({item.task for item in intent.task_candidates}))


def _measure_expression_count(intent: VisualizationIntentIR) -> int:
    """Count raw or derived analytical measures available to a task.

    M3's public IR currently stores one unambiguous aggregation directly and
    preserves multiple aggregate operations as typed transform alternatives.
    Count-without-field is a valid row-count measure.
    """

    raw_count = len({ref.field_name for ref in intent.measures})
    aggregate_transforms = tuple(
        item for item in intent.transforms if item.transform_type == "aggregate"
    )
    if aggregate_transforms:
        derived_count = sum(
            len(item.field_candidates)
            if item.field_candidates
            else (1 if item.operation == "count" else 0)
            for item in aggregate_transforms
        )
        if (
            intent.primary_task == "correlation"
            and len(aggregate_transforms) == 1
            and aggregate_transforms[0].operation == "count"
        ):
            return raw_count + derived_count
        return derived_count
    if intent.aggregation is not None:
        if intent.aggregation.field_candidates:
            derived_count = len(intent.aggregation.field_candidates)
            if intent.primary_task == "correlation" and (
                intent.aggregation.operation == "count"
            ):
                return raw_count + derived_count
            return derived_count
        return 1 if intent.aggregation.operation == "count" else 0
    transform_alternatives = tuple(
        item
        for item in intent.ambiguity
        if item.dimension == "transform" and item.alternatives
    )
    if transform_alternatives:
        operations = {
            operation
            for item in transform_alternatives
            for operation in item.alternatives
        }
        per_operation = raw_count or (1 if "count" in operations else 0)
        return len(operations) * per_operation
    return raw_count


def _missing_roles(intent: VisualizationIntentIR) -> tuple[str, ...]:
    task = intent.primary_task
    missing: list[str] = []
    measure_count = _measure_expression_count(intent)
    if task in {"ranking", "extremum"} and measure_count == 0:
        missing.append("measure")
    if task == "trend":
        if measure_count == 0:
            missing.append("measure")
        if not intent.temporal_fields:
            missing.append("temporal")
    if task == "correlation" and measure_count < 2:
        missing.append("second_measure")
    if task == "comparison":
        if measure_count == 0:
            missing.append("measure")
        elif measure_count == 1 and not intent.dimensions:
            missing.append("comparison_counterpart")
    if task == "composition":
        if measure_count == 0:
            missing.append("measure")
        if not intent.dimensions:
            missing.append("dimension")
    return tuple(sorted(set(missing)))


def classify_ambiguity(
    intent: VisualizationIntentIR,
    unsupported: tuple[UnsupportedScopeRecord, ...],
    registry: RuleRegistry,
) -> tuple[TypedAmbiguityState, tuple[str, ...]]:
    fields = _affected_fields(intent)
    tasks = _affected_tasks(intent)

    if unsupported:
        applied = registry.validate_ids({"R-SA-003", "R-SA-004"})
        spans = _dedupe_spans(
            span for item in unsupported for span in item.evidence_spans
        )
        return (
            TypedAmbiguityState(
                state=AmbiguityStateType.UNSUPPORTED_SCOPE,
                evidence_spans=spans,
                triggering_rules=applied,
                confidence=1.0,
                affected_fields=fields,
                affected_tasks=tasks,
                resolution_strategy="abstain_unsupported",
                explanation=(
                    "The requested governing operation is outside the declared "
                    "system capability boundary."
                ),
            ),
            applied,
        )

    material_spans = _dedupe_spans(
        _span(
            intent.query,
            match.start(),
            match.end(),
            kind="material_ambiguity",
            rule_id="R-SA-002",
        )
        for pattern in _MATERIAL_PATTERNS
        for match in pattern.finditer(intent.query)
    )
    if material_spans:
        applied = registry.validate_ids({"R-SA-002"})
        return (
            TypedAmbiguityState(
                state=AmbiguityStateType.MATERIAL_AMBIGUITY,
                evidence_spans=material_spans,
                triggering_rules=applied,
                confidence=0.9,
                affected_fields=fields,
                affected_tasks=tasks,
                resolution_strategy="request_clarification",
                explanation=(
                    "The query contains a qualitative criterion without a "
                    "declared threshold, baseline, or operational definition."
                ),
            ),
            applied,
        )

    # A directly observed material qualifier is the earliest actionable
    # clarification target.  Missing roles remain recorded in the upstream IR
    # and are evaluated immediately after the material criterion.
    missing = _missing_roles(intent)
    if missing:
        applied = registry.validate_ids({"R-SA-002"})
        return (
            TypedAmbiguityState(
                state=AmbiguityStateType.MISSING_REQUIRED_ROLE,
                evidence_spans=(_full_query_span(intent, "missing_required_role"),),
                triggering_rules=applied,
                confidence=0.95,
                affected_fields=tuple(sorted(set(fields) | set(missing))),
                affected_tasks=tasks,
                resolution_strategy="request_clarification",
                explanation=f"Required semantic roles are missing: {', '.join(missing)}.",
            ),
            applied,
        )

    represented_operations = {
        item.operation for item in intent.transforms
    } | {
        "time_unit"
        for item in intent.transforms
        if item.transform_type == "time_unit"
    }
    unresolved_constraint_spans = _dedupe_spans(
        _span(
            intent.query,
            match.start(),
            match.end(),
            kind="unresolved_constraint",
            rule_id="R-SA-002",
        )
        for required_operations, pattern in _UNRESOLVED_CONSTRAINT_PATTERNS
        if not required_operations & represented_operations
        for match in pattern.finditer(intent.query)
    )
    if unresolved_constraint_spans:
        applied = registry.validate_ids({"R-SA-002"})
        return (
            TypedAmbiguityState(
                state=AmbiguityStateType.MATERIAL_AMBIGUITY,
                evidence_spans=unresolved_constraint_spans,
                triggering_rules=applied,
                confidence=0.9,
                affected_fields=fields,
                affected_tasks=tasks,
                resolution_strategy="request_clarification",
                explanation=(
                    "A query constraint is not represented by a typed upstream "
                    "transform and could materially change the result."
                ),
            ),
            applied,
        )

    semantic_alternatives = tuple(
        item
        for item in intent.ambiguity
        if item.dimension in {"field", "task", "transform", "relation"}
    )
    if semantic_alternatives:
        applied = registry.validate_ids({"R-SA-002"})
        legacy_rules = {
            rule_id
            for item in semantic_alternatives
            for rule_id in item.rule_ids
        }
        return (
            TypedAmbiguityState(
                state=AmbiguityStateType.MULTIPLE_VALID_INTERPRETATIONS,
                evidence_spans=(
                    _full_query_span(intent, "multiple_valid_interpretations"),
                ),
                triggering_rules=tuple(sorted(set(applied) | legacy_rules)),
                confidence=0.9,
                affected_fields=fields,
                affected_tasks=tasks,
                resolution_strategy="preserve_alternatives",
                explanation=(
                    "Multiple semantically distinct upstream alternatives remain "
                    "valid and are preserved without selecting one."
                ),
            ),
            applied,
        )

    unresolved = tuple(intent.unresolved_mentions)
    if unresolved:
        applied = registry.validate_ids({"R-SA-001"})
        spans = tuple(
            _span(
                intent.query,
                mention.start,
                mention.end,
                kind="lexical_residue",
                rule_id="R-SA-001",
            )
            for mention in unresolved
        )
        return (
            TypedAmbiguityState(
                state=AmbiguityStateType.LEXICAL_RESIDUE,
                evidence_spans=_dedupe_spans(spans),
                triggering_rules=applied,
                confidence=(
                    0.95
                    if all(
                        mention.normalized in _LEXICAL_RESIDUE
                        for mention in unresolved
                    )
                    else 0.8
                ),
                affected_fields=(),
                affected_tasks=(),
                resolution_strategy="ignore_lexical_residue",
                explanation=(
                    "All required task/field roles are complete and no unresolved "
                    "analytical constraint remains; residual tokens cannot change "
                    "the current typed interpretation."
                ),
            ),
            applied,
        )

    applied = registry.validate_ids({"R-SA-005"})
    return (
        TypedAmbiguityState(
            state=AmbiguityStateType.NONE,
            evidence_spans=(),
            triggering_rules=applied,
            confidence=1.0,
            affected_fields=(),
            affected_tasks=(),
            resolution_strategy="proceed",
            explanation="No material semantic ambiguity was detected.",
        ),
        applied,
    )


def _warning(state: TypedAmbiguityState) -> tuple[SafetyWarning, ...]:
    if state.state == AmbiguityStateType.NONE:
        return ()
    severity = {
        AmbiguityStateType.LEXICAL_RESIDUE: "info",
        AmbiguityStateType.MULTIPLE_VALID_INTERPRETATIONS: "warning",
        AmbiguityStateType.MATERIAL_AMBIGUITY: "error",
        AmbiguityStateType.MISSING_REQUIRED_ROLE: "error",
        AmbiguityStateType.UNSUPPORTED_SCOPE: "critical",
    }[state.state]
    warning_type = state.state.value.casefold()
    return (
        SafetyWarning(
            warning_type=warning_type,
            severity=severity,
            message=state.explanation,
            evidence_spans=state.evidence_spans,
            rule_ids=state.triggering_rules,
        ),
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
        abstention = AbstentionState(
            abstained=True,
            stage="semantic_safety",
            reason=reason,
        )
    elif ambiguity.state in {
        AmbiguityStateType.MATERIAL_AMBIGUITY,
        AmbiguityStateType.MISSING_REQUIRED_ROLE,
    }:
        reason = f"semantic_safety:{ambiguity.state.value.casefold()}"
        abstention = AbstentionState(
            abstained=True,
            stage="semantic_safety",
            reason=reason,
        )

    return SemanticSafetyResult(
        ambiguity_state=ambiguity,
        unsupported_scope=unsupported,
        capability_assessment=capability,
        safety_warnings=_warning(ambiguity),
        abstention_state=abstention,
        abstention_reason=reason,
        applied_rule_ids=applied,
    )
