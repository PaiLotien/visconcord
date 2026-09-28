"""M3 Analytical Task Profiling and deterministic query decomposition."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .config import SkillVISV2Config
from .contracts import (
    AmbiguityRecord,
    AnalyticalTaskResult,
    EvidenceSpan,
    FieldCandidate,
    FieldResolutionResult,
    QueryNormalizationResult,
    SchemaProfile,
    TaskCandidate,
    TransformIntent,
)
from .rules import RuleRegistry


_TASK_ORDER = {
    "extremum": 0,
    "ranking": 1,
    "trend": 2,
    "correlation": 3,
    "distribution": 4,
    "composition": 5,
    "comparison": 6,
    "aggregation": 7,
    "filtering": 8,
    "sorting": 9,
    "temporal_analysis": 10,
    "unresolved": 11,
}


def _term_span(term, kind: str) -> EvidenceSpan:
    return EvidenceSpan(
        start=term.start,
        end=term.end,
        text=term.text,
        normalized=term.canonical,
        kind=kind,
        rule_ids=term.rule_ids,
    )


def _dedupe_spans(spans: Iterable[EvidenceSpan]) -> tuple[EvidenceSpan, ...]:
    unique = {
        (span.start, span.end, span.kind, span.normalized, span.rule_ids): span
        for span in spans
    }
    return tuple(unique[key] for key in sorted(unique))


def _field_spans(candidate: FieldCandidate) -> tuple[EvidenceSpan, ...]:
    return tuple(
        evidence.query_span
        for evidence in candidate.evidence
        if evidence.query_span is not None
    )


def _field_names_by_role(
    fields: FieldResolutionResult,
    role: str,
) -> tuple[str, ...]:
    return tuple(
        candidate.field_name
        for candidate in fields.candidate_fields
        if role in candidate.intended_roles
        and candidate.field_name in fields.required_field_candidates
    )


def _nearest_aggregate_fields(
    term,
    candidates: tuple[FieldCandidate, ...],
) -> tuple[str, ...]:
    """Bind an aggregate phrase to the nearest grounded quantitative field.

    COUNT is allowed to remain fieldless and then means row count.  Other
    aggregate operators require a quantitative field.  The association is a
    deterministic query-span decision, not a schema-name guess.
    """

    distances: list[tuple[int, int, str]] = []
    for candidate in candidates:
        for span in _field_spans(candidate):
            if span.start >= term.end:
                distance = span.start - term.end
                direction = 0
            elif span.end <= term.start:
                distance = term.start - span.end
                direction = 1
            else:
                distance = 0
                direction = 0
            distances.append((distance, direction, candidate.field_name))
    if not distances:
        return ()
    best = min(distances)
    # A distant field is not silently attached to an operator.  This matters
    # for COUNT-by-dimension queries, where fieldless COUNT is the correct role.
    if best[0] > 32:
        return ()
    return (best[2],)


def _nearest_fields_for_span(
    start: int,
    end: int,
    fields: FieldResolutionResult,
    schema: SchemaProfile,
    allowed_types: set[str],
) -> tuple[str, ...]:
    required = set(fields.required_field_candidates)
    profile_by_name = {item.field_name: item for item in schema.fields}
    distances: list[tuple[int, int, str]] = []
    for candidate in fields.candidate_fields:
        if candidate.field_name not in required:
            continue
        if profile_by_name[candidate.field_name].semantic_type not in allowed_types:
            continue
        for span in _field_spans(candidate):
            if span.end <= start:
                distance, direction = start - span.end, 0
            elif span.start >= end:
                distance, direction = span.start - end, 1
            else:
                distance, direction = 0, 0
            distances.append((distance, direction, candidate.field_name))
    if not distances:
        return ()
    best_distance, best_direction, _ = min(distances)
    return tuple(
        sorted(
            {
                field_name
                for distance, direction, field_name in distances
                if (distance, direction) == (best_distance, best_direction)
            }
        )
    )


def _near_negation(
    normalized: QueryNormalizationResult,
    start: int,
) -> bool:
    return any(
        term.end <= start and start - term.end <= 15
        for term in normalized.limitation_terms
        if term.canonical in {"negation", "exclusion"}
    )


def _build_transforms(
    normalized: QueryNormalizationResult,
    fields: FieldResolutionResult,
    schema: SchemaProfile,
    registry: RuleRegistry,
) -> tuple[
    tuple[TransformIntent, ...],
    TransformIntent | None,
    tuple[TransformIntent, ...],
    TransformIntent | None,
    int | None,
    tuple[str, ...],
    tuple[AmbiguityRecord, ...],
    set[str],
]:
    transforms: list[TransformIntent] = []
    ambiguity: list[AmbiguityRecord] = []
    applied: set[str] = set()
    measures = _field_names_by_role(fields, "measure")
    measure_candidates = tuple(
        candidate
        for candidate in fields.candidate_fields
        if candidate.field_name in measures
    )

    aggregate_ops = tuple(
        dict.fromkeys(term.canonical for term in normalized.aggregation_terms)
    )
    aggregation = None
    if aggregate_ops:
        applied.add("R-QD-001")
        for operation in aggregate_ops:
            operation_terms = tuple(
                term
                for term in normalized.aggregation_terms
                if term.canonical == operation
            )
            related = tuple(
                _term_span(term, "aggregation") for term in operation_terms
            )
            associated_fields = tuple(
                dict.fromkeys(
                    field_name
                    for term in operation_terms
                    for field_name in _nearest_aggregate_fields(
                        term, measure_candidates
                    )
                )
            )
            if not associated_fields and operation != "count" and len(measures) == 1:
                associated_fields = measures
            effective_operation = operation
            if operation == "sum" and not associated_fields and not measures:
                # "total submissions/products" is a frequency request when no
                # quantitative field exists; SUM without an operand is invalid.
                effective_operation = "count"
            transforms.append(
                TransformIntent(
                    transform_type="aggregate",
                    operation=effective_operation,
                    field_candidates=associated_fields,
                    evidence_spans=related,
                    rule_ids=("R-QD-001",),
                )
            )
        if len(aggregate_ops) == 1:
            aggregation = transforms[-1]
        elif any(
            not item.field_candidates and item.operation != "count"
            for item in transforms
            if item.transform_type == "aggregate"
        ):
            ambiguity.append(
                AmbiguityRecord(
                    dimension="transform",
                    alternatives=tuple(aggregate_ops),
                    reason="multiple distinct aggregation operations are explicit",
                    rule_ids=("R-QD-001",),
                )
            )

    if not aggregate_ops:
        chart_hints = {term.canonical for term in normalized.chart_mentions}
        task_hints = {term.canonical for term in normalized.task_terms}
        implicit_count = (
            ("composition" in task_hints and not measures)
            or ("line" in chart_hints and not measures)
        )
        if implicit_count:
            applied.add("R-QD-001")
            evidence_terms = (
                tuple(normalized.task_terms)
                if "composition" in task_hints
                else tuple(normalized.chart_mentions)
            )
            aggregation = TransformIntent(
                transform_type="aggregate",
                operation="count",
                field_candidates=(),
                evidence_spans=tuple(
                    _term_span(term, "implicit_count") for term in evidence_terms
                ),
                rule_ids=("R-QD-001",),
            )
            transforms.append(aggregation)

    directions = sorted(
        {
            term.canonical
            for term in normalized.ranking_terms
            if term.canonical in {"ascending", "descending"}
        }
    )
    sort = None
    if directions:
        applied.update({"R-QD-001", "R-QD-003"})
        if len(directions) == 1:
            sort = TransformIntent(
                transform_type="sort",
                operation=directions[0],
                field_candidates=measures,
                evidence_spans=tuple(
                    _term_span(term, "ranking")
                    for term in normalized.ranking_terms
                    if term.canonical == directions[0]
                ),
                rule_ids=("R-QD-001", "R-QD-003"),
            )
            transforms.append(sort)
        else:
            ambiguity.append(
                AmbiguityRecord(
                    dimension="transform",
                    alternatives=tuple(directions),
                    reason="opposing sort directions are explicit",
                    rule_ids=("R-QD-001", "R-QD-003"),
                )
            )

    limit = None
    limit_match = re.search(
        r"\b(?:top|bottom)\s+(\d+)\b", normalized.normalized_query
    )
    if limit_match:
        limit = max(1, int(limit_match.group(1)))
        applied.add("R-QD-003")
    elif normalized.detected_superlatives:
        limit = 1
        applied.add("R-QD-003")
    if limit is not None:
        span = next(
            (
                _term_span(term, "limit")
                for term in normalized.ranking_terms
                if term.canonical in {"ascending", "descending"}
            ),
            EvidenceSpan(
                start=0,
                end=len(normalized.original_query),
                text=normalized.original_query,
                normalized=normalized.normalized_query,
                kind="limit_context",
                rule_ids=("R-QD-003",),
            ),
        )
        transforms.append(
            TransformIntent(
                transform_type="limit",
                operation="first_n",
                value=limit,
                evidence_spans=(span,),
                rule_ids=("R-QD-003",),
            )
        )

    filters: list[TransformIntent] = []
    for candidate in fields.candidate_fields:
        if "value" not in candidate.match_types:
            continue
        for evidence in candidate.evidence:
            if evidence.evidence_type != "value" or evidence.query_span is None:
                continue
            applied.update({"R-QD-001", "R-QD-003"})
            item = TransformIntent(
                transform_type="filter",
                operation="not_equal" if _near_negation(normalized, evidence.query_span.start) else "equal",
                field_candidates=(candidate.field_name,),
                value=evidence.value,
                negated=_near_negation(normalized, evidence.query_span.start),
                evidence_spans=(evidence.query_span,),
                rule_ids=("R-QD-001", "R-QD-003"),
            )
            filters.append(item)
            transforms.append(item)

    query = normalized.original_query
    numeric_filter_pattern = re.compile(
        r"\b(?P<operator>at\s+least|at\s+most|no\s+less\s+than|"
        r"no\s+more\s+than|above|below|greater\s+than|less\s+than|"
        r"bigger\s+than|shorter\s+than)\s+"
        r"(?P<value>-?\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )
    for match in numeric_filter_pattern.finditer(query):
        fields_for_filter = _nearest_fields_for_span(
            match.start(),
            match.end(),
            fields,
            schema,
            {"quantitative", "ordinal"},
        )
        if not fields_for_filter:
            continue
        operator = match.group("operator").casefold()
        if operator in {"at least", "no less than"}:
            operation = "greater_than_or_equal"
        elif operator in {"at most", "no more than"}:
            operation = "less_than_or_equal"
        elif operator in {"above", "greater than", "bigger than"}:
            operation = "greater_than"
        else:
            operation = "less_than"
        value_text = match.group("value")
        value = float(value_text) if "." in value_text else int(value_text)
        span = EvidenceSpan(
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            normalized=match.group(0).casefold(),
            kind="numeric_filter",
            rule_ids=("R-QD-001", "R-QD-003"),
        )
        item = TransformIntent(
            transform_type="filter",
            operation=operation,
            field_candidates=fields_for_filter,
            value=value,
            evidence_spans=(span,),
            rule_ids=("R-QD-001", "R-QD-003"),
        )
        filters.append(item)
        transforms.append(item)
        applied.update({"R-QD-001", "R-QD-003", "R-QD-004"})

    temporal_filter_pattern = re.compile(
        r"\b(?P<operator>before|after|since|until)\s+"
        r"(?P<value>(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?)",
        re.IGNORECASE,
    )
    for match in temporal_filter_pattern.finditer(query):
        fields_for_filter = _nearest_fields_for_span(
            match.start(), match.end(), fields, schema, {"temporal"}
        )
        if not fields_for_filter:
            continue
        operation = match.group("operator").casefold()
        span = EvidenceSpan(
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            normalized=match.group(0).casefold(),
            kind="temporal_filter",
            rule_ids=("R-QD-001", "R-QD-003"),
        )
        item = TransformIntent(
            transform_type="filter",
            operation=operation,
            field_candidates=fields_for_filter,
            value=match.group("value"),
            evidence_spans=(span,),
            rule_ids=("R-QD-001", "R-QD-003"),
        )
        filters.append(item)
        transforms.append(item)
        applied.update({"R-QD-001", "R-QD-003"})

    contains_pattern = re.compile(
        r"\b(?:include|includes|including|contain|contains|containing)\s+"
        r"(?:the\s+)?letter\s+['\"]?(?P<value>[A-Za-z])['\"]?",
        re.IGNORECASE,
    )
    for match in contains_pattern.finditer(query):
        fields_for_filter = _nearest_fields_for_span(
            match.start(),
            match.end(),
            fields,
            schema,
            {"categorical", "ordinal", "identifier", "geographic"},
        )
        if not fields_for_filter:
            continue
        span = EvidenceSpan(
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            normalized=match.group(0).casefold(),
            kind="contains_filter",
            rule_ids=("R-QD-001", "R-QD-003"),
        )
        item = TransformIntent(
            transform_type="filter",
            operation="contains",
            field_candidates=fields_for_filter,
            value=match.group("value"),
            evidence_spans=(span,),
            rule_ids=("R-QD-001", "R-QD-003"),
        )
        filters.append(item)
        transforms.append(item)
        applied.update({"R-QD-001", "R-QD-003"})

    onward_pattern = re.compile(
        r"\bfrom\s+(?P<value>(?:19|20)\d{2})\s+(?:onward|onwards)\b",
        re.IGNORECASE,
    )
    for match in onward_pattern.finditer(query):
        fields_for_filter = _nearest_fields_for_span(
            match.start(), match.end(), fields, schema, {"temporal"}
        )
        if not fields_for_filter:
            continue
        span = EvidenceSpan(
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            normalized=match.group(0).casefold(),
            kind="temporal_filter",
            rule_ids=("R-QD-001", "R-QD-003"),
        )
        item = TransformIntent(
            transform_type="filter",
            operation="on_or_after",
            field_candidates=fields_for_filter,
            value=match.group("value"),
            evidence_spans=(span,),
            rule_ids=("R-QD-001", "R-QD-003"),
        )
        filters.append(item)
        transforms.append(item)
        applied.update({"R-QD-001", "R-QD-003"})

    time_unit_pattern = re.compile(
        r"\bbin(?:s|ned|ning)?\b.{0,60}?\b(?:into|by)\s+(?:the\s+)?"
        r"(?P<unit>weekday|day|week|month|quarter|year)(?:\s+interval)?\b",
        re.IGNORECASE,
    )
    for match in time_unit_pattern.finditer(query):
        fields_for_transform = _nearest_fields_for_span(
            match.start(), match.end(), fields, schema, {"temporal"}
        )
        if not fields_for_transform:
            continue
        span = EvidenceSpan(
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            normalized=match.group(0).casefold(),
            kind="time_unit",
            rule_ids=("R-QD-001",),
        )
        transforms.append(
            TransformIntent(
                transform_type="time_unit",
                operation=match.group("unit").casefold(),
                field_candidates=fields_for_transform,
                evidence_spans=(span,),
                rule_ids=("R-QD-001",),
            )
        )
        applied.add("R-QD-001")

    temporal_scope: list[str] = []
    for match in re.finditer(r"\b(?:19|20)\d{2}\b", normalized.original_query):
        temporal_scope.append(match.group(0))
        applied.add("R-QD-003")
        span = EvidenceSpan(
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            normalized=match.group(0),
            kind="temporal_scope",
            rule_ids=("R-QD-003",),
        )
        transforms.append(
            TransformIntent(
                transform_type="filter",
                operation="within_year",
                value=match.group(0),
                evidence_spans=(span,),
                rule_ids=("R-QD-003",),
            )
        )

    return (
        tuple(transforms),
        aggregation,
        tuple(filters),
        sort,
        limit,
        tuple(sorted(set(temporal_scope))),
        tuple(ambiguity),
        applied,
    )


def run_task_profiling(
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    config: SkillVISV2Config,
    registry: RuleRegistry,
) -> AnalyticalTaskResult:
    if not config.feature_flags.enable_m3_task_profiling:
        raise RuntimeError("M3 task profiling feature flag is disabled")

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    spans: dict[str, list[EvidenceSpan]] = defaultdict(list)
    rules: dict[str, set[str]] = defaultdict(set)
    roles: dict[str, set[str]] = defaultdict(set)
    applied: set[str] = {"R-TP-001"}

    for term in normalized.task_terms:
        task = term.canonical
        if task not in _TASK_ORDER:
            continue
        scores[task]["explicit_phrase"] = max(
            0.85, scores[task].get("explicit_phrase", 0.0)
        )
        spans[task].append(_term_span(term, "task"))
        rules[task].add("R-TP-004")
        applied.add("R-TP-004")

    analytical_ranking_terms = tuple(
        term
        for term in normalized.ranking_terms
        if term.text.casefold()
        not in {"ascending", "descending", "asc", "desc"}
    )
    for term in analytical_ranking_terms:
        scores["ranking"]["degree_direction"] = 0.88
        spans["ranking"].append(_term_span(term, "ranking"))
        rules["ranking"].add("R-TP-005")
        roles["ranking"].update({"dimension", "measure"})
        applied.add("R-TP-005")
    extremum_terms = tuple(
        term
        for term in normalized.detected_superlatives
        if term.text.casefold() not in {"top", "bottom"}
    )
    if extremum_terms:
        scores["extremum"]["superlative"] = 0.95
        scores["ranking"]["superlative"] = 0.88
        for term in extremum_terms:
            evidence = _term_span(term, "superlative")
            spans["extremum"].append(evidence)
            spans["ranking"].append(evidence)
        rules["extremum"].add("R-TP-005")
        rules["ranking"].add("R-TP-005")
        roles["extremum"].update({"dimension", "measure"})
        roles["ranking"].update({"dimension", "measure"})
        applied.add("R-TP-005")

    if normalized.detected_comparatives:
        scores["comparison"]["comparative"] = 0.82
        spans["comparison"].extend(
            _term_span(term, "comparative")
            for term in normalized.detected_comparatives
        )
        rules["comparison"].add("R-TP-004")
        roles["comparison"].update({"measure"})
        applied.add("R-TP-004")

    required_fields = set(fields.required_field_candidates)
    if required_fields:
        # Field-role assignment is an explicit decomposition decision. Record
        # it even though candidate generation happened upstream.
        applied.add("R-QD-002")
    profile_by_name = {item.field_name: item for item in schema.fields}
    measure_candidates = [
        item
        for item in fields.candidate_fields
        if item.field_name in required_fields
        and profile_by_name[item.field_name].semantic_type == "quantitative"
    ]
    temporal_candidates = [
        item
        for item in fields.candidate_fields
        if item.field_name in required_fields
        and profile_by_name[item.field_name].semantic_type == "temporal"
    ]
    dimension_candidates = [
        item
        for item in fields.candidate_fields
        if item.field_name in required_fields
        and profile_by_name[item.field_name].semantic_type
        in {"categorical", "ordinal", "geographic", "identifier"}
    ]

    (
        transforms,
        aggregation,
        filters,
        sort,
        limit,
        temporal_scope,
        transform_ambiguity,
        decomposition_rules,
    ) = _build_transforms(normalized, fields, schema, registry)
    aggregate_transforms = tuple(
        item for item in transforms if item.transform_type == "aggregate"
    )
    aggregate_expression_count = sum(
        len(item.field_candidates)
        if item.field_candidates
        else (1 if item.operation == "count" else 0)
        for item in aggregate_transforms
    )
    raw_measure_count = len(measure_candidates)
    measure_expression_count = (
        aggregate_expression_count
        if aggregate_transforms
        else raw_measure_count
    )
    has_measure_expression = measure_expression_count > 0

    if (
        normalized.temporal_expressions
        and temporal_candidates
        and has_measure_expression
    ):
        scores["trend"]["temporal_structure"] = 0.9
        scores["temporal_analysis"]["temporal_structure"] = 0.72
        temporal_spans = [
            _term_span(term, "temporal") for term in normalized.temporal_expressions
        ]
        spans["trend"].extend(temporal_spans)
        spans["temporal_analysis"].extend(temporal_spans)
        for candidate in (*temporal_candidates, *measure_candidates):
            spans["trend"].extend(_field_spans(candidate))
        rules["trend"].add("R-TP-002")
        rules["temporal_analysis"].add("R-TP-002")
        roles["trend"].update({"temporal", "measure"})
        roles["temporal_analysis"].update({"temporal", "measure"})
        applied.add("R-TP-002")

    if normalized.aggregation_terms:
        scores["aggregation"]["explicit_operation"] = 0.76
        spans["aggregation"].extend(
            _term_span(term, "aggregation") for term in normalized.aggregation_terms
        )
        rules["aggregation"].add("R-TP-004")
        roles["aggregation"].add("measure")
        applied.add("R-TP-004")

    if any("value" in item.match_types for item in fields.candidate_fields):
        scores["filtering"]["value_grounding"] = 0.7
        for candidate in fields.candidate_fields:
            if "value" in candidate.match_types:
                spans["filtering"].extend(_field_spans(candidate))
        rules["filtering"].add("R-TP-001")
        roles["filtering"].add("filter")

    if normalized.ranking_terms:
        scores["sorting"]["direction"] = 0.7
        spans["sorting"].extend(
            _term_span(term, "sorting") for term in normalized.ranking_terms
        )
        rules["sorting"].add("R-TP-005")

    # An explicit chart idiom is supporting evidence only when compatible field
    # structure is also present. It never maps to a task by itself.
    chart_hints = {term.canonical for term in normalized.chart_mentions}
    scatter_expression_count = measure_expression_count
    if (
        len(aggregate_transforms) == 1
        and aggregate_transforms[0].operation == "count"
        and measure_candidates
    ):
        # "age versus number of rows at each age" is a relation between the raw
        # value and its derived frequency, not a missing second measure.
        scatter_expression_count += raw_measure_count
    if "scatter" in chart_hints and scatter_expression_count >= 2:
        scores["correlation"]["chart_and_measure_structure"] = 0.82
        rules["correlation"].add("R-TP-002")
        roles["correlation"].add("measure")
        spans["correlation"].extend(
            _term_span(term, "chart_support")
            for term in normalized.chart_mentions
            if term.canonical == "scatter"
        )
        applied.add("R-TP-002")
    if chart_hints & {"histogram", "boxplot"} and has_measure_expression:
        scores["distribution"]["chart_and_measure_structure"] = 0.82
        rules["distribution"].add("R-TP-002")
        roles["distribution"].add("measure")
        spans["distribution"].extend(
            _term_span(term, "chart_support")
            for term in normalized.chart_mentions
            if term.canonical in {"histogram", "boxplot"}
        )
        applied.add("R-TP-002")
    composition_dimensions = (*dimension_candidates, *temporal_candidates)
    if "pie" in chart_hints and composition_dimensions and has_measure_expression:
        scores["composition"]["chart_and_dimension_structure"] = 0.8
        rules["composition"].add("R-TP-002")
        roles["composition"].update({"dimension", "measure"})
        spans["composition"].extend(
            _term_span(term, "chart_support")
            for term in normalized.chart_mentions
            if term.canonical == "pie"
        )
        applied.add("R-TP-002")
    if (
        "bar" in chart_hints
        and dimension_candidates
        and has_measure_expression
    ):
        scores["comparison"]["chart_dimension_aggregate_structure"] = 0.68
        rules["comparison"].add("R-TP-002")
        roles["comparison"].add("dimension")
        spans["comparison"].extend(
            _term_span(term, "chart_support")
            for term in normalized.chart_mentions
            if term.canonical == "bar"
        )
        applied.add("R-TP-002")

    if "line" in chart_hints and temporal_candidates and has_measure_expression:
        scores["trend"]["chart_temporal_measure_structure"] = max(
            0.84,
            scores["trend"].get("chart_temporal_measure_structure", 0.0),
        )
        rules["trend"].add("R-TP-002")
        roles["trend"].update({"temporal", "measure"})
        spans["trend"].extend(
            _term_span(term, "chart_support")
            for term in normalized.chart_mentions
            if term.canonical == "line"
        )
        applied.add("R-TP-002")

    primary_analysis_tasks = {
        "comparison", "ranking", "trend", "correlation",
        "distribution", "composition", "extremum",
    }
    has_primary_analysis = any(task_name in scores for task_name in primary_analysis_tasks)
    if (
        not has_primary_analysis
        and measure_candidates
        and dimension_candidates
        and "filtering" not in scores
    ):
        score = 0.78 if normalized.aggregation_terms else 0.58
        scores["comparison"]["measure_dimension_structure"] = score
        rules["comparison"].add("R-TP-001")
        roles["comparison"].update({"measure", "dimension"})
        for candidate in (*measure_candidates, *dimension_candidates):
            spans["comparison"].extend(_field_spans(candidate))
        has_primary_analysis = True

    explicit_primary = bool(scores)
    ambiguity: list[AmbiguityRecord] = list(fields.ambiguity)
    if not explicit_primary and required_fields:
        applied.add("R-TP-003")
        if len(measure_candidates) == 1 and not dimension_candidates:
            scores["distribution"]["underspecified_structure"] = 0.5
            scores["aggregation"]["underspecified_structure"] = max(
                scores["aggregation"].get("underspecified_structure", 0.0), 0.48
            )
            rules["distribution"].add("R-TP-003")
            rules["aggregation"].add("R-TP-003")
            roles["distribution"].add("measure")
            roles["aggregation"].add("measure")
            ambiguity.append(
                AmbiguityRecord(
                    dimension="task",
                    alternatives=("distribution", "aggregation"),
                    reason="a bare quantitative field does not determine the analytical goal",
                    rule_ids=("R-TP-003",),
                )
            )
        elif len(measure_candidates) >= 2 and not dimension_candidates:
            scores["correlation"]["underspecified_structure"] = 0.52
            scores["distribution"]["underspecified_structure"] = 0.48
            rules["correlation"].add("R-TP-003")
            rules["distribution"].add("R-TP-003")
            roles["correlation"].add("measure")
            roles["distribution"].add("measure")
            ambiguity.append(
                AmbiguityRecord(
                    dimension="task",
                    alternatives=("correlation", "distribution"),
                    reason="multiple bare measures support relationship or distribution analyses",
                    rule_ids=("R-TP-003",),
                )
            )
        elif dimension_candidates and not measure_candidates:
            scores["composition"]["underspecified_structure"] = 0.48
            scores["distribution"]["underspecified_structure"] = 0.48
            rules["composition"].add("R-TP-003")
            rules["distribution"].add("R-TP-003")
            roles["composition"].add("dimension")
            roles["distribution"].add("dimension")
            ambiguity.append(
                AmbiguityRecord(
                    dimension="task",
                    alternatives=("composition", "distribution"),
                    reason="a bare dimension supports composition or frequency distribution",
                    rule_ids=("R-TP-003",),
                )
            )

    ambiguity.extend(transform_ambiguity)
    applied.update(decomposition_rules)

    candidates: list[TaskCandidate] = []
    for task, components in scores.items():
        score = max(components.values()) if components else 0.0
        if score < config.task_candidate_threshold:
            continue
        task_rules = rules[task] or {"R-TP-001"}
        candidates.append(
            TaskCandidate(
                task=task,
                score=round(score, 6),
                evidence_spans=_dedupe_spans(spans[task]),
                required_roles=tuple(sorted(roles[task])),
                rule_ids=registry.validate_ids(task_rules),
                score_components={
                    key: round(value, 6) for key, value in sorted(components.items())
                },
            )
        )

    if not candidates:
        candidates.append(
            TaskCandidate(
                task="unresolved",
                score=1.0,
                evidence_spans=(),
                required_roles=(),
                rule_ids=("R-TP-003",),
                score_components={"insufficient_evidence": 1.0},
            )
        )
        applied.add("R-TP-003")

    primary_priority = {
        "extremum", "ranking", "trend", "correlation",
        "distribution", "composition", "comparison",
    }
    candidates.sort(
        key=lambda item: (
            0 if item.task in primary_priority else 1,
            -item.score,
            _TASK_ORDER[item.task],
        )
    )
    primary = candidates[0].task
    secondary = tuple(item.task for item in candidates[1:])
    required_roles = candidates[0].required_roles
    failure_state = None
    if primary == "unresolved":
        failure_state = "insufficient_task_evidence"

    all_spans = _dedupe_spans(
        span for candidate in candidates for span in candidate.evidence_spans
    )
    return AnalyticalTaskResult(
        task_candidates=tuple(candidates),
        task_scores={item.task: item.score for item in candidates},
        primary_task=primary,
        secondary_tasks=secondary,
        query_evidence_spans=all_spans,
        required_roles=required_roles,
        transforms=transforms,
        aggregation=aggregation,
        filters=filters,
        sort=sort,
        limit=limit,
        temporal_scope=temporal_scope,
        ambiguity=tuple(ambiguity),
        applied_rule_ids=registry.validate_ids(applied),
        failure_state=failure_state,
    )
