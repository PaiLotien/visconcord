"""M4 deterministic, typed, provenance-preserving Candidate Planner."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .config import SkillVISV2Config, load_m4_config
from .contracts import (
    AmbiguityStateType,
    AnalyticalTaskResult,
    CandidateChartType,
    CandidateDiversitySummary,
    CandidateEncoding,
    CandidateFields,
    CandidatePlanningResult,
    CandidateScoreComponents,
    CandidateTransform,
    CapabilityAssessment,
    EvidenceSpan,
    FieldResolutionResult,
    IntentFieldRef,
    ProvenanceRecord,
    SchemaProfile,
    TransformIntent,
    VisualizationCandidate,
    VisualizationIntentIR,
)
from .rules import RuleRegistry


_CORE_TASKS = {
    "comparison",
    "ranking",
    "trend",
    "correlation",
    "distribution",
    "composition",
    "extremum",
}

_MARK_TYPES = {
    "bar": "bar",
    "grouped_bar": "bar",
    "dot_plot": "point",
    "horizontal_bar": "bar",
    "top_k_bar": "bar",
    "line": "line",
    "multi_line": "line",
    "scatter": "point",
    "grouped_scatter": "point",
    "histogram": "bar",
    "box_plot": "boxplot",
    "stacked_bar": "bar",
    "normalized_stacked_bar": "bar",
    "pie": "arc",
    "ranked_bar": "bar",
    "highlighted_extremum": "bar",
}

_CHART_HINT_FAMILY = {
    "bar": "bar",
    "grouped_bar": "bar",
    "horizontal_bar": "bar",
    "top_k_bar": "bar",
    "stacked_bar": "bar",
    "normalized_stacked_bar": "bar",
    "ranked_bar": "bar",
    "highlighted_extremum": "bar",
    "line": "line",
    "multi_line": "line",
    "scatter": "scatter",
    "grouped_scatter": "scatter",
    "histogram": "histogram",
    "box_plot": "boxplot",
    "pie": "pie",
    "dot_plot": "dot_plot",
}

_DIVERSITY_FAMILY = {
    "bar": "bar",
    "grouped_bar": "bar",
    "horizontal_bar": "bar",
    "top_k_bar": "bar",
    "stacked_bar": "bar",
    "normalized_stacked_bar": "bar",
    "ranked_bar": "bar",
    "highlighted_extremum": "bar",
    "dot_plot": "point",
    "scatter": "point",
    "grouped_scatter": "point",
    "line": "line",
    "multi_line": "line",
    "histogram": "histogram",
    "box_plot": "box_plot",
    "pie": "pie",
}


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


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


def _convert_transform(item: TransformIntent) -> CandidateTransform:
    return CandidateTransform(
        transform_type=item.transform_type,
        operation=item.operation,
        field_candidates=item.field_candidates,
        value=item.value,
        negated=item.negated,
        evidence_spans=item.evidence_spans,
        rule_ids=item.rule_ids,
    )


def _planner_transform(
    transform_type: str,
    operation: str,
    *,
    fields: tuple[str, ...] = (),
    value: Any | None = None,
    evidence: tuple[EvidenceSpan, ...] = (),
    rule_ids: tuple[str, ...] = ("R-CP-003",),
) -> CandidateTransform:
    return CandidateTransform(
        transform_type=transform_type,
        operation=operation,
        field_candidates=fields,
        value=value,
        evidence_spans=evidence,
        rule_ids=rule_ids,
    )


_GROUP_GRAIN_CHARTS = {
    "bar",
    "grouped_bar",
    "horizontal_bar",
    "top_k_bar",
    "ranked_bar",
    "highlighted_extremum",
    "line",
    "multi_line",
    "stacked_bar",
    "normalized_stacked_bar",
    "pie",
}
_FOLD_KEY_FIELD = "__skillvis_measure_name"
_FOLD_VALUE_FIELD = "__skillvis_measure_value"
_NON_ADDITIVE_TOKENS = {
    "age", "average", "avg", "day", "days", "duration", "hour", "hours",
    "latency", "margin", "mean", "minute", "minutes", "pay", "pct",
    "percent", "percentage", "price", "rate", "ratio", "salary", "score",
    "second", "seconds", "time", "wage",
}
_ADDITIVE_TOKENS = {
    "amount", "charge", "consumption", "cost", "count", "demand", "expense",
    "fee", "income", "profit", "quantity", "revenue", "sale", "sales",
    "total", "unit", "units", "view", "views", "volume",
}


def _default_aggregation_operation(
    schema: SchemaProfile,
    measure_fields: tuple[str, ...],
) -> str:
    decisions = []
    for field_name in measure_fields:
        profile = schema.field(field_name)
        tokens = {
            *profile.normalized_name.split(),
            *(token for alias in profile.aliases for token in alias.split()),
        }
        if tokens & _NON_ADDITIVE_TOKENS:
            decisions.append("mean")
        elif tokens & _ADDITIVE_TOKENS:
            decisions.append("sum")
        else:
            # Mean is the conservative default for an unknown quantitative
            # unit: it avoids assuming additivity that the schema did not state.
            decisions.append("mean")
    return decisions[0] if len(set(decisions)) == 1 else "mean"


def _certain_duplicate_groups(
    schema: SchemaProfile,
    fields: CandidateFields,
) -> bool:
    known_grouping = tuple(
        dict.fromkeys(
            field
            for field in (
                *fields.dimension,
                *fields.temporal,
                *fields.grouping,
            )
            if field not in {_FOLD_KEY_FIELD, _FOLD_VALUE_FIELD}
        )
    )
    if not known_grouping:
        return False
    capacity = math.prod(
        max(1, schema.field(field).cardinality) for field in known_grouping
    )
    return schema.row_count > capacity


def _encoding(
    channel: str,
    field: str | None,
    semantic_type: str,
    *,
    aggregate: str | None = None,
    bin: bool = False,
    sort: str | None = None,
) -> CandidateEncoding:
    return CandidateEncoding(
        channel=channel,
        field=field,
        semantic_type=semantic_type,
        aggregate=aggregate,
        bin=bin,
        sort=sort,
    )


def _field_type(schema: SchemaProfile, field: str) -> str:
    return schema.field(field).semantic_type


def _field_names(refs: Iterable[IntentFieldRef]) -> tuple[str, ...]:
    def evidence_order(ref: IntentFieldRef) -> tuple[int, int, str]:
        starts = [
            evidence.query_span.start
            for evidence in ref.evidence
            if evidence.query_span is not None
        ]
        return (
            min(starts) if starts else 10**9,
            ref.candidate_rank,
            ref.field_name,
        )

    return tuple(
        dict.fromkeys(
            ref.field_name for ref in sorted(refs, key=evidence_order)
        )
    )


@dataclass(frozen=True)
class _MeasureExpression:
    """A raw or derived quantitative role consumed by a chart template."""

    field: str | None
    aggregate: str | None = None


def _analytical_measure_expressions(
    intent: VisualizationIntentIR,
    task_result: AnalyticalTaskResult,
) -> tuple[_MeasureExpression, ...]:
    raw = tuple(_MeasureExpression(field=name) for name in _field_names(intent.measures))
    aggregate_transforms = tuple(
        item
        for item in task_result.transforms
        if item.transform_type == "aggregate"
    )
    if not aggregate_transforms:
        return raw

    derived: list[_MeasureExpression] = []
    for transform in aggregate_transforms:
        if transform.field_candidates:
            derived.extend(
                _MeasureExpression(
                    field=field,
                    aggregate=transform.operation,
                )
                for field in transform.field_candidates
            )
        elif transform.operation == "count":
            derived.append(_MeasureExpression(field=None, aggregate="count"))
    return tuple(dict.fromkeys(derived))


def _relation_measure_expressions(
    intent: VisualizationIntentIR,
    task_result: AnalyticalTaskResult,
) -> tuple[_MeasureExpression, ...]:
    derived = _analytical_measure_expressions(intent, task_result)
    # Frequency relations (e.g., age vs count at each age) intentionally use a
    # raw measure and its count-derived counterpart.
    if (
        len(derived) == 1
        and derived[0].aggregate == "count"
        and intent.measures
    ):
        raw = tuple(
            _MeasureExpression(field=name)
            for name in _field_names(intent.measures)
        )
        return tuple(dict.fromkeys((*raw, *derived)))
    return derived


def _measure_fields(
    expressions: Iterable[_MeasureExpression],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            expression.field
            for expression in expressions
            if expression.field is not None
        )
    )


def _measure_encoding(
    channel: str,
    expression: _MeasureExpression,
    *,
    sort: str | None = None,
) -> CandidateEncoding:
    return _encoding(
        channel,
        expression.field,
        "quantitative",
        aggregate=expression.aggregate,
        sort=sort,
    )


def _ref_by_name(intent: VisualizationIntentIR) -> dict[str, IntentFieldRef]:
    rows: dict[str, IntentFieldRef] = {}
    for ref in (
        *intent.measures,
        *intent.dimensions,
        *intent.temporal_fields,
        *intent.grouping_fields,
    ):
        rows.setdefault(ref.field_name, ref)
    return rows


def _candidate_evidence(
    intent: VisualizationIntentIR,
    task: str,
    fields: tuple[str, ...],
) -> tuple[EvidenceSpan, ...]:
    refs = _ref_by_name(intent)
    spans: list[EvidenceSpan] = []
    for field in fields:
        ref = refs.get(field)
        if ref is None:
            continue
        spans.extend(
            evidence.query_span
            for evidence in ref.evidence
            if evidence.query_span is not None
        )
    spans.extend(
        candidate.evidence_spans
        for candidate in intent.task_candidates
        if candidate.task == task
    )
    flattened: list[EvidenceSpan] = []
    for item in spans:
        if isinstance(item, tuple):
            flattened.extend(item)
        else:
            flattened.append(item)
    flattened.extend(
        EvidenceSpan(
            start=term.start,
            end=term.end,
            text=term.text,
            normalized=term.canonical,
            kind="chart_hint",
            rule_ids=term.rule_ids,
        )
        for term in intent.chart_mentions
    )
    if not flattened:
        flattened.extend(intent.evidence_spans)
    return _dedupe_spans(flattened)


def _task_score(intent: VisualizationIntentIR, task: str) -> float:
    for candidate in intent.task_candidates:
        if candidate.task == task:
            return candidate.score
    return 0.0


def _spec_signature(
    *,
    chart_type: str,
    mark_type: str,
    fields: CandidateFields,
    encodings: tuple[CandidateEncoding, ...],
    transforms: tuple[CandidateTransform, ...],
    aggregation: CandidateTransform | None,
    filters: tuple[CandidateTransform, ...],
    sort: CandidateTransform | None,
    limit: int | None,
    grouping: tuple[str, ...],
    task: str,
) -> str:
    def transform_payload(item: CandidateTransform | None):
        if item is None:
            return None
        return {
            "transform_type": item.transform_type,
            "operation": item.operation,
            "field_candidates": item.field_candidates,
            "value": item.value,
            "negated": item.negated,
        }

    payload = {
        "chart_type": chart_type,
        "mark_type": mark_type,
        "fields": fields.model_dump(mode="json"),
        "encodings": [
            item.model_dump(mode="json")
            for item in sorted(encodings, key=lambda row: row.channel)
        ],
        "transforms": [transform_payload(item) for item in transforms],
        "aggregation": transform_payload(aggregation),
        "filters": [transform_payload(item) for item in filters],
        "sort": transform_payload(sort),
        "limit": limit,
        "grouping": grouping,
        "task": task,
    }
    return _canonical_hash(payload)


def _score(
    *,
    task: str,
    chart_type: str,
    intent: VisualizationIntentIR,
    evidence: tuple[EvidenceSpan, ...],
    config: SkillVISV2Config,
) -> tuple[CandidateScoreComponents, float]:
    settings = config.candidate_planner
    assert settings is not None
    weights = settings.score_weights
    task_fit = _task_score(intent, task)
    effectiveness = settings.task_chart_effectiveness[task][chart_type]
    ambiguity = {
        AmbiguityStateType.NONE: 0.0,
        AmbiguityStateType.LEXICAL_RESIDUE: 0.02,
        AmbiguityStateType.MULTIPLE_VALID_INTERPRETATIONS: 0.15,
        AmbiguityStateType.MATERIAL_AMBIGUITY: 1.0,
        AmbiguityStateType.MISSING_REQUIRED_ROLE: 1.0,
        AmbiguityStateType.UNSUPPORTED_SCOPE: 1.0,
    }[intent.ambiguity_state.state]
    chart_index = settings.chart_vocabulary.index(chart_type)
    tie = (
        1.0
        if len(settings.chart_vocabulary) == 1
        else 1.0 - chart_index / (len(settings.chart_vocabulary) - 1)
    )
    score_rule_ids = {
        "task_fit_score": ("R-CP-004",),
        "field_role_fit_score": ("R-CP-003", "R-CP-004"),
        "type_compatibility_score": ("R-CP-001", "R-CP-004"),
        "chart_effectiveness_score": ("R-CP-002", "R-CP-004"),
        "ambiguity_penalty": ("R-CP-001", "R-CP-004"),
        "missing_role_penalty": ("R-CP-001", "R-CP-004"),
        "unsupported_feature_penalty": ("R-CP-001", "R-CP-004"),
        "complexity_penalty": ("R-CP-004",),
        "provenance_strength_score": ("R-CP-004",),
        "deterministic_tie_breaker": ("R-CP-004", "R-CP-005"),
    }
    components = CandidateScoreComponents(
        task_fit_score=round(task_fit, 6),
        field_role_fit_score=1.0,
        type_compatibility_score=1.0,
        chart_effectiveness_score=round(effectiveness, 6),
        ambiguity_penalty=ambiguity,
        missing_role_penalty=0.0,
        unsupported_feature_penalty=0.0,
        complexity_penalty=round(settings.chart_complexity[chart_type], 6),
        provenance_strength_score=1.0 if evidence else 0.0,
        deterministic_tie_breaker=round(tie, 6),
        rule_ids=score_rule_ids,
    )
    positive_weight = (
        weights.task_fit
        + weights.field_role_fit
        + weights.type_compatibility
        + weights.chart_effectiveness
        + weights.provenance_strength
        + weights.deterministic_tie_breaker
    )
    positive = (
        components.task_fit_score * weights.task_fit
        + components.field_role_fit_score * weights.field_role_fit
        + components.type_compatibility_score * weights.type_compatibility
        + components.chart_effectiveness_score * weights.chart_effectiveness
        + components.provenance_strength_score * weights.provenance_strength
        + components.deterministic_tie_breaker
        * weights.deterministic_tie_breaker
    ) / positive_weight
    penalties = (
        components.ambiguity_penalty * weights.ambiguity_penalty
        + components.missing_role_penalty * weights.missing_role_penalty
        + components.unsupported_feature_penalty
        * weights.unsupported_feature_penalty
        + components.complexity_penalty * weights.complexity_penalty
    )
    return components, round(max(0.0, min(1.0, positive - penalties)), 6)


def _candidate_fields(
    *,
    measure: tuple[str, ...] = (),
    dimension: tuple[str, ...] = (),
    temporal: tuple[str, ...] = (),
    grouping: tuple[str, ...] = (),
    color: tuple[str, ...] = (),
    size: tuple[str, ...] = (),
    facet: tuple[str, ...] = (),
) -> CandidateFields:
    return CandidateFields(
        measure=measure,
        dimension=dimension,
        temporal=temporal,
        grouping=grouping,
        color=color,
        size=size,
        facet=facet,
    )


def _enumerate_templates(
    intent: VisualizationIntentIR,
    schema: SchemaProfile,
    task_result: AnalyticalTaskResult,
    config: SkillVISV2Config,
) -> list[dict[str, Any]]:
    settings = config.candidate_planner
    assert settings is not None
    measures = _analytical_measure_expressions(intent, task_result)
    relation_measures = _relation_measure_expressions(intent, task_result)
    temporal = _field_names(intent.temporal_fields)
    temporal_set = set(temporal)
    dimensions = tuple(
        name
        for name in _field_names(intent.dimensions)
        if name not in temporal_set
    )
    preserve_task_alternatives = (
        intent.ambiguity_state.state
        == AmbiguityStateType.MULTIPLE_VALID_INTERPRETATIONS
        or bool(intent.chart_mentions)
    )
    task_names = (
        tuple(
            candidate.task
            for candidate in intent.task_candidates
            if candidate.task in settings.core_tasks
        )
        if preserve_task_alternatives
        else (intent.primary_task,)
    )
    task_names = tuple(
        dict.fromkeys(
            task
            for task in task_names
            if task in _CORE_TASKS and task in settings.core_tasks
        )
    )
    base_transforms = tuple(
        _convert_transform(item) for item in task_result.transforms
    )
    rows: list[dict[str, Any]] = []

    def add(
        task: str,
        chart: str,
        encodings: tuple[CandidateEncoding, ...],
        fields: CandidateFields,
        *,
        extra_transforms: tuple[CandidateTransform, ...] = (),
        rationale: str,
        assumptions: tuple[str, ...],
    ) -> None:
        transforms = (*base_transforms, *extra_transforms)
        if (
            chart in _GROUP_GRAIN_CHARTS
            and fields.measure
            and not any(
                item.transform_type == "aggregate" for item in transforms
            )
            and _certain_duplicate_groups(schema, fields)
        ):
            operation = _default_aggregation_operation(schema, fields.measure)
            aggregate = _planner_transform(
                "aggregate",
                operation,
                fields=fields.measure,
                rule_ids=("R-CP-008",),
            )
            transforms = (*transforms, aggregate)
            encodings = tuple(
                encoding.model_copy(update={"aggregate": operation})
                if (
                    encoding.semantic_type == "quantitative"
                    and (
                        encoding.field in set(fields.measure)
                        or encoding.field == _FOLD_VALUE_FIELD
                    )
                )
                else encoding
                for encoding in encodings
            )
            assumptions = (
                *assumptions,
                (
                    f"certain duplicate group grain requires declared {operation} "
                    "aggregation under the versioned planner policy"
                ),
            )
        rows.append(
            {
                "task": task,
                "chart_type": chart,
                "encodings": encodings,
                "fields": fields,
                "transforms": transforms,
                "rationale": rationale,
                "assumptions": assumptions,
            }
        )

    for task in task_names:
        if task == "comparison" and measures and dimensions:
            dimension = dimensions[0]
            raw_measure_fields = _measure_fields(measures)
            if len(raw_measure_fields) >= 2 and len(dimensions) == 1:
                fold = _planner_transform(
                    "fold",
                    "measures_to_long",
                    fields=raw_measure_fields,
                    value={
                        "key_field": _FOLD_KEY_FIELD,
                        "value_field": _FOLD_VALUE_FIELD,
                    },
                    rule_ids=("R-CP-009",),
                )
                add(
                    task,
                    "grouped_bar",
                    (
                        _encoding("x", dimension, _field_type(schema, dimension)),
                        _encoding("y", _FOLD_VALUE_FIELD, "quantitative"),
                        _encoding("color", _FOLD_KEY_FIELD, "categorical"),
                    ),
                    _candidate_fields(
                        measure=raw_measure_fields,
                        dimension=(dimension,),
                        grouping=(_FOLD_KEY_FIELD,),
                        color=(_FOLD_KEY_FIELD,),
                    ),
                    extra_transforms=(fold,),
                    rationale=(
                        "A typed fold preserves every requested measure before "
                        "grouped-bar encoding."
                    ),
                    assumptions=(
                        "measure names become a declared derived grouping field",
                    ),
                )
            else:
                measure = measures[0]
                if len(dimensions) >= 2:
                    group = dimensions[1]
                    base = (
                        _encoding("x", dimension, _field_type(schema, dimension)),
                        _measure_encoding("y", measure),
                    )
                    add(
                        task,
                        "grouped_bar",
                        (*base, _encoding("color", group, _field_type(schema, group))),
                        _candidate_fields(
                            measure=_measure_fields((measure,)),
                            dimension=(dimension, group),
                            grouping=(group,),
                            color=(group,),
                        ),
                        rationale="Grouped bars compare a measure across two resolved dimensions.",
                        assumptions=("grouping cardinality is suitable for color",),
                    )
                    add(
                        task,
                        "grouped_bar",
                        (*base, _encoding("column", group, _field_type(schema, group))),
                        _candidate_fields(
                            measure=_measure_fields((measure,)),
                            dimension=(dimension, group),
                            grouping=(group,),
                            facet=(group,),
                        ),
                        rationale="Faceted grouped bars preserve the second dimension without color.",
                        assumptions=("facet cardinality is manageable",),
                    )
                else:
                    for chart in ("bar", "dot_plot"):
                        add(
                            task,
                            chart,
                            (
                                _encoding("x", dimension, _field_type(schema, dimension)),
                                _measure_encoding("y", measure),
                            ),
                            _candidate_fields(
                                measure=_measure_fields((measure,)),
                                dimension=(dimension,),
                            ),
                            rationale=(
                                "Position encodes the resolved quantitative comparison "
                                "across the resolved dimension."
                            ),
                            assumptions=("category count is displayable",),
                        )

        elif task == "ranking" and measures and dimensions:
            measure, dimension = measures[0], dimensions[0]
            add(
                task,
                "horizontal_bar",
                (
                    _measure_encoding("x", measure),
                    _encoding("y", dimension, _field_type(schema, dimension)),
                ),
                _candidate_fields(
                    measure=_measure_fields((measure,)), dimension=(dimension,)
                ),
                rationale="Horizontal position supports ordered category comparison.",
                assumptions=("labels benefit from a horizontal layout",),
            )
            if intent.limit is not None:
                add(
                    task,
                    "top_k_bar",
                    (
                        _measure_encoding("x", measure),
                        _encoding("y", dimension, _field_type(schema, dimension)),
                        _measure_encoding("color", measure),
                    ),
                    _candidate_fields(
                        measure=_measure_fields((measure,)),
                        dimension=(dimension,),
                        color=_measure_fields((measure,)),
                    ),
                    rationale="A bounded ranked bar retains top-k and quantitative emphasis.",
                    assumptions=("continuous color is supplementary to position",),
                )
            add(
                task,
                "ranked_bar",
                (
                    _encoding("x", dimension, _field_type(schema, dimension)),
                    _measure_encoding("y", measure),
                ),
                _candidate_fields(
                    measure=_measure_fields((measure,)), dimension=(dimension,)
                ),
                rationale="A sorted vertical bar is a distinct ranked encoding.",
                assumptions=("category labels fit the x-axis",),
            )

        elif task == "trend" and measures and temporal:
            measure, time = measures[0], temporal[0]
            group = dimensions[0] if dimensions else None
            if group:
                base = (
                    _encoding("x", time, "temporal"),
                    _measure_encoding("y", measure),
                )
                add(
                    task,
                    "multi_line",
                    (*base, _encoding("color", group, _field_type(schema, group))),
                    _candidate_fields(
                        measure=_measure_fields((measure,)),
                        temporal=(time,),
                        grouping=(group,),
                        color=(group,),
                    ),
                    rationale="Color separates resolved temporal series.",
                    assumptions=("series cardinality is manageable",),
                )
                add(
                    task,
                    "multi_line",
                    (*base, _encoding("column", group, _field_type(schema, group))),
                    _candidate_fields(
                        measure=_measure_fields((measure,)),
                        temporal=(time,),
                        grouping=(group,),
                        facet=(group,),
                    ),
                    rationale="Facets separate resolved temporal series.",
                    assumptions=("facet count is manageable",),
                )
            else:
                add(
                    task,
                    "line",
                    (
                        _encoding("x", time, "temporal"),
                        _measure_encoding("y", measure),
                    ),
                    _candidate_fields(
                        measure=_measure_fields((measure,)), temporal=(time,)
                    ),
                    rationale="A line preserves ordered temporal position.",
                    assumptions=("temporal observations have an ordered sequence",),
                )

        elif task == "correlation" and len(relation_measures) >= 2:
            first, second = relation_measures[:2]
            group = dimensions[0] if dimensions else None
            base = (
                _measure_encoding("x", first),
                _measure_encoding("y", second),
            )
            if group:
                add(
                    task,
                    "grouped_scatter",
                    (*base, _encoding("color", group, _field_type(schema, group))),
                    _candidate_fields(
                        measure=_measure_fields((first, second)),
                        dimension=(group,),
                        grouping=(group,),
                        color=(group,),
                    ),
                    rationale="Color preserves a resolved grouping in a two-measure relation.",
                    assumptions=("grouping cardinality is suitable for color",),
                )
                add(
                    task,
                    "grouped_scatter",
                    (*base, _encoding("column", group, _field_type(schema, group))),
                    _candidate_fields(
                        measure=_measure_fields((first, second)),
                        dimension=(group,),
                        grouping=(group,),
                        facet=(group,),
                    ),
                    rationale="Facets preserve grouping without overloading point color.",
                    assumptions=("facet count is manageable",),
                )
            else:
                add(
                    task,
                    "scatter",
                    base,
                    _candidate_fields(measure=_measure_fields((first, second))),
                    rationale="Two resolved quantitative measures map directly to position.",
                    assumptions=("row-level measure pairs are comparable",),
                )

        elif task == "distribution" and measures and measures[0].field is None and dimensions:
            dimension = dimensions[0]
            add(
                task,
                "histogram",
                (
                    _encoding("x", dimension, _field_type(schema, dimension)),
                    _measure_encoding("y", measures[0]),
                ),
                _candidate_fields(dimension=(dimension,)),
                rationale=(
                    "A requested frequency view maps the resolved category to "
                    "position and row count to magnitude."
                ),
                assumptions=(
                    "categorical frequency is retained because histogram was explicit",
                ),
            )

        elif task == "distribution" and measures and measures[0].field is not None:
            measure = measures[0]
            assert measure.field is not None
            add(
                task,
                "histogram",
                (
                    _encoding(
                        "x",
                        measure.field,
                        "quantitative",
                        aggregate=measure.aggregate,
                        bin=True,
                    ),
                    _encoding("y", None, "quantitative", aggregate="count"),
                ),
                _candidate_fields(measure=(measure.field,)),
                extra_transforms=(
                    _planner_transform("bin", "auto", fields=(measure.field,)),
                ),
                rationale="Binned position and count expose a quantitative distribution.",
                assumptions=("automatic binning is deferred to execution grammar",),
            )
            add(
                task,
                "box_plot",
                (_measure_encoding("x", measure),),
                _candidate_fields(measure=(measure.field,)),
                rationale="A box plot summarizes distribution and robust spread.",
                assumptions=("summary statistics are supported by the execution grammar",),
            )

        elif task == "composition" and measures and (dimensions or temporal):
            measure = measures[0]
            dimension = (dimensions or temporal)[0]
            stack_fields = (*_measure_fields((measure,)), dimension)
            stack = _planner_transform(
                "stack", "zero", fields=stack_fields
            )
            normalized = _planner_transform(
                "normalize", "stack_normalize", fields=stack_fields
            )
            for chart, extra, rationale in (
                (
                    "normalized_stacked_bar",
                    (stack, normalized),
                    "Normalized stacking compares proportional contribution.",
                ),
                (
                    "stacked_bar",
                    (stack,),
                    "Stacking preserves absolute part-to-whole contribution.",
                ),
            ):
                add(
                    task,
                    chart,
                    (
                        _measure_encoding("y", measure),
                        _encoding(
                            "color", dimension, _field_type(schema, dimension)
                        ),
                    ),
                    _candidate_fields(
                        measure=_measure_fields((measure,)),
                        dimension=(dimension,),
                        grouping=(dimension,),
                        color=(dimension,),
                    ),
                    extra_transforms=extra,
                    rationale=rationale,
                    assumptions=("parts are mutually comparable within the whole",),
                )
            explicit_pie = any(
                term.canonical == "pie" for term in intent.chart_mentions
            )
            if (
                schema.field(dimension).cardinality <= settings.pie_max_categories
                or explicit_pie
            ):
                add(
                    task,
                    "pie",
                    (
                        _measure_encoding("theta", measure),
                        _encoding(
                            "color", dimension, _field_type(schema, dimension)
                        ),
                    ),
                    _candidate_fields(
                        measure=_measure_fields((measure,)),
                        dimension=(dimension,),
                        grouping=(dimension,),
                        color=(dimension,),
                    ),
                    rationale="A bounded category count permits a part-to-whole arc encoding.",
                    assumptions=(
                        (
                            f"category cardinality <= {settings.pie_max_categories}"
                            if schema.field(dimension).cardinality
                            <= settings.pie_max_categories
                            else "explicit pie request retained for M5 cardinality review"
                        ),
                    ),
                )

        elif task == "extremum" and measures and dimensions:
            measure, dimension = measures[0], dimensions[0]
            add(
                task,
                "highlighted_extremum",
                (
                    _encoding("x", dimension, _field_type(schema, dimension)),
                    _measure_encoding("y", measure),
                    _measure_encoding("color", measure),
                ),
                _candidate_fields(
                    measure=_measure_fields((measure,)),
                    dimension=(dimension,),
                    color=_measure_fields((measure,)),
                ),
                extra_transforms=(
                    _planner_transform(
                        "highlight",
                        "extremum",
                        fields=_measure_fields((measure,)),
                        value=intent.limit or 1,
                    ),
                ),
                rationale="Position plus highlight emphasizes the requested extremum.",
                assumptions=("color is supplementary to positional comparison",),
            )
            add(
                task,
                "horizontal_bar",
                (
                    _measure_encoding("x", measure),
                    _encoding("y", dimension, _field_type(schema, dimension)),
                ),
                _candidate_fields(
                    measure=_measure_fields((measure,)), dimension=(dimension,)
                ),
                rationale="A sorted horizontal bar exposes the extremum among categories.",
                assumptions=("labels benefit from a horizontal layout",),
            )
            add(
                task,
                "ranked_bar",
                (
                    _encoding("x", dimension, _field_type(schema, dimension)),
                    _measure_encoding("y", measure),
                ),
                _candidate_fields(
                    measure=_measure_fields((measure,)), dimension=(dimension,)
                ),
                rationale="A ranked vertical bar provides an alternative extremum view.",
                assumptions=("category labels fit the x-axis",),
            )
    return rows


def _explicit_hint_filter(
    rows: list[dict[str, Any]],
    intent: VisualizationIntentIR,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    hints = tuple(sorted({item.canonical for item in intent.chart_mentions}))
    if not hints:
        return rows, ()
    if len(hints) > 1:
        return [], (
            "conflicting_explicit_chart_hints:" + ",".join(hints),
        )
    hint = hints[0]
    filtered = [
        row
        for row in rows
        if _CHART_HINT_FAMILY.get(row["chart_type"]) == hint
    ]
    if not filtered:
        return [], (f"incompatible_or_unsupported_chart_hint:{hint}",)
    return filtered, ()


def _candidate_from_row(
    row: dict[str, Any],
    *,
    intent: VisualizationIntentIR,
    schema: SchemaProfile,
    task_result: AnalyticalTaskResult,
    config: SkillVISV2Config,
    registry: RuleRegistry,
) -> VisualizationCandidate:
    chart = row["chart_type"]
    fields: CandidateFields = row["fields"]
    used_fields = tuple(
        dict.fromkeys(
            (
                *fields.measure,
                *fields.dimension,
                *fields.temporal,
                *fields.grouping,
                *fields.color,
                *fields.size,
                *fields.facet,
            )
        )
    )
    evidence = _candidate_evidence(intent, row["task"], used_fields)
    explicit_hint = bool(intent.chart_mentions)
    applied = {
        "R-CP-001",
        "R-CP-002",
        "R-CP-003",
        "R-CP-004",
        "R-CP-005",
        "R-CP-006",
    }
    applied.update(
        rule_id
        for transform in row["transforms"]
        for rule_id in transform.rule_ids
        if rule_id.startswith("R-CP-")
    )
    if explicit_hint:
        applied.add("R-CP-007")
    applied_ids = registry.validate_ids(applied)
    source_ids = tuple(
        sorted(
            {
                source_id
                for rule_id in applied_ids
                for source_id in registry.get(rule_id).source_ids
            }
        )
    )
    transforms: tuple[CandidateTransform, ...] = tuple(row["transforms"])
    aggregate_transforms = tuple(
        item for item in transforms if item.transform_type == "aggregate"
    )
    # The singular aggregation field is a projection only when the transform
    # set is genuinely singular.  Selecting the first of multiple aggregates
    # created a false internal-consistency violation in M5.
    aggregation = (
        aggregate_transforms[0] if len(aggregate_transforms) == 1 else None
    )
    filters = tuple(
        item for item in transforms if item.transform_type == "filter"
    )
    sort = next(
        (item for item in transforms if item.transform_type == "sort"),
        None,
    )
    encodings = tuple(
        sorted(row["encodings"], key=lambda item: item.channel)
    )
    signature = _spec_signature(
        chart_type=chart,
        mark_type=_MARK_TYPES[chart],
        fields=fields,
        encodings=encodings,
        transforms=transforms,
        aggregation=aggregation,
        filters=filters,
        sort=sort,
        limit=intent.limit,
        grouping=fields.grouping,
        task=row["task"],
    )
    score_components, total = _score(
        task=row["task"],
        chart_type=chart,
        intent=intent,
        evidence=evidence,
        config=config,
    )
    warnings = (
        ("multiple_valid_interpretations_preserved",)
        if intent.ambiguity_state.state
        == AmbiguityStateType.MULTIPLE_VALID_INTERPRETATIONS
        else ()
    )
    return VisualizationCandidate(
        candidate_id=f"m4-{signature[:20]}",
        chart_type=CandidateChartType(chart),
        mark_type=_MARK_TYPES[chart],
        fields=fields,
        encodings=encodings,
        transforms=transforms,
        aggregation=aggregation,
        filters=filters,
        sort=sort,
        limit=intent.limit,
        grouping=fields.grouping,
        task_alignment=row["task"],
        field_role_alignment={
            key: value
            for key, value in fields.model_dump(mode="python").items()
            if value
        },
        chart_compatibility_assumptions=(
            "fields and roles are inherited from VisualizationIntentIR",
            "candidate remains unvalidated until M5",
            *row["assumptions"],
        ),
        score_components=score_components,
        total_score=total,
        rank=1,
        rationale=row["rationale"],
        evidence_spans=evidence,
        applied_rule_ids=applied_ids,
        source_ids=source_ids,
        ambiguity_state=intent.ambiguity_state.state,
        warnings=warnings,
        unsupported_features=(),
        executable=True,
        audit_only=False,
        rejection_reasons=(),
        deterministic_signature=signature,
    )


def deduplicate_candidates(
    candidates: Iterable[VisualizationCandidate],
) -> tuple[VisualizationCandidate, ...]:
    """Keep one deterministic representative for each semantic signature."""

    rows: dict[str, VisualizationCandidate] = {}
    for candidate in candidates:
        current = rows.get(candidate.deterministic_signature)
        if current is None or (
            candidate.total_score,
            candidate.candidate_id,
        ) > (
            current.total_score,
            current.candidate_id,
        ):
            rows[candidate.deterministic_signature] = candidate
    return tuple(rows[key] for key in sorted(rows))


def _diversity(
    candidates: tuple[VisualizationCandidate, ...],
) -> CandidateDiversitySummary:
    encoding_signatures = {
        tuple(
            (
                item.channel,
                item.field,
                item.aggregate,
                item.bin,
                item.time_unit,
                item.sort,
            )
            for item in candidate.encodings
        )
        for candidate in candidates
    }
    transform_signatures = {
        tuple(
            (
                item.transform_type,
                item.operation,
                item.field_candidates,
                json.dumps(item.value, sort_keys=True),
                item.negated,
            )
            for item in candidate.transforms
        )
        for candidate in candidates
    }
    return CandidateDiversitySummary(
        chart_family_count=len(
            {
                _DIVERSITY_FAMILY[candidate.chart_type.value]
                for candidate in candidates
            }
        ),
        encoding_signature_count=len(encoding_signatures),
        transform_signature_count=len(transform_signatures),
        task_interpretation_count=len(
            {candidate.task_alignment for candidate in candidates}
        ),
    )


def _planner_provenance(
    registry: RuleRegistry,
    rule_ids: Iterable[str],
    intent: VisualizationIntentIR,
) -> tuple[ProvenanceRecord, ...]:
    return tuple(
        registry.trace(
            "candidate_planning",
            rule_id,
            inputs=(f"intent:{intent.deterministic_id}",),
            outputs=("visualization_candidates",),
        )
        for rule_id in registry.validate_ids(rule_ids)
    )


def _empty_result(
    *,
    intent: VisualizationIntentIR,
    config: SkillVISV2Config,
    registry: RuleRegistry,
    reason: str,
    rejection_reasons: tuple[str, ...],
) -> CandidatePlanningResult:
    applied = {"R-CP-001", "R-CP-005", "R-CP-006"}
    if intent.chart_mentions and reason == "no_hard_valid_candidate":
        applied.add("R-CP-007")
    provenance = _planner_provenance(registry, applied, intent)
    payload = {
        "intent_id": intent.deterministic_id,
        "reason": reason,
        "rejections": rejection_reasons,
        "candidate_signatures": [],
    }
    return CandidatePlanningResult(
        intent_id=intent.deterministic_id,
        config_version=config.config_version,
        registry_hash=registry.sha256,
        candidates=(),
        raw_candidate_count=0,
        post_dedup_count=0,
        post_ranking_count=0,
        duplicate_count=0,
        diversity=CandidateDiversitySummary(
            chart_family_count=0,
            encoding_signature_count=0,
            transform_signature_count=0,
            task_interpretation_count=0,
        ),
        abstained=True,
        abstention_reason=reason,
        warnings=(),
        rejection_reasons=rejection_reasons,
        provenance=provenance,
        deterministic_signature=_canonical_hash(payload),
    )


def plan_visualization_candidates(
    *,
    intent: VisualizationIntentIR,
    capability: CapabilityAssessment,
    schema: SchemaProfile,
    field_resolution: FieldResolutionResult,
    task_result: AnalyticalTaskResult,
    config: SkillVISV2Config | None = None,
    registry: RuleRegistry | None = None,
) -> CandidatePlanningResult:
    """Enumerate, score, deduplicate, and rank M4 candidates without validation."""

    config = config or load_m4_config()
    if not config.feature_flags.enable_candidate_planner:
        raise RuntimeError("M4 Candidate Planner feature flag is disabled")
    if config.feature_flags.enable_semantic_validator:
        raise RuntimeError("M5 Validator must remain disabled during M4")
    settings = config.candidate_planner
    if settings is None:
        raise RuntimeError("M4 Candidate Planner settings are unavailable")
    registry = registry or RuleRegistry.from_config(config)
    if capability != intent.capability_assessment:
        raise ValueError("capability input must match VisualizationIntentIR")

    schema_names = {field.field_name for field in schema.fields}
    intent_names = {
        ref.field_name
        for ref in (
            *intent.measures,
            *intent.dimensions,
            *intent.temporal_fields,
            *intent.grouping_fields,
        )
    }
    resolved_names = set(field_resolution.required_field_candidates)
    if not intent_names <= schema_names or not intent_names <= resolved_names:
        return _empty_result(
            intent=intent,
            config=config,
            registry=registry,
            reason="inconsistent_upstream_field_contract",
            rejection_reasons=(
                "intent fields must exist in schema and required field candidates",
            ),
        )

    if intent.abstention_state.abstained or capability.abstention_required:
        reason = (
            intent.abstention_reason
            or intent.abstention_state.reason
            or "required_abstention"
        )
        return _empty_result(
            intent=intent,
            config=config,
            registry=registry,
            reason=reason,
            rejection_reasons=(
                "upstream safety or semantic layer requires abstention",
            ),
        )

    if intent.ambiguity_state.state in {
        AmbiguityStateType.MATERIAL_AMBIGUITY,
        AmbiguityStateType.MISSING_REQUIRED_ROLE,
        AmbiguityStateType.UNSUPPORTED_SCOPE,
    }:
        return _empty_result(
            intent=intent,
            config=config,
            registry=registry,
            reason=f"blocked_ambiguity_state:{intent.ambiguity_state.state.value}",
            rejection_reasons=("typed ambiguity state forbids execution",),
        )

    rows = _enumerate_templates(intent, schema, task_result, config)
    rows, hint_rejections = _explicit_hint_filter(rows, intent)
    if not rows:
        return _empty_result(
            intent=intent,
            config=config,
            registry=registry,
            reason="no_hard_valid_candidate",
            rejection_reasons=hint_rejections
            or ("no template satisfied all task and field-role constraints",),
        )

    raw = tuple(
        _candidate_from_row(
            row,
            intent=intent,
            schema=schema,
            task_result=task_result,
            config=config,
            registry=registry,
        )
        for row in rows
        if row["chart_type"] in settings.chart_vocabulary
    )
    deduped = deduplicate_candidates(raw)

    def stable_rank_key(item: VisualizationCandidate):
        channels = {encoding.channel for encoding in item.encodings}
        # Within an otherwise tied chart template, a color grouping is the
        # lower-complexity default and a facet remains the genuine alternative.
        encoding_branch_priority = 0 if "color" in channels else (
            1 if {"row", "column"} & channels else 0
        )
        return (
            -item.total_score,
            encoding_branch_priority,
            item.deterministic_signature,
        )

    ranked_source = sorted(
        deduped,
        key=stable_rank_key,
    )[: settings.max_candidates]
    ranked = tuple(
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(ranked_source, start=1)
    )
    rule_ids = {
        rule_id
        for candidate in ranked
        for rule_id in candidate.applied_rule_ids
    }
    provenance = _planner_provenance(registry, rule_ids, intent)
    result_payload = {
        "intent_id": intent.deterministic_id,
        "candidate_signatures": [
            candidate.deterministic_signature for candidate in ranked
        ],
        "scores": [candidate.total_score for candidate in ranked],
        "ranks": [candidate.rank for candidate in ranked],
        "raw": len(raw),
        "dedup": len(deduped),
    }
    result = CandidatePlanningResult(
        intent_id=intent.deterministic_id,
        config_version=config.config_version,
        registry_hash=registry.sha256,
        candidates=ranked,
        raw_candidate_count=len(raw),
        post_dedup_count=len(deduped),
        post_ranking_count=len(ranked),
        duplicate_count=len(raw) - len(deduped),
        diversity=_diversity(ranked),
        abstained=False,
        abstention_reason=None,
        warnings=(
            ("candidate_budget_applied",)
            if len(deduped) > settings.max_candidates
            else ()
        ),
        rejection_reasons=(),
        provenance=provenance,
        deterministic_signature=_canonical_hash(result_payload),
    )
    return CandidatePlanningResult.model_validate(
        result.model_dump(mode="json")
    )
