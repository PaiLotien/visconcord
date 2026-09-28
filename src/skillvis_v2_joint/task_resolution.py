"""M3 analytical-goal modeling and joint field-task hypothesis resolution."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

from skillvis_v2_accuracy.contracts import (
    AmbiguityRecord,
    AnalyticalTaskResult,
    EvidenceSpan,
    FieldResolutionResult,
    QueryNormalizationResult,
    SchemaProfile,
    TaskCandidate,
    TransformIntent,
)
from skillvis_v2_accuracy.rules import RuleRegistry

from .config import JointSemanticSettings
from .contracts import (
    FieldRoleBinding,
    InterpretationHypothesis,
    PresentationIntent,
    SemanticFieldMention,
    SemanticGraphEdge,
    SemanticGraphNode,
)


_GOAL_ORDER = {
    "comparison": 0,
    "composition": 1,
    "trend": 2,
    "correlation": 3,
    "distribution": 4,
    "extremum": 5,
    "ranking": 6,
    "unresolved": 7,
}
_GROUP_CUE = re.compile(
    r"\b(each|per|by|across|grouped|group|categorized|categorised|divided|based)\b",
    re.IGNORECASE,
)
_PART_WHOLE_CUE = re.compile(
    r"\b(proportion|share|percentage|percent|composition|part(?:s)? of|pie)\b",
    re.IGNORECASE,
)
_TYPICAL_CUE = re.compile(r"\b(typical|typically|usual|usually)\b", re.IGNORECASE)
_SUPERLATIVE_MAX = re.compile(r"\b(maximum|largest|highest|greatest)\b", re.IGNORECASE)
_SUPERLATIVE_MIN = re.compile(r"\b(minimum|smallest|lowest|least)\b", re.IGNORECASE)
_ENTITY_ANSWER_CUE = re.compile(r"^\s*(which|who|what\s+(?:region|category|group|item))\b", re.IGNORECASE)


def _full_span(normalized: QueryNormalizationResult, kind: str) -> EvidenceSpan:
    return EvidenceSpan(
        start=0,
        end=len(normalized.original_query),
        text=normalized.original_query,
        normalized=normalized.normalized_query,
        kind=kind,
        rule_ids=("R-TP-021", "R-TP-022"),
    )


def _dedupe_spans(items) -> tuple[EvidenceSpan, ...]:
    rows = {}
    for item in items:
        rows[(item.start, item.end, item.kind, item.normalized, item.rule_ids)] = item
    return tuple(rows[key] for key in sorted(rows))


def _field_spans(candidate) -> tuple[EvidenceSpan, ...]:
    return _dedupe_spans(
        item.query_span for item in candidate.evidence if item.query_span is not None
    )


def _role_bindings(
    fields: FieldResolutionResult,
    schema: SchemaProfile,
    has_group_structure: bool,
) -> tuple[FieldRoleBinding, ...]:
    required = set(fields.required_field_candidates)
    schema_by_name = {item.field_name: item for item in schema.fields}
    rows: list[FieldRoleBinding] = []
    for candidate in fields.candidate_fields:
        if candidate.field_name not in required:
            continue
        semantic_type = schema_by_name[candidate.field_name].semantic_type
        spans = _field_spans(candidate)
        if semantic_type == "quantitative":
            roles = ("measure",)
        elif semantic_type == "temporal":
            roles = ("temporal", "dimension")
        elif semantic_type == "identifier":
            roles = ("identifier", "dimension")
        else:
            roles = ("dimension",)
        if has_group_structure and semantic_type in {
            "categorical", "ordinal", "geographic", "identifier", "temporal"
        }:
            roles = tuple(dict.fromkeys((*roles, "grouping")))
        for role in roles:
            rows.append(
                FieldRoleBinding(
                    field_name=candidate.field_name,
                    role=role,
                    mention_ids=candidate.mention_ids,
                    evidence_spans=spans,
                )
            )
    return tuple(sorted(rows, key=lambda item: (item.field_name.casefold(), item.role)))


def _presentation(normalized: QueryNormalizationResult) -> PresentationIntent:
    hints = tuple(sorted({item.canonical for item in normalized.chart_mentions}))
    spans = _dedupe_spans(
        EvidenceSpan(
            start=item.start,
            end=item.end,
            text=item.text,
            normalized=item.canonical,
            kind="presentation_hint",
            rule_ids=("R-TP-020",),
        )
        for item in normalized.chart_mentions
    )
    return PresentationIntent(
        chart_hints=hints,
        explicit=bool(hints),
        evidence_spans=spans,
    )


def _effective_operations(
    *,
    normalized: QueryNormalizationResult,
    base: AnalyticalTaskResult,
    bindings: tuple[FieldRoleBinding, ...],
    presentation: PresentationIntent,
) -> tuple[tuple[TransformIntent, ...], TransformIntent | None, object, int | None, tuple[AmbiguityRecord, ...]]:
    operations = list(base.transforms)
    measures = tuple(sorted({item.field_name for item in bindings if item.role == "measure"}))
    dimensions = tuple(sorted({item.field_name for item in bindings if item.role == "dimension"}))
    full = _full_span(normalized, "joint_operation_evidence")
    aggregate_ops = [item for item in operations if item.transform_type == "aggregate"]
    has_group = bool(dimensions) and bool(_GROUP_CUE.search(normalized.original_query))

    # Operation words can also be legitimate schema fields (for example a
    # quantitative column literally named ``count``). When a more specific
    # aggregate such as mean is explicitly present, the field mention must not
    # also create a row-count obligation over itself.
    non_count_aggregates = {
        item.operation for item in aggregate_ops if item.operation != "count"
    }
    if non_count_aggregates:
        operations = [
            item
            for item in operations
            if not (
                item.transform_type == "aggregate"
                and item.operation == "count"
                and item.field_candidates
                and set(item.field_candidates) <= set(measures)
            )
        ]
        aggregate_ops = [item for item in operations if item.transform_type == "aggregate"]

    # Natural row-count expressions name an entity, not an aggregate operand.
    if re.search(r"\b(number\s+of|how\s+many|total\s+number\s+of)\b", normalized.original_query, re.IGNORECASE):
        operations = [
            item.model_copy(update={"field_candidates": ()})
            if item.transform_type == "aggregate" and item.operation == "count"
            else item
            for item in operations
        ]
        aggregate_ops = [item for item in operations if item.transform_type == "aggregate"]

    # A categorical-only bar/pie request denotes a frequency display. This is
    # an explicit derived row-count obligation, not an invented schema field.
    explicit_frequency_goal = any(
        item.canonical in {"comparison", "composition", "distribution"}
        for item in normalized.task_terms
    )
    if (
        not aggregate_ops
        and dimensions
        and not measures
        and (
            (presentation.chart_hints and set(presentation.chart_hints) <= {"bar", "pie"})
            or explicit_frequency_goal
        )
    ):
        operations.append(
            TransformIntent(
                transform_type="aggregate",
                operation="count",
                field_candidates=(),
                evidence_spans=(full,),
                rule_ids=("R-TP-021",),
            )
        )
        aggregate_ops = [operations[-1]]

    # Grouped superlatives describe one aggregate per group. They are not a
    # request to sort the whole table and return a single row.
    grouped_superlative = has_group and bool(measures) and bool(
        _SUPERLATIVE_MAX.search(normalized.original_query)
        or _SUPERLATIVE_MIN.search(normalized.original_query)
    )
    if grouped_superlative:
        operation = "max" if _SUPERLATIVE_MAX.search(normalized.original_query) else "min"
        operations = [
            item for item in operations
            if item.transform_type not in {"sort", "limit"}
        ]
        if not any(
            item.transform_type == "aggregate" and item.operation == operation
            for item in operations
        ):
            operations.append(
                TransformIntent(
                    transform_type="aggregate",
                    operation=operation,
                    field_candidates=(measures[0],),
                    evidence_spans=(full,),
                    rule_ids=("R-TP-021",),
                )
            )

    transform_ambiguity = [
        item for item in base.ambiguity if item.dimension == "transform"
    ]
    if _TYPICAL_CUE.search(normalized.original_query) and measures and not any(
        item.transform_type == "aggregate" for item in operations
    ):
        for operation in ("mean", "median"):
            operations.append(
                TransformIntent(
                    transform_type="aggregate",
                    operation=operation,
                    field_candidates=(measures[0],),
                    evidence_spans=(full,),
                    rule_ids=("R-TP-023",),
                )
            )
        transform_ambiguity.append(
            AmbiguityRecord(
                dimension="transform",
                alternatives=("mean", "median"),
                reason="typical value permits mean or median without further clarification",
                rule_ids=("R-TP-023",),
            )
        )

    # Stable de-duplication.
    unique = {}
    for item in operations:
        key = (
            item.transform_type,
            item.operation,
            item.field_candidates,
            json.dumps(item.value, ensure_ascii=False, sort_keys=True),
            item.negated,
        )
        unique[key] = item
    transforms = tuple(unique[key] for key in sorted(unique, key=str))
    aggregates = tuple(item for item in transforms if item.transform_type == "aggregate")
    aggregation = aggregates[0] if len(aggregates) == 1 else None
    sort = next((item for item in transforms if item.transform_type == "sort"), None)
    limit_transform = next((item for item in transforms if item.transform_type == "limit"), None)
    limit = int(limit_transform.value) if limit_transform and limit_transform.value else None
    return transforms, aggregation, sort, limit, tuple(transform_ambiguity)


def _hypothesis_id(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "hyp-" + hashlib.sha256(raw.encode()).hexdigest()[:18]


def run_joint_task_resolution(
    *,
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    base: AnalyticalTaskResult,
    semantic_mentions: tuple[SemanticFieldMention, ...],
    settings: JointSemanticSettings,
    registry: RuleRegistry,
) -> tuple[
    AnalyticalTaskResult,
    tuple[InterpretationHypothesis, ...],
    str | None,
    tuple[SemanticGraphNode, ...],
    tuple[SemanticGraphEdge, ...],
]:
    query = normalized.original_query
    presentation = _presentation(normalized)
    has_group_structure = bool(_GROUP_CUE.search(query))
    bindings = _role_bindings(fields, schema, has_group_structure)
    transforms, aggregation, sort, limit, transform_ambiguity = _effective_operations(
        normalized=normalized,
        base=base,
        bindings=bindings,
        presentation=presentation,
    )
    measures = {item.field_name for item in bindings if item.role == "measure"}
    dimensions = {item.field_name for item in bindings if item.role == "dimension"}
    temporal = {item.field_name for item in bindings if item.role == "temporal"}
    non_temporal_dimensions = dimensions - temporal
    aggregate_transforms = tuple(
        item for item in transforms if item.transform_type == "aggregate"
    )
    derived_measure_count = sum(
        len(item.field_candidates) if item.field_candidates else (1 if item.operation == "count" else 0)
        for item in aggregate_transforms
    )
    measure_count = derived_measure_count if aggregate_transforms else len(measures)
    relation_cue = any(item.canonical == "correlation" for item in normalized.task_terms)
    part_whole = bool(_PART_WHOLE_CUE.search(query)) or "pie" in presentation.chart_hints
    grouped_superlative = bool(dimensions) and has_group_structure and bool(
        _SUPERLATIVE_MAX.search(query) or _SUPERLATIVE_MIN.search(query)
    )

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    constraints: dict[str, set[str]] = defaultdict(set)

    def support(goal: str, component: str, score: float, *satisfied: str) -> None:
        scores[goal][component] = max(score, scores[goal].get(component, 0.0))
        constraints[goal].update(satisfied)

    if aggregate_transforms and non_temporal_dimensions:
        if part_whole and any(item.operation == "count" for item in aggregate_transforms):
            support("composition", "count_dimension_part_whole", 0.94, "dimension_present", "derived_measure_present", "part_whole_supported")
        else:
            support("comparison", "aggregate_by_dimension", 0.92, "dimension_present", "derived_measure_present")
    elif measures and non_temporal_dimensions:
        support("comparison", "measure_dimension_structure", 0.84, "dimension_present", "measure_present")

    if grouped_superlative:
        support("comparison", "grouped_superlative_aggregate", 0.96, "dimension_present", "derived_measure_present", "grouped_superlative")

    if temporal and measure_count:
        support("trend", "temporal_measure_structure", 0.92, "temporal_present", "measure_present")

    # A relationship lexeme alone is insufficient. Two executable measure
    # expressions are a hard prerequisite for correlation.
    correlation_measure_count = len(measures)
    if aggregate_transforms and any(item.operation == "count" for item in aggregate_transforms):
        correlation_measure_count += 1
    if relation_cue and correlation_measure_count >= 2:
        support("correlation", "relation_two_measure_structure", 0.94, "two_measures_present", "relation_cue")
    if "scatter" in presentation.chart_hints and correlation_measure_count >= 2:
        support("correlation", "scatter_two_measure_structure", 0.93, "two_measures_present", "scatter_hint")

    if presentation.chart_hints and {"histogram", "boxplot"} & set(presentation.chart_hints) and measures:
        support("distribution", "distribution_chart_measure", 0.94, "measure_present", "distribution_chart_hint")
    explicit_distribution = any(item.canonical == "distribution" for item in normalized.task_terms)
    if explicit_distribution and measures and not dimensions:
        support("distribution", "explicit_statistical_distribution", 0.88, "measure_present", "distribution_cue")

    if "pie" in presentation.chart_hints and dimensions and measure_count:
        support("composition", "pie_dimension_measure", 0.95, "dimension_present", "measure_present", "part_whole_supported")
    if "bar" in presentation.chart_hints and dimensions and measure_count and not part_whole:
        support("comparison", "bar_dimension_measure", 0.91, "dimension_present", "measure_present", "bar_hint")
    if "line" in presentation.chart_hints and temporal and measure_count:
        support("trend", "line_temporal_measure", 0.95, "temporal_present", "measure_present", "line_hint")

    for term in normalized.task_terms:
        if term.canonical == "comparison" and dimensions and measure_count:
            support("comparison", "explicit_goal", 0.90, "explicit_comparison")
        elif term.canonical == "composition" and dimensions and measure_count:
            support("composition", "explicit_goal", 0.90, "explicit_composition")
        elif term.canonical == "trend" and temporal and measure_count:
            support("trend", "explicit_goal", 0.90, "explicit_trend")
        elif term.canonical == "correlation" and correlation_measure_count >= 2:
            support("correlation", "explicit_goal", 0.90, "explicit_correlation")

    # Entity-answer superlatives are ranking/extremum; max/min per group remains
    # a comparison with an aggregate operation.
    if (
        not grouped_superlative
        and dimensions
        and measure_count
        and normalized.detected_superlatives
        and (_ENTITY_ANSWER_CUE.search(query) or normalized.ranking_terms)
    ):
        support("extremum", "entity_superlative", 0.95, "dimension_present", "measure_present", "entity_answer")
        support("ranking", "rank_structure", 0.88, "dimension_present", "measure_present", "sort_supported")

    if not scores and len(measures) == 1 and not dimensions:
        support("distribution", "bare_measure", 0.56, "measure_present")
    if not scores and len(measures) >= 2 and not dimensions:
        support("correlation", "bare_multi_measure", 0.54, "two_measures_present")

    required_roles = {
        "comparison": ("dimension", "measure"),
        "composition": ("dimension", "measure"),
        "trend": ("measure", "temporal"),
        "correlation": ("measure",),
        "distribution": ("measure",),
        "ranking": ("dimension", "measure"),
        "extremum": ("dimension", "measure"),
    }
    full = _full_span(normalized, "joint_goal_evidence")
    hypotheses: list[InterpretationHypothesis] = []
    task_candidates: list[TaskCandidate] = []
    for goal, components in scores.items():
        score = max(components.values())
        violated: list[str] = []
        if goal in {"comparison", "composition", "ranking", "extremum"} and not dimensions:
            violated.append("missing_dimension")
        if goal in {"comparison", "composition", "trend", "distribution", "ranking", "extremum"} and measure_count == 0:
            violated.append("missing_measure_expression")
        if goal == "trend" and not temporal:
            violated.append("missing_temporal")
        if goal == "correlation" and correlation_measure_count < 2:
            violated.append("missing_second_measure")
        complete = not violated
        payload = {
            "goal": goal,
            "fields": [(item.field_name, item.role) for item in bindings],
            "operations": [(item.transform_type, item.operation, item.field_candidates) for item in transforms],
            "presentation": presentation.chart_hints,
            "score": score,
        }
        hypothesis = InterpretationHypothesis(
            hypothesis_id=_hypothesis_id(payload),
            analytical_goal=goal,
            field_bindings=bindings,
            operations=transforms,
            presentation=presentation,
            satisfied_constraints=tuple(sorted(constraints[goal])),
            violated_constraints=tuple(sorted(violated)),
            evidence_spans=_dedupe_spans((full, *presentation.evidence_spans)),
            score=round(score, 6),
            complete=complete,
            rule_ids=registry.validate_ids({"R-TP-020", "R-TP-021", "R-TP-022"}),
        )
        hypotheses.append(hypothesis)
        if complete:
            task_candidates.append(
                TaskCandidate(
                    task=goal,
                    score=round(score, 6),
                    evidence_spans=hypothesis.evidence_spans,
                    required_roles=required_roles[goal],
                    rule_ids=registry.validate_ids({"R-TP-020", "R-TP-021", "R-TP-022"}),
                    score_components={key: round(value, 6) for key, value in sorted(components.items())},
                )
            )

    task_candidates.sort(key=lambda item: (-item.score, _GOAL_ORDER[item.task]))
    hypotheses.sort(key=lambda item: (not item.complete, -item.score, _GOAL_ORDER[item.analytical_goal]))
    ambiguity = [item for item in fields.ambiguity]
    ambiguity.extend(transform_ambiguity)
    if not task_candidates:
        unresolved = TaskCandidate(
            task="unresolved",
            score=1.0,
            evidence_spans=(),
            required_roles=(),
            rule_ids=("R-TP-023",),
            score_components={"no_complete_hypothesis": 1.0},
        )
        task_candidates = [unresolved]
        primary = "unresolved"
        selected_id = None
        failure_state = "no_complete_field_task_hypothesis"
    else:
        primary = task_candidates[0].task
        selected = next(item for item in hypotheses if item.complete and item.analytical_goal == primary)
        selected_id = selected.hypothesis_id
        failure_state = None
        if len(task_candidates) > 1 and task_candidates[0].score - task_candidates[1].score <= settings.hypothesis_ambiguity_margin:
            ambiguity.append(
                AmbiguityRecord(
                    dimension="task",
                    alternatives=(task_candidates[0].task, task_candidates[1].task),
                    reason="multiple complete analytical-goal hypotheses remain within the frozen margin",
                    rule_ids=("R-TP-023",),
                )
            )

    applied = set(base.applied_rule_ids)
    applied.update({"R-TP-020", "R-TP-021", "R-TP-022", "R-TP-023"})
    result = AnalyticalTaskResult(
        task_candidates=tuple(task_candidates),
        task_scores={item.task: item.score for item in task_candidates},
        primary_task=primary,
        secondary_tasks=tuple(item.task for item in task_candidates[1:]),
        query_evidence_spans=_dedupe_spans(
            span for item in task_candidates for span in item.evidence_spans
        ),
        required_roles=task_candidates[0].required_roles,
        transforms=transforms,
        aggregation=aggregation,
        filters=tuple(item for item in transforms if item.transform_type == "filter"),
        sort=sort,
        limit=limit,
        temporal_scope=base.temporal_scope,
        ambiguity=tuple(
            sorted(
                {(
                    item.dimension,
                    item.alternatives,
                    item.reason,
                    item.rule_ids,
                ): item for item in ambiguity}.values(),
                key=lambda item: (item.dimension, item.alternatives, item.reason),
            )
        ),
        applied_rule_ids=registry.validate_ids(applied),
        failure_state=failure_state,
    )
    nodes, edges = build_semantic_graph(
        mentions=semantic_mentions,
        fields=fields,
        hypotheses=tuple(hypotheses),
        selected_hypothesis_id=selected_id,
    )
    return result, tuple(hypotheses), selected_id, nodes, edges


def _node_id(kind: str, value: str) -> str:
    return f"{kind}-{hashlib.sha1(value.encode()).hexdigest()[:12]}"


def build_semantic_graph(
    *,
    mentions: tuple[SemanticFieldMention, ...],
    fields: FieldResolutionResult,
    hypotheses: tuple[InterpretationHypothesis, ...],
    selected_hypothesis_id: str | None,
) -> tuple[tuple[SemanticGraphNode, ...], tuple[SemanticGraphEdge, ...]]:
    nodes: dict[str, SemanticGraphNode] = {}
    edges: dict[str, SemanticGraphEdge] = {}

    def add_node(node_type: str, label: str, attributes: dict[str, str] | None = None) -> str:
        node_id = _node_id(node_type, label)
        nodes[node_id] = SemanticGraphNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            attributes=attributes or {},
        )
        return node_id

    def add_edge(source: str, target: str, relation: str, rule_ids: tuple[str, ...]) -> None:
        key = f"{source}:{target}:{relation}:{','.join(rule_ids)}"
        edge_id = "edge-" + hashlib.sha1(key.encode()).hexdigest()[:14]
        edges[edge_id] = SemanticGraphEdge(
            edge_id=edge_id,
            source=source,
            target=target,
            relation=relation,
            rule_ids=rule_ids,
        )

    semantic_by_id = {item.mention_id: item for item in mentions}
    mention_nodes = {}
    for item in fields.query_mentions:
        semantic = semantic_by_id.get(item.mention_id)
        attributes = (
            {"concept": semantic.concept_id, "kind": semantic.concept_kind}
            if semantic
            else {"kind": item.mention_kind, "surface": item.text}
        )
        mention_nodes[item.mention_id] = add_node("mention", item.mention_id, attributes)
    required = set(fields.required_field_candidates)
    field_nodes = {
        item.field_name: add_node("field", item.field_name)
        for item in fields.candidate_fields
        if item.field_name in required
    }
    for candidate in fields.candidate_fields:
        if candidate.field_name not in field_nodes:
            continue
        for mention_id in candidate.mention_ids:
            if mention_id in mention_nodes:
                add_edge(mention_nodes[mention_id], field_nodes[candidate.field_name], "grounded_as", ("R-FR-021", "R-FR-022"))

    selected = next(
        (item for item in hypotheses if item.hypothesis_id == selected_hypothesis_id),
        None,
    )
    if selected is not None:
        goal_node = add_node("goal", selected.analytical_goal, {"hypothesis_id": selected.hypothesis_id})
        for binding in selected.field_bindings:
            if binding.field_name not in field_nodes:
                continue
            role_label = f"{binding.field_name}:{binding.role}"
            role_node = add_node("role", role_label)
            add_edge(field_nodes[binding.field_name], role_node, "fills_role", ("R-TP-021",))
            add_edge(role_node, goal_node, "supports_goal", ("R-TP-021", "R-TP-022"))
        for operation in selected.operations:
            label = f"{operation.transform_type}:{operation.operation}:{','.join(operation.field_candidates)}"
            operation_node = add_node("operation", label)
            add_edge(goal_node, operation_node, "requires_operation", ("R-TP-020", "R-TP-022"))
        for hint in selected.presentation.chart_hints:
            hint_node = add_node("presentation", hint)
            add_edge(hint_node, goal_node, "constrains_presentation", ("R-TP-020", "R-TP-021"))
    return (
        tuple(nodes[key] for key in sorted(nodes)),
        tuple(edges[key] for key in sorted(edges)),
    )


__all__ = ["build_semantic_graph", "run_joint_task_resolution"]
