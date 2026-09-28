"""Structurally guarded compilation of explicit ranking directives."""

from __future__ import annotations

import hashlib
import json

from skillvis_v2_accuracy.contracts import (
    AnalyticalTaskResult,
    EvidenceSpan,
    FieldResolutionResult,
    QueryNormalizationResult,
    SchemaProfile,
    TaskCandidate,
)
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_joint.config import JointSemanticSettings
from skillvis_v2_joint.contracts import InterpretationHypothesis, SemanticFieldMention
from skillvis_v2_joint.task_resolution import (
    build_semantic_graph,
    run_joint_task_resolution,
)


def _hypothesis_id(payload: dict) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "hyp-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18]


def _explicit_rank_spans(normalized: QueryNormalizationResult) -> tuple[EvidenceSpan, ...]:
    rows = {}
    for term in normalized.task_terms:
        if term.canonical != "ranking" or term.text.casefold() not in {"rank", "ranking"}:
            continue
        span = EvidenceSpan(
            start=term.start,
            end=term.end,
            text=term.text,
            normalized="ranking",
            kind="explicit_ranking_goal",
            rule_ids=("R-TP-030",),
        )
        rows[(span.start, span.end, span.kind)] = span
    return tuple(rows[key] for key in sorted(rows))


def _dedupe_spans(items) -> tuple[EvidenceSpan, ...]:
    rows = {
        (item.start, item.end, item.kind, item.normalized, item.rule_ids): item
        for item in items
    }
    return tuple(rows[key] for key in sorted(rows))


def run_joint_v3_task_resolution(
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
    tuple,
    tuple,
    bool,
]:
    result, hypotheses, selected_id, nodes, edges = run_joint_task_resolution(
        normalized=normalized,
        schema=schema,
        fields=fields,
        base=base,
        semantic_mentions=semantic_mentions,
        settings=settings,
        registry=registry,
    )
    rank_spans = _explicit_rank_spans(normalized)
    if not rank_spans:
        return result, hypotheses, selected_id, nodes, edges, False

    required = set(fields.required_field_candidates)
    profile = {item.field_name: item.semantic_type for item in schema.fields}
    measures = tuple(
        sorted(name for name in required if profile.get(name) == "quantitative")
    )
    dimensions = tuple(
        sorted(
            name
            for name in required
            if profile.get(name)
            in {"categorical", "ordinal", "geographic", "identifier", "temporal"}
        )
    )
    # Lexical rank evidence is not allowed to override missing structural roles.
    if not measures or not dimensions:
        return result, hypotheses, selected_id, nodes, edges, False

    seed = next((item for item in hypotheses if item.complete), None)
    if seed is None:
        return result, hypotheses, selected_id, nodes, edges, False

    score = 0.99
    payload = {
        "goal": "ranking",
        "fields": [(item.field_name, item.role) for item in seed.field_bindings],
        "operations": [
            (item.transform_type, item.operation, item.field_candidates)
            for item in seed.operations
        ],
        "presentation": seed.presentation.chart_hints,
        "explicit_spans": [(item.start, item.end) for item in rank_spans],
        "score": score,
    }
    ranking_hypothesis = seed.model_copy(
        update={
            "hypothesis_id": _hypothesis_id(payload),
            "analytical_goal": "ranking",
            "satisfied_constraints": tuple(
                sorted(
                    {
                        *seed.satisfied_constraints,
                        "explicit_ranking_goal",
                        "dimension_present",
                        "measure_present",
                        "sort_compiled_downstream",
                    }
                )
            ),
            "violated_constraints": (),
            "evidence_spans": _dedupe_spans((*seed.evidence_spans, *rank_spans)),
            "score": score,
            "complete": True,
            "rule_ids": registry.validate_ids((*seed.rule_ids, "R-TP-030")),
        }
    )
    ranking_candidate = TaskCandidate(
        task="ranking",
        score=score,
        evidence_spans=rank_spans,
        required_roles=("dimension", "measure"),
        rule_ids=("R-TP-030",),
        score_components={
            "explicit_ranking_directive": 0.99,
            "structural_role_closure": 1.0,
        },
    )
    candidates = [
        ranking_candidate,
        *(item for item in result.task_candidates if item.task != "ranking"),
    ]
    candidates.sort(key=lambda item: (-item.score, item.task))
    ambiguity = tuple(
        item
        for item in result.ambiguity
        if not (
            item.dimension == "task"
            and {"ranking", "comparison"} <= set(item.alternatives)
        )
    )
    applied = registry.validate_ids((*result.applied_rule_ids, "R-TP-030"))
    result = AnalyticalTaskResult.model_validate(
        result.model_copy(
            update={
                "task_candidates": tuple(candidates),
                "task_scores": {item.task: item.score for item in candidates},
                "primary_task": "ranking",
                "secondary_tasks": tuple(
                    item.task for item in candidates if item.task != "ranking"
                ),
                "query_evidence_spans": _dedupe_spans(
                    span for item in candidates for span in item.evidence_spans
                ),
                "required_roles": ("dimension", "measure"),
                "ambiguity": ambiguity,
                "applied_rule_ids": applied,
                "failure_state": None,
            }
        ).model_dump(mode="json")
    )
    hypotheses = tuple(
        sorted(
            (
                ranking_hypothesis,
                *(item for item in hypotheses if item.analytical_goal != "ranking"),
            ),
            key=lambda item: (not item.complete, -item.score, item.analytical_goal),
        )
    )
    selected_id = ranking_hypothesis.hypothesis_id
    nodes, edges = build_semantic_graph(
        mentions=semantic_mentions,
        fields=fields,
        hypotheses=hypotheses,
        selected_hypothesis_id=selected_id,
    )
    return result, hypotheses, selected_id, nodes, edges, True


__all__ = ["run_joint_v3_task_resolution"]
