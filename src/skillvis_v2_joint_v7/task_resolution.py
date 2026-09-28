"""Specific analytical-task dominance for structurally closed ranking directives."""

from __future__ import annotations

import hashlib
import json
import re

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
from skillvis_v2_joint.task_resolution import build_semantic_graph
from skillvis_v2_joint_v3.task_resolution import run_joint_v3_task_resolution
from skillvis_v2_joint_v6.evidence_strength import EvidenceStrength, classify_evidence


_DIRECTIVE = re.compile(r"\b(rank(?:ing)?|order|sort)\b", re.IGNORECASE)
_BY = re.compile(r"\bby\b", re.IGNORECASE)


def _hypothesis_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "hyp-" + hashlib.sha256(raw.encode()).hexdigest()[:18]


def _dedupe_spans(items) -> tuple[EvidenceSpan, ...]:
    rows = {
        (item.start, item.end, item.kind, item.normalized, item.rule_ids): item
        for item in items
    }
    return tuple(rows[key] for key in sorted(rows))


def _field_owned_spans(fields: FieldResolutionResult) -> tuple[object, ...]:
    return tuple(
        evidence.query_span
        for candidate in fields.candidate_fields
        for evidence in candidate.evidence
        if evidence.query_span is not None
        and classify_evidence(evidence.evidence_type)
        in {EvidenceStrength.EXPLICIT_GROUNDING, EvidenceStrength.SEMANTIC_GROUNDING}
    )


def _overlaps(start: int, end: int, span: object) -> bool:
    return start < span.end and span.start < end


def _directive_spans(
    normalized: QueryNormalizationResult,
    fields: FieldResolutionResult,
    task: AnalyticalTaskResult,
) -> tuple[EvidenceSpan, ...]:
    query = normalized.original_query
    owners = _field_owned_spans(fields)
    rows = []
    for match in _DIRECTIVE.finditer(query):
        if any(_overlaps(match.start(), match.end(), span) for span in owners):
            continue
        lexeme = match.group(1).casefold()
        if lexeme in {"order", "sort"}:
            by_match = _BY.search(query, match.end())
            if by_match is None or task.aggregation is None:
                continue
        rows.append(
            EvidenceSpan(
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                normalized="ranking",
                kind="specific_ranking_directive",
                rule_ids=("R-TP-040",),
            )
        )
    return _dedupe_spans(rows)


def run_joint_v7_task_resolution(
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
    result, hypotheses, selected_id, nodes, edges, _parent_promoted = (
        run_joint_v3_task_resolution(
            normalized=normalized,
            schema=schema,
            fields=fields,
            base=base,
            semantic_mentions=semantic_mentions,
            settings=settings,
            registry=registry,
        )
    )
    directive_spans = _directive_spans(normalized, fields, result)
    if not directive_spans:
        return result, hypotheses, selected_id, nodes, edges, False

    required = set(fields.required_field_candidates)
    profile = {item.field_name: item.semantic_type for item in schema.fields}
    measures = tuple(sorted(name for name in required if profile.get(name) == "quantitative"))
    dimensions = tuple(
        sorted(
            name
            for name in required
            if profile.get(name)
            in {"categorical", "ordinal", "geographic", "identifier", "temporal"}
        )
    )
    if not measures or not dimensions:
        return result, hypotheses, selected_id, nodes, edges, False

    seed = next(
        (item for item in hypotheses if item.analytical_goal == "ranking" and item.complete),
        next((item for item in hypotheses if item.complete), None),
    )
    if seed is None:
        return result, hypotheses, selected_id, nodes, edges, False

    score = 0.995
    payload = {
        "goal": "ranking",
        "fields": [(item.field_name, item.role) for item in seed.field_bindings],
        "operations": [
            (item.transform_type, item.operation, item.field_candidates)
            for item in seed.operations
        ],
        "directives": [(item.start, item.end, item.text) for item in directive_spans],
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
                        "specific_ranking_directive",
                        "dimension_present",
                        "measure_present",
                        "generic_comparison_subsumed",
                    }
                )
            ),
            "violated_constraints": (),
            "evidence_spans": _dedupe_spans((*seed.evidence_spans, *directive_spans)),
            "score": score,
            "complete": True,
            "rule_ids": registry.validate_ids((*seed.rule_ids, "R-TP-040")),
        }
    )
    ranking_candidate = TaskCandidate(
        task="ranking",
        score=score,
        evidence_spans=directive_spans,
        required_roles=("dimension", "measure"),
        rule_ids=("R-TP-040",),
        score_components={
            "specific_ranking_directive": score,
            "structural_role_closure": 1.0,
            "specificity_dominance": 1.0,
        },
    )
    candidates = [
        ranking_candidate,
        *(
            item
            for item in result.task_candidates
            if item.task not in {"ranking", "comparison"}
        ),
    ]
    candidates.sort(key=lambda item: (-item.score, item.task))
    ambiguity = tuple(
        item
        for item in result.ambiguity
        if not (
            item.dimension == "task"
            and ({"ranking", "comparison"} & set(item.alternatives))
        )
    )
    result = AnalyticalTaskResult.model_validate(
        result.model_copy(
            update={
                "task_candidates": tuple(candidates),
                "task_scores": {item.task: item.score for item in candidates},
                "primary_task": "ranking",
                "secondary_tasks": tuple(item.task for item in candidates if item.task != "ranking"),
                "query_evidence_spans": _dedupe_spans(
                    span for item in candidates for span in item.evidence_spans
                ),
                "required_roles": ("dimension", "measure"),
                "ambiguity": ambiguity,
                "applied_rule_ids": registry.validate_ids(
                    (*result.applied_rule_ids, "R-TP-040")
                ),
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


__all__ = ["run_joint_v7_task_resolution"]
