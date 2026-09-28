"""Specific analytical-task closure from discourse and typed roles."""

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
from skillvis_v2_joint.contracts import InterpretationHypothesis
from skillvis_v2_joint.task_resolution import build_semantic_graph


_CONTRAST = re.compile(r"\b(compare|contrast)\b", re.IGNORECASE)
_PART_TO_WHOLE = re.compile(r"\b(shares?|proportions?)\b", re.IGNORECASE)


def _hypothesis_id(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "hyp-" + hashlib.sha256(raw.encode()).hexdigest()[:18]


def _span(match: re.Match, goal: str) -> EvidenceSpan:
    return EvidenceSpan(
        start=match.start(), end=match.end(), text=match.group(0), normalized=goal,
        kind="analytical_discourse_directive", rule_ids=("R-TP-050",),
    )


def apply_discourse_task_closure(
    *,
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    task: AnalyticalTaskResult,
    hypotheses: tuple[InterpretationHypothesis, ...],
    semantic_mentions: tuple,
    registry: RuleRegistry,
) -> tuple[AnalyticalTaskResult, tuple[InterpretationHypothesis, ...], str | None, tuple, tuple, bool]:
    query = normalized.original_query
    profiles = {item.field_name: item.semantic_type for item in schema.fields}
    required = set(fields.required_field_candidates)
    measures = sorted(name for name in required if profiles.get(name) == "quantitative")
    dimensions = sorted(name for name in required if profiles.get(name) in {"categorical", "ordinal", "geographic", "identifier", "temporal"})
    part_match = _PART_TO_WHOLE.search(query)
    contrast_match = _CONTRAST.search(query)
    pie = any(item.canonical == "pie" for item in normalized.chart_mentions)
    goal = None
    directive = None
    required_roles: tuple[str, ...] = ()
    if part_match and pie and len(dimensions) == 1 and not measures:
        goal, directive, required_roles = "composition", _span(part_match, "composition"), ("dimension",)
    elif contrast_match and task.aggregation is not None and measures and dimensions:
        goal, directive, required_roles = "comparison", _span(contrast_match, "comparison"), ("dimension", "measure")
    if goal is None or (task.primary_task == goal and [item.task for item in task.task_candidates] == [goal]):
        selected = next((item.hypothesis_id for item in hypotheses if item.complete and item.analytical_goal == task.primary_task), None)
        nodes, edges = build_semantic_graph(mentions=semantic_mentions, fields=fields, hypotheses=hypotheses, selected_hypothesis_id=selected)
        return task, hypotheses, selected, nodes, edges, False

    score = 0.997
    candidate = TaskCandidate(
        task=goal,
        score=score,
        evidence_spans=(directive,),
        required_roles=required_roles,
        rule_ids=("R-TP-050",),
        score_components={"analytical_discourse": score, "typed_role_closure": 1.0, "specificity_dominance": 1.0},
    )
    updated = AnalyticalTaskResult.model_validate(
        task.model_copy(
            update={
                "task_candidates": (candidate,), "task_scores": {goal: score}, "primary_task": goal,
                "secondary_tasks": (), "query_evidence_spans": (directive,), "required_roles": required_roles,
                "ambiguity": tuple(item for item in task.ambiguity if item.dimension != "task"),
                "applied_rule_ids": registry.validate_ids((*task.applied_rule_ids, "R-TP-050")), "failure_state": None,
            }
        ).model_dump(mode="json")
    )
    seed = next((item for item in hypotheses if item.complete), None)
    if seed is None:
        return updated, hypotheses, None, (), (), True
    payload = {"goal": goal, "seed": seed.hypothesis_id, "directive": (directive.start, directive.end), "score": score}
    promoted = seed.model_copy(
        update={
            "hypothesis_id": _hypothesis_id(payload), "analytical_goal": goal,
            "satisfied_constraints": tuple(sorted({*seed.satisfied_constraints, "analytical_discourse_closure", "specific_task_dominance"})),
            "violated_constraints": (), "evidence_spans": tuple(sorted({*seed.evidence_spans, directive}, key=lambda item: (item.start, item.end, item.kind))),
            "score": score, "complete": True, "rule_ids": registry.validate_ids((*seed.rule_ids, "R-TP-050")),
        }
    )
    new_hypotheses = tuple(sorted((promoted, *hypotheses), key=lambda item: (not item.complete, -item.score, item.analytical_goal, item.hypothesis_id)))
    selected = promoted.hypothesis_id
    nodes, edges = build_semantic_graph(mentions=semantic_mentions, fields=fields, hypotheses=new_hypotheses, selected_hypothesis_id=selected)
    return updated, new_hypotheses, selected, nodes, edges, True


__all__ = ["apply_discourse_task_closure"]
