"""Close ranking intent from a typed target-score relation."""

from __future__ import annotations

import hashlib
import json
import re

from skillvis_v2_accuracy.contracts import AnalyticalTaskResult, EvidenceSpan, FieldResolutionResult, QueryNormalizationResult, SchemaProfile, TaskCandidate
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_joint.contracts import InterpretationHypothesis
from skillvis_v2_joint.task_resolution import build_semantic_graph


_DIRECTIVE = re.compile(r"\b(rank|order)\b", re.IGNORECASE)


def _hypothesis_id(payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "hyp-" + hashlib.sha256(raw.encode()).hexdigest()[:18]


def apply_ranking_role_closure(*, normalized: QueryNormalizationResult, schema: SchemaProfile, fields: FieldResolutionResult, task: AnalyticalTaskResult, hypotheses: tuple[InterpretationHypothesis, ...], semantic_mentions: tuple, registry: RuleRegistry):
    match = _DIRECTIVE.search(normalized.original_query)
    profiles = {item.field_name: item.semantic_type for item in schema.fields}
    required = set(fields.required_field_candidates)
    dimensions = sorted(name for name in required if profiles.get(name) in {"categorical", "ordinal", "geographic", "identifier", "temporal"})
    measures = sorted(name for name in required if profiles.get(name) == "quantitative")
    eligible = bool(match and "R-FR-120" in fields.applied_rule_ids and task.aggregation is not None and dimensions and measures)
    if not eligible or (task.primary_task == "ranking" and [item.task for item in task.task_candidates] == ["ranking"]):
        selected = next((item.hypothesis_id for item in hypotheses if item.complete and item.analytical_goal == task.primary_task), None)
        nodes, edges = build_semantic_graph(mentions=semantic_mentions, fields=fields, hypotheses=hypotheses, selected_hypothesis_id=selected)
        return task, hypotheses, selected, nodes, edges, False
    directive = EvidenceSpan(start=match.start(), end=match.end(), text=match.group(0), normalized="ranking", kind="ranking_role_directive", rule_ids=("R-TP-130",))
    score = 0.998
    candidate = TaskCandidate(task="ranking", score=score, evidence_spans=(directive,), required_roles=("dimension", "measure"), rule_ids=("R-TP-130",), score_components={"ranking_directive": score, "typed_target_score_relation": 1.0, "aggregate_score": 1.0})
    updated = AnalyticalTaskResult.model_validate(task.model_copy(update={"task_candidates": (candidate,), "task_scores": {"ranking": score}, "primary_task": "ranking", "secondary_tasks": (), "query_evidence_spans": (directive,), "required_roles": ("dimension", "measure"), "ambiguity": tuple(item for item in task.ambiguity if item.dimension != "task"), "applied_rule_ids": registry.validate_ids((*task.applied_rule_ids, "R-TP-130")), "failure_state": None}).model_dump(mode="json"))
    seed = next((item for item in hypotheses if item.complete), None)
    if seed is None:
        return updated, hypotheses, None, (), (), True
    payload = {"goal": "ranking", "seed": seed.hypothesis_id, "directive": (directive.start, directive.end), "score": score}
    promoted = seed.model_copy(update={"hypothesis_id": _hypothesis_id(payload), "analytical_goal": "ranking", "satisfied_constraints": tuple(sorted({*seed.satisfied_constraints, "ranking_target_score_closure", "specific_task_dominance"})), "violated_constraints": (), "evidence_spans": tuple(sorted({*seed.evidence_spans, directive}, key=lambda item: (item.start, item.end, item.kind))), "score": score, "complete": True, "rule_ids": registry.validate_ids((*seed.rule_ids, "R-TP-130"))})
    new_hypotheses = tuple(sorted((promoted, *hypotheses), key=lambda item: (not item.complete, -item.score, item.analytical_goal, item.hypothesis_id)))
    selected = promoted.hypothesis_id
    nodes, edges = build_semantic_graph(mentions=semantic_mentions, fields=fields, hypotheses=new_hypotheses, selected_hypothesis_id=selected)
    return updated, new_hypotheses, selected, nodes, edges, True


__all__ = ["apply_ranking_role_closure"]
