"""Close ranking before directional superlatives can become aggregates."""

from __future__ import annotations

import hashlib
import json
import re

from skillvis_v2_accuracy.contracts import (
    AnalyticalTaskResult,
    EvidenceSpan,
    FieldResolutionResult,
    SchemaProfile,
    TaskCandidate,
    TransformIntent,
)
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_joint.contracts import InterpretationHypothesis
from skillvis_v2_joint.task_resolution import build_semantic_graph


_DIRECTIVE = re.compile(r"\b(rank|order)\b", re.IGNORECASE)
_LINK = re.compile(r"\b(by|using|based\s+on|according\s+to)\b", re.IGNORECASE)
_DESCENDING = re.compile(
    r"\b(from\s+(?:highest|largest)\s+to\s+(?:lowest|smallest)|"
    r"(?:highest|largest)\s+first|descending(?:\s+order)?)\b",
    re.IGNORECASE,
)
_ASCENDING = re.compile(
    r"\b(from\s+(?:lowest|smallest)\s+to\s+(?:highest|largest)|"
    r"(?:lowest|smallest)\s+first|ascending(?:\s+order)?)\b",
    re.IGNORECASE,
)
_STRONG = {
    "exact",
    "normalized_lexical",
    "alias",
    "acronym_expansion",
    "value",
    "semantic_concept",
    "ranking_target_role",
    "set_noun_grouping_role",
}


def _hypothesis_id(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "hyp-" + hashlib.sha256(raw.encode()).hexdigest()[:18]


def _direction(query: str):
    descending = _DESCENDING.search(query)
    ascending = _ASCENDING.search(query)
    if bool(descending) == bool(ascending):
        return None
    return ("descending", descending) if descending else ("ascending", ascending)


def _grounded_spans(candidate):
    return tuple(
        evidence.query_span
        for evidence in candidate.evidence
        if evidence.query_span is not None and evidence.evidence_type in _STRONG
    )


def close_typed_ranking_relation(
    *,
    query: str,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    base: AnalyticalTaskResult,
    task: AnalyticalTaskResult,
    hypotheses: tuple[InterpretationHypothesis, ...],
    semantic_mentions: tuple,
    registry: RuleRegistry,
):
    directive = _DIRECTIVE.search(query)
    link = _LINK.search(query, directive.end() if directive else 0)
    direction = _direction(query)
    explicit = base.aggregation
    if directive is None or link is None or direction is None or explicit is None:
        selected = next(
            (
                item.hypothesis_id
                for item in hypotheses
                if item.complete and item.analytical_goal == task.primary_task
            ),
            None,
        )
        nodes, edges = build_semantic_graph(
            mentions=semantic_mentions,
            fields=fields,
            hypotheses=hypotheses,
            selected_hypothesis_id=selected,
        )
        return task, hypotheses, selected, nodes, edges, False, ()

    profile_types = {item.field_name: item.semantic_type for item in schema.fields}
    required = set(fields.required_field_candidates)
    candidates = {
        item.field_name: item
        for item in fields.candidate_fields
        if item.field_name in required
    }
    targets = []
    scores = []
    for name, candidate in candidates.items():
        spans = _grounded_spans(candidate)
        if any(directive.end() <= span.start and span.end <= link.start() for span in spans):
            targets.append(name)
        if any(span.start >= link.end() for span in spans):
            scores.append(name)
    targets = sorted(set(targets))
    scores = sorted(set(scores))
    if (
        len(targets) != 1
        or len(scores) != 1
        or targets[0] == scores[0]
        or scores[0] not in explicit.field_candidates
        or profile_types.get(targets[0])
        not in {"categorical", "ordinal", "geographic", "identifier", "temporal"}
        or profile_types.get(scores[0]) != "quantitative"
    ):
        selected = next(
            (
                item.hypothesis_id
                for item in hypotheses
                if item.complete and item.analytical_goal == task.primary_task
            ),
            None,
        )
        nodes, edges = build_semantic_graph(
            mentions=semantic_mentions,
            fields=fields,
            hypotheses=hypotheses,
            selected_hypothesis_id=selected,
        )
        return task, hypotheses, selected, nodes, edges, False, ()

    sort_direction, direction_match = direction
    direction_span = EvidenceSpan(
        start=direction_match.start(),
        end=direction_match.end(),
        text=direction_match.group(0),
        normalized=sort_direction,
        kind="ranking_direction",
        rule_ids=("R-TP-140", "R-TP-141"),
    )
    directive_span = EvidenceSpan(
        start=directive.start(),
        end=directive.end(),
        text=directive.group(0),
        normalized="ranking",
        kind="typed_ranking_directive",
        rule_ids=("R-TP-141",),
    )
    sort = TransformIntent(
        transform_type="sort",
        operation=sort_direction,
        field_candidates=explicit.field_candidates,
        evidence_spans=(direction_span,),
        rule_ids=("R-TP-140", "R-TP-141"),
    )
    transforms = tuple(
        sorted(
            (
                explicit,
                sort,
                *(
                    item
                    for item in task.transforms
                    if item.transform_type not in {"aggregate", "sort", "limit"}
                ),
            ),
            key=lambda item: (
                item.transform_type,
                item.operation,
                item.field_candidates,
            ),
        )
    )
    score = 0.999
    ranking = TaskCandidate(
        task="ranking",
        score=score,
        evidence_spans=(directive_span, direction_span),
        required_roles=("dimension", "measure"),
        rule_ids=("R-TP-141",),
        score_components={
            "typed_target_link_score_relation": 1.0,
            "explicit_aggregate_score": 1.0,
            "explicit_order_direction": 1.0,
            "specificity_dominance": score,
        },
    )
    ambiguity = tuple(
        item
        for item in task.ambiguity
        if not (
            item.dimension == "transform"
            and set(item.alternatives) == {"ascending", "descending"}
        )
        and not (
            item.dimension == "task"
            and ({"ranking", "comparison", "extremum"} & set(item.alternatives))
        )
    )
    updated = AnalyticalTaskResult.model_validate(
        task.model_copy(
            update={
                "task_candidates": (ranking,),
                "task_scores": {"ranking": score},
                "primary_task": "ranking",
                "secondary_tasks": (),
                "query_evidence_spans": (directive_span, direction_span),
                "required_roles": ("dimension", "measure"),
                "transforms": transforms,
                "aggregation": explicit,
                "sort": sort,
                "limit": None,
                "ambiguity": ambiguity,
                "applied_rule_ids": registry.validate_ids(
                    (*task.applied_rule_ids, "R-TP-140", "R-TP-141")
                ),
                "failure_state": None,
            }
        ).model_dump(mode="json")
    )

    seed = next(
        (
            item
            for item in hypotheses
            if item.complete and item.analytical_goal == "ranking"
        ),
        next((item for item in hypotheses if item.complete), None),
    )
    if seed is None:
        return updated, hypotheses, None, (), (), True, (
            "R-TP-141 task closed without complete hypothesis seed",
        )
    payload = {
        "goal": "ranking",
        "seed": seed.hypothesis_id,
        "target": targets[0],
        "score_field": scores[0],
        "aggregation": explicit.operation,
        "direction": sort_direction,
    }
    promoted = seed.model_copy(
        update={
            "hypothesis_id": _hypothesis_id(payload),
            "analytical_goal": "ranking",
            "operations": transforms,
            "satisfied_constraints": tuple(
                sorted(
                    {
                        *seed.satisfied_constraints,
                        "typed_target_link_score_relation",
                        "explicit_aggregation_precedes_order_direction",
                        "specific_task_dominance",
                    }
                )
            ),
            "violated_constraints": (),
            "evidence_spans": tuple(
                sorted(
                    {*seed.evidence_spans, directive_span, direction_span},
                    key=lambda item: (item.start, item.end, item.kind),
                )
            ),
            "score": score,
            "complete": True,
            "rule_ids": registry.validate_ids(
                (*seed.rule_ids, "R-TP-140", "R-TP-141")
            ),
        }
    )
    updated_hypotheses = tuple(
        sorted(
            (
                promoted,
                *(
                    item
                    for item in hypotheses
                    if item.analytical_goal not in {"ranking", "comparison", "extremum"}
                ),
            ),
            key=lambda item: (
                not item.complete,
                -item.score,
                item.analytical_goal,
                item.hypothesis_id,
            ),
        )
    )
    selected = promoted.hypothesis_id
    nodes, edges = build_semantic_graph(
        mentions=semantic_mentions,
        fields=fields,
        hypotheses=updated_hypotheses,
        selected_hypothesis_id=selected,
    )
    suppressed = sorted(
        {item.task for item in task.task_candidates if item.task != "ranking"}
    )
    trace = (
        f"R-TP-141 typed ranking:{targets[0]}:{link.group(1).casefold()}:{scores[0]}:aggregate={explicit.operation}:sort={sort_direction}:suppressed={','.join(suppressed) or 'none'}",
    )
    return updated, updated_hypotheses, selected, nodes, edges, True, trace


__all__ = ["close_typed_ranking_relation"]
