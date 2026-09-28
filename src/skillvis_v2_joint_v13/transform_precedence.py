"""Keep score aggregation separate from ranking direction."""

from __future__ import annotations

import re

from skillvis_v2_accuracy.contracts import (
    AnalyticalTaskResult,
    EvidenceSpan,
    TransformIntent,
)
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_joint.contracts import InterpretationHypothesis
from skillvis_v2_joint.task_resolution import build_semantic_graph


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


def _direction(query: str):
    descending = _DESCENDING.search(query)
    ascending = _ASCENDING.search(query)
    if bool(descending) == bool(ascending):
        return None
    return ("descending", descending) if descending else ("ascending", ascending)


def apply_ordering_aggregation_precedence(
    *,
    query: str,
    base: AnalyticalTaskResult,
    task: AnalyticalTaskResult,
    hypotheses: tuple[InterpretationHypothesis, ...],
    semantic_mentions: tuple,
    fields,
    registry: RuleRegistry,
):
    direction = _direction(query)
    explicit = base.aggregation
    if task.primary_task != "ranking" or explicit is None or direction is None:
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

    sort_direction, match = direction
    removed = tuple(
        item
        for item in task.transforms
        if item.transform_type == "aggregate"
        and item.operation in {"max", "min"}
        and "R-TP-021" in item.rule_ids
        and item.operation != explicit.operation
    )
    transforms = tuple(
        item for item in task.transforms if item not in removed and item.transform_type != "sort"
    )
    if explicit not in transforms:
        transforms = (*transforms, explicit)
    direction_span = EvidenceSpan(
        start=match.start(),
        end=match.end(),
        text=match.group(0),
        normalized=sort_direction,
        kind="ranking_direction",
        rule_ids=("R-TP-140",),
    )
    score_fields = explicit.field_candidates
    sort = TransformIntent(
        transform_type="sort",
        operation=sort_direction,
        field_candidates=score_fields,
        evidence_spans=(direction_span,),
        rule_ids=("R-TP-140",),
    )
    transforms = tuple(
        sorted(
            (*transforms, sort),
            key=lambda item: (
                item.transform_type,
                item.operation,
                item.field_candidates,
            ),
        )
    )
    ambiguity = tuple(
        item
        for item in task.ambiguity
        if not (
            item.dimension == "transform"
            and set(item.alternatives) == {"ascending", "descending"}
        )
    )
    updated = AnalyticalTaskResult.model_validate(
        task.model_copy(
            update={
                "transforms": transforms,
                "aggregation": explicit,
                "sort": sort,
                "ambiguity": ambiguity,
                "applied_rule_ids": registry.validate_ids(
                    (*task.applied_rule_ids, "R-TP-140")
                ),
            }
        ).model_dump(mode="json")
    )

    updated_hypotheses = []
    for hypothesis in hypotheses:
        if not hypothesis.complete or hypothesis.analytical_goal != "ranking":
            updated_hypotheses.append(hypothesis)
            continue
        updated_hypotheses.append(
            hypothesis.model_copy(
                update={
                    "operations": transforms,
                    "satisfied_constraints": tuple(
                        sorted(
                            {
                                *hypothesis.satisfied_constraints,
                                "explicit_aggregation_precedes_order_direction",
                            }
                        )
                    ),
                    "rule_ids": registry.validate_ids(
                        (*hypothesis.rule_ids, "R-TP-140")
                    ),
                }
            )
        )
    updated_hypotheses = tuple(updated_hypotheses)
    selected = next(
        (
            item.hypothesis_id
            for item in updated_hypotheses
            if item.complete and item.analytical_goal == "ranking"
        ),
        None,
    )
    nodes, edges = build_semantic_graph(
        mentions=semantic_mentions,
        fields=fields,
        hypotheses=updated_hypotheses,
        selected_hypothesis_id=selected,
    )
    trace = (
        f"R-TP-140 ordering precedence:aggregate={explicit.operation}:"
        f"sort={sort_direction}:removed={','.join(item.operation for item in removed) or 'none'}",
    )
    return updated, updated_hypotheses, selected, nodes, edges, True, trace


__all__ = ["apply_ordering_aggregation_precedence"]
