"""Typed intent-obligation compilation before semantic safety and M4 planning."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from skillvis_v2_accuracy.contracts import (
    AnalyticalTaskResult,
    EvidenceSpan,
    IntentFieldRef,
    TaskCandidate,
    TransformIntent,
    VisualizationIntentIR,
)
from skillvis_v2_accuracy.rules import RuleRegistry


_CORE_TASKS = {
    "comparison", "ranking", "trend", "correlation", "distribution",
    "composition", "extremum",
}
_STRONG_GROUNDING = {
    "exact", "normalized_lexical", "alias", "semantic_concept",
    "acronym_expansion",
}
_TIME_UNITS = {
    "daily": "day",
    "weekly": "week",
    "monthly": "month",
    "quarterly": "quarter",
    "yearly": "year",
    "annual": "year",
    "annually": "year",
}


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FieldObligation(FrozenModel):
    field_name: str
    permitted_roles: tuple[str, ...]
    filter_only: bool
    evidence_spans: tuple[EvidenceSpan, ...]


class TransformObligation(FrozenModel):
    transform_type: str
    operation: str
    field_candidates: tuple[str, ...]
    value: object | None = None


class VisualizationObligationSet(FrozenModel):
    schema_version: str = "skillvis-v2-visualization-obligations-v1.0.0"
    obligation_id: str
    intent_id: str
    planning_task: str
    field_obligations: tuple[FieldObligation, ...]
    transform_obligations: tuple[TransformObligation, ...]
    applied_rule_ids: tuple[str, ...]


class IntentCompilationResult(FrozenModel):
    intent: VisualizationIntentIR
    task_result: AnalyticalTaskResult
    obligations: VisualizationObligationSet
    applied_rule_ids: tuple[str, ...]


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _strong_spans(ref: IntentFieldRef) -> tuple[EvidenceSpan, ...]:
    return tuple(
        evidence.query_span
        for evidence in ref.evidence
        if evidence.query_span is not None and evidence.evidence_type in _STRONG_GROUNDING
    )


def _remove_shadowed(refs: tuple[IntentFieldRef, ...]) -> tuple[tuple[IntentFieldRef, ...], bool]:
    shadowed: set[str] = set()
    for short in refs:
        for long in refs:
            if short.field_name == long.field_name or short.role != long.role:
                continue
            for short_span in _strong_spans(short):
                for long_span in _strong_spans(long):
                    if (
                        long_span.start <= short_span.start
                        and short_span.end <= long_span.end
                        and (long_span.end - long_span.start) > (short_span.end - short_span.start)
                    ):
                        shadowed.add(short.field_name)
    return tuple(ref for ref in refs if ref.field_name not in shadowed), bool(shadowed)


def _span(query: str, start: int, end: int, kind: str) -> EvidenceSpan:
    return EvidenceSpan(
        start=start,
        end=end,
        text=query[start:end],
        normalized=query[start:end].casefold(),
        kind=kind,
        rule_ids=("R-IC-002",),
    )


def _dedupe_transforms(items: Iterable[TransformIntent]) -> tuple[TransformIntent, ...]:
    rows = {}
    for item in items:
        key = (
            item.transform_type,
            item.operation,
            item.field_candidates,
            json.dumps(item.value, ensure_ascii=False, sort_keys=True),
            item.negated,
        )
        rows.setdefault(key, item)
    return tuple(rows[key] for key in sorted(rows, key=str))


def compile_intent_obligations(
    intent: VisualizationIntentIR,
    task_result: AnalyticalTaskResult,
    registry: RuleRegistry,
) -> IntentCompilationResult:
    """Close only obligations that are uniquely determined by typed evidence."""

    applied: set[str] = set()
    measures, changed_measure = _remove_shadowed(intent.measures)
    dimensions, changed_dimension = _remove_shadowed(intent.dimensions)
    temporal, changed_temporal = _remove_shadowed(intent.temporal_fields)
    grouping, changed_grouping = _remove_shadowed(intent.grouping_fields)
    if changed_measure or changed_dimension or changed_temporal or changed_grouping:
        applied.add("R-IC-001")

    transforms = list(task_result.transforms)
    sort = task_result.sort
    query = intent.query
    temporal_names = tuple(ref.field_name for ref in temporal)
    if temporal_names and not any(item.transform_type == "time_unit" for item in transforms):
        for match in re.finditer(r"\b(daily|weekly|monthly|quarterly|yearly|annual|annually)\b", query, re.IGNORECASE):
            transforms.append(
                TransformIntent(
                    transform_type="time_unit",
                    operation=_TIME_UNITS[match.group(1).casefold()],
                    field_candidates=(temporal_names[0],),
                    value=None,
                    evidence_spans=(_span(query, match.start(), match.end(), "time_unit_obligation"),),
                    rule_ids=("R-IC-002",),
                )
            )
            applied.add("R-IC-002")
            break

    if intent.primary_task == "ranking" and sort is None and measures:
        evidence = next(
            (item.evidence_spans for item in intent.task_candidates if item.task == "ranking"),
            (),
        ) or (_span(query, 0, len(query), "ranking_sort_obligation"),)
        sort = TransformIntent(
            transform_type="sort",
            operation="descending",
            field_candidates=(measures[0].field_name,),
            value=None,
            evidence_spans=evidence,
            rule_ids=("R-IC-002",),
        )
        transforms.append(sort)
        applied.add("R-IC-002")

    filter_fields = {
        field
        for item in transforms
        if item.transform_type == "filter"
        for field in item.field_candidates
    }
    visual_dimensions = tuple(
        ref for ref in dimensions
        if ref.field_name not in filter_fields and ref.field_name not in set(temporal_names)
    )
    primary_task = intent.primary_task
    task_candidates = list(intent.task_candidates)
    if primary_task not in _CORE_TASKS and measures and visual_dimensions:
        primary_task = "comparison"
        task_candidates.append(
            TaskCandidate(
                task="comparison",
                score=0.8,
                evidence_spans=(_span(query, 0, len(query), "visual_task_obligation"),),
                required_roles=("measure", "dimension"),
                rule_ids=("R-IC-002",),
                score_components={"typed_measure": 0.4, "typed_dimension": 0.4},
            )
        )
        applied.add("R-IC-002")

    transforms_tuple = _dedupe_transforms(transforms)
    task_candidates_tuple = tuple(
        sorted(
            {item.task: item for item in task_candidates}.values(),
            key=lambda item: (-item.score, item.task),
        )
    )
    updated_task = task_result.model_copy(
        update={
            "primary_task": primary_task,
            "task_candidates": task_candidates_tuple,
            "transforms": transforms_tuple,
            "sort": sort,
            "secondary_tasks": tuple(
                item.task for item in task_candidates_tuple if item.task != primary_task
            ),
            "applied_rule_ids": tuple(sorted(set(task_result.applied_rule_ids) | applied)),
        }
    )
    updated_intent = intent.model_copy(
        update={
            "measures": measures,
            "dimensions": dimensions,
            "temporal_fields": temporal,
            "grouping_fields": grouping,
            "primary_task": primary_task,
            "task_candidates": task_candidates_tuple,
            "transforms": transforms_tuple,
            "sort": sort,
        }
    )

    role_map: dict[str, set[str]] = {}
    evidence_map: dict[str, list[EvidenceSpan]] = {}
    for ref in (*measures, *dimensions, *temporal, *grouping):
        role_map.setdefault(ref.field_name, set()).add(ref.role)
        evidence_map.setdefault(ref.field_name, []).extend(_strong_spans(ref))
    for field in filter_fields:
        role_map.setdefault(field, set()).add("filter")
        evidence_map.setdefault(field, [])
    field_obligations = tuple(
        FieldObligation(
            field_name=name,
            permitted_roles=tuple(sorted(roles)),
            filter_only=name in filter_fields and roles <= {"filter", "dimension", "geographic", "identifier"},
            evidence_spans=tuple(evidence_map[name]),
        )
        for name, roles in sorted(role_map.items())
    )
    transform_obligations = tuple(
        TransformObligation(
            transform_type=item.transform_type,
            operation=item.operation,
            field_candidates=item.field_candidates,
            value=item.value,
        )
        for item in transforms_tuple
    )
    obligation_payload = {
        "intent_id": intent.deterministic_id,
        "planning_task": primary_task,
        "fields": [item.model_dump(mode="json") for item in field_obligations],
        "transforms": [item.model_dump(mode="json") for item in transform_obligations],
        "rules": sorted(applied | {"R-CP-010", "R-CP-011"}),
    }
    obligations = VisualizationObligationSet(
        obligation_id=f"obl-{_canonical_hash(obligation_payload)[:20]}",
        intent_id=intent.deterministic_id,
        planning_task=primary_task,
        field_obligations=field_obligations,
        transform_obligations=transform_obligations,
        applied_rule_ids=tuple(sorted(applied | {"R-CP-010", "R-CP-011"})),
    )
    registry.validate_ids(obligations.applied_rule_ids)
    return IntentCompilationResult(
        intent=updated_intent,
        task_result=updated_task,
        obligations=obligations,
        applied_rule_ids=tuple(sorted(applied)),
    )


__all__ = [
    "FieldObligation",
    "IntentCompilationResult",
    "TransformObligation",
    "VisualizationObligationSet",
    "compile_intent_obligations",
]
