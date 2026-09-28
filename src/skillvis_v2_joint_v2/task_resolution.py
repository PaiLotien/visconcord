"""Typed query-role and operation-field collision refinements for v1.1."""

from __future__ import annotations

import re

from skillvis_v2_accuracy.contracts import (
    AnalyticalTaskResult,
    FieldResolutionResult,
    QueryNormalizationResult,
    SchemaProfile,
)


_GROUP_CUE = re.compile(
    r"\b(each|per|by|across|grouped|groups?|categorized|categorised|divided|based)\b",
    re.IGNORECASE,
)
_STRONG = {"exact", "normalized_lexical", "alias", "value"}
_TEMPORAL_FIELD_NAME = re.compile(
    r"(?:^|[_\-\s])(year|date|time|month|quarter|week|day)(?:$|[_\-\s])",
    re.IGNORECASE,
)


def _is_explicit_grouping_field(query: str, candidate) -> bool:
    """Return true only when a grounded field is locally marked as a group key."""
    for evidence in candidate.evidence:
        span = evidence.query_span
        if span is None or evidence.evidence_type not in _STRONG:
            continue
        before = query[max(0, span.start - 16) : span.start]
        after = query[span.end : min(len(query), span.end + 18)]
        if re.search(r"\b(by|per|each|across)\s*$", before, re.IGNORECASE):
            return True
        if re.match(r"\s+(groups?|categories)\b", after, re.IGNORECASE):
            return True
    return False


def refine_schema_roles(
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
) -> tuple[SchemaProfile, tuple[str, ...]]:
    has_group_cue = bool(_GROUP_CUE.search(normalized.original_query))
    candidates = {row.field_name: row for row in fields.candidate_fields}
    refined: list[str] = []
    rows = []
    for field in schema.fields:
        candidate = candidates.get(field.field_name)
        intended = set(candidate.intended_roles) if candidate else set()
        locally_grouped = bool(
            candidate
            and _is_explicit_grouping_field(normalized.original_query, candidate)
        )
        if (
            field.field_name in fields.required_field_candidates
            and field.semantic_type == "quantitative"
            and candidate
            and (_STRONG & set(candidate.match_types))
            and _TEMPORAL_FIELD_NAME.search(field.field_name)
        ):
            rows.append(field.model_copy(update={"semantic_type": "temporal"}))
            refined.append(f"temporal:{field.field_name}")
        elif (
            field.field_name in fields.required_field_candidates
            and field.semantic_type == "quantitative"
            and has_group_cue
            and ("dimension" in intended or locally_grouped)
        ):
            rows.append(field.model_copy(update={"semantic_type": "categorical"}))
            refined.append(f"categorical:{field.field_name}")
        else:
            rows.append(field)
    if not refined:
        return schema, ()
    return (
        SchemaProfile.model_validate(
            schema.model_copy(update={"fields": tuple(rows)}).model_dump(mode="json")
        ),
        tuple(sorted(refined)),
    )


def remove_operation_field_collisions(
    base: AnalyticalTaskResult,
    fields: FieldResolutionResult,
) -> tuple[AnalyticalTaskResult, tuple[str, ...]]:
    candidates = {row.field_name: row for row in fields.candidate_fields}
    exact_spans: dict[str, tuple] = {}
    for name, candidate in candidates.items():
        if not (_STRONG & set(candidate.match_types)):
            continue
        exact_spans[name] = tuple(
            evidence.query_span
            for evidence in candidate.evidence
            if evidence.query_span is not None
            and evidence.evidence_type in _STRONG
        )

    removed: list[str] = []
    kept = []
    for operation in base.transforms:
        collision = False
        if operation.transform_type == "aggregate" and operation.field_candidates:
            for field_name in operation.field_candidates:
                spans = exact_spans.get(field_name, ())
                if spans and operation.evidence_spans and all(
                    any(
                        field_span.start <= evidence.start
                        and evidence.end <= field_span.end
                        for field_span in spans
                    )
                    for evidence in operation.evidence_spans
                ):
                    collision = True
                    break
        if collision:
            removed.append(f"{operation.operation}:{','.join(operation.field_candidates)}")
        else:
            kept.append(operation)
    if not removed:
        return base, ()
    aggregation = next(
        (row for row in kept if row.transform_type == "aggregate"), None
    )
    updated = AnalyticalTaskResult.model_validate(
        base.model_copy(
            update={
                "transforms": tuple(kept),
                "aggregation": aggregation,
            }
        ).model_dump(mode="json")
    )
    return updated, tuple(sorted(removed))


__all__ = ["refine_schema_roles", "remove_operation_field_collisions"]
