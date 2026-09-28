"""Query-local discourse grouping roles for physically numeric category codes."""

from __future__ import annotations

import re

from skillvis_v2_accuracy.contracts import (
    AnalyticalTaskResult,
    FieldEvidence,
    FieldResolutionResult,
    QueryNormalizationResult,
    SchemaProfile,
)
from skillvis_v2_accuracy.rules import RuleRegistry


_STRONG = {"exact", "normalized_lexical", "alias", "acronym_expansion", "value", "semantic_concept"}
_BEFORE = re.compile(r"\b(across|among|by|per|for)\s*$", re.IGNORECASE)
_AFTER = re.compile(r"^\s+(groups?|categories|classes|types|codes?|bands?)\b", re.IGNORECASE)
_PART_TO_WHOLE = re.compile(r"\b(shares?|proportions?)\b", re.IGNORECASE)


def _local_grouping_evidence(query: str, candidate, *, composition: bool):
    for evidence in candidate.evidence:
        span = evidence.query_span
        if span is None or evidence.evidence_type not in _STRONG:
            continue
        before = query[max(0, span.start - 24) : span.start]
        after = query[span.end : min(len(query), span.end + 24)]
        before_match = _BEFORE.search(before)
        after_match = _AFTER.search(after)
        if before_match:
            return evidence, f"before:{before_match.group(1).casefold()}"
        if after_match:
            return evidence, f"after:{after_match.group(1).casefold()}"
        if composition:
            return evidence, "part_to_whole_single_field"
    return None


def resolve_discourse_grouping_roles(
    *,
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    task: AnalyticalTaskResult,
    registry: RuleRegistry,
) -> tuple[SchemaProfile, FieldResolutionResult, tuple[str, ...]]:
    query = normalized.original_query
    aggregate_targets = {
        field_name
        for operation in task.transforms
        if operation.transform_type == "aggregate" and not operation.negated
        for field_name in operation.field_candidates
    }
    composition = bool(normalized.chart_mentions) and any(
        item.canonical == "pie" for item in normalized.chart_mentions
    ) and bool(_PART_TO_WHOLE.search(query)) and len(fields.required_field_candidates) == 1
    candidates = {item.field_name: item for item in fields.candidate_fields}
    required = set(fields.required_field_candidates)
    decisions: dict[str, tuple[object, str]] = {}
    schema_rows = []
    for profile in schema.fields:
        candidate = candidates.get(profile.field_name)
        local = _local_grouping_evidence(query, candidate, composition=composition) if candidate else None
        should_convert = bool(
            profile.field_name in required
            and profile.field_name not in aggregate_targets
            and profile.semantic_type == "quantitative"
            and profile.numeric_summary is not None
            and candidate
            and local
        )
        if not should_convert:
            schema_rows.append(profile)
            continue
        decisions[profile.field_name] = local
        schema_rows.append(
            profile.model_copy(
                update={
                    "semantic_type": "categorical",
                    "role_candidates": ("dimension", "grouping", "filter"),
                    "source_rule_ids": registry.validate_ids((*profile.source_rule_ids, "R-FR-080")),
                    "warnings": tuple(sorted({*profile.warnings, "query-local discourse role overrides numeric physical type"})),
                }
            )
        )
    if not decisions:
        return schema, fields, ()

    candidate_rows = []
    evidence_map = dict(fields.evidence)
    intended_map = dict(fields.intended_roles)
    match_map = dict(fields.match_types)
    traces = []
    for candidate in fields.candidate_fields:
        decision = decisions.get(candidate.field_name)
        if decision is None:
            candidate_rows.append(candidate)
            continue
        grounding, cue = decision
        evidence = tuple(
            sorted(
                (
                    *candidate.evidence,
                    FieldEvidence(
                        evidence_type="discourse_grouping_role",
                        query_span=grounding.query_span,
                        value=f"dimension:{cue}",
                        rule_id="R-FR-080",
                        source_ids=registry.get("R-FR-080").source_ids,
                    ),
                ),
                key=lambda item: (item.query_span.start if item.query_span else -1, item.evidence_type, item.value),
            )
        )
        roles = ("dimension", "filter", "grouping")
        match_types = tuple(sorted({*candidate.match_types, "discourse_grouping_role"}))
        updated = candidate.model_copy(
            update={
                "evidence": evidence,
                "intended_roles": roles,
                "match_types": match_types,
                "source_rule_ids": registry.validate_ids((*candidate.source_rule_ids, "R-FR-080")),
            }
        )
        candidate_rows.append(updated)
        evidence_map[candidate.field_name] = evidence
        intended_map[candidate.field_name] = roles
        match_map[candidate.field_name] = match_types
        traces.append(f"R-FR-080 discourse grouping role:{candidate.field_name}:cue={cue}")

    updated_schema = SchemaProfile.model_validate(
        schema.model_copy(
            update={
                "fields": tuple(schema_rows),
                "applied_rule_ids": registry.validate_ids((*schema.applied_rule_ids, "R-FR-080")),
            }
        ).model_dump(mode="json")
    )
    updated_fields = FieldResolutionResult.model_validate(
        fields.model_copy(
            update={
                "candidate_fields": tuple(candidate_rows),
                "evidence": evidence_map,
                "intended_roles": intended_map,
                "match_types": match_map,
                "applied_rule_ids": registry.validate_ids((*fields.applied_rule_ids, "R-FR-080")),
            }
        ).model_dump(mode="json")
    )
    return updated_schema, updated_fields, tuple(sorted(traces))


__all__ = ["resolve_discourse_grouping_roles"]
