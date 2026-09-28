"""Restore numeric duration roles for explicit grouped extrema."""

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


_GROUP = re.compile(r"\b(by|per|across|among|between|for)\b", re.IGNORECASE)
_TREND = re.compile(
    r"\b(over\s+time|through\s+time|time\s+series|trend|by\s+(?:year|month|week|day|date)|chronolog(?:y|ical))\b",
    re.IGNORECASE,
)
_STRONG = {
    "exact",
    "normalized_lexical",
    "alias",
    "acronym_expansion",
    "value",
    "semantic_concept",
}


def restore_grouped_extremum_numeric_roles(
    *,
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    task: AnalyticalTaskResult,
    registry: RuleRegistry,
) -> tuple[SchemaProfile, FieldResolutionResult, tuple[str, ...]]:
    aggregation = task.aggregation
    if (
        aggregation is None
        or aggregation.operation not in {"min", "max"}
        or not aggregation.field_candidates
        or _GROUP.search(normalized.original_query) is None
        or _TREND.search(normalized.original_query) is not None
    ):
        return schema, fields, ()
    required = set(fields.required_field_candidates)
    targets = set(aggregation.field_candidates)
    profiles = {item.field_name: item for item in schema.fields}
    grouping = {
        name
        for name in required - targets
        if profiles[name].semantic_type
        in {"categorical", "ordinal", "geographic", "identifier"}
    }
    if not grouping:
        return schema, fields, ()
    candidates = {item.field_name: item for item in fields.candidate_fields}
    decisions = {}
    schema_rows = []
    for profile in schema.fields:
        candidate = candidates.get(profile.field_name)
        grounding = next(
            (
                evidence
                for evidence in (candidate.evidence if candidate else ())
                if evidence.query_span is not None
                and evidence.evidence_type in _STRONG
            ),
            None,
        )
        should_restore = bool(
            profile.field_name in targets
            and profile.semantic_type == "temporal"
            and profile.numeric_summary is not None
            and profile.temporal_summary is None
            and profile.physical_type not in {"datetime64", "datetime64[ns]"}
            and grounding is not None
        )
        if not should_restore:
            schema_rows.append(profile)
            continue
        decisions[profile.field_name] = grounding
        schema_rows.append(
            profile.model_copy(
                update={
                    "semantic_type": "quantitative",
                    "role_candidates": ("measure",),
                    "source_rule_ids": registry.validate_ids(
                        (*profile.source_rule_ids, "R-SP-150")
                    ),
                    "warnings": tuple(
                        sorted(
                            {
                                *profile.warnings,
                                "grouped numeric extremum overrides lexical temporal name hint",
                            }
                        )
                    ),
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
        grounding = decisions.get(candidate.field_name)
        if grounding is None:
            candidate_rows.append(candidate)
            continue
        evidence = tuple(
            sorted(
                (
                    *candidate.evidence,
                    FieldEvidence(
                        evidence_type="grouped_extremum_numeric_role",
                        query_span=grounding.query_span,
                        value=f"measure:{aggregation.operation}:groups={','.join(sorted(grouping))}",
                        rule_id="R-SP-150",
                        source_ids=registry.get("R-SP-150").source_ids,
                    ),
                ),
                key=lambda item: (
                    item.query_span.start if item.query_span else -1,
                    item.evidence_type,
                    item.value,
                ),
            )
        )
        roles = ("measure",)
        match_types = tuple(
            sorted({*candidate.match_types, "grouped_extremum_numeric_role"})
        )
        updated = candidate.model_copy(
            update={
                "evidence": evidence,
                "intended_roles": roles,
                "match_types": match_types,
                "source_rule_ids": registry.validate_ids(
                    (*candidate.source_rule_ids, "R-SP-150")
                ),
            }
        )
        candidate_rows.append(updated)
        evidence_map[candidate.field_name] = evidence
        intended_map[candidate.field_name] = roles
        match_map[candidate.field_name] = match_types
        traces.append(
            f"R-SP-150 grouped numeric extremum:{candidate.field_name}:operation={aggregation.operation}:groups={','.join(sorted(grouping))}"
        )

    updated_schema = SchemaProfile.model_validate(
        schema.model_copy(
            update={
                "fields": tuple(schema_rows),
                "applied_rule_ids": registry.validate_ids(
                    (*schema.applied_rule_ids, "R-SP-150")
                ),
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
                "applied_rule_ids": registry.validate_ids(
                    (*fields.applied_rule_ids, "R-SP-150")
                ),
            }
        ).model_dump(mode="json")
    )
    return updated_schema, updated_fields, tuple(sorted(traces))


__all__ = ["restore_grouped_extremum_numeric_roles"]
