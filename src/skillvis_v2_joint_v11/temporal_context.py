"""Query-scoped reconciliation of numeric value shape and temporal name hints."""

from __future__ import annotations

import re

from skillvis_v2_accuracy.contracts import (
    FieldEvidence,
    FieldResolutionResult,
    QueryNormalizationResult,
    SchemaProfile,
)
from skillvis_v2_accuracy.rules import RuleRegistry


_CORRELATION = re.compile(
    r"\b(against|versus|relationship|association|correlat(?:e|ed|es|ing|ion))\b",
    re.IGNORECASE,
)
_TREND = re.compile(
    r"\b(over\s+time|over|through|trend|time\s+series|across\s+time|"
    r"by\s+(?:year|month|week|day|date)|chronolog(?:y|ical)|evolution)\b",
    re.IGNORECASE,
)
_STRONG_GROUNDING = {
    "exact",
    "normalized_lexical",
    "alias",
    "acronym_expansion",
    "value",
    "semantic_concept",
}


def _strongest_grounding(candidate):
    grounded = [
        evidence
        for evidence in candidate.evidence
        if evidence.query_span is not None
        and evidence.evidence_type in _STRONG_GROUNDING
    ]
    return min(
        grounded,
        key=lambda evidence: (
            evidence.query_span.start,
            evidence.query_span.end,
            evidence.evidence_type,
        ),
        default=None,
    )


def restore_correlation_numeric_roles(
    *,
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    registry: RuleRegistry,
) -> tuple[SchemaProfile, FieldResolutionResult, tuple[str, ...]]:
    """Restore numeric roles only under positive correlation-task evidence."""

    query = normalized.original_query
    correlation_match = _CORRELATION.search(query)
    trend_match = _TREND.search(query)
    if correlation_match is None or trend_match is not None:
        return schema, fields, ()

    required = set(fields.required_field_candidates)
    candidates = {item.field_name: item for item in fields.candidate_fields}
    numeric_shaped_required = {
        profile.field_name
        for profile in schema.fields
        if profile.field_name in required and profile.numeric_summary is not None
    }
    if len(numeric_shaped_required) < 2:
        return schema, fields, ()

    decisions: dict[str, object] = {}
    schema_rows = []
    for profile in schema.fields:
        candidate = candidates.get(profile.field_name)
        grounding = _strongest_grounding(candidate) if candidate else None
        should_restore = bool(
            profile.field_name in required
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
                        (*profile.source_rule_ids, "R-SP-110")
                    ),
                    "warnings": tuple(
                        sorted(
                            {
                                *profile.warnings,
                                "correlation-scoped numeric role overrides lexical temporal name hint",
                            }
                        )
                    ),
                }
            )
        )

    if not decisions:
        return schema, fields, ()

    updated_schema = SchemaProfile.model_validate(
        schema.model_copy(
            update={
                "fields": tuple(schema_rows),
                "applied_rule_ids": registry.validate_ids(
                    (*schema.applied_rule_ids, "R-SP-110")
                ),
            }
        ).model_dump(mode="json")
    )

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
                        evidence_type="correlation_scoped_numeric_role",
                        query_span=grounding.query_span,
                        value="measure:correlation_discourse+numeric_value_shape",
                        rule_id="R-SP-110",
                        source_ids=registry.get("R-SP-110").source_ids,
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
            sorted({*candidate.match_types, "correlation_scoped_numeric_role"})
        )
        updated = candidate.model_copy(
            update={
                "evidence": evidence,
                "intended_roles": roles,
                "match_types": match_types,
                "source_rule_ids": registry.validate_ids(
                    (*candidate.source_rule_ids, "R-SP-110")
                ),
            }
        )
        candidate_rows.append(updated)
        evidence_map[candidate.field_name] = evidence
        intended_map[candidate.field_name] = roles
        match_map[candidate.field_name] = match_types
        traces.append(
            "R-SP-110 correlation-scoped numeric role restored:"
            f"{candidate.field_name}:cue={correlation_match.group(1).casefold()}"
        )

    updated_fields = FieldResolutionResult.model_validate(
        fields.model_copy(
            update={
                "candidate_fields": tuple(candidate_rows),
                "evidence": evidence_map,
                "intended_roles": intended_map,
                "match_types": match_map,
                "applied_rule_ids": registry.validate_ids(
                    (*fields.applied_rule_ids, "R-SP-110")
                ),
            }
        ).model_dump(mode="json")
    )
    return updated_schema, updated_fields, tuple(sorted(traces))


__all__ = ["restore_correlation_numeric_roles"]
