"""Query-local grouping roles from explicit set nouns."""

from __future__ import annotations

import re

from skillvis_v2_accuracy.contracts import (
    AnalyticalTaskResult,
    FieldEvidence,
    FieldResolutionResult,
    SchemaProfile,
)
from skillvis_v2_accuracy.rules import RuleRegistry


_SET_NOUN = re.compile(r"^\s+(levels?|tiers?|strata|buckets?)\b", re.IGNORECASE)
_STRONG = {
    "exact",
    "normalized_lexical",
    "alias",
    "acronym_expansion",
    "value",
    "semantic_concept",
}


def _set_evidence(query: str, candidate):
    for evidence in candidate.evidence:
        span = evidence.query_span
        if span is None or evidence.evidence_type not in _STRONG:
            continue
        suffix = query[span.end : min(len(query), span.end + 18)]
        match = _SET_NOUN.match(suffix)
        if match:
            return evidence, match.group(1).casefold(), (
                span.end + match.start(1),
                span.end + match.end(1),
            )
    return None


def resolve_set_noun_grouping_roles(
    *,
    query: str,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    task: AnalyticalTaskResult,
    registry: RuleRegistry,
) -> tuple[SchemaProfile, FieldResolutionResult, tuple[str, ...]]:
    aggregate_targets = {
        field_name
        for operation in task.transforms
        if operation.transform_type == "aggregate" and not operation.negated
        for field_name in operation.field_candidates
    }
    required = set(fields.required_field_candidates)
    candidates = {item.field_name: item for item in fields.candidate_fields}
    decisions: dict[str, tuple[object, str, tuple[int, int]]] = {}
    schema_rows = []
    for profile in schema.fields:
        candidate = candidates.get(profile.field_name)
        evidence = _set_evidence(query, candidate) if candidate else None
        should_convert = bool(
            profile.field_name in required
            and profile.field_name not in aggregate_targets
            and profile.semantic_type == "quantitative"
            and profile.numeric_summary is not None
            and evidence
        )
        if not should_convert:
            schema_rows.append(profile)
            continue
        decisions[profile.field_name] = evidence
        schema_rows.append(
            profile.model_copy(
                update={
                    "semantic_type": "categorical",
                    "role_candidates": ("dimension", "grouping", "filter"),
                    "source_rule_ids": registry.validate_ids(
                        (*profile.source_rule_ids, "R-FR-140")
                    ),
                    "warnings": tuple(
                        sorted(
                            {
                                *profile.warnings,
                                "explicit set noun overrides numeric physical type",
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
        decision = decisions.get(candidate.field_name)
        if decision is None:
            candidate_rows.append(candidate)
            continue
        grounding, noun, noun_span = decision
        evidence = tuple(
            sorted(
                (
                    *candidate.evidence,
                    FieldEvidence(
                        evidence_type="set_noun_grouping_role",
                        query_span=grounding.query_span,
                        value=f"dimension:{noun}@{noun_span[0]}:{noun_span[1]}",
                        rule_id="R-FR-140",
                        source_ids=registry.get("R-FR-140").source_ids,
                    ),
                ),
                key=lambda item: (
                    item.query_span.start if item.query_span else -1,
                    item.evidence_type,
                    item.value,
                ),
            )
        )
        roles = ("dimension", "filter", "grouping")
        match_types = tuple(sorted({*candidate.match_types, "set_noun_grouping_role"}))
        updated = candidate.model_copy(
            update={
                "evidence": evidence,
                "intended_roles": roles,
                "match_types": match_types,
                "source_rule_ids": registry.validate_ids(
                    (*candidate.source_rule_ids, "R-FR-140")
                ),
            }
        )
        candidate_rows.append(updated)
        evidence_map[candidate.field_name] = evidence
        intended_map[candidate.field_name] = roles
        match_map[candidate.field_name] = match_types
        traces.append(
            f"R-FR-140 set noun role:{candidate.field_name}:noun={noun}:span={noun_span[0]}-{noun_span[1]}"
        )

    updated_schema = SchemaProfile.model_validate(
        schema.model_copy(
            update={
                "fields": tuple(schema_rows),
                "applied_rule_ids": registry.validate_ids(
                    (*schema.applied_rule_ids, "R-FR-140")
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
                    (*fields.applied_rule_ids, "R-FR-140")
                ),
            }
        ).model_dump(mode="json")
    )
    return updated_schema, updated_fields, tuple(sorted(traces))


__all__ = ["resolve_set_noun_grouping_roles"]
