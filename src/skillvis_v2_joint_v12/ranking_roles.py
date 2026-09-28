"""Bind numerical category codes to ranking-target roles from query structure."""

from __future__ import annotations

import re

from skillvis_v2_accuracy.contracts import FieldEvidence, FieldResolutionResult, QueryNormalizationResult, SchemaProfile
from skillvis_v2_accuracy.rules import RuleRegistry


_DIRECTIVE = re.compile(r"\b(rank|order)\b", re.IGNORECASE)
_LINK = re.compile(r"\b(by|using|based\s+on|according\s+to)\b", re.IGNORECASE)
_STRONG = {"exact", "normalized_lexical", "alias", "acronym_expansion", "value", "semantic_concept"}


def _groundings(candidate):
    return tuple(
        evidence
        for evidence in candidate.evidence
        if evidence.query_span is not None and evidence.evidence_type in _STRONG
    )


def resolve_ranking_target_roles(*, normalized: QueryNormalizationResult, schema: SchemaProfile, fields: FieldResolutionResult, registry: RuleRegistry) -> tuple[SchemaProfile, FieldResolutionResult, tuple[str, ...]]:
    query = normalized.original_query
    directive = _DIRECTIVE.search(query)
    link = _LINK.search(query, directive.end() if directive else 0)
    if directive is None or link is None or not normalized.aggregation_terms:
        return schema, fields, ()
    required = set(fields.required_field_candidates)
    candidates = {item.field_name: item for item in fields.candidate_fields if item.field_name in required}
    targets = {}
    score_fields = []
    for name, candidate in candidates.items():
        for evidence in _groundings(candidate):
            span = evidence.query_span
            if directive.end() <= span.start and span.end <= link.start():
                targets[name] = evidence
            elif span.start >= link.end():
                score_fields.append((name, evidence))
    if len(targets) != 1 or not score_fields:
        return schema, fields, ()
    target_name, target_evidence = next(iter(targets.items()))
    distinct_scores = sorted({name for name, _ in score_fields if name != target_name})
    if not distinct_scores:
        return schema, fields, ()

    schema_rows = []
    changed = False
    for profile in schema.fields:
        if profile.field_name != target_name or profile.semantic_type != "quantitative" or profile.numeric_summary is None:
            schema_rows.append(profile)
            continue
        changed = True
        schema_rows.append(profile.model_copy(update={"semantic_type": "categorical", "role_candidates": ("dimension", "grouping", "filter"), "source_rule_ids": registry.validate_ids((*profile.source_rule_ids, "R-FR-120")), "warnings": tuple(sorted({*profile.warnings, "ranking target role overrides numeric physical type"}))}))
    if not changed:
        return schema, fields, ()
    updated_schema = SchemaProfile.model_validate(schema.model_copy(update={"fields": tuple(schema_rows), "applied_rule_ids": registry.validate_ids((*schema.applied_rule_ids, "R-FR-120"))}).model_dump(mode="json"))

    candidate_rows = []
    evidence_map = dict(fields.evidence)
    intended_map = dict(fields.intended_roles)
    match_map = dict(fields.match_types)
    for candidate in fields.candidate_fields:
        if candidate.field_name != target_name:
            candidate_rows.append(candidate)
            continue
        evidence = tuple(sorted((*candidate.evidence, FieldEvidence(evidence_type="ranking_target_role", query_span=target_evidence.query_span, value=f"dimension:score_fields={','.join(distinct_scores)}", rule_id="R-FR-120", source_ids=registry.get("R-FR-120").source_ids)), key=lambda item: (item.query_span.start if item.query_span else -1, item.evidence_type, item.value)))
        roles = ("dimension", "filter", "grouping")
        match_types = tuple(sorted({*candidate.match_types, "ranking_target_role"}))
        updated = candidate.model_copy(update={"evidence": evidence, "intended_roles": roles, "match_types": match_types, "source_rule_ids": registry.validate_ids((*candidate.source_rule_ids, "R-FR-120"))})
        candidate_rows.append(updated)
        evidence_map[target_name] = evidence
        intended_map[target_name] = roles
        match_map[target_name] = match_types
    updated_fields = FieldResolutionResult.model_validate(fields.model_copy(update={"candidate_fields": tuple(candidate_rows), "evidence": evidence_map, "intended_roles": intended_map, "match_types": match_map, "applied_rule_ids": registry.validate_ids((*fields.applied_rule_ids, "R-FR-120"))}).model_dump(mode="json"))
    trace = f"R-FR-120 ranking target role:{target_name}:scores={','.join(distinct_scores)}:link={link.group(1).casefold()}"
    return updated_schema, updated_fields, (trace,)


__all__ = ["resolve_ranking_target_roles"]
