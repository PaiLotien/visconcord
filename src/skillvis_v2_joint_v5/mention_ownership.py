"""Evidence-ownership closure for M2 required-field obligations.

Candidates remain in the audit contract.  This module changes only which
candidates are mandatory downstream, and only when all field-specific support
is weak and demonstrably owned by a stronger mention or analytical operation.
"""

from __future__ import annotations

import re
from collections import Counter

from skillvis_v2_accuracy.contracts import FieldResolutionResult, SchemaProfile
from skillvis_v2_accuracy.query_normalization import normalize_field_phrase
from skillvis_v2_accuracy.rules import RuleRegistry


_STRONG_EVIDENCE = {
    "exact",
    "normalized_lexical",
    "alias",
    "acronym_expansion",
    "semantic_concept",
    "schema_semantic_concept",
    "value",
}
_WEAK_EVIDENCE = {"token_overlap"}
_OPERATION_TOKENS = {
    "average", "compare", "comparison", "correlation", "distribution",
    "extremum", "highest", "lowest", "maximum", "mean", "minimum",
    "rank", "ranking", "sum", "total", "trend",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalize_field_phrase(value)))


def _spans(candidate, allowed: set[str]):
    return tuple(
        evidence.query_span
        for evidence in candidate.evidence
        if evidence.evidence_type in allowed and evidence.query_span is not None
    )


def _strong_owners(fields: FieldResolutionResult):
    return tuple(
        (candidate, span)
        for candidate in fields.candidate_fields
        for span in _spans(candidate, _STRONG_EVIDENCE)
    )


def _owned_by_complete_mention(
    candidate,
    weak_span,
    owners,
    shared_tokens: set[str],
):
    field_tokens = _tokens(candidate.field_name)
    observed = field_tokens & _tokens(weak_span.normalized)
    if not observed or not observed <= shared_tokens:
        return None
    matches = []
    for owner, owner_span in owners:
        if owner.field_name == candidate.field_name:
            continue
        owner_tokens = _tokens(owner.field_name)
        if observed <= owner_tokens and owner_span.start >= weak_span.start and owner_span.end <= weak_span.end:
            matches.append((owner_span.end - owner_span.start, owner.field_name, owner_span))
    return max(matches, default=None)


def _owned_by_operation(candidate, weak_span, owners):
    field_tokens = _tokens(candidate.field_name)
    observed = field_tokens & _tokens(weak_span.normalized)
    operation = observed & _OPERATION_TOKENS
    base = observed - _OPERATION_TOKENS
    if not operation or not base:
        return None
    matches = []
    for owner, owner_span in owners:
        if owner.field_name == candidate.field_name:
            continue
        if base <= _tokens(owner.field_name) and owner_span.start >= weak_span.start and owner_span.end <= weak_span.end:
            matches.append((owner_span.end - owner_span.start, owner.field_name, owner_span, operation))
    return max(matches, default=None)


def resolve_field_mention_ownership(
    *,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    registry: RuleRegistry,
) -> tuple[FieldResolutionResult, tuple[str, ...]]:
    required = set(fields.required_field_candidates)
    token_frequency = Counter(
        token
        for profile in schema.fields
        for token in _tokens(profile.normalized_name or profile.field_name)
    )
    shared_tokens = {token for token, count in token_frequency.items() if count >= 2}
    owners = _strong_owners(fields)
    suppressions: dict[str, tuple[str, str]] = {}

    for candidate in fields.candidate_fields:
        if candidate.field_name not in required or _spans(candidate, _STRONG_EVIDENCE):
            continue
        weak_spans = _spans(candidate, _WEAK_EVIDENCE)
        if not weak_spans:
            continue
        decisions = []
        for span in weak_spans:
            shared_owner = _owned_by_complete_mention(
                candidate, span, owners, shared_tokens
            )
            if shared_owner is not None:
                decisions.append(("R-FR-050", shared_owner[1]))
                continue
            operation_owner = _owned_by_operation(candidate, span, owners)
            if operation_owner is not None:
                decisions.append(("R-FR-051", operation_owner[1]))
                continue
            decisions.append(None)
        # Any independent weak span preserves the obligation.
        if decisions and all(decision is not None for decision in decisions):
            rule_ids = sorted({decision[0] for decision in decisions if decision})
            owner_names = sorted({decision[1] for decision in decisions if decision})
            suppressions[candidate.field_name] = (
                ",".join(rule_ids), ",".join(owner_names)
            )

    if not suppressions:
        return fields, ()
    surviving = tuple(
        name for name in fields.required_field_candidates if name not in suppressions
    )
    ambiguity = []
    for item in fields.ambiguity:
        if item.dimension != "field":
            ambiguity.append(item)
            continue
        alternatives = tuple(name for name in item.alternatives if name in surviving)
        if len(alternatives) > 1:
            ambiguity.append(item.model_copy(update={"alternatives": alternatives}))
    applied_ids = registry.validate_ids(
        (*fields.applied_rule_ids, *(rule for value in suppressions.values() for rule in value[0].split(",")))
    )
    updated = FieldResolutionResult.model_validate(
        fields.model_copy(
            update={
                "required_field_candidates": surviving,
                "ambiguity": tuple(ambiguity),
                "applied_rule_ids": applied_ids,
                "abstention_reason": None if surviving else "no_field_candidate_after_mention_ownership",
            }
        ).model_dump(mode="json")
    )
    warnings = tuple(
        f"{rule_ids} mention ownership suppressed {field_name}; owner={owners_}"
        for field_name, (rule_ids, owners_) in sorted(suppressions.items())
    )
    return updated, warnings


__all__ = ["resolve_field_mention_ownership"]
