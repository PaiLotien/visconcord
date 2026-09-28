"""Adjacent generic-head ownership for required field obligations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from skillvis_v2_accuracy.contracts import FieldResolutionResult, SchemaProfile
from skillvis_v2_accuracy.query_normalization import normalize_field_phrase
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_joint_v6.evidence_strength import EvidenceStrength, classify_evidence


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_OWNER_STRENGTHS = {
    EvidenceStrength.EXPLICIT_GROUNDING,
    EvidenceStrength.SEMANTIC_GROUNDING,
}


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalize_field_phrase(value)))


@dataclass(frozen=True)
class _Owner:
    field_name: str
    span: object
    tokens: frozenset[str]


def _owners(fields: FieldResolutionResult) -> tuple[_Owner, ...]:
    rows = []
    for candidate in fields.candidate_fields:
        for evidence in candidate.evidence:
            if (
                evidence.query_span is not None
                and classify_evidence(evidence.evidence_type) in _OWNER_STRENGTHS
            ):
                rows.append(
                    _Owner(
                        field_name=candidate.field_name,
                        span=evidence.query_span,
                        tokens=frozenset(_tokens(candidate.field_name)),
                    )
                )
    return tuple(rows)


def _adjacent_owner(candidate, evidence, owners, shared_tokens: set[str]):
    span = evidence.query_span
    if span is None or classify_evidence(evidence.evidence_type) in _OWNER_STRENGTHS:
        return None
    observed = _tokens(candidate.field_name) & _tokens(span.normalized)
    if not observed or not observed <= shared_tokens:
        return None
    matches = []
    for owner in owners:
        if owner.field_name == candidate.field_name or not observed <= set(owner.tokens):
            continue
        gap = span.start - owner.span.end
        if 0 <= gap <= 2:
            matches.append((gap, owner.field_name, owner.span, tuple(sorted(observed))))
    return min(matches, default=None)


def resolve_adjacent_generic_head_ownership(
    *,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    registry: RuleRegistry,
) -> tuple[FieldResolutionResult, tuple[str, ...]]:
    """Change only required obligations; retain candidates and raw evidence."""

    token_frequency = Counter(
        token
        for profile in schema.fields
        for token in _tokens(profile.normalized_name or profile.field_name)
    )
    shared_tokens = {token for token, count in token_frequency.items() if count >= 2}
    owners = _owners(fields)
    required = set(fields.required_field_candidates)
    suppressions: dict[str, tuple[tuple[str, str, int], ...]] = {}

    for candidate in fields.candidate_fields:
        if candidate.field_name not in required:
            continue
        grounded = tuple(
            item for item in candidate.evidence if item.query_span is not None
        )
        if not grounded:
            continue
        if any(classify_evidence(item.evidence_type) in _OWNER_STRENGTHS for item in grounded):
            continue
        decisions = []
        for evidence in grounded:
            owner = _adjacent_owner(candidate, evidence, owners, shared_tokens)
            if owner is None:
                decisions.append(None)
            else:
                decisions.append((owner[1], ",".join(owner[3]), owner[0]))
        if decisions and all(item is not None for item in decisions):
            suppressions[candidate.field_name] = tuple(
                item for item in decisions if item is not None
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
    updated = FieldResolutionResult.model_validate(
        fields.model_copy(
            update={
                "required_field_candidates": surviving,
                "ambiguity": tuple(ambiguity),
                "applied_rule_ids": registry.validate_ids(
                    (*fields.applied_rule_ids, "R-FR-070")
                ),
                "abstention_reason": (
                    None if surviving else "no_field_candidate_after_adjacent_ownership"
                ),
            }
        ).model_dump(mode="json")
    )
    traces = tuple(
        "R-FR-070 adjacent generic-head ownership suppressed "
        f"{field_name}; owners="
        + ";".join(
            f"{owner}:tokens={tokens}:gap={gap}"
            for owner, tokens, gap in decisions
        )
        for field_name, decisions in sorted(suppressions.items())
    )
    return updated, traces


__all__ = ["resolve_adjacent_generic_head_ownership"]
