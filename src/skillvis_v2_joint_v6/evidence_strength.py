"""Typed evidence strength and token-level mention ownership."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from skillvis_v2_accuracy.contracts import FieldResolutionResult, SchemaProfile
from skillvis_v2_accuracy.query_normalization import normalize_field_phrase
from skillvis_v2_accuracy.rules import RuleRegistry


class EvidenceStrength(str, Enum):
    EXPLICIT_GROUNDING = "EXPLICIT_GROUNDING"
    SEMANTIC_GROUNDING = "SEMANTIC_GROUNDING"
    SCHEMA_INFERENCE = "SCHEMA_INFERENCE"
    CONTEXTUAL_SUPPORT = "CONTEXTUAL_SUPPORT"
    WEAK_OVERLAP = "WEAK_OVERLAP"


_EXPLICIT = {"exact", "normalized_lexical", "alias", "acronym_expansion", "value"}
_SEMANTIC = {"semantic_concept"}
_SCHEMA = {"schema_semantic_concept", "schema_temporal_profile"}
_WEAK = {"token_overlap"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def classify_evidence(evidence_type: str) -> EvidenceStrength:
    if evidence_type in _EXPLICIT:
        return EvidenceStrength.EXPLICIT_GROUNDING
    if evidence_type in _SEMANTIC:
        return EvidenceStrength.SEMANTIC_GROUNDING
    if evidence_type in _SCHEMA:
        return EvidenceStrength.SCHEMA_INFERENCE
    if evidence_type in _WEAK:
        return EvidenceStrength.WEAK_OVERLAP
    return EvidenceStrength.CONTEXTUAL_SUPPORT


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalize_field_phrase(value)))


def _overlaps(left, right) -> bool:
    return left.start < right.end and right.start < left.end


@dataclass(frozen=True)
class _Owner:
    field_name: str
    span: object
    field_tokens: frozenset[str]


def _explicit_owners(fields: FieldResolutionResult) -> tuple[_Owner, ...]:
    owners = []
    for candidate in fields.candidate_fields:
        for evidence in candidate.evidence:
            if (
                classify_evidence(evidence.evidence_type)
                is EvidenceStrength.EXPLICIT_GROUNDING
                and evidence.query_span is not None
            ):
                owners.append(
                    _Owner(
                        field_name=candidate.field_name,
                        span=evidence.query_span,
                        field_tokens=frozenset(_tokens(candidate.field_name)),
                    )
                )
    return tuple(owners)


def _independent_support(candidate, evidence, owners: tuple[_Owner, ...]) -> bool:
    span = evidence.query_span
    if span is None:
        return False
    strength = classify_evidence(evidence.evidence_type)
    if strength in {EvidenceStrength.SCHEMA_INFERENCE, EvidenceStrength.CONTEXTUAL_SUPPORT}:
        return False
    field_tokens = _tokens(candidate.field_name)
    observed = field_tokens & _tokens(span.normalized)
    if not observed:
        return False
    relevant = tuple(
        owner
        for owner in owners
        if owner.field_name != candidate.field_name and _overlaps(owner.span, span)
    )
    if strength is EvidenceStrength.EXPLICIT_GROUNDING:
        normalized_field = normalize_field_phrase(candidate.field_name)
        complete = normalize_field_phrase(span.normalized) == normalized_field
        if complete:
            nested = any(
                owner.span.start <= span.start
                and span.end <= owner.span.end
                and (owner.span.end - owner.span.start) > (span.end - span.start)
                and observed <= owner.field_tokens
                for owner in relevant
            )
            return not nested
    owned_tokens = set().union(*(owner.field_tokens for owner in relevant)) if relevant else set()
    return not observed <= owned_tokens


def resolve_typed_evidence_ownership(
    *,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    registry: RuleRegistry,
) -> tuple[FieldResolutionResult, tuple[str, ...]]:
    """Update obligations only; retain all candidates and their raw evidence."""

    del schema  # The decision uses observed candidate evidence, never name lists.
    owners = _explicit_owners(fields)
    required = set(fields.required_field_candidates)
    suppressions: dict[str, tuple[str, ...]] = {}
    audit_rows: list[str] = []
    for candidate in fields.candidate_fields:
        strengths = tuple(
            sorted({classify_evidence(item.evidence_type).value for item in candidate.evidence})
        )
        audit_rows.append(
            f"R-FR-060 evidence strengths: {candidate.field_name}={','.join(strengths)}"
        )
        if candidate.field_name not in required:
            continue
        grounded = tuple(item for item in candidate.evidence if item.query_span is not None)
        independent = tuple(
            item for item in grounded if _independent_support(candidate, item, owners)
        )
        if independent:
            continue
        observed_tokens = set().union(
            *(
                _tokens(candidate.field_name) & _tokens(item.query_span.normalized)
                for item in grounded
                if item.query_span is not None
            )
        ) if grounded else set()
        owner_fields = tuple(
            sorted(
                {
                    owner.field_name
                    for owner in owners
                    if owner.field_name != candidate.field_name
                    and any(_overlaps(owner.span, item.query_span) for item in grounded if item.query_span)
                    and observed_tokens & set(owner.field_tokens)
                }
            )
        )
        # No query-grounded evidence is an upstream scoring concern, not proof
        # that another mention owns the candidate.
        if grounded and owner_fields and observed_tokens:
            suppressions[candidate.field_name] = owner_fields

    applied = {"R-FR-060"}
    if suppressions:
        applied.add("R-FR-061")
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
                    (*fields.applied_rule_ids, *applied)
                ),
                "abstention_reason": (
                    None if surviving else "no_field_candidate_after_typed_ownership"
                ),
            }
        ).model_dump(mode="json")
    )
    audit_rows.extend(
        f"R-FR-061 typed ownership suppressed {field_name}; owners={','.join(owner_names)}"
        for field_name, owner_names in sorted(suppressions.items())
    )
    return updated, tuple(sorted(audit_rows))


__all__ = [
    "EvidenceStrength",
    "classify_evidence",
    "resolve_typed_evidence_ownership",
]
