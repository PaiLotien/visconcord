"""Context-sensitive M2 obligation repair without altering candidate evidence."""

from __future__ import annotations

import re

from skillvis_v2_accuracy.contracts import (
    FieldResolutionResult,
    QueryNormalizationResult,
)
from skillvis_v2_accuracy.query_normalization import normalize_field_phrase
from skillvis_v2_accuracy.rules import RuleRegistry


_GENERIC_MEASURE_HEADS = {"value"}
_GENERIC_LEXICAL_EVIDENCE = {"exact", "normalized_lexical"}
_EXPLICIT_HEAD_EVIDENCE = {
    "exact",
    "normalized_lexical",
    "acronym_expansion",
    "alias",
    "semantic_concept",
    "schema_semantic_concept",
}


def _evidence_spans(candidate, allowed: set[str]):
    return tuple(
        evidence.query_span
        for evidence in candidate.evidence
        if evidence.query_span is not None
        and evidence.evidence_type in allowed
    )


def _adjacent_head(
    *,
    query: str,
    generic_span,
    candidates,
    required: set[str],
    generic_field: str,
):
    matches = []
    for candidate in candidates:
        if candidate.field_name == generic_field:
            continue
        if candidate.field_name not in required:
            continue
        for head_span in _evidence_spans(candidate, _EXPLICIT_HEAD_EVIDENCE):
            if head_span.end > generic_span.start:
                continue
            between = query[head_span.end : generic_span.start]
            if re.fullmatch(r"\s+", between):
                matches.append((head_span.end, candidate, head_span, between))
    if not matches:
        return None
    return max(matches, key=lambda row: (row[0], row[1].field_name.casefold()))


def suppress_generic_measure_head_obligations(
    *,
    normalized: QueryNormalizationResult,
    fields: FieldResolutionResult,
    registry: RuleRegistry,
) -> tuple[FieldResolutionResult, tuple[str, ...]]:
    """Remove only false independent obligations such as ``profit values``.

    The candidate and its positive lexical evidence remain in the contract for
    audit.  A standalone ``value`` or coordinated ``profit and value`` remains
    required because it has no immediately adjacent grounded head field.
    """

    required = set(fields.required_field_candidates)
    suppressions: dict[str, list[str]] = {}
    for candidate in fields.candidate_fields:
        if candidate.field_name not in required:
            continue
        if normalize_field_phrase(candidate.field_name) not in _GENERIC_MEASURE_HEADS:
            continue
        generic_spans = _evidence_spans(candidate, _GENERIC_LEXICAL_EVIDENCE)
        if not generic_spans:
            continue
        heads = [
            _adjacent_head(
                query=normalized.original_query,
                generic_span=span,
                candidates=fields.candidate_fields,
                required=required,
                generic_field=candidate.field_name,
            )
            for span in generic_spans
        ]
        # A single independent mention is sufficient to preserve the field.
        if not heads or any(head is None for head in heads):
            continue
        suppressions[candidate.field_name] = [
            (
                f"{head_candidate.field_name}@{head_span.start}:{head_span.end}"
                f"->{candidate.field_name}@{generic_span.start}:{generic_span.end}"
            )
            for generic_span, (_, head_candidate, head_span, _) in zip(
                generic_spans, heads, strict=True
            )
        ]

    if not suppressions:
        return fields, ()

    surviving = tuple(
        name
        for name in fields.required_field_candidates
        if name not in suppressions
    )
    ambiguity = []
    for item in fields.ambiguity:
        if item.dimension != "field":
            ambiguity.append(item)
            continue
        alternatives = tuple(name for name in item.alternatives if name in surviving)
        if len(alternatives) > 1:
            ambiguity.append(item.model_copy(update={"alternatives": alternatives}))
    applied = registry.validate_ids((*fields.applied_rule_ids, "R-FR-030"))
    updated = FieldResolutionResult.model_validate(
        fields.model_copy(
            update={
                "required_field_candidates": surviving,
                "ambiguity": tuple(ambiguity),
                "applied_rule_ids": applied,
                "abstention_reason": (
                    None if surviving else "no_field_candidate_after_contextual_suppression"
                ),
            }
        ).model_dump(mode="json")
    )
    warnings = tuple(
        "R-FR-030 generic-head obligation suppressed: "
        + field_name
        + " ["
        + ", ".join(rows)
        + "]"
        for field_name, rows in sorted(suppressions.items())
    )
    return updated, warnings


__all__ = ["suppress_generic_measure_head_obligations"]
