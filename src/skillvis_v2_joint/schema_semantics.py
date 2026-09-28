"""Versioned schema decomposition and schema-independent mention extraction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from skillvis_v2_accuracy.contracts import EvidenceSpan, QueryNormalizationResult, SchemaProfile
from skillvis_v2_accuracy.query_normalization import normalize_field_phrase

from .config import JointSemanticSettings, load_semantic_registry_payload, load_settings
from .contracts import SchemaSemanticProfile, SemanticFieldMention


@dataclass(frozen=True)
class SemanticConcept:
    concept_id: str
    kind: str
    query_terms: tuple[str, ...]
    schema_terms: tuple[str, ...]


def _contains_phrase(tokens: tuple[str, ...], phrase: str) -> bool:
    target = tuple(normalize_field_phrase(phrase).split())
    if not target:
        return False
    return any(
        tokens[index : index + len(target)] == target
        for index in range(len(tokens) - len(target) + 1)
    )


def load_concepts(
    settings: JointSemanticSettings | None = None,
) -> tuple[SemanticConcept, ...]:
    payload = load_semantic_registry_payload(settings or load_settings())
    if payload.get("status") != "development_only_versioned":
        raise RuntimeError("schema semantic registry is not versioned for development")
    return tuple(
        SemanticConcept(
            concept_id=item["concept_id"],
            kind=item["kind"],
            query_terms=tuple(item["query_terms"]),
            schema_terms=tuple(item["schema_terms"]),
        )
        for item in payload["concepts"]
    )


def decompose_schema(
    schema: SchemaProfile,
    settings: JointSemanticSettings | None = None,
) -> tuple[SchemaSemanticProfile, ...]:
    concepts = load_concepts(settings)
    profiles = []
    for field in schema.fields:
        components = tuple(field.normalized_name.split())
        matched = tuple(
            concept
            for concept in concepts
            if any(_contains_phrase(components, term) for term in concept.schema_terms)
        )
        by_kind = {
            kind: tuple(sorted(item.concept_id for item in matched if item.kind == kind))
            for kind in ("entity", "property", "qualifier", "measure", "generic")
        }
        # Measures are properties of a field for schema-decomposition purposes,
        # while their kind remains available in the concept registry and query mention.
        properties = tuple(sorted((*by_kind["property"], *by_kind["measure"])))
        profiles.append(
            SchemaSemanticProfile(
                field_name=field.field_name,
                components=components,
                entity_concepts=by_kind["entity"],
                property_concepts=properties,
                qualifier_concepts=by_kind["qualifier"],
                generic_concepts=by_kind["generic"],
            )
        )
    return tuple(sorted(profiles, key=lambda item: item.field_name.casefold()))


def _expected_roles(kind: str) -> tuple[str, ...]:
    if kind == "measure":
        return ("measure",)
    if kind == "qualifier":
        return ("dimension", "identifier")
    if kind == "entity":
        return ("dimension", "grouping")
    return ("dimension", "grouping", "filter")


def extract_semantic_mentions(
    normalized: QueryNormalizationResult,
    settings: JointSemanticSettings | None = None,
) -> tuple[SemanticFieldMention, ...]:
    concepts = load_concepts(settings)
    lemmas = tuple(normalized.lemmas_or_normalized_terms)
    spans = tuple(normalized.query_spans)
    mentions: dict[tuple[str, int, int], SemanticFieldMention] = {}
    for concept in concepts:
        for surface in concept.query_terms:
            target = tuple(normalize_field_phrase(surface).split())
            if not target:
                continue
            for index in range(len(lemmas) - len(target) + 1):
                if lemmas[index : index + len(target)] != target:
                    continue
                start = spans[index].start
                end = spans[index + len(target) - 1].end
                text = normalized.original_query[start:end]
                digest = hashlib.sha1(
                    f"{concept.concept_id}:{start}:{end}:{text.casefold()}".encode()
                ).hexdigest()[:12]
                evidence = EvidenceSpan(
                    start=start,
                    end=end,
                    text=text,
                    normalized=" ".join(target),
                    kind="semantic_field_mention",
                    rule_ids=("R-FR-020",),
                )
                mentions[(concept.concept_id, start, end)] = SemanticFieldMention(
                    mention_id=f"sm-{digest}",
                    concept_id=concept.concept_id,
                    concept_kind=concept.kind,
                    query_span=evidence,
                    expected_roles=_expected_roles(concept.kind),
                )
    return tuple(
        sorted(
            mentions.values(),
            key=lambda item: (
                item.query_span.start,
                -(item.query_span.end - item.query_span.start),
                item.concept_id,
            ),
        )
    )


__all__ = [
    "SemanticConcept",
    "decompose_schema",
    "extract_semantic_mentions",
    "load_concepts",
]
