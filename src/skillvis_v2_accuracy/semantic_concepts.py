"""Versioned deterministic semantic bridge for schema-local field grounding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .contracts import QueryNormalizationResult, SchemaFieldProfile
from .query_normalization import normalize_field_phrase


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONCEPT_REGISTRY = (
    ROOT / "configs/skillvis_v2_accuracy/semantic_concept_registry_v3.json"
)


@dataclass(frozen=True)
class SemanticConcept:
    concept_id: str
    query_terms: tuple[str, ...]
    schema_terms: tuple[str, ...]
    allowed_semantic_types: tuple[str, ...]


@dataclass(frozen=True)
class QueryConceptMatch:
    concept_id: str
    text: str
    normalized: str
    start: int
    end: int


@dataclass(frozen=True)
class SemanticConceptRegistry:
    concepts: tuple[SemanticConcept, ...]
    sha256: str

    def concepts_for_field(self, field: SchemaFieldProfile) -> tuple[str, ...]:
        surfaces = {
            field.normalized_name,
            *field.normalized_name.split(),
            *field.aliases,
            *(token for alias in field.aliases for token in alias.split()),
        }
        matched = {
            concept.concept_id
            for concept in self.concepts
            if field.semantic_type in concept.allowed_semantic_types
            if any(
                schema_term in surfaces
                or any(
                    schema_term in surface.split()
                    for surface in surfaces
                )
                for schema_term in concept.schema_terms
            )
        }
        return tuple(sorted(matched))

    def match_query(
        self,
        normalized: QueryNormalizationResult,
    ) -> tuple[QueryConceptMatch, ...]:
        lemmas = list(normalized.lemmas_or_normalized_terms)
        spans = list(normalized.query_spans)
        candidates: list[tuple[int, int, QueryConceptMatch]] = []
        for concept in self.concepts:
            for phrase in concept.query_terms:
                phrase_tokens = phrase.split()
                width = len(phrase_tokens)
                for index in range(len(lemmas) - width + 1):
                    if lemmas[index : index + width] != phrase_tokens:
                        continue
                    start = spans[index].start
                    end = spans[index + width - 1].end
                    candidates.append(
                        (
                            -width,
                            start,
                            QueryConceptMatch(
                                concept_id=concept.concept_id,
                                text=normalized.original_query[start:end],
                                normalized=phrase,
                                start=start,
                                end=end,
                            ),
                        )
                    )
        # Keep the longest concept phrase at a given overlapping surface.  A
        # phrase may still map to multiple schema fields through one concept.
        selected: list[QueryConceptMatch] = []
        for _, _, item in sorted(candidates, key=lambda row: (row[0], row[1], row[2].concept_id)):
            if any(
                item.start < current.end and current.start < item.end
                for current in selected
            ):
                continue
            selected.append(item)
        return tuple(sorted(selected, key=lambda item: (item.start, item.end, item.concept_id)))


@lru_cache(maxsize=1)
def load_semantic_concept_registry(
    path: Path = DEFAULT_CONCEPT_REGISTRY,
) -> SemanticConceptRegistry:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("status") != "INTERNAL_VERSIONED_SEMANTIC_BRIDGE":
        raise ValueError("semantic concept registry is not versioned for execution")
    concepts = tuple(
        SemanticConcept(
            concept_id=item["concept_id"],
            query_terms=tuple(
                sorted(
                    {
                        normalize_field_phrase(term)
                        for term in item["query_terms"]
                        if normalize_field_phrase(term)
                    }
                )
            ),
            schema_terms=tuple(
                sorted(
                    {
                        normalize_field_phrase(term)
                        for term in item["schema_terms"]
                        if normalize_field_phrase(term)
                    }
                )
            ),
            allowed_semantic_types=tuple(sorted(item["allowed_semantic_types"])),
        )
        for item in payload["concepts"]
    )
    ids = [item.concept_id for item in concepts]
    if len(ids) != len(set(ids)):
        raise ValueError("semantic concept IDs must be unique")
    return SemanticConceptRegistry(
        concepts=concepts,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "DEFAULT_CONCEPT_REGISTRY",
    "QueryConceptMatch",
    "SemanticConcept",
    "SemanticConceptRegistry",
    "load_semantic_concept_registry",
]
