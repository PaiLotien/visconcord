"""High-recall M2 field candidates with contrastive schema-local selection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from skillvis_v2_accuracy.config import SkillVISV2Config
from skillvis_v2_accuracy.contracts import (
    AmbiguityRecord,
    CandidateScore,
    FieldCandidate,
    FieldEvidence,
    FieldResolutionResult,
    QueryMention,
    QueryNormalizationResult,
    SchemaProfile,
)
from skillvis_v2_accuracy.rules import RuleRegistry

from .config import JointSemanticSettings
from .contracts import SchemaSemanticProfile, SemanticFieldMention
from .schema_semantics import decompose_schema, extract_semantic_mentions


_CONTRIBUTION = {
    "entity": 0.38,
    "property": 0.66,
    "measure": 0.72,
    "qualifier": 0.52,
    "generic": 0.28,
}
_GROUP_CUE = re.compile(
    r"\b(each|per|by|across|grouped|group|categorized|categorised|divided|based)\b",
    re.IGNORECASE,
)
_NAMESPACE_CUE = re.compile(r"\bnamespaces?\b", re.IGNORECASE)
_IDENTIFIER_CUE = re.compile(r"\b(identifier|identifiers|identity|identities|ids?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class JointFieldArtifacts:
    result: FieldResolutionResult
    schema_profiles: tuple[SchemaSemanticProfile, ...]
    semantic_mentions: tuple[SemanticFieldMention, ...]
    warnings: tuple[str, ...]


def _concepts(profile: SchemaSemanticProfile) -> set[str]:
    return {
        *profile.entity_concepts,
        *profile.property_concepts,
        *profile.qualifier_concepts,
        *profile.generic_concepts,
    }


def _query_mention(mention: SemanticFieldMention) -> QueryMention:
    span = mention.query_span
    return QueryMention(
        mention_id=mention.mention_id,
        text=span.text,
        normalized=span.normalized,
        start=span.start,
        end=span.end,
        mention_kind="alias",
        source_rule_ids=("R-FR-020",),
    )


def _semantic_type_roles(semantic_type: str) -> set[str]:
    if semantic_type == "quantitative":
        return {"measure"}
    if semantic_type == "temporal":
        return {"temporal", "dimension"}
    if semantic_type == "identifier":
        return {"identifier", "dimension"}
    if semantic_type == "geographic":
        return {"geographic", "dimension"}
    if semantic_type in {"categorical", "ordinal"}:
        return {"dimension"}
    return {"unknown"}


def _semantic_score(
    *,
    profile: SchemaSemanticProfile,
    mentions: tuple[SemanticFieldMention, ...],
    query: str,
) -> tuple[float, float, tuple[str, ...]]:
    field_concepts = _concepts(profile)
    matched = tuple(item for item in mentions if item.concept_id in field_concepts)
    if not matched:
        return 0.0, 0.0, ()
    kinds = {item.concept_kind for item in matched}
    # Independent entity + property evidence is additive. Repeated synonyms for
    # one concept are not, which avoids long queries inflating scores.
    concept_ids = {item.concept_id for item in matched}
    contribution = 0.0
    for concept_id in sorted(concept_ids):
        kind = next(item.concept_kind for item in matched if item.concept_id == concept_id)
        value = _CONTRIBUTION[kind]
        if contribution:
            value *= 0.48
        contribution += value
    positive: list[str] = [f"concept:{item}" for item in sorted(concept_ids)]
    penalty = 0.0
    components = set(profile.components)
    has_group_cue = bool(_GROUP_CUE.search(query))
    has_identifier_cue = bool(_IDENTIFIER_CUE.search(query))
    has_namespace_cue = bool(_NAMESPACE_CUE.search(query))
    if {"local", "id"} <= components and (has_group_cue or has_identifier_cue):
        contribution += 0.18
        positive.append("contrastive:entity_instance_identifier")
    if "namespace" in components:
        if has_namespace_cue:
            contribution += 0.18
            positive.append("contrastive:explicit_namespace")
        elif has_group_cue or has_identifier_cue:
            penalty = 0.30
            positive.append("negative:namespace_without_namespace_cue")
    return min(0.84, contribution), penalty, tuple(positive)


def _dedupe_evidence(items: tuple[FieldEvidence, ...] | list[FieldEvidence]) -> tuple[FieldEvidence, ...]:
    rows = {}
    for item in items:
        span = item.query_span
        key = (
            item.evidence_type,
            span.start if span else -1,
            span.end if span else -1,
            item.value,
            item.rule_id,
        )
        rows[key] = item
    return tuple(rows[key] for key in sorted(rows))


def _overlaps(left, right) -> bool:
    return left.start < right.end and right.start < left.end


def run_joint_field_resolution(
    *,
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    base: FieldResolutionResult,
    config: SkillVISV2Config,
    settings: JointSemanticSettings,
    registry: RuleRegistry,
) -> JointFieldArtifacts:
    profiles = decompose_schema(schema, settings)
    mentions = extract_semantic_mentions(normalized, settings)
    specific_mentions = tuple(item for item in mentions if item.concept_kind != "generic")
    active_mentions = tuple(
        item
        for item in mentions
        if item.concept_kind != "generic"
        or not any(_overlaps(item.query_span, other.query_span) for other in specific_mentions)
    )
    profile_by_name = {item.field_name: item for item in profiles}
    schema_by_name = {item.field_name: item for item in schema.fields}
    base_by_name = {item.field_name: item for item in base.candidate_fields}
    query_mentions = {item.mention_id: item for item in base.query_mentions}
    query_mentions.update({item.mention_id: _query_mention(item) for item in mentions})
    candidates: list[FieldCandidate] = []
    semantic_fields: set[str] = set()
    warning_rows: list[str] = []

    for field_name in sorted(schema_by_name, key=str.casefold):
        old = base_by_name.get(field_name)
        semantic_raw, semantic_penalty, reasons = _semantic_score(
            profile=profile_by_name[field_name],
            mentions=active_mentions,
            query=normalized.original_query,
        )
        matched_mentions = tuple(
            item for item in active_mentions if item.concept_id in _concepts(profile_by_name[field_name])
        )
        if old is None and not matched_mentions:
            continue
        if matched_mentions:
            semantic_fields.add(field_name)
        old_score = old.score_components if old else CandidateScore(
            lexical_score=0.0,
            alias_score=0.0,
            value_score=0.0,
            type_score=0.0,
            task_role_score=0.0,
            conflict_penalty=0.0,
            total_score=0.0,
        )
        evidence = list(old.evidence if old else ())
        mention_ids = set(old.mention_ids if old else ())
        match_types = set(old.match_types if old else ())
        roles = set(old.intended_roles if old else ())
        for mention in matched_mentions:
            mention_ids.add(mention.mention_id)
            match_types.add("schema_semantic_concept")
            roles.update(mention.expected_roles)
            evidence.append(
                FieldEvidence(
                    evidence_type="schema_semantic_concept",
                    query_span=mention.query_span,
                    value=mention.concept_id,
                    rule_id="R-FR-021",
                    source_ids=registry.get("R-FR-021").source_ids,
                )
            )
        roles.update(_semantic_type_roles(schema_by_name[field_name].semantic_type))
        if (
            matched_mentions
            and _GROUP_CUE.search(normalized.original_query)
            and schema_by_name[field_name].semantic_type
            in {"categorical", "ordinal", "geographic", "identifier", "temporal"}
        ):
            roles.update({"dimension", "grouping"})
        for reason in reasons:
            evidence.append(
                FieldEvidence(
                    evidence_type="contrastive_semantic_evidence",
                    query_span=matched_mentions[0].query_span if matched_mentions else None,
                    value=reason,
                    rule_id="R-FR-022",
                    source_ids=registry.get("R-FR-022").source_ids,
                )
            )
        alias_score = max(old_score.alias_score, semantic_raw)
        type_score = max(old_score.type_score, 0.08 if matched_mentions else 0.0)
        role_score = max(
            old_score.task_role_score,
            0.12 if matched_mentions and _GROUP_CUE.search(normalized.original_query) else 0.0,
        )
        penalty = min(1.0, max(old_score.conflict_penalty, semantic_penalty))
        total = min(
            1.0,
            max(
                0.0,
                old_score.lexical_score
                + alias_score
                + old_score.value_score
                + type_score
                + role_score
                - penalty,
            ),
        )
        score = CandidateScore(
            lexical_score=old_score.lexical_score,
            alias_score=round(alias_score, 6),
            value_score=old_score.value_score,
            type_score=round(type_score, 6),
            task_role_score=round(role_score, 6),
            conflict_penalty=round(penalty, 6),
            total_score=round(total, 6),
        )
        rule_ids = set(old.source_rule_ids if old else ())
        if matched_mentions:
            rule_ids.update({"R-SD-001", "R-FR-020", "R-FR-021", "R-FR-022"})
        candidates.append(
            FieldCandidate(
                field_name=field_name,
                mention_ids=tuple(sorted(mention_ids)),
                candidate_rank=1,
                score_components=score,
                match_types=tuple(sorted(match_types)),
                evidence=_dedupe_evidence(evidence),
                intended_roles=tuple(sorted(roles)),
                source_rule_ids=registry.validate_ids(rule_ids),
            )
        )

    candidates.sort(key=lambda item: (-item.score_components.total_score, item.field_name.casefold()))
    # Retain top-k for every mention rather than truncating the full schema.
    per_mention: dict[str, list[FieldCandidate]] = {}
    for item in candidates:
        for mention_id in item.mention_ids:
            per_mention.setdefault(mention_id, []).append(item)
    keep_names = {
        row.field_name
        for rows in per_mention.values()
        for row in sorted(
            rows,
            key=lambda item: (-item.score_components.total_score, item.field_name.casefold()),
        )[: settings.field_top_k]
    }
    # Always keep explicitly grounded base candidates as audit evidence.
    keep_names.update(
        item.field_name
        for item in base.candidate_fields
        if {"exact", "normalized_lexical", "alias", "value"} & set(item.match_types)
    )
    candidates = [item for item in candidates if item.field_name in keep_names]
    candidates = [
        item.model_copy(update={"candidate_rank": rank})
        for rank, item in enumerate(candidates, start=1)
    ]

    # Specific semantic mentions shadow generic type/kind mentions only when
    # their spans overlap and the selected field has a clear score margin.
    generic_mentions = {
        item.mention_id: item for item in mentions if item.concept_kind == "generic"
    }
    required: set[str] = set()
    ambiguity: list[AmbiguityRecord] = [
        item for item in base.ambiguity if item.dimension != "field"
    ]
    candidate_by_mention: dict[str, list[FieldCandidate]] = {}
    for candidate in candidates:
        for mention_id in candidate.mention_ids:
            candidate_by_mention.setdefault(mention_id, []).append(candidate)

    for mention_id, rows in sorted(candidate_by_mention.items()):
        generic = generic_mentions.get(mention_id)
        if generic and any(
            _overlaps(generic.query_span, item.query_span) for item in specific_mentions
        ):
            continue
        eligible = [
            item
            for item in rows
            if item.score_components.total_score >= settings.semantic_candidate_threshold
        ]
        if not eligible:
            continue
        eligible.sort(key=lambda item: (-item.score_components.total_score, item.field_name.casefold()))
        top = eligible[0].score_components.total_score
        tied = [
            item for item in eligible
            if top - item.score_components.total_score <= config.ambiguity_margin
        ]
        required.update(item.field_name for item in tied)
        if len(tied) > 1:
            ambiguity.append(
                AmbiguityRecord(
                    dimension="field",
                    alternatives=tuple(item.field_name for item in tied),
                    reason=(
                        "multiple schema fields remain contrastively equivalent for "
                        f"mention {mention_id}"
                    ),
                    evidence_span_ids=(mention_id,),
                    rule_ids=("R-FR-022",),
                )
            )

    # Remove a generic token-only base selection when a stronger semantic field
    # covers the same text. Candidate evidence remains in candidate_fields.
    strong_semantic = [
        item
        for item in candidates
        if "schema_semantic_concept" in item.match_types
        and item.score_components.total_score >= settings.semantic_candidate_threshold
    ]
    for candidate in candidates:
        if not (
            "token_overlap" in candidate.match_types
            and not ({"exact", "alias", "value", "schema_semantic_concept"} & set(candidate.match_types))
        ):
            continue
        spans = [item.query_span for item in candidate.evidence if item.query_span]
        if any(
            candidate.score_components.total_score + config.ambiguity_margin
            < stronger.score_components.total_score
            and any(
                _overlaps(left, right.query_span)
                for left in spans
                for right in stronger.evidence
                if right.query_span is not None
            )
            for stronger in strong_semantic
        ):
            required.discard(candidate.field_name)

    # Entity nouns inside "number of/how many" denote the counted row grain.
    # A partial field-name overlap (orders -> order_total, incidents ->
    # incident_id) must not turn that noun into an aggregate operand.
    count_entity_pattern = re.compile(
        r"(?:number\s+of|how\s+many|count(?:s|ed|ing)?|total\s+number\s+of)\s+$",
        re.IGNORECASE,
    )
    for candidate in candidates:
        strong_types = {
            "exact", "normalized_lexical", "alias", "value"
        } & set(candidate.match_types)
        specific_semantic = any(
            evidence.evidence_type == "schema_semantic_concept"
            and evidence.value != "property.category_type"
            for evidence in candidate.evidence
        )
        if strong_types or specific_semantic:
            continue
        for evidence in candidate.evidence:
            span = evidence.query_span
            if span is None or evidence.evidence_type != "token_overlap":
                continue
            prefix = normalized.original_query[max(0, span.start - 28) : span.start]
            if count_entity_pattern.search(prefix):
                required.discard(candidate.field_name)
                break

    covered = [item.query_span for item in mentions]
    unresolved = tuple(
        item for item in base.unresolved_mentions
        if not any(span.start <= item.start and item.end <= span.end for span in covered)
    )
    if semantic_fields:
        warning_rows.append(
            "development joint semantic registry contributed field candidates; "
            "formal protocol evidence is not authorized"
        )
    abstention_reason = None
    if not candidates:
        abstention_reason = "no_reliable_field_match"
    elif not required:
        abstention_reason = "no_field_candidate_above_threshold"
    applied = set(base.applied_rule_ids)
    if profiles:
        applied.add("R-SD-001")
    if mentions:
        applied.update({"R-FR-020", "R-FR-021", "R-FR-022"})
    resolved_ambiguity: list[AmbiguityRecord] = []
    for item in ambiguity:
        if item.dimension != "field":
            resolved_ambiguity.append(item)
            continue
        surviving = tuple(name for name in item.alternatives if name in required)
        if len(surviving) > 1:
            resolved_ambiguity.append(item.model_copy(update={"alternatives": surviving}))
    result = FieldResolutionResult(
        query_mentions=tuple(
            sorted(query_mentions.values(), key=lambda item: (item.start, item.end, item.mention_id))
        ),
        candidate_fields=tuple(candidates),
        candidate_scores={item.field_name: item.score_components for item in candidates},
        match_types={item.field_name: item.match_types for item in candidates},
        evidence={item.field_name: item.evidence for item in candidates},
        intended_roles={item.field_name: item.intended_roles for item in candidates},
        ambiguity=tuple(
            sorted(
                resolved_ambiguity,
                key=lambda item: (item.dimension, item.alternatives, item.reason),
            )
        ),
        required_field_candidates=tuple(sorted(required)),
        unresolved_mentions=unresolved,
        applied_rule_ids=registry.validate_ids(applied),
        abstention_reason=abstention_reason,
    )
    return JointFieldArtifacts(
        result=result,
        schema_profiles=profiles,
        semantic_mentions=mentions,
        warnings=tuple(sorted(warning_rows)),
    )


__all__ = ["JointFieldArtifacts", "run_joint_field_resolution"]
