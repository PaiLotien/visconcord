"""M2 Semantic Field Resolver with auditable score components."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .config import SkillVISV2Config
from .contracts import (
    AmbiguityRecord,
    CandidateScore,
    EvidenceSpan,
    FieldCandidate,
    FieldEvidence,
    FieldResolutionResult,
    QueryMention,
    QueryNormalizationResult,
    SchemaFieldProfile,
    SchemaProfile,
)
from .query_normalization import normalize_field_phrase
from .rules import RuleRegistry
from .semantic_concepts import load_semantic_concept_registry


_CONTENT_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "chart", "compare",
    "comparison", "correlation", "data", "display", "distribution", "do", "each",
    "for", "from", "graph", "has", "have", "in", "is", "it", "line", "make",
    "me", "of", "on", "over", "plot", "rank", "ranking", "show", "sort", "the",
    "to", "trend", "versus", "visualize", "what", "which", "with", "year",
    "years", "time", "top", "bottom", "highest", "lowest", "average", "avg",
    "mean", "total", "sum", "count", "number", "how", "many", "not", "only",
    "excluding", "without", "scatter", "histogram", "boxplot", "pie", "bar",
    "analyze", "draw", "between", "than", "into", "using", "return",
}
_GENERIC_OVERLAP = {
    "value", "values", "measure", "measures", "dimension", "dimensions",
    "amount", "amounts", "number", "numbers", "score", "scores", "total",
    "rating", "ratings",
}


@dataclass
class _AccumulatedCandidate:
    field: SchemaFieldProfile
    mention_ids: set[str] = field(default_factory=set)
    match_types: set[str] = field(default_factory=set)
    evidence: list[FieldEvidence] = field(default_factory=list)
    lexical_raw: float = 0.0
    alias_raw: float = 0.0
    value_raw: float = 0.0
    structural_recovery_raw: float = 0.0
    intended_roles: set[str] = field(default_factory=set)


def _find_phrase(query: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase:
        return []
    query_cf = query.casefold()
    phrase_cf = phrase.casefold()
    pattern = re.compile(rf"(?<!\w){re.escape(phrase_cf)}(?!\w)")
    return [(match.start(), match.end()) for match in pattern.finditer(query_cf)]


def _mention(
    query: str,
    start: int,
    end: int,
    kind: str,
    rule_id: str,
) -> QueryMention:
    text = query[start:end]
    digest = hashlib.sha1(
        f"{start}:{end}:{kind}:{text.casefold()}".encode("utf-8")
    ).hexdigest()[:8]
    return QueryMention(
        mention_id=f"m-{start}-{end}-{kind}-{digest}",
        text=text,
        normalized=normalize_field_phrase(text),
        start=start,
        end=end,
        mention_kind=kind,
        source_rule_ids=(rule_id,),
    )


def _span_from_mention(mention: QueryMention, rule_id: str) -> EvidenceSpan:
    return EvidenceSpan(
        start=mention.start,
        end=mention.end,
        text=mention.text,
        normalized=mention.normalized,
        kind=f"field_{mention.mention_kind}",
        rule_ids=(rule_id,),
    )


def _source_ids(registry: RuleRegistry, rule_id: str) -> tuple[str, ...]:
    return registry.get(rule_id).source_ids


def _role_context(
    normalized: QueryNormalizationResult,
    profile: SchemaFieldProfile,
    mention: QueryMention,
    match_type: str,
) -> set[str]:
    roles = set(profile.role_candidates)
    prefix = normalized.original_query[max(0, mention.start - 12) : mention.start].casefold()
    if re.search(r"(?:\bby|\bacross|\bper|\bfor each)\s+$", prefix):
        roles.update({"grouping", "dimension"})
    if match_type == "value":
        roles.update({"filter", "dimension"})
    if profile.semantic_type == "quantitative":
        roles.add("measure")
    if profile.semantic_type == "temporal":
        roles.add("temporal")
    if profile.semantic_type == "geographic":
        roles.update({"geographic", "dimension"})
    if profile.semantic_type == "identifier":
        roles.add("identifier")
    return roles


def _task_hints(normalized: QueryNormalizationResult) -> set[str]:
    hints = {item.canonical for item in normalized.task_terms}
    if normalized.temporal_expressions:
        hints.add("temporal_analysis")
    if normalized.ranking_terms or normalized.detected_superlatives:
        hints.add("ranking")
    if normalized.aggregation_terms:
        hints.add("aggregation")
    return hints


def _type_and_task_raw(
    field: SchemaFieldProfile,
    roles: set[str],
    hints: set[str],
) -> tuple[float, float, float]:
    type_raw = 1.0 if field.semantic_type != "unknown" else 0.0
    task_raw = 0.0
    conflict = 0.0
    if "temporal_analysis" in hints:
        if field.semantic_type in {"temporal", "quantitative"}:
            task_raw = 1.0
        elif field.semantic_type == "identifier":
            conflict = 1.0
    if "ranking" in hints or "aggregation" in hints:
        if field.semantic_type in {
            "quantitative", "categorical", "ordinal", "geographic"
        }:
            task_raw = max(task_raw, 1.0)
        elif field.semantic_type == "identifier" and "measure" in roles:
            conflict = 1.0
    if hints & {"correlation"} and field.semantic_type == "quantitative":
        task_raw = max(task_raw, 1.0)
    return type_raw, task_raw, conflict


def _add_evidence(
    accumulated: _AccumulatedCandidate,
    normalized: QueryNormalizationResult,
    mention: QueryMention,
    match_type: str,
    raw_score: float,
    rule_id: str,
    registry: RuleRegistry,
) -> None:
    accumulated.mention_ids.add(mention.mention_id)
    accumulated.match_types.add(match_type)
    if match_type in {
        "exact",
        "normalized_lexical",
        "token_overlap",
        "acronym_expansion",
    }:
        accumulated.lexical_raw = max(accumulated.lexical_raw, raw_score)
    elif match_type == "alias":
        accumulated.alias_raw = max(accumulated.alias_raw, raw_score)
    elif match_type == "semantic_concept":
        accumulated.alias_raw = max(accumulated.alias_raw, raw_score)
    elif match_type == "value":
        accumulated.value_raw = max(accumulated.value_raw, raw_score)
    elif match_type == "task_conditioned_recovery":
        accumulated.structural_recovery_raw = max(
            accumulated.structural_recovery_raw, raw_score
        )
    accumulated.evidence.append(
        FieldEvidence(
            evidence_type=match_type,
            query_span=_span_from_mention(mention, rule_id),
            value=mention.text,
            rule_id=rule_id,
            source_ids=_source_ids(registry, rule_id),
        )
    )
    accumulated.intended_roles.update(
        _role_context(
            normalized,
            accumulated.field,
            mention,
            match_type,
        )
    )


def _contiguous_lemma_span(
    normalized: QueryNormalizationResult,
    phrase_tokens: list[str],
) -> tuple[int, int] | None:
    if not phrase_tokens:
        return None
    lemmas = list(normalized.lemmas_or_normalized_terms)
    spans = list(normalized.query_spans)
    for index in range(0, len(lemmas) - len(phrase_tokens) + 1):
        if lemmas[index : index + len(phrase_tokens)] == phrase_tokens:
            return spans[index].start, spans[index + len(phrase_tokens) - 1].end
    return None


def run_field_resolution(
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    config: SkillVISV2Config,
    registry: RuleRegistry,
) -> FieldResolutionResult:
    if not config.feature_flags.enable_m2_field_resolution:
        raise RuntimeError("M2 field resolution feature flag is disabled")
    if normalized.failure_state or schema.failure_state:
        reason = "normalization_or_schema_failure"
        return FieldResolutionResult(
            query_mentions=(),
            candidate_fields=(),
            candidate_scores={},
            match_types={},
            evidence={},
            intended_roles={},
            ambiguity=(),
            required_field_candidates=(),
            unresolved_mentions=(),
            applied_rule_ids=(),
            abstention_reason=reason,
        )

    query = normalized.original_query
    mentions: dict[str, QueryMention] = {}
    candidates: dict[str, _AccumulatedCandidate] = {}
    applied: set[str] = {"R-FR-004"}

    def candidate_for(field_profile: SchemaFieldProfile) -> _AccumulatedCandidate:
        if field_profile.field_name not in candidates:
            acc = _AccumulatedCandidate(field=field_profile)
            candidates[field_profile.field_name] = acc
        return candidates[field_profile.field_name]

    for field_profile in schema.fields:
        acc = candidate_for(field_profile)
        exact_phrases = {
            field_profile.field_name.casefold().replace("_", " "),
            field_profile.field_name.casefold(),
        }
        exact_found = False
        for phrase in sorted(exact_phrases):
            for start, end in _find_phrase(query.replace("_", " "), phrase):
                mention = _mention(query, start, end, "field", "R-FR-001")
                mentions[mention.mention_id] = mention
                _add_evidence(acc, normalized, mention, "exact", 1.0, "R-FR-001", registry)
                exact_found = True
                applied.add("R-FR-001")

        normalized_tokens = field_profile.normalized_name.split()
        lemma_span = _contiguous_lemma_span(normalized, normalized_tokens)
        if lemma_span and not exact_found:
            mention = _mention(
                query, lemma_span[0], lemma_span[1], "field", "R-FR-005"
            )
            mentions[mention.mention_id] = mention
            _add_evidence(
                acc, normalized, mention, "normalized_lexical", 0.9, "R-FR-005", registry
            )
            applied.add("R-FR-005")

        for alias in field_profile.aliases:
            alias_span = _contiguous_lemma_span(normalized, alias.split())
            if alias_span:
                mention = _mention(
                    query, alias_span[0], alias_span[1], "alias", "R-FR-002"
                )
                mentions[mention.mention_id] = mention
                _add_evidence(acc, normalized, mention, "alias", 1.0, "R-FR-002", registry)
                applied.add("R-FR-002")

        if (
            config.feature_flags.enable_value_matching
            and field_profile.cardinality <= config.max_value_cardinality
        ):
            for value in field_profile.sample_values:
                text_value = str(value).strip()
                if (
                    len(text_value) < 2
                    or text_value.casefold() in {"true", "false", "none"}
                ):
                    continue
                for start, end in _find_phrase(query, text_value):
                    mention = _mention(query, start, end, "value", "R-FR-003")
                    mentions[mention.mention_id] = mention
                    _add_evidence(
                        acc, normalized, mention, "value", 1.0, "R-FR-003", registry
                    )
                    applied.add("R-FR-003")

        if not acc.mention_ids and normalized_tokens:
            acronym = field_profile.field_name.strip()
            if (
                acronym.isalpha()
                and acronym.isupper()
                and 2 <= len(acronym) <= 5
            ):
                lemmas = list(normalized.lemmas_or_normalized_terms)
                spans = list(normalized.query_spans)
                width = len(acronym)
                for index in range(len(lemmas) - width + 1):
                    terms = lemmas[index : index + width]
                    if (
                        all(term.isalpha() and len(term) >= 2 for term in terms)
                        and "".join(term[0] for term in terms).casefold()
                        == acronym.casefold()
                    ):
                        mention = _mention(
                            query,
                            spans[index].start,
                            spans[index + width - 1].end,
                            "field",
                            "R-FR-006",
                        )
                        mentions[mention.mention_id] = mention
                        _add_evidence(
                            acc,
                            normalized,
                            mention,
                            "acronym_expansion",
                            0.85,
                            "R-FR-006",
                            registry,
                        )
                        applied.add("R-FR-006")
                        break

        if not acc.mention_ids and normalized_tokens:
            query_terms = set(normalized.lemmas_or_normalized_terms)
            field_terms = set(normalized_tokens)
            overlap = (query_terms & field_terms) - _GENERIC_OVERLAP
            if overlap:
                raw = len(overlap) / len(field_terms)
                # A non-generic head noun can identify a multi-token field when
                # natural language omits a schema modifier (e.g. "budget" for
                # "Production Budget"). The partial nature remains visible in
                # the score and token_overlap evidence.
                if (
                    len(overlap) == 1
                    and normalized_tokens[-1] in overlap
                    and normalized_tokens[-1] not in _GENERIC_OVERLAP
                ):
                    raw = max(raw, 0.75)
                if raw >= 0.5:
                    token_spans = [
                        span
                        for lemma, span in zip(
                            normalized.lemmas_or_normalized_terms,
                            normalized.query_spans,
                            strict=True,
                        )
                        if lemma in overlap
                    ]
                    mention = _mention(
                        query,
                        min(span.start for span in token_spans),
                        max(span.end for span in token_spans),
                        "field",
                        "R-FR-006",
                    )
                    mentions[mention.mention_id] = mention
                    _add_evidence(
                        acc,
                        normalized,
                        mention,
                        "token_overlap",
                        min(0.75, raw),
                        "R-FR-006",
                        registry,
                    )
                    applied.add("R-FR-006")

    # A bounded concept registry aligns query words with schema-local names.
    # It is intentionally weaker than explicit aliases and retains ties rather
    # than using embedding similarity or a schema-order fallback.
    semantic_registry = load_semantic_concept_registry()
    field_concepts = {
        field_profile.field_name: set(
            semantic_registry.concepts_for_field(field_profile)
        )
        for field_profile in schema.fields
    }
    strong_grounded_spans = tuple(
        evidence.query_span
        for acc in candidates.values()
        for evidence in acc.evidence
        if evidence.query_span is not None
        and evidence.evidence_type
        in {"exact", "normalized_lexical", "alias", "acronym_expansion"}
    )
    for concept_match in semantic_registry.match_query(normalized):
        if any(
            span.start <= concept_match.start and concept_match.end <= span.end
            for span in strong_grounded_spans
        ):
            continue
        matched_fields = [
            field_profile
            for field_profile in schema.fields
            if concept_match.concept_id in field_concepts[field_profile.field_name]
        ]
        for field_profile in matched_fields:
            acc = candidate_for(field_profile)
            already_grounded = any(
                evidence.query_span is not None
                and evidence.query_span.start == concept_match.start
                and evidence.query_span.end == concept_match.end
                for evidence in acc.evidence
            )
            if already_grounded:
                continue
            mention = _mention(
                query,
                concept_match.start,
                concept_match.end,
                "alias",
                "R-FR-010",
            )
            mentions[mention.mention_id] = mention
            _add_evidence(
                acc,
                normalized,
                mention,
                "semantic_concept",
                0.9,
                "R-FR-010",
                registry,
            )
            # Record the concept identity without changing the public score
            # contract or inventing an unobserved schema alias.
            acc.evidence.append(
                FieldEvidence(
                    evidence_type="semantic_concept_registry",
                    query_span=_span_from_mention(mention, "R-FR-010"),
                    value=(
                        f"{concept_match.concept_id}@"
                        f"{semantic_registry.sha256[:16]}"
                    ),
                    rule_id="R-FR-010",
                    source_ids=_source_ids(registry, "R-FR-010"),
                )
            )
            applied.add("R-FR-010")

    hints = _task_hints(normalized)
    # A unique temporal field may be recovered from an explicit temporal phrase.
    # This is registered structural evidence, not a schema-order/type-only fallback.
    if normalized.temporal_expressions:
        temporal_profiles = [
            item for item in schema.fields if item.semantic_type == "temporal"
        ]
        has_temporal_evidence = any(
            acc.mention_ids and acc.field.semantic_type == "temporal"
            for acc in candidates.values()
        )
        temporal_term_consumed_by_field = any(
            mention.start <= term.start and term.end <= mention.end
            for mention in mentions.values()
            if mention.mention_kind in {"field", "alias"}
            for term in normalized.temporal_expressions
        )
        if (
            len(temporal_profiles) == 1
            and not has_temporal_evidence
            and not temporal_term_consumed_by_field
        ):
            profile = temporal_profiles[0]
            acc = candidate_for(profile)
            term = normalized.temporal_expressions[0]
            mention = _mention(
                query, term.start, term.end, "structural", "R-FR-009"
            )
            mentions[mention.mention_id] = mention
            _add_evidence(
                acc,
                normalized,
                mention,
                "task_conditioned_recovery",
                1.0,
                "R-FR-009",
                registry,
            )
            acc.intended_roles.add("temporal")
            applied.add("R-FR-009")

    # "each <entity>" requires a row label for one-mark-per-entity designs.
    # Recover a unique human-readable label field only when the entity noun has
    # no strong exact/alias grounding of its own.  The structural decision and
    # source span remain explicit for later validation.
    lemmas = list(normalized.lemmas_or_normalized_terms)
    spans = list(normalized.query_spans)
    each_indices = [index for index, lemma in enumerate(lemmas[:-1]) if lemma == "each"]
    label_profiles = [
        item
        for item in schema.fields
        if item.semantic_type in {"categorical", "identifier"}
        and item.normalized_name.split()[-1:] in [["name"], ["label"], ["title"]]
    ]
    exact_label_profiles = [
        item
        for item in label_profiles
        if item.normalized_name in {"name", "label", "title"}
    ]
    if exact_label_profiles:
        label_profiles = exact_label_profiles
    if len(label_profiles) == 1:
        for index in each_indices:
            entity_span = spans[index + 1]
            strong_entity_match = any(
                acc.lexical_raw >= 0.9
                and any(
                    evidence.query_span is not None
                    and evidence.query_span.start <= entity_span.start
                    and entity_span.end <= evidence.query_span.end
                    for evidence in acc.evidence
                )
                for acc in candidates.values()
            )
            label_acc = candidate_for(label_profiles[0])
            if strong_entity_match or label_acc.mention_ids:
                continue
            mention = _mention(
                query,
                entity_span.start,
                entity_span.end,
                "structural",
                "R-FR-009",
            )
            mentions[mention.mention_id] = mention
            _add_evidence(
                label_acc,
                normalized,
                mention,
                "task_conditioned_recovery",
                0.85,
                "R-FR-009",
                registry,
            )
            label_acc.intended_roles.update({"dimension", "grouping"})
            applied.add("R-FR-009")

    weights = config.resolver_weights
    ranked_rows: list[tuple[_AccumulatedCandidate, CandidateScore]] = []
    for acc in candidates.values():
        if not acc.mention_ids:
            continue
        type_raw, task_raw, conflict_raw = _type_and_task_raw(
            acc.field, acc.intended_roles, hints
        )
        lexical = min(1.0, acc.lexical_raw * weights.lexical)
        alias = min(1.0, acc.alias_raw * weights.alias)
        value = min(1.0, acc.value_raw * weights.value)
        type_score = min(1.0, type_raw * weights.type_compatible)
        task_score = min(
            1.0,
            max(
                task_raw * weights.task_role,
                acc.structural_recovery_raw * weights.structural_recovery,
            ),
        )
        penalty = min(1.0, conflict_raw * weights.conflict)
        total = max(0.0, min(1.0, lexical + alias + value + type_score + task_score - penalty))
        score = CandidateScore(
            lexical_score=round(lexical, 6),
            alias_score=round(alias, 6),
            value_score=round(value, 6),
            type_score=round(type_score, 6),
            task_role_score=round(task_score, 6),
            conflict_penalty=round(penalty, 6),
            total_score=round(total, 6),
        )
        ranked_rows.append((acc, score))
        acc.match_types.add("type_compatible")
        acc.evidence.append(
            FieldEvidence(
                evidence_type="type_compatible",
                query_span=None,
                value=acc.field.semantic_type,
                rule_id="R-FR-007",
                source_ids=_source_ids(registry, "R-FR-007"),
            )
        )
        if hints or acc.structural_recovery_raw:
            acc.match_types.add("task_conditioned")
            acc.evidence.append(
                FieldEvidence(
                    evidence_type="task_conditioned",
                    query_span=None,
                    value=",".join(sorted(hints)) or "structural_temporal_recovery",
                    rule_id="R-FR-008",
                    source_ids=_source_ids(registry, "R-FR-008"),
                )
            )
        applied.update({"R-FR-007", "R-FR-008"})

    ranked_rows.sort(
        key=lambda item: (-item[1].total_score, item[0].field.field_name.casefold())
    )
    field_candidates: list[FieldCandidate] = []
    for rank, (acc, score) in enumerate(ranked_rows, 1):
        unique_evidence = {
            (
                ev.evidence_type,
                ev.query_span.start if ev.query_span else -1,
                ev.query_span.end if ev.query_span else -1,
                ev.value,
                ev.rule_id,
            ): ev
            for ev in acc.evidence
        }
        field_candidates.append(
            FieldCandidate(
                field_name=acc.field.field_name,
                mention_ids=tuple(sorted(acc.mention_ids)),
                candidate_rank=rank,
                score_components=score,
                match_types=tuple(sorted(acc.match_types)),
                evidence=tuple(
                    unique_evidence[key]
                    for key in sorted(unique_evidence)
                ),
                intended_roles=tuple(sorted(acc.intended_roles)),
                source_rule_ids=registry.validate_ids(
                    ev.rule_id for ev in unique_evidence.values()
                ),
            )
        )

    mention_to_candidates: dict[str, list[FieldCandidate]] = defaultdict(list)
    for candidate in field_candidates:
        for mention_id in candidate.mention_ids:
            mention_to_candidates[mention_id].append(candidate)

    ambiguity: list[AmbiguityRecord] = []
    required: set[str] = set()
    for mention_id, rows in sorted(mention_to_candidates.items()):
        rows.sort(key=lambda item: (-item.score_components.total_score, item.field_name))
        eligible = [
            item
            for item in rows
            if item.score_components.total_score >= config.field_candidate_threshold
        ]
        if eligible:
            top_score = eligible[0].score_components.total_score
            tied = [
                item
                for item in eligible
                if top_score - item.score_components.total_score <= config.ambiguity_margin
            ]
            selected = tied
            # In "number of tickets by queue", a partial match to ticket_id
            # is entity evidence for row count, not a requested visual role.
            # Keep the candidate/evidence in the trace while excluding it from
            # required visual fields when the match is identifier-only.
            count_entity_rows = []
            for item in tied:
                profile = schema.field(item.field_name)
                mention = mentions[mention_id]
                prefix = query[max(0, mention.start - 24) : mention.start].casefold()
                identifier_only = (
                    profile.semantic_type == "identifier"
                    and "token_overlap" in item.match_types
                    and not ({"exact", "alias", "semantic_concept", "value"} & set(item.match_types))
                )
                if identifier_only and re.search(
                    r"(?:number\s+of|how\s+many|count(?:s|ed|ing)?|total\s+number\s+of)\s+$",
                    prefix,
                ):
                    count_entity_rows.append(item)
            if count_entity_rows and len(count_entity_rows) == len(tied):
                selected = []
                applied.add("R-FR-011")
            required.update(item.field_name for item in selected)
            if len(tied) > 1:
                ambiguity.append(
                    AmbiguityRecord(
                        dimension="field",
                        alternatives=tuple(item.field_name for item in tied),
                        reason=f"multiple fields match mention {mention_id} within configured margin",
                        evidence_span_ids=(mention_id,),
                        rule_ids=("R-FR-001", "R-FR-002", "R-FR-003"),
                    )
                )

    covered_ranges = {
        (mention.start, mention.end)
        for mention in mentions.values()
        if mention.mention_kind != "unresolved"
    }
    operator_terms = (
        *normalized.detected_comparatives,
        *normalized.detected_superlatives,
        *normalized.temporal_expressions,
        *normalized.aggregation_terms,
        *normalized.ranking_terms,
        *normalized.task_terms,
        *normalized.chart_mentions,
        *normalized.limitation_terms,
    )
    covered_ranges.update((term.start, term.end) for term in operator_terms)
    unresolved: list[QueryMention] = []
    for token, lemma, span in zip(
        normalized.tokens,
        normalized.lemmas_or_normalized_terms,
        normalized.query_spans,
        strict=True,
    ):
        if (
            not any(start <= span.start and span.end <= end for start, end in covered_ranges)
            and lemma.isalpha()
            and len(lemma) >= 3
            and lemma not in _CONTENT_STOP
        ):
            mention = _mention(
                query, span.start, span.end, "unresolved", "R-FR-009"
            )
            unresolved.append(mention)
    if len(field_candidates) > 1 or unresolved:
        applied.add("R-FR-009")
    if unresolved:
        ambiguity.append(
            AmbiguityRecord(
                dimension="query",
                alternatives=tuple(item.text for item in unresolved),
                reason="content-bearing query tokens have no reliable schema grounding",
                evidence_span_ids=tuple(item.mention_id for item in unresolved),
                rule_ids=("R-FR-009",),
            )
        )

    abstention_reason = None
    if not field_candidates:
        abstention_reason = "no_reliable_field_match"
    elif not required:
        abstention_reason = "no_field_candidate_above_threshold"

    registry_ids = registry.validate_ids(applied)
    return FieldResolutionResult(
        query_mentions=tuple(
            sorted(
                (*mentions.values(), *unresolved),
                key=lambda item: (item.start, item.end, item.mention_kind, item.mention_id),
            )
        ),
        candidate_fields=tuple(field_candidates),
        candidate_scores={
            item.field_name: item.score_components for item in field_candidates
        },
        match_types={
            item.field_name: item.match_types for item in field_candidates
        },
        evidence={item.field_name: item.evidence for item in field_candidates},
        intended_roles={
            item.field_name: item.intended_roles for item in field_candidates
        },
        ambiguity=tuple(ambiguity),
        required_field_candidates=tuple(sorted(required)),
        unresolved_mentions=tuple(unresolved),
        applied_rule_ids=registry_ids,
        abstention_reason=abstention_reason,
    )
