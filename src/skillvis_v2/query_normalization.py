"""M1 Query Normalization — deterministic and span preserving."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .contracts import DetectedTerm, EvidenceSpan, QueryNormalizationResult
from .rules import RuleRegistry


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[_'-][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)

_CHART_TERMS = {
    "bar chart": "bar",
    "bar graph": "bar",
    "barchart": "bar",
    "bargraph": "bar",
    "line chart": "line",
    "line graph": "line",
    "linechart": "line",
    "linegraph": "line",
    "scatter plot": "scatter",
    "scatterplot": "scatter",
    "scatter": "scatter",
    "histogram": "histogram",
    "box plot": "boxplot",
    "boxplot": "boxplot",
    "pie chart": "pie",
    "piechart": "pie",
    "area chart": "area",
    "heat map": "heatmap",
    "heatmap": "heatmap",
    "table": "table",
}

_AGGREGATION_TERMS = {
    "average": "mean",
    "avg": "mean",
    "mean": "mean",
    "total": "sum",
    "sum": "sum",
    "count": "count",
    "number of": "count",
    "how many": "count",
    "median": "median",
    "minimum": "min",
    "maximum": "max",
}

_TASK_TERMS = {
    "compare": "comparison",
    "comparison": "comparison",
    "versus": "comparison",
    "vs": "comparison",
    "relationship": "correlation",
    "correlation": "correlation",
    "correlate": "correlation",
    "trend": "trend",
    "change over time": "trend",
    "distribution": "distribution",
    "spread": "distribution",
    "composition": "composition",
    "proportion": "composition",
    "share": "composition",
    "rank": "ranking",
    "ranking": "ranking",
    "filter": "filtering",
    "sort": "sorting",
}

_TEMPORAL_TERMS = {
    "over time": "over_time",
    "over the years": "over_years",
    "by year": "year",
    "by month": "month",
    "by quarter": "quarter",
    "daily": "day",
    "weekly": "week",
    "monthly": "month",
    "quarterly": "quarter",
    "yearly": "year",
    "year": "year",
    "date": "date",
    "time": "time",
}

_RANKING_TERMS = {
    "top": "descending",
    "bottom": "ascending",
    "highest": "descending",
    "lowest": "ascending",
    "largest": "descending",
    "smallest": "ascending",
    "most": "descending",
    "least": "ascending",
    "rank": "unspecified",
    "ranking": "unspecified",
    "descending": "descending",
    "ascending": "ascending",
}

_COMPARATIVE_TERMS = {
    "higher": "greater",
    "lower": "less",
    "greater": "greater",
    "less": "less",
    "more": "greater",
    "fewer": "less",
    "above": "greater",
    "below": "less",
}

_SUPERLATIVE_TERMS = {
    "highest": "maximum",
    "lowest": "minimum",
    "largest": "maximum",
    "smallest": "minimum",
    "most": "maximum",
    "least": "minimum",
    "top": "maximum",
    "bottom": "minimum",
}

_LIMITATION_TERMS = {
    "not": "negation",
    "no": "negation",
    "without": "exclusion",
    "excluding": "exclusion",
    "except": "exclusion",
    "only": "restriction",
    "within": "restriction",
}

_IRREGULAR_LEMMAS = {
    "categories": "category",
    "countries": "country",
    "cities": "city",
    "companies": "company",
    "salaries": "salary",
    "values": "value",
    "sales": "sales",
    "series": "series",
    "data": "data",
    "indices": "index",
    "highest": "high",
    "lowest": "low",
    "better": "good",
    "worse": "bad",
}

_INFLECTION_ALLOWLIST = {
    "regions",
    "origins",
    "products",
    "customers",
    "orders",
    "years",
    "months",
    "quarters",
    "ratings",
    "budgets",
    "profits",
    "costs",
    "prices",
    "measures",
    "dimensions",
    "groups",
    "segments",
    "genres",
    "scores",
    "amounts",
    "counts",
    "turnovers",
    "weights",
}


@dataclass(frozen=True)
class _Token:
    text: str
    normalized: str
    lemma: str
    start: int
    end: int


def normalize_lexical_token(token: str) -> str:
    term = unicodedata.normalize("NFKC", token).casefold().strip()
    term = term.replace("_", " ")
    if term in _IRREGULAR_LEMMAS:
        return _IRREGULAR_LEMMAS[term]
    if term in _INFLECTION_ALLOWLIST and term.endswith("s") and len(term) > 3:
        return term[:-1]
    return term


def normalize_field_phrase(text: str) -> str:
    tokens = [
        normalize_lexical_token(match.group(0))
        for match in _TOKEN_RE.finditer(unicodedata.normalize("NFKC", text))
        if match.group(0).isalnum() or any(ch.isalnum() for ch in match.group(0))
    ]
    return " ".join(tokens)


def _build_tokens(query: str) -> tuple[_Token, ...]:
    result = []
    compact_chart_surface = {
        "barchart": "bar chart",
        "bargraph": "bar graph",
        "linechart": "line chart",
        "linegraph": "line graph",
        "scatterplot": "scatter plot",
        "boxplot": "box plot",
        "piechart": "pie chart",
        "heatmap": "heat map",
    }
    for match in _TOKEN_RE.finditer(query):
        original = match.group(0)
        normalized = unicodedata.normalize("NFKC", original).casefold()
        normalized = compact_chart_surface.get(normalized, normalized)
        result.append(
            _Token(
                text=original,
                normalized=normalized,
                lemma=compact_chart_surface.get(
                    unicodedata.normalize("NFKC", original).casefold(),
                    normalize_lexical_token(original),
                ),
                start=match.start(),
                end=match.end(),
            )
        )
    return tuple(result)


def _normalized_query(tokens: tuple[_Token, ...]) -> str:
    output = ""
    no_space_before = {".", ",", ":", ";", "?", "!", ")", "]", "}"}
    no_space_after = {"(", "[", "{"}
    for token in tokens:
        if not output or token.normalized in no_space_before or output[-1] in no_space_after:
            output += token.normalized
        else:
            output += " " + token.normalized
    return output


def _detect_phrases(
    query: str,
    vocabulary: dict[str, str],
    rule_id: str,
) -> tuple[DetectedTerm, ...]:
    lowered = unicodedata.normalize("NFKC", query).casefold()
    detected: list[DetectedTerm] = []
    for phrase, canonical in sorted(
        vocabulary.items(), key=lambda item: (-len(item[0]), item[0])
    ):
        pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)")
        for match in pattern.finditer(lowered):
            detected.append(
                DetectedTerm(
                    canonical=canonical,
                    text=query[match.start() : match.end()],
                    start=match.start(),
                    end=match.end(),
                    rule_ids=(rule_id,),
                )
            )
    unique = {
        (item.start, item.end, item.canonical, item.text.casefold()): item
        for item in detected
    }
    return tuple(
        sorted(unique.values(), key=lambda item: (item.start, item.end, item.canonical))
    )


def run_query_normalization(
    query: str,
    registry: RuleRegistry,
) -> QueryNormalizationResult:
    if not isinstance(query, str) or not query.strip():
        registry.get("R-QN-003")
        return QueryNormalizationResult(
            original_query=query if isinstance(query, str) else "",
            normalized_query="",
            tokens=(),
            lemmas_or_normalized_terms=(),
            query_spans=(),
            applied_rule_ids=("R-QN-003",),
            warnings=("query is empty",),
            failure_state="empty_query",
        )

    tokens = _build_tokens(query)
    spans = tuple(
        EvidenceSpan(
            start=token.start,
            end=token.end,
            text=token.text,
            normalized=token.normalized,
            kind="token",
            rule_ids=("R-QN-003",)
            + (("R-QN-004",) if token.lemma != token.normalized else ()),
        )
        for token in tokens
    )
    chart_mentions = _detect_phrases(query, _CHART_TERMS, "R-QN-001")
    aggregations = _detect_phrases(query, _AGGREGATION_TERMS, "R-QN-002")
    task_terms = _detect_phrases(query, _TASK_TERMS, "R-QN-002")
    temporal = _detect_phrases(query, _TEMPORAL_TERMS, "R-QN-002")
    ranking = _detect_phrases(query, _RANKING_TERMS, "R-QN-005")
    comparatives = _detect_phrases(query, _COMPARATIVE_TERMS, "R-QN-005")
    superlatives = _detect_phrases(query, _SUPERLATIVE_TERMS, "R-QN-005")
    limitations = _detect_phrases(query, _LIMITATION_TERMS, "R-QN-006")

    applied = {"R-QN-003"}
    if any(token.lemma != token.normalized for token in tokens):
        applied.add("R-QN-004")
    if chart_mentions:
        applied.add("R-QN-001")
    if aggregations or task_terms or temporal:
        applied.add("R-QN-002")
    if ranking or comparatives or superlatives:
        applied.add("R-QN-005")
    if limitations:
        applied.add("R-QN-006")
    applied_ids = registry.validate_ids(applied)

    warnings = []
    directions = {term.canonical for term in ranking if term.canonical != "unspecified"}
    if {"ascending", "descending"}.issubset(directions):
        warnings.append("conflicting ranking directions preserved")

    return QueryNormalizationResult(
        original_query=query,
        normalized_query=_normalized_query(tokens),
        tokens=tuple(token.normalized for token in tokens),
        lemmas_or_normalized_terms=tuple(token.lemma for token in tokens),
        detected_comparatives=comparatives,
        detected_superlatives=superlatives,
        temporal_expressions=temporal,
        aggregation_terms=aggregations,
        ranking_terms=ranking,
        task_terms=task_terms,
        chart_mentions=chart_mentions,
        limitation_terms=limitations,
        query_spans=spans,
        applied_rule_ids=applied_ids,
        warnings=tuple(warnings),
        failure_state=None,
    )
