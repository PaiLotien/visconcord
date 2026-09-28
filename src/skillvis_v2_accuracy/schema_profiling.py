"""M1 Schema Profiling using dtype, cardinality and bounded value evidence."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import SkillVISV2Config
from .contracts import (
    CategoricalSummary,
    CategoryValueCount,
    NumericSummary,
    SchemaFieldProfile,
    SchemaProfile,
    TemporalSummary,
)
from .query_normalization import normalize_field_phrase
from .rules import RuleRegistry


_ID_NAME_TOKENS = {
    "id", "identifier", "key", "uuid", "code", "number", "no", "num"
}
_TEMPORAL_NAME_TOKENS = {
    "date",
    "datetime",
    "time",
    "timestamp",
    "year",
    "month",
    "quarter",
    "day",
}
_ORDINAL_NAME_TOKENS = {"level", "grade", "rank", "rating", "stage", "priority", "tier"}
_GEO_NAME_TOKENS = {
    "country",
    "state",
    "province",
    "region",
    "continent",
    "city",
    "county",
    "latitude",
    "longitude",
}
_KNOWN_GEO_VALUES = {
    "africa",
    "asia",
    "europe",
    "north america",
    "south america",
    "oceania",
    "north",
    "south",
    "east",
    "west",
    "united states",
    "usa",
    "canada",
    "china",
    "india",
    "france",
    "germany",
    "brazil",
    "japan",
    "australia",
}
_KNOWN_ORDINAL_VALUES = {
    "very low",
    "low",
    "medium",
    "high",
    "very high",
    "poor",
    "fair",
    "good",
    "excellent",
    "first",
    "second",
    "third",
}
_DATE_SHAPE = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"[A-Za-z]{3,9}\s+(?:\d{1,2},?\s+)?\d{4}|\d{4})$"
)


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_unique(series: pd.Series, limit: int) -> tuple[Any, ...]:
    values = {_canonical_value(value): _safe_scalar(value) for value in series.dropna()}
    return tuple(values[key] for key in sorted(values)[:limit])


def _canonical_value(value: Any) -> str:
    return f"{type(value).__name__}:{_safe_scalar(value)!s}".casefold()


def _name_tokens(name: str) -> set[str]:
    return set(normalize_field_phrase(name).split())


def _date_parse(series: pd.Series) -> tuple[float, pd.Series]:
    strings = series.dropna().astype(str).str.strip()
    if strings.empty:
        return 0.0, pd.Series(dtype="datetime64[ns]")
    shaped = strings[strings.map(lambda item: bool(_DATE_SHAPE.match(item)))]
    if shaped.empty:
        return 0.0, pd.Series(dtype="datetime64[ns]")
    parsed = pd.to_datetime(shaped, errors="coerce", format="mixed", utc=False)
    ratio = float(parsed.notna().sum() / len(strings))
    return ratio, parsed.dropna()


def _numeric_summary(series: pd.Series) -> NumericSummary | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    std = float(numeric.std(ddof=0)) if len(numeric) else None
    return NumericSummary(
        count=int(len(numeric)),
        minimum=float(numeric.min()),
        maximum=float(numeric.max()),
        mean=float(numeric.mean()),
        median=float(numeric.median()),
        standard_deviation=std,
    )


def _categorical_summary(series: pd.Series) -> CategoricalSummary:
    non_null = series.dropna()
    counts = non_null.map(_safe_scalar).value_counts(dropna=False)
    pairs = sorted(
        ((_safe_scalar(value), int(count)) for value, count in counts.items()),
        key=lambda item: (-item[1], str(item[0]).casefold()),
    )
    return CategoricalSummary(
        unique_count=int(non_null.nunique(dropna=True)),
        top_values=tuple(
            CategoryValueCount(value=value, count=count) for value, count in pairs[:8]
        ),
    )


def _semantic_type(
    name: str,
    series: pd.Series,
) -> tuple[str, float, tuple[str, ...], TemporalSummary | None]:
    warnings: list[str] = []
    tokens = _name_tokens(name)
    non_null = series.dropna()
    physical = str(series.dtype)
    if non_null.empty:
        return "unknown", 0.0, ("all values are null",), None

    cardinality = int(non_null.nunique(dropna=True))
    uniqueness_ratio = cardinality / len(non_null)
    temporal_summary = None

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        converted = pd.to_datetime(non_null, errors="coerce")
        temporal_summary = TemporalSummary(
            parse_ratio=float(converted.notna().mean()),
            minimum=converted.min().isoformat() if converted.notna().any() else None,
            maximum=converted.max().isoformat() if converted.notna().any() else None,
            timezone_observed=getattr(series.dt, "tz", None) is not None,
        )
        return "temporal", 1.0, (), temporal_summary

    if pd.api.types.is_bool_dtype(series.dtype):
        return "categorical", 1.0, (), None

    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        return "ordinal", 1.0, (), None

    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        integer_like = bool(
            len(numeric)
            and np.isclose(numeric.to_numpy(dtype=float) % 1, 0).all()
        )
        plausible_temporal = False
        if "year" in tokens and integer_like and cardinality >= 2:
            plausible_temporal = bool(numeric.between(1000, 2999).all())
        elif "month" in tokens and integer_like and cardinality >= 2:
            plausible_temporal = bool(numeric.between(1, 12).all())
        elif "quarter" in tokens and integer_like and cardinality >= 2:
            plausible_temporal = bool(numeric.between(1, 4).all())
        elif (
            not tokens & _ID_NAME_TOKENS
            and integer_like
            and cardinality >= 2
            and numeric.between(1500, 2200).all()
        ):
            # A bounded all-year value domain is temporal evidence even when
            # legacy schemas use names such as Creation, Season, or Launch.
            plausible_temporal = True
        if plausible_temporal:
            temporal_summary = TemporalSummary(
                parse_ratio=1.0,
                minimum=str(_safe_scalar(numeric.min())),
                maximum=str(_safe_scalar(numeric.max())),
                timezone_observed=False,
            )
            return "temporal", 0.92, (), temporal_summary
        if (
            tokens & _ID_NAME_TOKENS
            and uniqueness_ratio >= 0.95
            and len(non_null) >= 5
        ):
            return "identifier", 0.95, (), None
        if tokens & _TEMPORAL_NAME_TOKENS:
            warnings.append(
                "temporal name hint conflicts with numeric values; semantic type remains quantitative"
            )
        return "quantitative", 0.98, tuple(warnings), None

    date_ratio, parsed = _date_parse(series)
    if date_ratio >= 0.8 and len(parsed) >= 2:
        temporal_summary = TemporalSummary(
            parse_ratio=date_ratio,
            minimum=parsed.min().isoformat(),
            maximum=parsed.max().isoformat(),
            timezone_observed=False,
        )
        return "temporal", min(0.98, date_ratio), (), temporal_summary

    values = {str(value).strip().casefold() for value in non_null.unique()}
    if tokens & _GEO_NAME_TOKENS and len(values & _KNOWN_GEO_VALUES) >= 2:
        return "geographic", 0.9, (), None
    if tokens & _GEO_NAME_TOKENS:
        warnings.append(
            "geographic name hint lacks sufficient value evidence; retained as categorical"
        )

    if (
        (tokens & _ORDINAL_NAME_TOKENS and len(values & _KNOWN_ORDINAL_VALUES) >= 2)
        or len(values & _KNOWN_ORDINAL_VALUES) >= 3
    ):
        return "ordinal", 0.88, tuple(warnings), None

    if (
        tokens & _ID_NAME_TOKENS
        and uniqueness_ratio >= 0.95
        and len(non_null) >= 5
    ):
        return "identifier", 0.92, tuple(warnings), None
    if tokens & _ID_NAME_TOKENS and uniqueness_ratio < 0.95:
        warnings.append(
            "identifier name hint lacks uniqueness evidence; retained as categorical"
        )
    return "categorical", 0.9, tuple(warnings), None


def _role_candidates(semantic_type: str, cardinality: int, row_count: int) -> tuple[str, ...]:
    if semantic_type == "quantitative":
        roles = ["measure"]
        if row_count and cardinality <= min(20, max(2, row_count // 10)):
            roles.append("dimension")
        return tuple(roles)
    if semantic_type in {"categorical", "ordinal"}:
        return ("dimension", "grouping", "filter")
    if semantic_type == "temporal":
        return ("temporal", "dimension", "grouping", "filter")
    if semantic_type == "identifier":
        return ("identifier", "dimension", "filter")
    if semantic_type == "geographic":
        return ("geographic", "dimension", "grouping", "filter")
    return ("unknown",)


def run_schema_profiling(
    dataset_id: str,
    data: pd.DataFrame | Mapping[str, Sequence[Any]],
    aliases: Mapping[str, Sequence[str]] | None,
    config: SkillVISV2Config,
    registry: RuleRegistry,
) -> SchemaProfile:
    if not config.feature_flags.enable_m1_schema_profiling:
        raise RuntimeError("M1 schema profiling feature flag is disabled")
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    aliases = aliases or {}
    warnings: list[str] = []
    if len(df) > config.max_profile_rows:
        df = df.iloc[: config.max_profile_rows].copy()
        warnings.append(
            f"profile limited to first {config.max_profile_rows} rows by config"
        )

    fields: list[SchemaFieldProfile] = []
    for name in df.columns:
        field_name = str(name)
        series = df[name]
        non_null = series.dropna()
        cardinality = int(non_null.nunique(dropna=True))
        semantic, confidence, field_warnings, temporal_summary = _semantic_type(
            field_name, series
        )
        source_rules = {"R-SP-001", "R-SP-003"}
        if temporal_summary is not None or _name_tokens(field_name) & _TEMPORAL_NAME_TOKENS:
            source_rules.add("R-SP-004")
        if semantic in {"identifier", "ordinal", "geographic"} or field_warnings:
            source_rules.add("R-SP-005")
        if (
            config.feature_flags.enable_value_matching
            and cardinality <= config.max_value_cardinality
        ):
            source_rules.add("R-SP-002")

        numeric_summary = (
            _numeric_summary(series)
            if pd.api.types.is_numeric_dtype(series.dtype)
            else None
        )
        categorical_summary = (
            _categorical_summary(series)
            if semantic in {"categorical", "ordinal", "geographic", "identifier"}
            else None
        )
        normalized_aliases = tuple(
            sorted(
                {
                    normalize_field_phrase(alias)
                    for alias in aliases.get(field_name, ())
                    if normalize_field_phrase(alias)
                }
            )
        )
        fields.append(
            SchemaFieldProfile(
                field_name=field_name,
                normalized_name=normalize_field_phrase(field_name),
                aliases=normalized_aliases,
                semantic_type=semantic,
                physical_type=str(series.dtype),
                cardinality=cardinality,
                null_rate=float(series.isna().mean()) if len(series) else 0.0,
                sample_values=_stable_unique(series, config.max_sample_values),
                numeric_summary=numeric_summary,
                temporal_summary=temporal_summary,
                categorical_summary=categorical_summary,
                role_candidates=_role_candidates(semantic, cardinality, len(df)),
                confidence=confidence,
                source_rule_ids=registry.validate_ids(source_rules),
                warnings=field_warnings,
            )
        )

    applied = registry.validate_ids(
        rule_id for field in fields for rule_id in field.source_rule_ids
    )
    failure_state = None
    if not len(df.columns):
        warnings.append("schema contains no fields")
        failure_state = "empty_schema"
    return SchemaProfile(
        dataset_id=dataset_id,
        row_count=int(len(df)),
        fields=tuple(fields),
        data_access_policy=(
            "bounded_profile_values_enabled"
            if config.feature_flags.enable_value_matching
            else "schema_only_value_matching_disabled"
        ),
        applied_rule_ids=applied,
        warnings=tuple(warnings),
        failure_state=failure_state,
    )
