"""Bounded, value-shape temporal profiling without source-value mutation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd

from skillvis_v2_accuracy.contracts import (
    FieldEvidence,
    FieldResolutionResult,
    SchemaProfile,
    TemporalSummary,
)
from skillvis_v2_accuracy.rules import RuleRegistry


_MAX_TEMPORAL_SAMPLE = 512
_COMPACT_ISO = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?P<time>\d{1,2}:\d{2}(?::\d{2})?)$"
)
_FORMATS: tuple[tuple[str, str], ...] = (
    ("iso_datetime_t", "%Y-%m-%dT%H:%M:%S"),
    ("iso_datetime_space", "%Y-%m-%d %H:%M:%S"),
    ("iso_datetime_minute_t", "%Y-%m-%dT%H:%M"),
    ("iso_datetime_minute_space", "%Y-%m-%d %H:%M"),
    ("iso_date", "%Y-%m-%d"),
    ("year_first_slash", "%Y/%m/%d"),
    ("day_first_slash", "%d/%m/%Y"),
    ("month_first_slash", "%m/%d/%Y"),
    ("day_first_dash", "%d-%m-%Y"),
    ("month_first_dash", "%m-%d-%Y"),
    ("month_name", "%b %d, %Y"),
    ("month_name_long", "%B %d, %Y"),
)


def _parse_temporal_value(value: object) -> tuple[datetime | None, str | None]:
    text = str(value).strip()
    compact = _COMPACT_ISO.fullmatch(text)
    if compact:
        normalized = f"{compact.group('date')}T{compact.group('time')}"
        try:
            return datetime.strptime(normalized, "%Y-%m-%dT%H:%M:%S"), "compact_iso"
        except ValueError:
            return None, None
    for label, fmt in _FORMATS:
        try:
            return datetime.strptime(text, fmt), label
        except ValueError:
            continue
    return None, None


def promote_temporal_schema_fields(
    *,
    schema: SchemaProfile,
    data: pd.DataFrame | Mapping[str, Sequence[object]],
    registry: RuleRegistry,
) -> tuple[SchemaProfile, tuple[str, ...], tuple[str, ...]]:
    """Promote only string-like fields with independently sufficient values."""

    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    promoted: list[str] = []
    traces: list[str] = []
    rows = []
    for profile in schema.fields:
        if profile.semantic_type == "temporal" or profile.field_name not in df.columns:
            rows.append(profile)
            continue
        series = df[profile.field_name]
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            rows.append(profile)
            continue
        values = tuple(series.dropna().astype(str).str.strip().iloc[:_MAX_TEMPORAL_SAMPLE])
        if not values:
            rows.append(profile)
            continue
        parsed: list[datetime] = []
        morphologies: set[str] = set()
        for value in values:
            item, morphology = _parse_temporal_value(value)
            if item is not None and morphology is not None:
                parsed.append(item)
                morphologies.add(morphology)
        ratio = len(parsed) / len(values)
        if ratio < 0.8 or len(set(parsed)) < 2:
            rows.append(profile)
            continue
        promoted.append(profile.field_name)
        source_rule_ids = registry.validate_ids((*profile.source_rule_ids, "R-SP-040"))
        rows.append(
            profile.model_copy(
                update={
                    "semantic_type": "temporal",
                    "temporal_summary": TemporalSummary(
                        parse_ratio=round(ratio, 6),
                        minimum=min(parsed).isoformat(),
                        maximum=max(parsed).isoformat(),
                        timezone_observed=False,
                    ),
                    "categorical_summary": None,
                    "role_candidates": ("temporal", "dimension", "grouping", "filter"),
                    "confidence": min(0.99, max(profile.confidence, ratio)),
                    "source_rule_ids": source_rule_ids,
                    "warnings": tuple(
                        sorted(
                            {
                                *profile.warnings,
                                "temporal semantic type promoted from bounded value morphology; source values unchanged",
                            }
                        )
                    ),
                }
            )
        )
        traces.append(
            "R-SP-040 temporal promotion: "
            f"{profile.field_name} sample={len(values)} parse_ratio={ratio:.6f} "
            f"morphologies={','.join(sorted(morphologies))}"
        )
    if not promoted:
        return schema, (), ()
    updated = SchemaProfile.model_validate(
        schema.model_copy(
            update={
                "fields": tuple(rows),
                "applied_rule_ids": registry.validate_ids(
                    (*schema.applied_rule_ids, "R-SP-040")
                ),
            }
        ).model_dump(mode="json")
    )
    return updated, tuple(sorted(promoted)), tuple(sorted(traces))


def align_temporal_field_roles(
    *,
    fields: FieldResolutionResult,
    promoted_fields: tuple[str, ...],
    registry: RuleRegistry,
) -> FieldResolutionResult:
    """Align existing candidates with promoted schema roles without adding fields."""

    if not promoted_fields:
        return fields
    promoted = set(promoted_fields)
    candidates = []
    evidence_map = dict(fields.evidence)
    intended_map = dict(fields.intended_roles)
    match_map = dict(fields.match_types)
    for candidate in fields.candidate_fields:
        if candidate.field_name not in promoted:
            candidates.append(candidate)
            continue
        evidence = tuple(
            sorted(
                (
                    *candidate.evidence,
                    FieldEvidence(
                        evidence_type="schema_temporal_profile",
                        query_span=None,
                        value="temporal:value_morphology",
                        rule_id="R-SP-040",
                        source_ids=registry.get("R-SP-040").source_ids,
                    ),
                ),
                key=lambda item: (
                    item.query_span.start if item.query_span else -1,
                    item.evidence_type,
                    item.value,
                ),
            )
        )
        roles = tuple(
            sorted(
                {
                    *(role for role in candidate.intended_roles if role != "measure"),
                    "temporal",
                    "dimension",
                }
            )
        )
        match_types = tuple(sorted({*candidate.match_types, "schema_temporal_profile"}))
        updated = candidate.model_copy(
            update={
                "evidence": evidence,
                "intended_roles": roles,
                "match_types": match_types,
                "source_rule_ids": registry.validate_ids(
                    (*candidate.source_rule_ids, "R-SP-040")
                ),
            }
        )
        candidates.append(updated)
        evidence_map[candidate.field_name] = evidence
        intended_map[candidate.field_name] = roles
        match_map[candidate.field_name] = match_types
    return FieldResolutionResult.model_validate(
        fields.model_copy(
            update={
                "candidate_fields": tuple(candidates),
                "evidence": evidence_map,
                "intended_roles": intended_map,
                "match_types": match_map,
                "applied_rule_ids": registry.validate_ids(
                    (*fields.applied_rule_ids, "R-SP-040")
                ),
            }
        ).model_dump(mode="json")
    )


__all__ = [
    "align_temporal_field_roles",
    "promote_temporal_schema_fields",
]
