"""Resolve query-role evidence against weak temporal name hints."""

from __future__ import annotations

from skillvis_v2_accuracy.contracts import AnalyticalTaskResult, SchemaProfile
from skillvis_v2_accuracy.rules import RuleRegistry


_MEASURE_AGGREGATES = {"mean", "sum", "median", "count", "distinct_count"}


def restore_aggregate_target_numeric_measures(
    *,
    schema: SchemaProfile,
    task: AnalyticalTaskResult,
    registry: RuleRegistry,
) -> tuple[SchemaProfile, tuple[str, ...]]:
    """Let typed aggregate-role evidence dominate a lexical temporal-name hint."""

    aggregation = task.aggregation
    if aggregation is None or aggregation.operation not in _MEASURE_AGGREGATES:
        return schema, ()
    targets = set(aggregation.field_candidates)
    restored = []
    rows = []
    for profile in schema.fields:
        should_restore = (
            profile.field_name in targets
            and profile.semantic_type == "temporal"
            and profile.numeric_summary is not None
            and profile.temporal_summary is None
            and profile.physical_type not in {"datetime64", "datetime64[ns]"}
        )
        if not should_restore:
            rows.append(profile)
            continue
        restored.append(profile.field_name)
        rows.append(
            profile.model_copy(
                update={
                    "semantic_type": "quantitative",
                    "role_candidates": ("measure",),
                    "source_rule_ids": registry.validate_ids(
                        (*profile.source_rule_ids, "R-SP-041")
                    ),
                    "warnings": tuple(
                        sorted(
                            {
                                *profile.warnings,
                                "aggregate-target numeric role overrides lexical temporal name hint",
                            }
                        )
                    ),
                }
            )
        )
    if not restored:
        return schema, ()
    updated = SchemaProfile.model_validate(
        schema.model_copy(
            update={
                "fields": tuple(rows),
                "applied_rule_ids": registry.validate_ids(
                    (*schema.applied_rule_ids, "R-SP-041")
                ),
            }
        ).model_dump(mode="json")
    )
    traces = tuple(
        f"R-SP-041 aggregate-target numeric measure restored:{name}:operation={aggregation.operation}"
        for name in sorted(restored)
    )
    return updated, traces


__all__ = ["restore_aggregate_target_numeric_measures"]
