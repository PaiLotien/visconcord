"""Post-enumeration common-plan semantic deduplication for M4-v3."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from skillvis_v2_accuracy import candidate_planner as base
from skillvis_v2_accuracy.contracts import (
    AnalyticalTaskResult,
    CandidatePlanningResult,
    CapabilityAssessment,
    FieldResolutionResult,
    SchemaProfile,
    VisualizationCandidate,
    VisualizationIntentIR,
)
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_obligation.config import SkillVISV2Config, load_m4_config
from skillvis_v2_obligation_v2 import candidate_planner as parent


_CHART_FAMILY = {
    "bar": "bar",
    "grouped_bar": "bar",
    "stacked_bar": "bar",
    "normalized_stacked_bar": "bar",
    "horizontal_bar": "bar",
    "top_k_bar": "bar",
    "ranked_bar": "bar",
    "highlighted_extremum": "bar",
    "line": "line",
    "multi_line": "line",
    "scatter": "scatter",
    "grouped_scatter": "scatter",
    "histogram": "histogram",
    "box_plot": "box_plot",
    "pie": "pie",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _channel_group(channel: str) -> str:
    if channel in {"x", "y"}:
        return "position"
    if channel in {"row", "column", "facet"}:
        return "facet"
    return channel


def _field_set(candidate: VisualizationCandidate) -> tuple[str, ...]:
    payload = candidate.fields.model_dump(mode="python")
    return tuple(sorted({field for values in payload.values() for field in values}))


def common_plan_signature(candidate: VisualizationCandidate) -> str:
    """Cross-system semantic signature, intentionally coarser than native identity."""

    chart = candidate.chart_type.value
    payload = {
        "chart_family": _CHART_FAMILY.get(chart, chart),
        "fields": _field_set(candidate),
        "encodings": sorted(
            [
                (
                _channel_group(item.channel),
                item.field,
                item.aggregate,
                bool(item.bin),
                item.time_unit,
                )
                for item in candidate.encodings
            ],
            key=_canonical,
        ),
        "transforms": sorted(
            (
                item.transform_type,
                item.operation,
                tuple(sorted(item.field_candidates)),
                _canonical(item.value),
                item.negated,
            )
            for item in candidate.transforms
        ),
        "filters": sorted(
            (
                item.transform_type,
                item.operation,
                tuple(sorted(item.field_candidates)),
                _canonical(item.value),
                item.negated,
            )
            for item in candidate.filters
        ),
        "aggregation": (
            None
            if candidate.aggregation is None
            else (
                candidate.aggregation.operation,
                tuple(sorted(candidate.aggregation.field_candidates)),
            )
        ),
        "sort": (
            None
            if candidate.sort is None
            else (
                candidate.sort.operation,
                tuple(sorted(candidate.sort.field_candidates)),
            )
        ),
        "limit": candidate.limit,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _deduplicate_common(
    candidates: tuple[VisualizationCandidate, ...],
) -> tuple[tuple[VisualizationCandidate, ...], dict[str, tuple[str, ...]]]:
    groups: dict[str, list[VisualizationCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(common_plan_signature(candidate), []).append(candidate)
    retained = []
    removed: dict[str, tuple[str, ...]] = {}
    for signature, rows in sorted(groups.items()):
        rows.sort(key=lambda item: (-item.total_score, item.deterministic_signature))
        retained.append(rows[0])
        if len(rows) > 1:
            removed[signature] = tuple(item.deterministic_signature for item in rows[1:])
    retained.sort(key=lambda item: (item.rank, -item.total_score, item.deterministic_signature))
    return tuple(retained), removed


def plan_visualization_candidates(
    *,
    intent: VisualizationIntentIR,
    capability: CapabilityAssessment,
    schema: SchemaProfile,
    field_resolution: FieldResolutionResult,
    task_result: AnalyticalTaskResult,
    config: SkillVISV2Config | None = None,
    registry: RuleRegistry | None = None,
) -> CandidatePlanningResult:
    config = config or load_m4_config()
    if registry is None:
        raise ValueError("M4-v3 requires the Joint v1.6 registry")
    result = parent.plan_visualization_candidates(
        intent=intent,
        capability=capability,
        schema=schema,
        field_resolution=field_resolution,
        task_result=task_result,
        config=config,
        registry=registry,
    )
    if not result.candidates:
        return result
    retained, removed = _deduplicate_common(result.candidates)
    ranked = tuple(
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(retained, start=1)
    )
    rule_ids = {
        "R-CP-020",
        *(rule_id for candidate in ranked for rule_id in candidate.applied_rule_ids),
    }
    provenance = base._planner_provenance(registry, rule_ids, intent)
    result_payload = {
        "parent_signature": result.deterministic_signature,
        "common_signatures": [common_plan_signature(item) for item in ranked],
        "candidate_signatures": [item.deterministic_signature for item in ranked],
        "removed": removed,
        "ranks": [item.rank for item in ranked],
    }
    dedup_warnings = (
        (
            "R-CP-020 common semantic duplicates removed:"
            f"{sum(map(len, removed.values()))}"
        ),
    ) if removed else ()
    return CandidatePlanningResult(
        intent_id=result.intent_id,
        config_version=result.config_version,
        registry_hash=registry.sha256,
        candidates=ranked,
        raw_candidate_count=result.raw_candidate_count,
        post_dedup_count=len(ranked),
        post_ranking_count=len(ranked),
        duplicate_count=result.raw_candidate_count - len(ranked),
        diversity=base._diversity(ranked),
        abstained=result.abstained,
        abstention_reason=result.abstention_reason,
        warnings=tuple(
            dict.fromkeys(
                (
                    *result.warnings,
                    *dedup_warnings,
                )
            )
        ),
        rejection_reasons=result.rejection_reasons,
        provenance=provenance,
        deterministic_signature=base._canonical_hash(result_payload),
    )


__all__ = ["common_plan_signature", "plan_visualization_candidates"]
