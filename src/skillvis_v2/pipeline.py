"""Explicit deterministic M1→M2→M3 entry point."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .config import SkillVISV2Config, load_config
from .contracts import (
    AbstentionState,
    EvidenceSpan,
    IntentFieldRef,
    M1M3PipelineResult,
    ProvenanceRecord,
    VisualizationIntentIR,
)
from .field_resolution import run_field_resolution
from .query_normalization import run_query_normalization
from .rules import RuleRegistry
from .schema_profiling import run_schema_profiling
from .semantic_safety import run_semantic_safety
from .task_profiling import run_task_profiling


def _stable_unique(items, key):
    unique = {key(item): item for item in items}
    return tuple(unique[item_key] for item_key in sorted(unique))


def _build_field_refs(pipeline_fields, schema, task):
    profile_by_name = {field.field_name: field for field in schema.fields}
    required = set(pipeline_fields.required_field_candidates)
    measures = []
    dimensions = []
    temporal = []
    grouping = []
    grouping_tasks = {"comparison", "ranking", "composition", "extremum"}
    for candidate in pipeline_fields.candidate_fields:
        if candidate.field_name not in required:
            continue
        profile = profile_by_name[candidate.field_name]
        base = {
            "field_name": candidate.field_name,
            "score": candidate.score_components.total_score,
            "candidate_rank": candidate.candidate_rank,
            "evidence": candidate.evidence,
            "rule_ids": candidate.source_rule_ids,
        }
        if profile.semantic_type == "quantitative":
            measures.append(IntentFieldRef(role="measure", **base))
        elif profile.semantic_type == "temporal":
            temporal.append(IntentFieldRef(role="temporal", **base))
            dimensions.append(IntentFieldRef(role="dimension", **base))
        elif profile.semantic_type == "identifier":
            dimensions.append(IntentFieldRef(role="identifier", **base))
        elif profile.semantic_type == "geographic":
            dimensions.append(IntentFieldRef(role="geographic", **base))
        elif profile.semantic_type in {"categorical", "ordinal"}:
            dimensions.append(IntentFieldRef(role="dimension", **base))

        # Grouping can be structural for ranking/comparison over a dimension.
        should_group = (
            profile.semantic_type in {
                "categorical", "ordinal", "geographic", "temporal", "identifier"
            }
            and task.primary_task in grouping_tasks
        )
        if should_group:
            grouping.append(IntentFieldRef(role="grouping", **base))
    key = lambda item: (item.candidate_rank, item.field_name, item.role)
    return (
        tuple(sorted(measures, key=key)),
        tuple(sorted(dimensions, key=key)),
        tuple(sorted(temporal, key=key)),
        tuple(sorted(grouping, key=key)),
    )


def _grouping_from_query(
    query: str,
    dimensions: tuple[IntentFieldRef, ...],
) -> tuple[IntentFieldRef, ...]:
    grouping = []
    for ref in dimensions:
        for evidence in ref.evidence:
            span = evidence.query_span
            if span is None:
                continue
            prefix = query[max(0, span.start - 14) : span.start].casefold()
            if re.search(r"(?:\bby|\bacross|\bper|\bfor each)\s+$", prefix):
                grouping.append(ref.model_copy(update={"role": "grouping"}))
                break
    return tuple(
        sorted(
            {item.field_name: item for item in grouping}.values(),
            key=lambda item: (item.candidate_rank, item.field_name),
        )
    )


def _stage_for_rule(rule_id: str) -> str:
    if rule_id.startswith("R-QN-"):
        return "query_normalization"
    if rule_id.startswith("R-SP-"):
        return "schema_profiling"
    if rule_id.startswith("R-FR-"):
        return "field_resolution"
    if rule_id.startswith("R-TP-"):
        return "task_profiling"
    if rule_id.startswith("R-QD-"):
        return "query_decomposition"
    if rule_id.startswith("R-SA-"):
        return "semantic_safety"
    return "intent_assembly"


def _provenance(
    registry: RuleRegistry,
    rule_ids: set[str],
    dataset_id: str,
) -> tuple[ProvenanceRecord, ...]:
    records = []
    for rule_id in sorted(rule_ids):
        stage = _stage_for_rule(rule_id)
        records.append(
            registry.trace(
                stage,
                rule_id,
                inputs=("query", f"dataset:{dataset_id}"),
                outputs=(stage,),
            )
        )
    return tuple(records)


def _abstention_state(normalization, schema, fields, task) -> AbstentionState:
    if normalization.failure_state:
        return AbstentionState(
            abstained=True,
            stage="query_normalization",
            reason=normalization.failure_state,
        )
    if schema.failure_state:
        return AbstentionState(
            abstained=True,
            stage="schema_profiling",
            reason=schema.failure_state,
        )
    if fields.abstention_reason:
        return AbstentionState(
            abstained=True,
            stage="field_resolution",
            reason=fields.abstention_reason,
        )
    if task.primary_task == "unresolved":
        return AbstentionState(
            abstained=True,
            stage="task_profiling",
            reason=task.failure_state,
        )
    return AbstentionState(abstained=False)


def run_m1_m3_pipeline(
    *,
    query: str,
    dataset_id: str,
    data: pd.DataFrame | Mapping[str, Sequence[object]],
    aliases: Mapping[str, Sequence[str]] | None = None,
    config: SkillVISV2Config | None = None,
) -> M1M3PipelineResult:
    config = config or load_config()
    if not config.enabled:
        raise RuntimeError("SkillVIS v2 requires an explicitly enabled config")
    registry = RuleRegistry.from_config(config)
    normalization = run_query_normalization(query, registry)
    schema = run_schema_profiling(dataset_id, data, aliases, config, registry)
    fields = run_field_resolution(normalization, schema, config, registry)
    task = run_task_profiling(normalization, schema, fields, config, registry)

    measures, dimensions, temporal, structural_grouping = _build_field_refs(
        fields, schema, task
    )
    explicit_grouping = _grouping_from_query(query, dimensions)
    grouping = _stable_unique(
        (*structural_grouping, *explicit_grouping),
        key=lambda item: (item.field_name, item.role),
    )

    all_rule_ids = set(normalization.applied_rule_ids)
    all_rule_ids.update(schema.applied_rule_ids)
    all_rule_ids.update(fields.applied_rule_ids)
    all_rule_ids.update(task.applied_rule_ids)
    all_rule_ids.update({"R-MR-001", "R-MR-002"})
    registry.validate_ids(all_rule_ids)

    evidence_spans = _stable_unique(
        (
            *normalization.query_spans,
            *task.query_evidence_spans,
            *(
                evidence.query_span
                for candidate in fields.candidate_fields
                for evidence in candidate.evidence
                if evidence.query_span is not None
            ),
        ),
        key=lambda span: (
            span.start,
            span.end,
            span.kind,
            span.normalized,
            span.rule_ids,
        ),
    )
    ambiguity = _stable_unique(
        (*fields.ambiguity, *task.ambiguity),
        key=lambda item: (
            item.dimension,
            item.alternatives,
            item.reason,
            item.rule_ids,
        ),
    )
    warnings = tuple(
        sorted(
            {
                *normalization.warnings,
                *schema.warnings,
                *(
                    warning
                    for field_profile in schema.fields
                    for warning in field_profile.warnings
                ),
            }
        )
    )
    provenance = _provenance(registry, all_rule_ids, dataset_id)
    abstention = _abstention_state(normalization, schema, fields, task)

    payload = {
        "config_version": config.config_version,
        "registry_hash": registry.sha256,
        "deterministic_id": "",
        "query": query,
        "normalized_query": normalization.normalized_query,
        "dataset_id": dataset_id,
        "measures": measures,
        "dimensions": dimensions,
        "temporal_fields": temporal,
        "grouping_fields": grouping,
        "task_candidates": task.task_candidates,
        "primary_task": task.primary_task,
        "aggregation": task.aggregation,
        "filters": task.filters,
        "sort": task.sort,
        "limit": task.limit,
        "chart_mentions": normalization.chart_mentions,
        "field_candidates": fields.candidate_fields,
        "unresolved_mentions": fields.unresolved_mentions,
        "ambiguity": ambiguity,
        "abstention_reason": abstention.reason,
        "evidence_spans": evidence_spans,
        "provenance": provenance,
        "warnings": warnings,
        "abstention_state": abstention,
    }
    draft = VisualizationIntentIR.model_validate(payload)
    if config.feature_flags.enable_m3_5_semantic_safety:
        safety = run_semantic_safety(draft, registry)
        all_rule_ids.update(safety.applied_rule_ids)
        registry.validate_ids(all_rule_ids)
        provenance = _provenance(registry, all_rule_ids, dataset_id)
        draft = draft.model_copy(
            update={
                "ambiguity_state": safety.ambiguity_state,
                "unsupported_scope": safety.unsupported_scope,
                "capability_assessment": safety.capability_assessment,
                "safety_warnings": safety.safety_warnings,
                "abstention_reason": safety.abstention_reason,
                "abstention_state": safety.abstention_state,
                "provenance": provenance,
            }
        )
    identity_payload = draft.model_dump(mode="json")
    identity_payload.pop("deterministic_id", None)
    deterministic_id = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    ir = draft.model_copy(update={"deterministic_id": deterministic_id})
    # Round-trip validation is an implementation invariant, not a later validator.
    ir = VisualizationIntentIR.validate_serialized(ir.model_dump(mode="json"))
    return M1M3PipelineResult(
        normalization=normalization,
        schema_profile=schema,
        field_resolution=fields,
        task_profile=task,
        intent_ir=ir,
    )


def run_m1_m3_5_pipeline(
    *,
    query: str,
    dataset_id: str,
    data: pd.DataFrame | Mapping[str, Sequence[object]],
    aliases: Mapping[str, Sequence[str]] | None = None,
    config: SkillVISV2Config | None = None,
) -> M1M3PipelineResult:
    """Run M1–M3 plus the M3.5 safety gate; downstream modules remain absent."""

    if config is None:
        from .config import load_m3_5_config

        config = load_m3_5_config()
    if not config.feature_flags.enable_m3_5_semantic_safety:
        raise RuntimeError("run_m1_m3_5_pipeline requires M3.5 safety enabled")
    return run_m1_m3_pipeline(
        query=query,
        dataset_id=dataset_id,
        data=data,
        aliases=aliases,
        config=config,
    )


def build_visualization_intent_ir(
    *,
    query: str,
    dataset_id: str,
    data: pd.DataFrame | Mapping[str, Sequence[object]],
    aliases: Mapping[str, Sequence[str]] | None = None,
    config_path: str | Path | None = None,
) -> VisualizationIntentIR:
    """Explicit v2 entry point. SkillVIS v1 remains the default elsewhere."""

    config = load_config(config_path)
    return run_m1_m3_pipeline(
        query=query,
        dataset_id=dataset_id,
        data=data,
        aliases=aliases,
        config=config,
    ).intent_ir


def build_safe_visualization_intent_ir(
    *,
    query: str,
    dataset_id: str,
    data: pd.DataFrame | Mapping[str, Sequence[object]],
    aliases: Mapping[str, Sequence[str]] | None = None,
    config_path: str | Path | None = None,
) -> VisualizationIntentIR:
    """Explicit M3.5 entry point; no candidate-planning output is produced."""

    from .config import load_m3_5_config

    config = load_config(config_path) if config_path else load_m3_5_config()
    return run_m1_m3_5_pipeline(
        query=query,
        dataset_id=dataset_id,
        data=data,
        aliases=aliases,
        config=config,
    ).intent_ir
