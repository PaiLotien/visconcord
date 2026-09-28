"""Isolated v1.1 M1 -> joint M2/M3 -> M3.5 pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from skillvis_v2_accuracy.config import SkillVISV2Config
from skillvis_v2_accuracy.contracts import M1M3PipelineResult, VisualizationIntentIR
from skillvis_v2_accuracy.field_resolution import run_field_resolution
from skillvis_v2_accuracy.pipeline import (
    _abstention_state,
    _build_field_refs,
    _grouping_from_query,
    _provenance,
    _stable_unique,
)
from skillvis_v2_accuracy.query_normalization import run_query_normalization
from skillvis_v2_accuracy.schema_profiling import run_schema_profiling
from skillvis_v2_accuracy.task_profiling import run_task_profiling
from skillvis_v2_joint.config import JointSemanticSettings, load_config, load_settings
from skillvis_v2_joint.contracts import JointPipelineResult, JointSemanticResolution
from skillvis_v2_joint.registry import build_joint_registry
from skillvis_v2_joint.task_resolution import run_joint_task_resolution
from skillvis_v2_obligation.candidate_planner import plan_visualization_candidates
from skillvis_v2_obligation.intent_compiler import compile_intent_obligations
from skillvis_v2_obligation.semantic_safety import run_semantic_safety

from .field_resolution import run_joint_v2_field_resolution
from .task_resolution import refine_schema_roles, remove_operation_field_collisions


CONFIG_VERSION = "skillvis-v2-joint-field-task-v1.1.0"


def _identity(intent: VisualizationIntentIR) -> VisualizationIntentIR:
    payload = intent.model_dump(mode="json")
    payload.pop("deterministic_id", None)
    deterministic_id = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return VisualizationIntentIR.validate_serialized(
        intent.model_copy(update={"deterministic_id": deterministic_id}).model_dump(mode="json")
    )


def run_joint_v2_m1_m3_pipeline(
    *,
    query: str,
    dataset_id: str,
    data: pd.DataFrame | Mapping[str, Sequence[object]],
    aliases: Mapping[str, Sequence[str]] | None = None,
    config: SkillVISV2Config | None = None,
    settings: JointSemanticSettings | None = None,
) -> JointPipelineResult:
    settings = settings or load_settings()
    config = (config or load_config(settings)).model_copy(
        update={"config_version": CONFIG_VERSION}
    )
    registry = build_joint_registry(settings)
    normalization = run_query_normalization(query, registry)
    original_schema = run_schema_profiling(dataset_id, data, aliases, config, registry)
    base_fields = run_field_resolution(normalization, original_schema, config, registry)
    field_artifacts = run_joint_v2_field_resolution(
        normalized=normalization,
        schema=original_schema,
        base=base_fields,
        config=config,
        settings=settings,
        registry=registry,
    )
    fields = field_artifacts.result
    schema, role_refinements = refine_schema_roles(normalization, original_schema, fields)
    base_task = run_task_profiling(normalization, schema, fields, config, registry)
    base_task, operation_collisions = remove_operation_field_collisions(base_task, fields)
    (
        task,
        hypotheses,
        selected_hypothesis_id,
        graph_nodes,
        graph_edges,
    ) = run_joint_task_resolution(
        normalized=normalization,
        schema=schema,
        fields=fields,
        base=base_task,
        semantic_mentions=field_artifacts.semantic_mentions,
        settings=settings,
        registry=registry,
    )

    measures, dimensions, temporal, structural_grouping = _build_field_refs(fields, schema, task)
    explicit_grouping = _grouping_from_query(query, dimensions)
    grouping = _stable_unique(
        (*structural_grouping, *explicit_grouping),
        key=lambda item: (item.field_name, item.role),
    )
    all_rule_ids = {
        *normalization.applied_rule_ids,
        *schema.applied_rule_ids,
        *fields.applied_rule_ids,
        *task.applied_rule_ids,
        "R-MR-001",
        "R-MR-002",
        "R-MR-020",
        "R-MR-021",
    }
    registry.validate_ids(all_rule_ids)
    evidence_spans = _stable_unique(
        (
            *normalization.query_spans,
            *task.query_evidence_spans,
            *(item.query_span for item in field_artifacts.semantic_mentions),
            *(
                evidence.query_span
                for candidate in fields.candidate_fields
                for evidence in candidate.evidence
                if evidence.query_span is not None
            ),
        ),
        key=lambda span: (span.start, span.end, span.kind, span.normalized, span.rule_ids),
    )
    ambiguity = _stable_unique(
        (*fields.ambiguity, *task.ambiguity),
        key=lambda item: (item.dimension, item.alternatives, item.reason, item.rule_ids),
    )
    warnings = tuple(
        sorted(
            {
                *normalization.warnings,
                *schema.warnings,
                *field_artifacts.warnings,
                *(f"query_role_refinement:{name}" for name in role_refinements),
                *(f"operation_field_collision_removed:{name}" for name in operation_collisions),
                "joint v1.1 development revision; external protocol lock required",
            }
        )
    )
    provenance = _provenance(registry, all_rule_ids, dataset_id)
    abstention = _abstention_state(normalization, schema, fields, task)
    draft = VisualizationIntentIR.model_validate(
        {
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
            "transforms": task.transforms,
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
    )
    compiled = compile_intent_obligations(draft, task, registry)
    safety = run_semantic_safety(compiled.intent, registry)
    all_rule_ids.update(compiled.applied_rule_ids)
    all_rule_ids.update(safety.applied_rule_ids)
    provenance = _provenance(registry, all_rule_ids, dataset_id)
    completed = compiled.intent.model_copy(
        update={
            "ambiguity_state": safety.ambiguity_state,
            "unsupported_scope": safety.unsupported_scope,
            "capability_assessment": safety.capability_assessment,
            "safety_warnings": safety.safety_warnings,
            "abstention_state": safety.abstention_state,
            "abstention_reason": safety.abstention_reason,
            "provenance": provenance,
        }
    )
    completed = _identity(completed)
    pipeline = M1M3PipelineResult(
        normalization=normalization,
        schema_profile=schema,
        field_resolution=fields,
        task_profile=compiled.task_result,
        intent_ir=completed,
    )
    joint = JointSemanticResolution(
        schema_profiles=field_artifacts.schema_profiles,
        semantic_mentions=field_artifacts.semantic_mentions,
        base_field_resolution=base_fields,
        hypotheses=hypotheses,
        selected_hypothesis_id=selected_hypothesis_id,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        applied_rule_ids=registry.validate_ids(
            {
                "R-SD-001",
                "R-FR-020",
                "R-FR-021",
                "R-FR-022",
                "R-TP-020",
                "R-TP-021",
                "R-TP-022",
                "R-TP-023",
                "R-MR-020",
                "R-MR-021",
            }
        ),
        warnings=warnings,
    )
    return JointPipelineResult(pipeline=pipeline, joint_semantics=joint)


def plan_joint_v2_candidates(result: JointPipelineResult):
    settings = load_settings()
    config = load_config(settings).model_copy(update={"config_version": CONFIG_VERSION})
    registry = build_joint_registry(settings)
    pipeline = result.pipeline
    return plan_visualization_candidates(
        intent=pipeline.intent_ir,
        capability=pipeline.intent_ir.capability_assessment,
        schema=pipeline.schema_profile,
        field_resolution=pipeline.field_resolution,
        task_result=pipeline.task_profile,
        config=config,
        registry=registry,
    )


__all__ = ["plan_joint_v2_candidates", "run_joint_v2_m1_m3_pipeline"]
