"""Joint v1.12: typed set roles and ordering/aggregation precedence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from skillvis_v2_accuracy.contracts import M1M3PipelineResult, VisualizationIntentIR
from skillvis_v2_accuracy.pipeline import (
    _abstention_state,
    _build_field_refs,
    _grouping_from_query,
    _provenance,
    _stable_unique,
)
from skillvis_v2_accuracy.task_profiling import run_task_profiling
from skillvis_v2_joint.config import JointSemanticSettings, load_config, load_settings
from skillvis_v2_joint.contracts import JointPipelineResult, JointSemanticResolution
from skillvis_v2_joint_v2.task_resolution import remove_operation_field_collisions
from skillvis_v2_joint_v6.safety_taxonomy import run_semantic_safety_v15
from skillvis_v2_joint_v7.task_resolution import run_joint_v7_task_resolution
from skillvis_v2_joint_v8.task_resolution import apply_discourse_task_closure
from skillvis_v2_joint_v12 import run_joint_v12_m1_m3_pipeline
from skillvis_v2_joint_v12.ranking_roles import resolve_ranking_target_roles
from skillvis_v2_joint_v12.task_closure import apply_ranking_role_closure
from skillvis_v2_obligation.intent_compiler import compile_intent_obligations
from skillvis_v2_obligation_v3.candidate_planner import plan_visualization_candidates

from .registry import build_joint_v13_registry
from .set_roles import resolve_set_noun_grouping_roles
from .transform_precedence import apply_ordering_aggregation_precedence


CONFIG_VERSION = "skillvis-v2-joint-semantic-precedence-v1.12.0"


def _identity(intent: VisualizationIntentIR) -> VisualizationIntentIR:
    payload = intent.model_dump(mode="json")
    payload.pop("deterministic_id", None)
    identity = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return VisualizationIntentIR.validate_serialized(
        intent.model_copy(update={"deterministic_id": identity}).model_dump(
            mode="json"
        )
    )


def run_joint_v13_m1_m3_pipeline(
    *,
    query: str,
    dataset_id: str,
    data: pd.DataFrame | Mapping[str, Sequence[object]],
    aliases: Mapping[str, Sequence[str]] | None = None,
    settings: JointSemanticSettings | None = None,
) -> JointPipelineResult:
    settings = settings or load_settings()
    parent = run_joint_v12_m1_m3_pipeline(
        query=query,
        dataset_id=dataset_id,
        data=data,
        aliases=aliases,
        settings=settings,
    )
    config = load_config(settings).model_copy(update={"config_version": CONFIG_VERSION})
    registry = build_joint_v13_registry(settings)
    normalization = parent.pipeline.normalization

    schema, fields, set_traces = resolve_set_noun_grouping_roles(
        query=query,
        schema=parent.pipeline.schema_profile,
        fields=parent.pipeline.field_resolution,
        task=parent.pipeline.task_profile,
        registry=registry,
    )
    schema, fields, ranking_role_traces = resolve_ranking_target_roles(
        normalized=normalization,
        schema=schema,
        fields=fields,
        registry=registry,
    )
    base = run_task_profiling(normalization, schema, fields, config, registry)
    base, collisions = remove_operation_field_collisions(base, fields)
    (
        task,
        hypotheses,
        selected,
        nodes,
        edges,
        ranking_applied,
    ) = run_joint_v7_task_resolution(
        normalized=normalization,
        schema=schema,
        fields=fields,
        base=base,
        semantic_mentions=parent.joint_semantics.semantic_mentions,
        settings=settings,
        registry=registry,
    )
    (
        task,
        hypotheses,
        selected,
        nodes,
        edges,
        discourse_applied,
    ) = apply_discourse_task_closure(
        normalized=normalization,
        schema=schema,
        fields=fields,
        task=task,
        hypotheses=hypotheses,
        semantic_mentions=parent.joint_semantics.semantic_mentions,
        registry=registry,
    )
    (
        task,
        hypotheses,
        selected,
        nodes,
        edges,
        ranking_role_closed,
    ) = apply_ranking_role_closure(
        normalized=normalization,
        schema=schema,
        fields=fields,
        task=task,
        hypotheses=hypotheses,
        semantic_mentions=parent.joint_semantics.semantic_mentions,
        registry=registry,
    )
    (
        task,
        hypotheses,
        selected,
        nodes,
        edges,
        precedence_applied,
        precedence_traces,
    ) = apply_ordering_aggregation_precedence(
        query=query,
        base=base,
        task=task,
        hypotheses=hypotheses,
        semantic_mentions=parent.joint_semantics.semantic_mentions,
        fields=fields,
        registry=registry,
    )

    measures, dimensions, temporal, structural = _build_field_refs(
        fields, schema, task
    )
    grouping = _stable_unique(
        (*structural, *_grouping_from_query(query, dimensions)),
        key=lambda item: (item.field_name, item.role),
    )
    rules = {
        *normalization.applied_rule_ids,
        *schema.applied_rule_ids,
        *fields.applied_rule_ids,
        *task.applied_rule_ids,
        "R-MR-001",
        "R-MR-002",
        "R-MR-020",
        "R-MR-021",
    }
    registry.validate_ids(rules)
    spans = _stable_unique(
        (
            *normalization.query_spans,
            *task.query_evidence_spans,
            *(item.query_span for item in parent.joint_semantics.semantic_mentions),
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
                *parent.joint_semantics.warnings,
                *set_traces,
                *ranking_role_traces,
                *precedence_traces,
                *(f"operation_field_collision_removed:{name}" for name in collisions),
                *(("R-TP-040 specific ranking dominance applied",) if ranking_applied else ()),
                *(("R-TP-050 analytical discourse closure applied",) if discourse_applied else ()),
                *(("R-TP-130 ranking target-score closure applied",) if ranking_role_closed else ()),
                *(("R-TP-140 ordering/aggregation precedence applied",) if precedence_applied else ()),
                "joint v1.12 semantic precedence; development only; independent qualification required",
            }
        )
    )
    provenance = _provenance(registry, rules, dataset_id)
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
            "evidence_spans": spans,
            "provenance": provenance,
            "warnings": warnings,
            "abstention_state": abstention,
        }
    )
    compiled = compile_intent_obligations(draft, task, registry)
    safety = run_semantic_safety_v15(compiled.intent, registry)
    rules.update(compiled.applied_rule_ids)
    rules.update(safety.applied_rule_ids)
    provenance = _provenance(registry, rules, dataset_id)
    completed = _identity(
        compiled.intent.model_copy(
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
    )
    pipeline = M1M3PipelineResult(
        normalization=normalization,
        schema_profile=schema,
        field_resolution=fields,
        task_profile=compiled.task_result,
        intent_ir=completed,
    )
    joint_ids = {
        rule
        for rule in rules
        if rule.startswith(
            (
                "R-SP-04",
                "R-SP-110",
                "R-SD-",
                "R-FR-02",
                "R-FR-03",
                "R-FR-04",
                "R-FR-05",
                "R-FR-06",
                "R-FR-07",
                "R-FR-08",
                "R-FR-09",
                "R-FR-10",
                "R-FR-120",
                "R-FR-140",
                "R-TP-02",
                "R-TP-03",
                "R-TP-04",
                "R-TP-05",
                "R-TP-130",
                "R-TP-140",
                "R-MR-02",
                "R-SA-02",
            )
        )
    }
    joint = JointSemanticResolution(
        schema_profiles=parent.joint_semantics.schema_profiles,
        semantic_mentions=parent.joint_semantics.semantic_mentions,
        base_field_resolution=parent.joint_semantics.base_field_resolution,
        hypotheses=hypotheses,
        selected_hypothesis_id=selected,
        graph_nodes=nodes,
        graph_edges=edges,
        applied_rule_ids=registry.validate_ids(joint_ids),
        warnings=warnings,
    )
    return JointPipelineResult(pipeline=pipeline, joint_semantics=joint)


def plan_joint_v13_candidates(result: JointPipelineResult):
    settings = load_settings()
    config = load_config(settings).model_copy(update={"config_version": CONFIG_VERSION})
    registry = build_joint_v13_registry(settings)
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


__all__ = [
    "CONFIG_VERSION",
    "plan_joint_v13_candidates",
    "run_joint_v13_m1_m3_pipeline",
]
