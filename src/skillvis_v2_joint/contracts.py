"""Typed contracts for joint field-task semantic resolution."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from skillvis_v2_accuracy.contracts import (
    EvidenceSpan,
    FieldResolutionResult,
    FrozenContract,
    M1M3PipelineResult,
    TransformIntent,
)


AnalyticalGoal = Literal[
    "comparison",
    "ranking",
    "trend",
    "correlation",
    "distribution",
    "composition",
    "extremum",
    "unresolved",
]


class SchemaSemanticProfile(FrozenContract):
    field_name: str
    components: tuple[str, ...]
    entity_concepts: tuple[str, ...]
    property_concepts: tuple[str, ...]
    qualifier_concepts: tuple[str, ...]
    generic_concepts: tuple[str, ...]
    rule_ids: tuple[str, ...] = ("R-SD-001",)


class SemanticFieldMention(FrozenContract):
    mention_id: str
    concept_id: str
    concept_kind: Literal["entity", "property", "qualifier", "measure", "generic"]
    query_span: EvidenceSpan
    expected_roles: tuple[str, ...]
    rule_ids: tuple[str, ...] = ("R-FR-020",)


class FieldRoleBinding(FrozenContract):
    field_name: str
    role: Literal[
        "measure", "dimension", "grouping", "temporal", "filter", "identifier"
    ]
    mention_ids: tuple[str, ...]
    evidence_spans: tuple[EvidenceSpan, ...]


class PresentationIntent(FrozenContract):
    chart_hints: tuple[str, ...]
    explicit: bool
    evidence_spans: tuple[EvidenceSpan, ...]


class InterpretationHypothesis(FrozenContract):
    hypothesis_id: str
    analytical_goal: AnalyticalGoal
    field_bindings: tuple[FieldRoleBinding, ...]
    operations: tuple[TransformIntent, ...]
    presentation: PresentationIntent
    satisfied_constraints: tuple[str, ...]
    violated_constraints: tuple[str, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    score: float = Field(ge=0.0, le=1.0)
    complete: bool
    rule_ids: tuple[str, ...]


class SemanticGraphNode(FrozenContract):
    node_id: str
    node_type: Literal["mention", "field", "role", "goal", "operation", "presentation"]
    label: str
    attributes: dict[str, str]


class SemanticGraphEdge(FrozenContract):
    edge_id: str
    source: str
    target: str
    relation: Literal[
        "grounded_as",
        "fills_role",
        "supports_goal",
        "requires_operation",
        "constrains_presentation",
        "conflicts_with",
    ]
    rule_ids: tuple[str, ...]


class JointSemanticResolution(FrozenContract):
    schema_version: Literal["skillvis-v2-joint-semantic-resolution-v1.0.0"] = (
        "skillvis-v2-joint-semantic-resolution-v1.0.0"
    )
    schema_profiles: tuple[SchemaSemanticProfile, ...]
    semantic_mentions: tuple[SemanticFieldMention, ...]
    base_field_resolution: FieldResolutionResult
    hypotheses: tuple[InterpretationHypothesis, ...]
    selected_hypothesis_id: str | None
    graph_nodes: tuple[SemanticGraphNode, ...]
    graph_edges: tuple[SemanticGraphEdge, ...]
    applied_rule_ids: tuple[str, ...]
    warnings: tuple[str, ...]


class JointPipelineResult(FrozenContract):
    schema_version: Literal["skillvis-v2-joint-pipeline-v1.0.0"] = (
        "skillvis-v2-joint-pipeline-v1.0.0"
    )
    pipeline: M1M3PipelineResult
    joint_semantics: JointSemanticResolution


__all__ = [
    "AnalyticalGoal",
    "FieldRoleBinding",
    "InterpretationHypothesis",
    "JointPipelineResult",
    "JointSemanticResolution",
    "PresentationIntent",
    "SchemaSemanticProfile",
    "SemanticFieldMention",
    "SemanticGraphEdge",
    "SemanticGraphNode",
]
