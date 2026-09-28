"""Typed, immutable contracts for the SkillVIS v2 deterministic M1–M3 path."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SemanticType = Literal[
    "quantitative",
    "categorical",
    "temporal",
    "identifier",
    "ordinal",
    "geographic",
    "unknown",
]
TaskName = Literal[
    "comparison",
    "ranking",
    "trend",
    "correlation",
    "distribution",
    "composition",
    "extremum",
    "aggregation",
    "filtering",
    "sorting",
    "temporal_analysis",
    "unresolved",
]
FieldRole = Literal[
    "measure",
    "dimension",
    "temporal",
    "grouping",
    "filter",
    "identifier",
    "geographic",
    "unknown",
]


class FrozenContract(BaseModel):
    """Base class with stable serialization and strict schema handling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class EvidenceSpan(FrozenContract):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    normalized: str
    kind: str
    rule_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_bounds(self) -> "EvidenceSpan":
        if self.end < self.start:
            raise ValueError("span end must be >= start")
        return self


class DetectedTerm(FrozenContract):
    canonical: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    rule_ids: tuple[str, ...]


class QueryNormalizationResult(FrozenContract):
    schema_version: Literal["skillvis-v2-query-normalization-v0.1.0"] = (
        "skillvis-v2-query-normalization-v0.1.0"
    )
    original_query: str
    normalized_query: str
    tokens: tuple[str, ...]
    lemmas_or_normalized_terms: tuple[str, ...]
    detected_comparatives: tuple[DetectedTerm, ...] = ()
    detected_superlatives: tuple[DetectedTerm, ...] = ()
    temporal_expressions: tuple[DetectedTerm, ...] = ()
    aggregation_terms: tuple[DetectedTerm, ...] = ()
    ranking_terms: tuple[DetectedTerm, ...] = ()
    task_terms: tuple[DetectedTerm, ...] = ()
    chart_mentions: tuple[DetectedTerm, ...] = ()
    limitation_terms: tuple[DetectedTerm, ...] = ()
    query_spans: tuple[EvidenceSpan, ...]
    applied_rule_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    failure_state: str | None = None


class NumericSummary(FrozenContract):
    count: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None


class TemporalSummary(FrozenContract):
    parse_ratio: float = Field(ge=0.0, le=1.0)
    minimum: str | None = None
    maximum: str | None = None
    timezone_observed: bool = False


class CategoryValueCount(FrozenContract):
    value: Any
    count: int = Field(ge=0)


class CategoricalSummary(FrozenContract):
    unique_count: int = Field(ge=0)
    top_values: tuple[CategoryValueCount, ...] = ()


class SchemaFieldProfile(FrozenContract):
    field_name: str
    normalized_name: str
    aliases: tuple[str, ...]
    semantic_type: SemanticType
    physical_type: str
    cardinality: int = Field(ge=0)
    null_rate: float = Field(ge=0.0, le=1.0)
    sample_values: tuple[Any, ...]
    numeric_summary: NumericSummary | None
    temporal_summary: TemporalSummary | None
    categorical_summary: CategoricalSummary | None
    role_candidates: tuple[FieldRole, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    source_rule_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class SchemaProfile(FrozenContract):
    schema_version: Literal["skillvis-v2-schema-profile-v0.1.0"] = (
        "skillvis-v2-schema-profile-v0.1.0"
    )
    dataset_id: str
    row_count: int = Field(ge=0)
    fields: tuple[SchemaFieldProfile, ...]
    data_access_policy: str
    applied_rule_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    failure_state: str | None = None

    def field(self, name: str) -> SchemaFieldProfile:
        for field in self.fields:
            if field.field_name == name:
                return field
        raise KeyError(name)


class QueryMention(FrozenContract):
    mention_id: str
    text: str
    normalized: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    mention_kind: Literal["field", "alias", "value", "structural", "unresolved"]
    source_rule_ids: tuple[str, ...]


class CandidateScore(FrozenContract):
    lexical_score: float = Field(ge=0.0, le=1.0)
    alias_score: float = Field(ge=0.0, le=1.0)
    value_score: float = Field(ge=0.0, le=1.0)
    type_score: float = Field(ge=0.0, le=1.0)
    task_role_score: float = Field(ge=0.0, le=1.0)
    conflict_penalty: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)


class FieldEvidence(FrozenContract):
    evidence_type: str
    query_span: EvidenceSpan | None
    value: str
    rule_id: str
    source_ids: tuple[str, ...]


class FieldCandidate(FrozenContract):
    field_name: str
    mention_ids: tuple[str, ...]
    candidate_rank: int = Field(ge=1)
    score_components: CandidateScore
    match_types: tuple[str, ...]
    evidence: tuple[FieldEvidence, ...]
    intended_roles: tuple[FieldRole, ...]
    source_rule_ids: tuple[str, ...]


class AmbiguityRecord(FrozenContract):
    dimension: Literal["field", "task", "transform", "relation", "query"]
    alternatives: tuple[str, ...]
    reason: str
    evidence_span_ids: tuple[str, ...] = ()
    rule_ids: tuple[str, ...]


class AmbiguityStateType(str, Enum):
    NONE = "NONE"
    LEXICAL_RESIDUE = "LEXICAL_RESIDUE"
    MATERIAL_AMBIGUITY = "MATERIAL_AMBIGUITY"
    MISSING_REQUIRED_ROLE = "MISSING_REQUIRED_ROLE"
    MULTIPLE_VALID_INTERPRETATIONS = "MULTIPLE_VALID_INTERPRETATIONS"
    UNSUPPORTED_SCOPE = "UNSUPPORTED_SCOPE"


class TypedAmbiguityState(FrozenContract):
    state: AmbiguityStateType
    evidence_spans: tuple[EvidenceSpan, ...]
    triggering_rules: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    affected_fields: tuple[str, ...]
    affected_tasks: tuple[str, ...]
    resolution_strategy: Literal[
        "proceed",
        "ignore_lexical_residue",
        "request_clarification",
        "preserve_alternatives",
        "abstain_unsupported",
    ]
    explanation: str


class UnsupportedScopeRecord(FrozenContract):
    operation: str
    evidence_spans: tuple[EvidenceSpan, ...]
    triggering_rules: tuple[str, ...]
    explanation: str


class CapabilityAssessment(FrozenContract):
    supported_operations: tuple[str, ...]
    unsupported_operations: tuple[str, ...]
    risk_level: Literal["unknown", "low", "medium", "high", "critical"]
    abstention_required: bool
    explanation: str
    provenance: tuple[str, ...]

    @model_validator(mode="after")
    def validate_operation_partition(self) -> "CapabilityAssessment":
        overlap = set(self.supported_operations) & set(self.unsupported_operations)
        if overlap:
            raise ValueError(
                f"supported and unsupported operations overlap: {sorted(overlap)}"
            )
        if self.abstention_required != bool(self.unsupported_operations):
            raise ValueError(
                "capability abstention must exactly reflect unsupported operations"
            )
        return self


class SafetyWarning(FrozenContract):
    warning_type: Literal[
        "lexical_residue",
        "material_ambiguity",
        "missing_required_role",
        "multiple_valid_interpretations",
        "unsupported_scope",
    ]
    severity: Literal["info", "warning", "error", "critical"]
    message: str
    evidence_spans: tuple[EvidenceSpan, ...]
    rule_ids: tuple[str, ...]


class SemanticSafetyResult(FrozenContract):
    schema_version: Literal["skillvis-v2-semantic-safety-v1.0.0"] = (
        "skillvis-v2-semantic-safety-v1.0.0"
    )
    ambiguity_state: TypedAmbiguityState
    unsupported_scope: tuple[UnsupportedScopeRecord, ...]
    capability_assessment: CapabilityAssessment
    safety_warnings: tuple[SafetyWarning, ...]
    abstention_state: AbstentionState
    abstention_reason: str | None
    applied_rule_ids: tuple[str, ...]


class FieldResolutionResult(FrozenContract):
    schema_version: Literal["skillvis-v2-field-resolution-v0.1.0"] = (
        "skillvis-v2-field-resolution-v0.1.0"
    )
    query_mentions: tuple[QueryMention, ...]
    candidate_fields: tuple[FieldCandidate, ...]
    candidate_scores: dict[str, CandidateScore]
    match_types: dict[str, tuple[str, ...]]
    evidence: dict[str, tuple[FieldEvidence, ...]]
    intended_roles: dict[str, tuple[FieldRole, ...]]
    ambiguity: tuple[AmbiguityRecord, ...]
    required_field_candidates: tuple[str, ...]
    unresolved_mentions: tuple[QueryMention, ...]
    applied_rule_ids: tuple[str, ...]
    abstention_reason: str | None = None


class TransformIntent(FrozenContract):
    transform_type: Literal[
        "aggregate", "filter", "sort", "limit", "bin", "time_unit"
    ]
    operation: str
    field_candidates: tuple[str, ...] = ()
    value: Any | None = None
    negated: bool = False
    evidence_spans: tuple[EvidenceSpan, ...]
    rule_ids: tuple[str, ...]


class TaskCandidate(FrozenContract):
    task: TaskName
    score: float = Field(ge=0.0, le=1.0)
    evidence_spans: tuple[EvidenceSpan, ...]
    required_roles: tuple[FieldRole, ...]
    rule_ids: tuple[str, ...]
    score_components: dict[str, float]


class AnalyticalTaskResult(FrozenContract):
    schema_version: Literal["skillvis-v2-task-result-v0.1.0"] = (
        "skillvis-v2-task-result-v0.1.0"
    )
    task_candidates: tuple[TaskCandidate, ...]
    task_scores: dict[str, float]
    primary_task: TaskName
    secondary_tasks: tuple[TaskName, ...]
    query_evidence_spans: tuple[EvidenceSpan, ...]
    required_roles: tuple[FieldRole, ...]
    transforms: tuple[TransformIntent, ...]
    aggregation: TransformIntent | None
    filters: tuple[TransformIntent, ...]
    sort: TransformIntent | None
    limit: int | None = Field(default=None, ge=1)
    temporal_scope: tuple[str, ...]
    ambiguity: tuple[AmbiguityRecord, ...]
    applied_rule_ids: tuple[str, ...]
    failure_state: str | None = None


class IntentFieldRef(FrozenContract):
    field_name: str
    role: FieldRole
    score: float = Field(ge=0.0, le=1.0)
    candidate_rank: int = Field(ge=1)
    evidence: tuple[FieldEvidence, ...]
    rule_ids: tuple[str, ...]


class ProvenanceRecord(FrozenContract):
    stage: Literal[
        "query_normalization",
        "schema_profiling",
        "field_resolution",
        "task_profiling",
        "query_decomposition",
        "intent_assembly",
        "semantic_safety",
        "candidate_planning",
    ]
    rule_id: str
    source_ids: tuple[str, ...]
    scope: str
    rationale: str
    priority: int
    failure_condition: str
    conflict_policy: str
    trace_fields: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


class AbstentionState(FrozenContract):
    abstained: bool
    stage: str | None = None
    reason: str | None = None


class CandidateChartType(str, Enum):
    BAR = "bar"
    GROUPED_BAR = "grouped_bar"
    DOT_PLOT = "dot_plot"
    HORIZONTAL_BAR = "horizontal_bar"
    TOP_K_BAR = "top_k_bar"
    LINE = "line"
    MULTI_LINE = "multi_line"
    SCATTER = "scatter"
    GROUPED_SCATTER = "grouped_scatter"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    STACKED_BAR = "stacked_bar"
    NORMALIZED_STACKED_BAR = "normalized_stacked_bar"
    PIE = "pie"
    RANKED_BAR = "ranked_bar"
    HIGHLIGHTED_EXTREMUM = "highlighted_extremum"


class CandidateFields(FrozenContract):
    measure: tuple[str, ...] = ()
    dimension: tuple[str, ...] = ()
    temporal: tuple[str, ...] = ()
    grouping: tuple[str, ...] = ()
    color: tuple[str, ...] = ()
    size: tuple[str, ...] = ()
    facet: tuple[str, ...] = ()


class CandidateEncoding(FrozenContract):
    channel: Literal[
        "x", "y", "color", "size", "row", "column", "theta", "detail"
    ]
    field: str | None = None
    semantic_type: SemanticType
    aggregate: str | None = None
    bin: bool = False
    time_unit: str | None = None
    sort: str | None = None


class CandidateTransform(FrozenContract):
    transform_type: Literal[
        "aggregate",
        "filter",
        "sort",
        "limit",
        "bin",
        "time_unit",
        "stack",
        "normalize",
        "highlight",
        "fold",
    ]
    operation: str
    field_candidates: tuple[str, ...] = ()
    value: Any | None = None
    negated: bool = False
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    rule_ids: tuple[str, ...]


class CandidateScoreComponents(FrozenContract):
    task_fit_score: float = Field(ge=0.0, le=1.0)
    field_role_fit_score: float = Field(ge=0.0, le=1.0)
    type_compatibility_score: float = Field(ge=0.0, le=1.0)
    chart_effectiveness_score: float = Field(ge=0.0, le=1.0)
    ambiguity_penalty: float = Field(ge=0.0, le=1.0)
    missing_role_penalty: float = Field(ge=0.0, le=1.0)
    unsupported_feature_penalty: float = Field(ge=0.0, le=1.0)
    complexity_penalty: float = Field(ge=0.0, le=1.0)
    provenance_strength_score: float = Field(ge=0.0, le=1.0)
    deterministic_tie_breaker: float = Field(ge=0.0, le=1.0)
    rule_ids: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def validate_score_rule_bindings(self) -> "CandidateScoreComponents":
        component_names = set(type(self).model_fields) - {"rule_ids"}
        if set(self.rule_ids) != component_names:
            raise ValueError(
                "score rule bindings must cover every score component exactly"
            )
        if any(not value for value in self.rule_ids.values()):
            raise ValueError("every score component requires at least one rule ID")
        return self


class VisualizationCandidate(FrozenContract):
    schema_version: Literal["skillvis-v2-visualization-candidate-v1.0.0"] = (
        "skillvis-v2-visualization-candidate-v1.0.0"
    )
    candidate_id: str
    chart_type: CandidateChartType
    mark_type: Literal["bar", "point", "line", "arc", "boxplot"]
    fields: CandidateFields
    encodings: tuple[CandidateEncoding, ...]
    transforms: tuple[CandidateTransform, ...]
    aggregation: CandidateTransform | None
    filters: tuple[CandidateTransform, ...]
    sort: CandidateTransform | None
    limit: int | None = Field(default=None, ge=1)
    grouping: tuple[str, ...]
    task_alignment: TaskName
    field_role_alignment: dict[str, tuple[str, ...]]
    chart_compatibility_assumptions: tuple[str, ...]
    score_components: CandidateScoreComponents
    total_score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    rationale: str
    evidence_spans: tuple[EvidenceSpan, ...]
    applied_rule_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    ambiguity_state: AmbiguityStateType
    warnings: tuple[str, ...]
    unsupported_features: tuple[str, ...]
    executable: bool
    audit_only: bool
    rejection_reasons: tuple[str, ...]
    deterministic_signature: str

    @model_validator(mode="after")
    def validate_candidate_integrity(self) -> "VisualizationCandidate":
        channels = [item.channel for item in self.encodings]
        if len(channels) != len(set(channels)):
            raise ValueError("candidate encoding channels must be unique")
        if self.executable and (
            self.audit_only
            or self.rejection_reasons
            or self.unsupported_features
        ):
            raise ValueError(
                "executable candidate cannot be audit-only, rejected, or unsupported"
            )
        if not self.deterministic_signature:
            raise ValueError("candidate deterministic signature is required")
        if not self.evidence_spans:
            raise ValueError("candidate evidence spans are required")
        if not self.applied_rule_ids or not self.source_ids:
            raise ValueError("candidate rule/source provenance is required")
        return self


class CandidateDiversitySummary(FrozenContract):
    chart_family_count: int = Field(ge=0)
    encoding_signature_count: int = Field(ge=0)
    transform_signature_count: int = Field(ge=0)
    task_interpretation_count: int = Field(ge=0)


class CandidatePlanningResult(FrozenContract):
    schema_version: Literal["skillvis-v2-candidate-planning-result-v1.0.0"] = (
        "skillvis-v2-candidate-planning-result-v1.0.0"
    )
    intent_id: str
    config_version: str
    registry_hash: str
    candidates: tuple[VisualizationCandidate, ...]
    raw_candidate_count: int = Field(ge=0)
    post_dedup_count: int = Field(ge=0)
    post_ranking_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    diversity: CandidateDiversitySummary
    abstained: bool
    abstention_reason: str | None
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    provenance: tuple[ProvenanceRecord, ...]
    deterministic_signature: str

    @model_validator(mode="after")
    def validate_result_counts(self) -> "CandidatePlanningResult":
        if self.post_ranking_count != len(self.candidates):
            raise ValueError("post-ranking count must equal candidate list length")
        if self.post_dedup_count > self.raw_candidate_count:
            raise ValueError("post-dedup count cannot exceed raw count")
        if self.duplicate_count != (
            self.raw_candidate_count - self.post_dedup_count
        ):
            raise ValueError("duplicate count does not match enumeration counts")
        if self.abstained and any(item.executable for item in self.candidates):
            raise ValueError("abstained planning result cannot execute candidates")
        ranks = [item.rank for item in self.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("candidate ranks must be contiguous and ordered")
        signatures = [item.deterministic_signature for item in self.candidates]
        if len(signatures) != len(set(signatures)):
            raise ValueError("ranked candidate signatures must be unique")
        return self


class VisualizationIntentIR(FrozenContract):
    schema_version: Literal[
        "skillvis-v2-intent-ir-v0.2.0",
        "skillvis-v2-intent-ir-v0.3.0",
    ] = (
        "skillvis-v2-intent-ir-v0.3.0"
    )
    config_version: str
    registry_hash: str
    deterministic_id: str
    query: str
    normalized_query: str
    dataset_id: str
    measures: tuple[IntentFieldRef, ...]
    dimensions: tuple[IntentFieldRef, ...]
    temporal_fields: tuple[IntentFieldRef, ...]
    grouping_fields: tuple[IntentFieldRef, ...]
    task_candidates: tuple[TaskCandidate, ...]
    primary_task: TaskName
    transforms: tuple[TransformIntent, ...] = ()
    aggregation: TransformIntent | None
    filters: tuple[TransformIntent, ...]
    sort: TransformIntent | None
    limit: int | None = Field(default=None, ge=1)
    chart_mentions: tuple[DetectedTerm, ...]
    field_candidates: tuple[FieldCandidate, ...]
    unresolved_mentions: tuple[QueryMention, ...]
    ambiguity: tuple[AmbiguityRecord, ...]
    ambiguity_state: TypedAmbiguityState = TypedAmbiguityState(
        state=AmbiguityStateType.NONE,
        evidence_spans=(),
        triggering_rules=(),
        confidence=1.0,
        affected_fields=(),
        affected_tasks=(),
        resolution_strategy="proceed",
        explanation="M3.5 semantic safety was not enabled for this execution.",
    )
    unsupported_scope: tuple[UnsupportedScopeRecord, ...] = ()
    capability_assessment: CapabilityAssessment = CapabilityAssessment(
        supported_operations=(),
        unsupported_operations=(),
        risk_level="unknown",
        abstention_required=False,
        explanation="M3.5 semantic safety was not enabled for this execution.",
        provenance=(),
    )
    safety_warnings: tuple[SafetyWarning, ...] = ()
    abstention_reason: str | None = None
    evidence_spans: tuple[EvidenceSpan, ...]
    provenance: tuple[ProvenanceRecord, ...]
    warnings: tuple[str, ...]
    abstention_state: AbstentionState

    @classmethod
    def validate_serialized(cls, payload: dict[str, Any]) -> "VisualizationIntentIR":
        return cls.model_validate(payload)

    @classmethod
    def json_contract_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()


class M1M3PipelineResult(FrozenContract):
    schema_version: Literal["skillvis-v2-m1-m3-pipeline-v0.1.0"] = (
        "skillvis-v2-m1-m3-pipeline-v0.1.0"
    )
    normalization: QueryNormalizationResult
    schema_profile: SchemaProfile
    field_resolution: FieldResolutionResult
    task_profile: AnalyticalTaskResult
    intent_ir: VisualizationIntentIR
