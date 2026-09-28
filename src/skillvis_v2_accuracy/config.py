"""Versioned configuration and feature flags for SkillVIS v2."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeatureFlags(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enable_m1_query_normalization: bool
    enable_m1_schema_profiling: bool
    enable_m2_field_resolution: bool
    enable_m3_task_profiling: bool
    enable_m3_query_decomposition: bool
    enable_m3_5_semantic_safety: bool = False
    enable_value_matching: bool
    enable_candidate_planner: bool
    enable_semantic_validator: bool
    enable_selective_recovery: bool
    enable_llm: bool


class ResolverWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lexical: float = Field(ge=0)
    alias: float = Field(ge=0)
    value: float = Field(ge=0)
    type_compatible: float = Field(ge=0)
    task_role: float = Field(ge=0)
    structural_recovery: float = Field(ge=0)
    conflict: float = Field(ge=0)


class CandidateScoreWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_fit: float = Field(ge=0)
    field_role_fit: float = Field(ge=0)
    type_compatibility: float = Field(ge=0)
    chart_effectiveness: float = Field(ge=0)
    provenance_strength: float = Field(ge=0)
    deterministic_tie_breaker: float = Field(ge=0)
    ambiguity_penalty: float = Field(ge=0)
    missing_role_penalty: float = Field(ge=0)
    unsupported_feature_penalty: float = Field(ge=0)
    complexity_penalty: float = Field(ge=0)


class CandidatePlannerGates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    top1_acceptable_rate_min: float = Field(ge=0.0, le=1.0)
    acceptable_candidate_recall_at_3_min: float = Field(ge=0.0, le=1.0)
    unsupported_executable_candidate_rate_max: float = Field(ge=0.0, le=1.0)
    duplicate_candidate_rate_max: float = Field(ge=0.0, le=1.0)
    deterministic_replay_rate_min: float = Field(ge=0.0, le=1.0)
    provenance_completeness_rate_min: float = Field(ge=0.0, le=1.0)
    score_decomposition_completeness_rate_min: float = Field(ge=0.0, le=1.0)
    schema_validation_rate_min: float = Field(ge=0.0, le=1.0)


class CandidatePlannerSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    max_candidates: int = Field(ge=1, le=20)
    pie_max_categories: int = Field(ge=2)
    explicit_chart_hints_are_hard_constraints: bool
    audit_only_on_missing_roles: bool
    core_tasks: tuple[str, ...]
    chart_vocabulary: tuple[str, ...]
    score_formula: str
    score_weights: CandidateScoreWeights
    task_chart_effectiveness: dict[str, dict[str, float]]
    chart_complexity: dict[str, float]
    gates: CandidatePlannerGates

    @model_validator(mode="after")
    def validate_candidate_settings(self) -> "CandidatePlannerSettings":
        if len(set(self.chart_vocabulary)) != len(self.chart_vocabulary):
            raise ValueError("candidate chart vocabulary contains duplicates")
        if len(set(self.core_tasks)) != len(self.core_tasks):
            raise ValueError("candidate core task vocabulary contains duplicates")
        missing_complexity = set(self.chart_vocabulary) - set(self.chart_complexity)
        if missing_complexity:
            raise ValueError(
                f"chart complexity missing for {sorted(missing_complexity)}"
            )
        declared_effectiveness = {
            chart
            for values in self.task_chart_effectiveness.values()
            for chart in values
        }
        if not declared_effectiveness <= set(self.chart_vocabulary):
            raise ValueError("effectiveness table references undeclared chart")
        positive = self.score_weights.model_dump()
        if sum(
            positive[name]
            for name in (
                "task_fit",
                "field_role_fit",
                "type_compatibility",
                "chart_effectiveness",
                "provenance_strength",
                "deterministic_tie_breaker",
            )
        ) <= 0:
            raise ValueError("candidate positive score weight sum must be > 0")
        return self


class SkillVISV2Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    config_version: str
    enabled: bool
    rule_registry_path: str
    safety_rule_registry_path: str | None = None
    candidate_rule_registry_path: str | None = None
    feature_flags: FeatureFlags
    resolver_weights: ResolverWeights
    field_candidate_threshold: float = Field(ge=0.0, le=1.0)
    ambiguity_margin: float = Field(ge=0.0, le=1.0)
    task_candidate_threshold: float = Field(ge=0.0, le=1.0)
    max_value_cardinality: int = Field(ge=1)
    max_sample_values: int = Field(ge=1)
    max_profile_rows: int = Field(ge=1)
    deterministic_seed: int
    candidate_planner: CandidatePlannerSettings | None = None

    @model_validator(mode="after")
    def enforce_phase_boundary(self) -> "SkillVISV2Config":
        flags = self.feature_flags
        forbidden = {
            "semantic_validator": flags.enable_semantic_validator,
            "selective_recovery": flags.enable_selective_recovery,
            "llm": flags.enable_llm,
        }
        enabled_forbidden = [name for name, value in forbidden.items() if value]
        if enabled_forbidden:
            raise ValueError(
                "SkillVIS v2 semantic config cannot enable deferred components: "
                + ", ".join(enabled_forbidden)
            )
        if flags.enable_candidate_planner:
            if not flags.enable_m3_5_semantic_safety:
                raise ValueError("M4 Candidate Planner requires M3.5 safety")
            if self.candidate_planner is None:
                raise ValueError("M4 Candidate Planner settings are missing")
            if not self.candidate_rule_registry_path:
                raise ValueError("M4 candidate rule registry path is missing")
        elif self.candidate_planner is not None or self.candidate_rule_registry_path:
            raise ValueError(
                "candidate settings/registry require enable_candidate_planner=true"
            )
        return self


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "configs/skillvis_v2_accuracy/candidate_planner_v2.json"


def m3_5_config_path() -> Path:
    return project_root() / "configs/skillvis_v2_accuracy/candidate_planner_v2.json"


def m4_config_path() -> Path:
    return project_root() / "configs/skillvis_v2_accuracy/candidate_planner_v2.json"


def load_config(path: str | Path | None = None) -> SkillVISV2Config:
    config_path = Path(path) if path else default_config_path()
    config = SkillVISV2Config.model_validate_json(config_path.read_text())
    if not config.enabled:
        raise RuntimeError(
            "SkillVIS v2 is disabled; use an explicitly enabled v2 configuration"
        )
    return config


def load_m3_5_config() -> SkillVISV2Config:
    config = load_config(m3_5_config_path())
    if not config.feature_flags.enable_m3_5_semantic_safety:
        raise RuntimeError("M3.5 semantic safety is not enabled")
    return config


def load_m4_config() -> SkillVISV2Config:
    config = load_config(m4_config_path())
    if not config.feature_flags.enable_candidate_planner:
        raise RuntimeError("M4 Candidate Planner is not enabled")
    if config.feature_flags.enable_semantic_validator:
        raise RuntimeError("M5 Validator must remain disabled during M4")
    return config
