"""Versioned configuration for the isolated obligation-closure revision."""

from pathlib import Path

from skillvis_v2_accuracy.config import (  # re-export frozen contracts
    CandidatePlannerGates,
    CandidatePlannerSettings,
    CandidateScoreWeights,
    FeatureFlags,
    ResolverWeights,
    SkillVISV2Config,
    project_root,
)


def default_config_path() -> Path:
    return project_root() / "configs/skillvis_v2_obligation/candidate_planner_v4.json"


def load_config(path: str | Path | None = None) -> SkillVISV2Config:
    config_path = Path(path) if path else default_config_path()
    config = SkillVISV2Config.model_validate_json(config_path.read_text())
    if not config.enabled:
        raise RuntimeError("SkillVIS obligation revision is disabled")
    return config


def load_m3_5_config() -> SkillVISV2Config:
    config = load_config()
    if not config.feature_flags.enable_m3_5_semantic_safety:
        raise RuntimeError("M3.5 semantic safety is not enabled")
    return config


def load_m4_config() -> SkillVISV2Config:
    config = load_config()
    if not config.feature_flags.enable_candidate_planner:
        raise RuntimeError("M4 Candidate Planner is not enabled")
    if config.feature_flags.enable_semantic_validator:
        raise RuntimeError("M5 Validator remains a separate downstream gate")
    return config


__all__ = [
    "CandidatePlannerGates",
    "CandidatePlannerSettings",
    "CandidateScoreWeights",
    "FeatureFlags",
    "ResolverWeights",
    "SkillVISV2Config",
    "load_config",
    "load_m3_5_config",
    "load_m4_config",
    "project_root",
]
