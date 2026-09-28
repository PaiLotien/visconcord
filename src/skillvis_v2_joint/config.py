"""Configuration boundary for the isolated joint M2/M3 revision."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from skillvis_v2_accuracy.config import SkillVISV2Config, project_root


class JointSemanticSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    config_version: str
    base_config_path: str
    joint_rule_registry_path: str
    semantic_registry_path: str
    field_top_k: int = Field(ge=1, le=20)
    semantic_candidate_threshold: float = Field(ge=0.0, le=1.0)
    hypothesis_ambiguity_margin: float = Field(ge=0.0, le=1.0)
    status: str


def settings_path() -> Path:
    return project_root() / "configs/skillvis_v2_joint/joint_semantic_v1.json"


def load_settings(path: str | Path | None = None) -> JointSemanticSettings:
    target = Path(path) if path else settings_path()
    settings = JointSemanticSettings.model_validate_json(target.read_text())
    if settings.status != "development_only_not_protocol_frozen":
        raise RuntimeError("joint semantic revision must remain development-only")
    return settings


def load_config(
    settings: JointSemanticSettings | None = None,
) -> SkillVISV2Config:
    settings = settings or load_settings()
    base_path = project_root() / settings.base_config_path
    base = SkillVISV2Config.model_validate_json(base_path.read_text())
    return base.model_copy(update={"config_version": settings.config_version})


def load_semantic_registry_payload(
    settings: JointSemanticSettings | None = None,
) -> dict:
    settings = settings or load_settings()
    return json.loads((project_root() / settings.semantic_registry_path).read_text())


__all__ = [
    "JointSemanticSettings",
    "load_config",
    "load_semantic_registry_payload",
    "load_settings",
]
