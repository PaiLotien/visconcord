"""Combined provenance registry without mutating frozen V3/V4 registries."""

from __future__ import annotations

import json

from skillvis_v2_accuracy.config import project_root
from skillvis_v2_accuracy.rules import RuleDefinition, RuleRegistry

from .config import JointSemanticSettings, load_config, load_settings


def build_joint_registry(
    settings: JointSemanticSettings | None = None,
) -> RuleRegistry:
    settings = settings or load_settings()
    base = RuleRegistry.from_config(load_config(settings))
    joint_path = project_root() / settings.joint_rule_registry_path
    payload = json.loads(joint_path.read_text())
    if payload.get("status") != "joint_semantic_development_implemented":
        raise ValueError("joint semantic rule registry has invalid status")
    rules = [base.get(rule_id) for rule_id in base.rule_ids]
    rules.extend(RuleDefinition.model_validate(item) for item in payload["rules"])
    return RuleRegistry(joint_path, rules)


__all__ = ["build_joint_registry"]
