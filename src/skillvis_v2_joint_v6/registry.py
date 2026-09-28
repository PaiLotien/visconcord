"""Versioned rule registry for the isolated Joint v1.5 closure."""

from __future__ import annotations

import json

from skillvis_v2_accuracy.config import project_root
from skillvis_v2_accuracy.rules import RuleDefinition, RuleRegistry
from skillvis_v2_joint.config import JointSemanticSettings
from skillvis_v2_joint_v5.registry import build_joint_v5_registry


V15_RULE_PATH = (
    project_root()
    / "configs/skillvis_v2_joint_v6/rules_joint_v15_closure.json"
)


def build_joint_v6_registry(
    settings: JointSemanticSettings | None = None,
) -> RuleRegistry:
    base = build_joint_v5_registry(settings)
    payload = json.loads(V15_RULE_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "development_only_not_protocol_frozen":
        raise ValueError("Joint v1.5 registry must remain development-only")
    rules = [base.get(rule_id) for rule_id in base.rule_ids]
    rules.extend(RuleDefinition.model_validate(item) for item in payload["rules"])
    return RuleRegistry(V15_RULE_PATH, rules)


__all__ = ["V15_RULE_PATH", "build_joint_v6_registry"]
