"""Versioned rule registry for Joint v1.4 mention ownership."""

from __future__ import annotations

import json

from skillvis_v2_accuracy.config import project_root
from skillvis_v2_accuracy.rules import RuleDefinition, RuleRegistry
from skillvis_v2_joint.config import JointSemanticSettings
from skillvis_v2_joint_v4.registry import build_joint_v4_registry


OWNERSHIP_RULE_PATH = (
    project_root()
    / "configs/skillvis_v2_joint_v5/rules_mention_ownership_v1.json"
)


def build_joint_v5_registry(
    settings: JointSemanticSettings | None = None,
) -> RuleRegistry:
    base = build_joint_v4_registry(settings)
    payload = json.loads(OWNERSHIP_RULE_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "development_only_not_protocol_frozen":
        raise ValueError("joint v5 registry must remain development-only")
    rules = [base.get(rule_id) for rule_id in base.rule_ids]
    rules.extend(RuleDefinition.model_validate(item) for item in payload["rules"])
    return RuleRegistry(OWNERSHIP_RULE_PATH, rules)


__all__ = ["OWNERSHIP_RULE_PATH", "build_joint_v5_registry"]
