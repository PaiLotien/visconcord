"""Versioned rule registry for Joint v1.3 operator-scoped field roles."""

from __future__ import annotations

import json

from skillvis_v2_accuracy.config import project_root
from skillvis_v2_accuracy.rules import RuleDefinition, RuleRegistry
from skillvis_v2_joint.config import JointSemanticSettings
from skillvis_v2_joint_v3.registry import build_joint_v3_registry


ROLE_RULE_PATH = (
    project_root()
    / "configs/skillvis_v2_joint_v4/rules_operator_scoped_roles_v1.json"
)


def build_joint_v4_registry(
    settings: JointSemanticSettings | None = None,
) -> RuleRegistry:
    base = build_joint_v3_registry(settings)
    payload = json.loads(ROLE_RULE_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "development_only_not_protocol_frozen":
        raise ValueError("joint v4 role registry must remain development-only")
    rules = [base.get(rule_id) for rule_id in base.rule_ids]
    rules.extend(RuleDefinition.model_validate(item) for item in payload["rules"])
    return RuleRegistry(ROLE_RULE_PATH, rules)


__all__ = ["ROLE_RULE_PATH", "build_joint_v4_registry"]
