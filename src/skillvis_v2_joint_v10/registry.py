"""Versioned registry for Joint v1.9."""
from __future__ import annotations
import json
from skillvis_v2_accuracy.config import project_root
from skillvis_v2_accuracy.rules import RuleDefinition, RuleRegistry
from skillvis_v2_joint.config import JointSemanticSettings
from skillvis_v2_joint_v9.registry import build_joint_v9_registry
RULE_PATH = project_root()/"configs/skillvis_v2_joint_v10/rules_joint_v19_composite_span.json"
def build_joint_v10_registry(settings: JointSemanticSettings|None=None)->RuleRegistry:
    parent=build_joint_v9_registry(settings); payload=json.loads(RULE_PATH.read_text())
    if payload.get("status")!="development_only_not_protocol_frozen": raise ValueError("Joint v1.9 registry must remain development-only")
    rules=[parent.get(rule_id) for rule_id in parent.rule_ids]; rules.extend(RuleDefinition.model_validate(item) for item in payload["rules"]); return RuleRegistry(RULE_PATH,rules)
__all__=["RULE_PATH","build_joint_v10_registry"]
