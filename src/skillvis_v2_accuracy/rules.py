"""Runtime rule registry with mandatory provenance metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from .config import SkillVISV2Config, project_root
from .contracts import ProvenanceRecord


class RuleDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str
    module: str
    source_ids: tuple[str, ...]
    scope: str
    rationale: str
    priority: int = Field(ge=1)
    failure_condition: str
    conflict_policy: str
    trace_fields: tuple[str, ...]
    unit_tests: tuple[str, ...]


class RuleRegistry:
    def __init__(self, path: Path, rules: Iterable[RuleDefinition]) -> None:
        self.path = path
        ordered = sorted(rules, key=lambda item: (item.priority, item.rule_id))
        self._rules = {rule.rule_id: rule for rule in ordered}
        if len(self._rules) != len(ordered):
            raise ValueError("duplicate rule_id in SkillVIS v2 registry")
        raw = json.dumps(
            [rule.model_dump(mode="json") for rule in ordered],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.sha256 = hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_config(cls, config: SkillVISV2Config) -> "RuleRegistry":
        path = project_root() / config.rule_registry_path
        payload = json.loads(path.read_text())
        if payload.get("status") not in {
            "m1_m3_implemented",
            "accuracy_repair_v2_implemented",
        }:
            raise ValueError("rule registry has an unsupported implementation status")
        rules = [
            RuleDefinition.model_validate(item) for item in payload["rules"]
        ]
        if config.safety_rule_registry_path:
            safety_path = project_root() / config.safety_rule_registry_path
            safety_payload = json.loads(safety_path.read_text())
            if safety_payload.get("status") not in {
                "m3_5_safety_implemented",
                "impact_aware_safety_v2_implemented",
            }:
                raise ValueError(
                    "safety rule registry is not marked m3_5_safety_implemented"
                )
            rules.extend(
                RuleDefinition.model_validate(item)
                for item in safety_payload["rules"]
            )
        if config.candidate_rule_registry_path:
            candidate_path = project_root() / config.candidate_rule_registry_path
            candidate_payload = json.loads(candidate_path.read_text())
            if candidate_payload.get("status") not in {
                "m4_candidate_planner_implemented",
                "derived_measure_planner_v2_implemented",
            }:
                raise ValueError(
                    "candidate rule registry is not marked "
                    "m4_candidate_planner_implemented"
                )
            rules.extend(
                RuleDefinition.model_validate(item)
                for item in candidate_payload["rules"]
            )
        return cls(path, rules)

    def get(self, rule_id: str) -> RuleDefinition:
        try:
            return self._rules[rule_id]
        except KeyError as exc:
            raise KeyError(f"unregistered SkillVIS v2 rule: {rule_id}") from exc

    def validate_ids(self, rule_ids: Iterable[str]) -> tuple[str, ...]:
        unique = tuple(sorted(set(rule_ids)))
        for rule_id in unique:
            self.get(rule_id)
        return unique

    def trace(
        self,
        stage: str,
        rule_id: str,
        *,
        inputs: Iterable[str],
        outputs: Iterable[str],
    ) -> ProvenanceRecord:
        rule = self.get(rule_id)
        return ProvenanceRecord(
            stage=stage,
            rule_id=rule.rule_id,
            source_ids=rule.source_ids,
            scope=rule.scope,
            rationale=rule.rationale,
            priority=rule.priority,
            failure_condition=rule.failure_condition,
            conflict_policy=rule.conflict_policy,
            trace_fields=rule.trace_fields,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
        )

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(self._rules)
