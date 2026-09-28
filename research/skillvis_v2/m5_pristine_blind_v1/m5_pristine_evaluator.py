#!/usr/bin/env python3
"""Bind the frozen M5 evaluator formulas to pristine human-adjudicated gold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.skillvis_v2 import m5_blind_evaluator as base


ROOT = base.ROOT
DATA_DIR = ROOT / "data/skillvis_v2/m5_pristine_blind_v1"
DATA = DATA_DIR / "candidate_set_v1.json"
GOLD = DATA_DIR / "m5_pristine_adjudicated_gold_v1.json"
FREEZE = DATA_DIR / "m5_pristine_freeze_manifest_v1.json"
CONFIG = ROOT / "configs/skillvis_v2/validator_formal_v1.json"


def _bind() -> dict[str, Any]:
    if not FREEZE.exists():
        raise RuntimeError("pristine freeze manifest is missing")
    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    assets = manifest.get("frozen_assets", {})
    required = {"candidate_set", "gold", "validator_config"}
    if not required <= set(assets):
        raise RuntimeError("pristine freeze manifest lacks evaluator assets")
    base.DATA = DATA
    base.GOLD = GOLD
    base.FREEZE = FREEZE
    base.CONFIG = CONFIG
    base.EXPECTED = {
        "data": assets["candidate_set"]["sha256"],
        "gold": assets["gold"]["sha256"],
        "config": assets["validator_config"]["sha256"],
    }
    return manifest


def validate_freeze_manifest() -> dict[str, Any]:
    expected = _bind()
    manifest = base.validate_freeze_manifest()
    if manifest != expected:
        raise RuntimeError("freeze manifest changed while binding evaluator")
    if manifest.get("version") != "m5_pristine_blind_freeze_v1":
        raise RuntimeError("not an M5 pristine freeze")
    if manifest.get("unresolved_case_count") != 0:
        raise RuntimeError("pristine gold has unresolved cases")
    scope = manifest.get("evidence_scope", {})
    if not scope.get("pristine_validator_execution"):
        raise RuntimeError("pristine execution scope is not frozen")
    if not scope.get("component_level_only"):
        raise RuntimeError("M5 evaluator is component-level only")
    return manifest


def evaluate_saved_outputs(output_root: Path) -> dict[str, Any]:
    validate_freeze_manifest()
    result = base.evaluate_saved_outputs(output_root)
    result["schema_version"] = "skillvis-v2-m5-pristine-evaluation-v1.0.0"
    result["status"] = "PRISTINE_HUMAN_GOLD_COMPONENT_VALIDATION_SCORED"
    result["scope"].update(
        {
            "pristine_validator_execution": True,
            "template_family_reused": True,
            "standalone_end_to_end_evidence": False,
        }
    )
    return result


sha256_file = base.sha256_file
write_json = base.write_json


__all__ = [
    "CONFIG",
    "DATA",
    "FREEZE",
    "GOLD",
    "ROOT",
    "evaluate_saved_outputs",
    "sha256_file",
    "validate_freeze_manifest",
    "write_json",
]
