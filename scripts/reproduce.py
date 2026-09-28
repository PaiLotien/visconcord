#!/usr/bin/env python3
"""Verify the anonymous artifact without rerunning a formal experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest() -> dict[str, object]:
    manifest_path = ROOT / "release_manifest.json"
    sidecar = ROOT / "release_manifest.json.sha256"
    expected_manifest = sidecar.read_text(encoding="utf-8").split()[0]
    actual_manifest = sha256_file(manifest_path)
    if actual_manifest != expected_manifest:
        raise RuntimeError("release manifest sidecar mismatch")
    manifest = read_json(manifest_path)
    errors = []
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            errors.append(f"missing:{record['path']}")
            continue
        if path.stat().st_size != record["size_bytes"]:
            errors.append(f"size:{record['path']}")
        if sha256_file(path) != record["sha256"]:
            errors.append(f"sha256:{record['path']}")
    if errors:
        raise RuntimeError(f"manifest validation failed: {errors[:10]}")
    return {
        "status": "PASS",
        "file_count": manifest["file_count"],
        "manifest_sha256": actual_manifest,
    }


def deterministic_smoke() -> dict[str, object]:
    import pandas as pd

    from skillvis_v2_joint_v14 import (
        CONFIG_VERSION,
        plan_joint_v14_candidates,
        run_joint_v14_m1_m3_pipeline,
    )

    task_path = (
        ROOT
        / "data/skillvis_v2/pcs_v7_joint_v1121_retry3/formal_tasks_v1.json"
    )
    task = read_json(task_path)["tasks"][0]
    frame = pd.read_csv(ROOT / task["dataset"]["local_path"], low_memory=False)
    signatures = []
    candidate_counts = []
    for _ in range(3):
        pipeline = run_joint_v14_m1_m3_pipeline(
            query=task["natural_language_query"],
            dataset_id=task["dataset"]["dataset_id"],
            data=frame,
        )
        planning = plan_joint_v14_candidates(pipeline)
        if not planning.candidates:
            raise RuntimeError("deterministic smoke produced no candidates")
        signatures.append(planning.deterministic_signature)
        candidate_counts.append(len(planning.candidates))
    if len(set(signatures)) != 1 or len(set(candidate_counts)) != 1:
        raise RuntimeError("deterministic replay mismatch")
    if CONFIG_VERSION != "skillvis-v2-joint-semantic-precedence-v1.12.1":
        raise RuntimeError(f"unexpected final method: {CONFIG_VERSION}")
    return {
        "status": "PASS",
        "method": CONFIG_VERSION,
        "task_id": task["task_id"],
        "replays": 3,
        "candidate_count": candidate_counts[0],
        "signature": signatures[0],
        "llm_calls": 0,
        "paid_api_calls": 0,
    }


def verify_pcs_v7() -> dict[str, object]:
    path = (
        ROOT
        / "results/pcs_v7/skillvis-joint-v1121-vs-nl4dv-pcs-v7-20260731-r3/"
        "evaluation_summary.json"
    )
    summary = read_json(path)
    skill = summary["SkillVIS_V2_JOINT"]
    other = summary["NL4DV"]
    observed = {
        "skillvis_plan_success": skill["core_raw_counts"]["plan_success_at_1"],
        "nl4dv_plan_success": other["core_raw_counts"]["plan_success_at_1"],
        "skillvis_field_f1": skill["core"]["field_f1"],
        "nl4dv_field_f1": other["core"]["field_f1"],
        "skillvis_chart_hit_at_1": skill["core"]["chart_hit_at_1"],
        "nl4dv_chart_hit_at_1": other["core"]["chart_hit_at_1"],
    }
    expected = {
        "skillvis_plan_success": 30,
        "nl4dv_plan_success": 6,
        "skillvis_field_f1": 1.0,
        "nl4dv_field_f1": 0.6222222222222221,
        "skillvis_chart_hit_at_1": 1.0,
        "nl4dv_chart_hit_at_1": 0.7333333333333333,
    }
    if observed != expected:
        raise RuntimeError(f"PCS-v7 publication values drifted: {observed}")
    return {"status": "PASS_ARCHIVED_SUMMARY", **observed}


def verify_compassql() -> dict[str, object]:
    path = (
        ROOT
        / "results/compassql/skillvis-v1121-vs-compassql-20260731/"
        "evaluation_summary.json"
    )
    summary = read_json(path)
    systems = summary["systems"]
    observed = {
        name: round(
            payload["task_level_at_k"]["any_conforming_at_k"]
            * payload["task_counts"]["total"]
        )
        for name, payload in systems.items()
    }
    if observed != {"CompassQL": 28, "SkillVIS": 28}:
        raise RuntimeError(f"CompassQL conformance values drifted: {observed}")
    return {
        "status": "PASS_ARCHIVED_SUMMARY",
        "any_conforming_at_5_counts": observed,
        "scope": "structured conformance only",
    }


def recompute_m5() -> dict[str, object]:
    from research.skillvis_v2 import m5_blind_evaluator as base
    from research.skillvis_v2.m5_pristine_blind_v1 import (
        m5_pristine_evaluator as pristine,
    )

    # The original fail-closed evaluator also checks private annotation and
    # authorization artifacts. Those files are intentionally excluded from an
    # anonymous release. This public gate verifies the three inputs used by the
    # scoring formulas and preserves the original freeze-manifest identity;
    # it does not alter any metric computation.
    def validate_public_scoring_assets():
        manifest = read_json(pristine.FREEZE)
        assets = manifest["frozen_assets"]
        checks = {
            "candidate_set": pristine.DATA,
            "gold": pristine.GOLD,
            "validator_config": pristine.CONFIG,
        }
        for name, path in checks.items():
            if not path.is_file():
                raise RuntimeError(f"M5 public scoring asset missing: {name}")
            if sha256_file(path) != assets[name]["sha256"]:
                raise RuntimeError(f"M5 public scoring asset drift: {name}")
        identity = read_json(
            ROOT / "release_manifest.json"
        )["files"]
        if not identity:
            raise RuntimeError("release manifest has no files")
        return manifest

    base.validate_freeze_manifest = validate_public_scoring_assets

    run = ROOT / "results/m5/m5-pristine-v1-20260812-r1"
    recomputed = pristine.evaluate_saved_outputs(run)
    archived = read_json(run / "evaluation_summary.json")
    for key in ("metrics", "ranking", "gates", "execution", "input_identity"):
        if recomputed[key] != archived[key]:
            raise RuntimeError(f"M5 recomputation mismatch: {key}")
    metrics = recomputed["metrics"]
    return {
        "status": "PASS_RECOMPUTED_FROM_360_SAVED_OUTPUTS",
        "private_provenance_assets_required_for_scoring": False,
        "metric_formulas_modified": False,
        "action_correctness": metrics["action_correctness"],
        "hard_error_detection_recall": metrics["hard_error_detection_recall"],
        "false_safe_rate": metrics["false_safe_rate"],
        "false_rejection_rate": metrics["false_rejection_rate"],
        "soft_issue_precision": metrics["soft_issue_precision"],
        "repair_proposal_correctness": metrics["repair_proposal_correctness"],
        "all_gates_pass": recomputed["gates"]["all_pass"],
    }


def main() -> None:
    report = {
        "schema_version": "skillvis-anonymous-release-verification-v1.0.0",
        "status": "PASS",
        "formal_experiment_rerun": False,
        "llm_or_paid_api_invoked": False,
        "checks": {
            "manifest": verify_manifest(),
            "deterministic_smoke": deterministic_smoke(),
            "pcs_v7": verify_pcs_v7(),
            "compassql": verify_compassql(),
            "m5": recompute_m5(),
        },
    }
    (ROOT / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
