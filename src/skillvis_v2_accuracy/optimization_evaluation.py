"""Internal, split-aware evaluation for the SkillVIS v2 optimization revision.

This evaluator is deliberately separate from E-S1/EV2 and from every frozen
paper evaluator.  Development cases may be used for debugging; the holdout
split is intended for one execution after a revision is fixed.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .candidate_planner import plan_visualization_candidates
from .config import SkillVISV2Config, load_m4_config
from .contracts import VisualizationIntentIR
from .pipeline import run_m1_m3_pipeline
from .validator_formal import (
    CandidateSetContext,
    FormalCandidateDecision,
    candidate_set_signature,
    validate_candidate_set,
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_optimization_set(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _frame(dataset: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {field["field_name"]: field["values"] for field in dataset["schema"]}
    )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _case_prf(predicted: set[str], expected: set[str]) -> dict[str, float]:
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    return _prf(
        len(predicted & expected),
        len(predicted - expected),
        len(expected - predicted),
    )


def _transform_signature(transform: Any) -> tuple[str, str, tuple[str, ...]]:
    return (
        transform.transform_type,
        transform.operation,
        tuple(sorted(transform.field_candidates)),
    )


def _required_transform_found(
    observed: Iterable[Any],
    expected: dict[str, Any],
) -> bool:
    expected_fields = set(expected.get("fields", ()))
    for item in observed:
        if item.transform_type != expected["type"]:
            continue
        if item.operation != expected["operation"]:
            continue
        if expected_fields and expected_fields != set(item.field_candidates):
            continue
        if "value" in expected and expected["value"] != item.value:
            continue
        return True
    return False


def _validation_context(case_id: str, upstream: Any, planning: Any) -> CandidateSetContext:
    signature = candidate_set_signature(planning.candidates)
    return CandidateSetContext(
        context_id=f"v2-opt:{case_id}",
        candidate_set_id=f"v2-opt:{case_id}:{signature[:16]}",
        source_id="skillvis_v2_accuracy_optimization",
        schema=upstream.schema_profile,
        intent=upstream.intent_ir,
        capability=upstream.intent_ir.capability_assessment,
        candidates=planning.candidates,
        candidate_set_signature=signature,
    )


def _execute_case(
    case: dict[str, Any],
    dataset: dict[str, Any],
    config: SkillVISV2Config,
) -> tuple[Any, Any, Any | None]:
    upstream = run_m1_m3_pipeline(
        query=case["query"],
        dataset_id=dataset["dataset_id"],
        data=_frame(dataset),
        aliases=dataset.get("aliases") or {},
        config=config,
    )
    planning = plan_visualization_candidates(
        intent=upstream.intent_ir,
        capability=upstream.intent_ir.capability_assessment,
        schema=upstream.schema_profile,
        field_resolution=upstream.field_resolution,
        task_result=upstream.task_profile,
        config=config,
    )
    validation = None
    if planning.candidates:
        validation = validate_candidate_set(
            _validation_context(case["case_id"], upstream, planning)
        )
    return upstream, planning, validation


def evaluate_optimization_set(
    payload: dict[str, Any],
    *,
    config: SkillVISV2Config | None = None,
    replay_count: int = 3,
) -> dict[str, Any]:
    """Evaluate semantic, planning, safety, and Validator contract quality."""

    config = config or load_m4_config()
    datasets = {item["dataset_id"]: item for item in payload["datasets"]}
    field_tp = field_fp = field_fn = 0
    field_macro: list[dict[str, float]] = []
    task_correct = ambiguity_correct = abstention_correct = 0
    transform_found = transform_total = 0
    coverage = chart_top1 = chart_recall_k = 0
    false_reject = false_safe = 0
    deterministic = schema_valid = provenance_complete = 0
    latencies: list[float] = []
    error_taxonomy: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    per_case: list[dict[str, Any]] = []

    for case in payload["cases"]:
        dataset = datasets[case["dataset_id"]]
        expected = case["expected"]
        strata[case["stratum"]] += 1
        started = time.perf_counter()
        try:
            upstream, planning, validation = _execute_case(case, dataset, config)
            runtime_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:  # pragma: no cover - retained for audit output
            error_taxonomy["exception"] += 1
            per_case.append(
                {
                    "case_id": case["case_id"],
                    "stratum": case["stratum"],
                    "query": case["query"],
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        latencies.append(runtime_ms)

        predicted_fields = set(upstream.field_resolution.required_field_candidates)
        expected_fields = set(expected["required_fields"])
        field_case = _case_prf(predicted_fields, expected_fields)
        field_macro.append(field_case)
        field_tp += len(predicted_fields & expected_fields)
        field_fp += len(predicted_fields - expected_fields)
        field_fn += len(expected_fields - predicted_fields)

        task_ok = upstream.intent_ir.primary_task in set(
            expected["acceptable_primary_tasks"]
        )
        ambiguity_value = upstream.intent_ir.ambiguity_state.state.value
        ambiguity_ok = ambiguity_value in set(expected["acceptable_ambiguity_states"])
        abstention_value = upstream.intent_ir.abstention_state.abstained
        abstention_ok = abstention_value == expected["abstained"]
        task_correct += int(task_ok)
        ambiguity_correct += int(ambiguity_ok)
        abstention_correct += int(abstention_ok)

        required_transforms = expected.get("required_transforms", ())
        transform_results = [
            _required_transform_found(upstream.intent_ir.transforms, item)
            for item in required_transforms
        ]
        transform_found += sum(transform_results)
        transform_total += len(transform_results)

        candidate_types = [item.chart_type.value for item in planning.candidates]
        eligible_types: list[str] = []
        validator_decisions: dict[str, str] = {}
        if validation is not None:
            candidate_by_id = {
                item.candidate_id: item for item in planning.candidates
            }
            result_by_id = {
                item.candidate_id: item for item in validation.candidate_results
            }
            for candidate_id in validation.validated_candidate_order:
                result = result_by_id[candidate_id]
                validator_decisions[candidate_id] = result.decision.value
                if result.decision != FormalCandidateDecision.REJECT:
                    eligible_types.append(
                        candidate_by_id[candidate_id].chart_type.value
                    )

        acceptable_charts = set(expected.get("acceptable_chart_types", ()))
        answered = bool(eligible_types)
        if answered:
            coverage += 1
        if acceptable_charts and eligible_types:
            chart_top1 += int(eligible_types[0] in acceptable_charts)
            chart_recall_k += int(bool(set(eligible_types[:3]) & acceptable_charts))

        expects_candidate = (
            not expected["abstained"] and bool(acceptable_charts)
        )
        if expected["abstained"]:
            false_safe += int(answered)
        elif expects_candidate:
            false_reject += int(not answered)

        schema_ok = (
            VisualizationIntentIR.validate_serialized(
                upstream.intent_ir.model_dump(mode="json")
            ).canonical_json()
            == upstream.intent_ir.canonical_json()
        )
        schema_valid += int(schema_ok)
        upstream_rules = {
            *upstream.normalization.applied_rule_ids,
            *upstream.schema_profile.applied_rule_ids,
            *upstream.field_resolution.applied_rule_ids,
            *upstream.task_profile.applied_rule_ids,
            "R-MR-001",
            "R-MR-002",
        }
        provenance_rules = {item.rule_id for item in upstream.intent_ir.provenance}
        upstream_provenance_ok = upstream_rules <= provenance_rules
        planner_provenance_ok = all(
            set(candidate.applied_rule_ids)
            <= {item.rule_id for item in planning.provenance}
            for candidate in planning.candidates
        )
        validator_audit_ok = validation is None or all(
            result.trace and result.deterministic_signature
            for result in validation.candidate_results
        )
        provenance_ok = (
            upstream_provenance_ok and planner_provenance_ok and validator_audit_ok
        )
        provenance_complete += int(provenance_ok)

        signatures = {
            (
                upstream.intent_ir.deterministic_id,
                planning.deterministic_signature,
                validation.deterministic_signature if validation else None,
            )
        }
        for _ in range(max(1, replay_count) - 1):
            replay_upstream, replay_planning, replay_validation = _execute_case(
                case, dataset, config
            )
            signatures.add(
                (
                    replay_upstream.intent_ir.deterministic_id,
                    replay_planning.deterministic_signature,
                    replay_validation.deterministic_signature
                    if replay_validation
                    else None,
                )
            )
        deterministic_ok = len(signatures) == 1
        deterministic += int(deterministic_ok)

        failures: list[str] = []
        if field_case["f1"] < 1.0:
            failures.append("field_grounding")
        if not task_ok:
            failures.append("task_understanding")
        if not ambiguity_ok:
            failures.append("ambiguity_classification")
        if not abstention_ok:
            failures.append("safety_decision")
        if required_transforms and not all(transform_results):
            failures.append("transform_binding")
        if expects_candidate and not answered:
            failures.append("no_eligible_candidate")
        if acceptable_charts and (
            not eligible_types or eligible_types[0] not in acceptable_charts
        ):
            failures.append("candidate_ranking")
        for failure in failures:
            error_taxonomy[failure] += 1

        per_case.append(
            {
                "case_id": case["case_id"],
                "stratum": case["stratum"],
                "query": case["query"],
                "expected_fields": sorted(expected_fields),
                "predicted_fields": sorted(predicted_fields),
                "field_metrics": field_case,
                "expected_primary_tasks": expected["acceptable_primary_tasks"],
                "primary_task": upstream.intent_ir.primary_task,
                "task_correct": task_ok,
                "expected_ambiguity_states": expected[
                    "acceptable_ambiguity_states"
                ],
                "ambiguity_state": ambiguity_value,
                "ambiguity_correct": ambiguity_ok,
                "expected_abstained": expected["abstained"],
                "abstained": abstention_value,
                "abstention_reason": upstream.intent_ir.abstention_reason,
                "required_transform_results": transform_results,
                "observed_transforms": [
                    _transform_signature(item)
                    for item in upstream.intent_ir.transforms
                ],
                "candidate_types": candidate_types,
                "eligible_candidate_types": eligible_types,
                "validator_decisions": validator_decisions,
                "schema_valid": schema_ok,
                "provenance_complete": provenance_ok,
                "deterministic_replay": deterministic_ok,
                "runtime_ms": runtime_ms,
                "failures": failures,
            }
        )

    n = len(payload["cases"])
    completed = len(per_case) - error_taxonomy["exception"]
    candidate_required_n = sum(
        not item["expected"]["abstained"]
        and bool(item["expected"].get("acceptable_chart_types"))
        for item in payload["cases"]
    )
    unsafe_n = sum(item["expected"]["abstained"] for item in payload["cases"])
    chart_n = sum(
        bool(item["expected"].get("acceptable_chart_types"))
        and not item["expected"]["abstained"]
        for item in payload["cases"]
    )
    return {
        "schema_version": "skillvis-v2-optimization-evaluation-v1.0.0",
        "status": "internal_generalization_evidence_not_paper_benchmark",
        "set_version": payload["schema_version"],
        "split": payload["split"],
        "case_count": n,
        "completed": completed,
        "config_version": config.config_version,
        "strata": dict(sorted(strata.items())),
        "field": {
            "raw": {"tp": field_tp, "fp": field_fp, "fn": field_fn},
            "micro": _prf(field_tp, field_fp, field_fn),
            "macro": {
                key: statistics.mean(item[key] for item in field_macro)
                if field_macro
                else 0.0
                for key in ("precision", "recall", "f1")
            },
        },
        "primary_task_accuracy": task_correct / n if n else 0.0,
        "typed_ambiguity_accuracy": ambiguity_correct / n if n else 0.0,
        "abstention_correctness": abstention_correct / n if n else 0.0,
        "transform_recall": transform_found / transform_total
        if transform_total
        else None,
        "transform_raw": {"found": transform_found, "required": transform_total},
        "answerable_coverage": (
            candidate_required_n - false_reject
        ) / candidate_required_n
        if candidate_required_n
        else None,
        "false_reject_rate": false_reject / candidate_required_n
        if candidate_required_n
        else None,
        "false_safe_rate": false_safe / unsafe_n if unsafe_n else None,
        "chart_hit_at_1": chart_top1 / chart_n if chart_n else None,
        "acceptable_chart_recall_at_3": chart_recall_k / chart_n
        if chart_n
        else None,
        "schema_validation_rate": schema_valid / n if n else 0.0,
        "provenance_completeness_rate": provenance_complete / n if n else 0.0,
        "deterministic_replay_rate": deterministic / n if n else 0.0,
        "resource_observation": {
            "runtime_ms_mean_first_pass": statistics.mean(latencies)
            if latencies
            else None,
            "runtime_ms_median_first_pass": statistics.median(latencies)
            if latencies
            else None,
            "note": "In-process internal diagnostic; not comparable to E-S1 resources.",
        },
        "error_taxonomy": dict(sorted(error_taxonomy.items())),
        "per_case": per_case,
    }


def write_evaluation(result: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


__all__ = [
    "evaluate_optimization_set",
    "load_optimization_set",
    "sha256_file",
    "write_evaluation",
]
