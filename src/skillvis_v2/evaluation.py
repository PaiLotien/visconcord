"""Development-only and diagnostic-only evaluation for M1–M3 readiness.

This module is deliberately separate from the frozen E-S1 evaluator.
"""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

from .config import SkillVISV2Config, load_config
from .contracts import M1M3PipelineResult, VisualizationIntentIR
from .pipeline import run_m1_m3_pipeline
from .rules import RuleRegistry


def dataframe_from_development_case(case: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {field["field_name"]: field["values"] for field in case["schema"]}
    )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _case_prf(predicted: set[str], expected: set[str]) -> dict[str, float]:
    # An intentionally empty prediction is fully correct when the case expects
    # safe abstention. Treating empty/empty as zero would make the macro score
    # contradict the raw counts and penalize the required failure behavior.
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    return _prf(
        len(predicted & expected),
        len(predicted - expected),
        len(expected - predicted),
    )


def _provenance_complete(
    result: M1M3PipelineResult,
    registry: RuleRegistry,
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    applied = {
        *result.normalization.applied_rule_ids,
        *result.schema_profile.applied_rule_ids,
        *result.field_resolution.applied_rule_ids,
        *result.task_profile.applied_rule_ids,
        "R-MR-001",
        "R-MR-002",
    }
    provenance_ids = {item.rule_id for item in result.intent_ir.provenance}
    if applied != provenance_ids:
        problems.append(
            f"provenance rule mismatch missing={sorted(applied-provenance_ids)} "
            f"extra={sorted(provenance_ids-applied)}"
        )
    for rule_id in applied:
        registry.get(rule_id)
    for candidate in result.field_resolution.candidate_fields:
        if not candidate.evidence or not candidate.source_rule_ids:
            problems.append(f"candidate {candidate.field_name} lacks evidence/provenance")
        if not candidate.mention_ids:
            problems.append(f"candidate {candidate.field_name} lacks query mention")
    try:
        VisualizationIntentIR.validate_serialized(
            json.loads(result.intent_ir.canonical_json())
        )
    except Exception as exc:  # pragma: no cover - defensive reporting
        problems.append(f"IR schema validation failed: {type(exc).__name__}: {exc}")
    return not problems, problems


def evaluate_development_set(
    payload: dict[str, Any],
    config: SkillVISV2Config | None = None,
    *,
    replay_count: int = 2,
) -> dict[str, Any]:
    """Evaluate only the declared development invariants and module metrics."""

    config = config or load_config()
    registry = RuleRegistry.from_config(config)
    field_tp = field_fp = field_fn = 0
    task_tp = task_fp = task_fn = 0
    field_macro = []
    task_macro = []
    topk_hits = {1: [], 3: [], 5: []}
    ambiguity_tp = ambiguity_fp = ambiguity_fn = ambiguity_tn = 0
    abstention_correct = 0
    primary_correct = 0
    aggregation_correct = aggregation_total = 0
    sort_correct = sort_total = 0
    limit_correct = limit_total = 0
    provenance_pass = 0
    deterministic_pass = 0
    schema_pass = 0
    latencies_ms: list[float] = []
    per_case = []

    tracemalloc.start()
    for case in payload["cases"]:
        df = dataframe_from_development_case(case)
        start = time.perf_counter()
        result = run_m1_m3_pipeline(
            query=case["query"],
            dataset_id=case["dataset_id"],
            data=df,
            aliases=case.get("aliases") or {},
            config=config,
        )
        elapsed = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed)
        expected = case["expected_intent"]
        expected_fields = set(expected["required_fields"])
        predicted_fields = set(result.field_resolution.required_field_candidates)
        fmetrics = _case_prf(predicted_fields, expected_fields)
        field_macro.append(fmetrics)
        field_tp += len(predicted_fields & expected_fields)
        field_fp += len(predicted_fields - expected_fields)
        field_fn += len(expected_fields - predicted_fields)

        ranked_fields = [
            candidate.field_name
            for candidate in result.field_resolution.candidate_fields
        ]
        for k in topk_hits:
            if not expected_fields:
                topk_hits[k].append(1.0 if not ranked_fields else 0.0)
            else:
                topk_hits[k].append(
                    len(set(ranked_fields[:k]) & expected_fields) / len(expected_fields)
                )

        expected_tasks = set(expected["acceptable_tasks"])
        predicted_tasks = set(result.task_profile.task_scores)
        tmetrics = _case_prf(predicted_tasks, expected_tasks)
        task_macro.append(tmetrics)
        task_tp += len(predicted_tasks & expected_tasks)
        task_fp += len(predicted_tasks - expected_tasks)
        task_fn += len(expected_tasks - predicted_tasks)
        primary_ok = result.task_profile.primary_task in set(
            expected["acceptable_primary_tasks"]
        )
        primary_correct += int(primary_ok)

        ambiguity_expected = bool(expected["ambiguity_expected"])
        ambiguity_predicted = bool(result.intent_ir.ambiguity)
        if ambiguity_expected and ambiguity_predicted:
            ambiguity_tp += 1
        elif ambiguity_expected:
            ambiguity_fn += 1
        elif ambiguity_predicted:
            ambiguity_fp += 1
        else:
            ambiguity_tn += 1

        abstention_predicted = result.intent_ir.abstention_state.abstained
        abstention_expected = bool(expected["abstention_expected"])
        abstention_ok = abstention_predicted == abstention_expected
        abstention_correct += int(abstention_ok)

        aggregation_observed = (
            result.task_profile.aggregation.operation
            if result.task_profile.aggregation
            else None
        )
        if expected["aggregation"] is not None:
            aggregation_total += 1
            aggregation_correct += int(aggregation_observed == expected["aggregation"])
        sort_observed = (
            result.task_profile.sort.operation if result.task_profile.sort else None
        )
        if expected["sort"] is not None:
            sort_total += 1
            sort_correct += int(sort_observed == expected["sort"])
        if expected["limit"] is not None:
            limit_total += 1
            limit_correct += int(result.task_profile.limit == expected["limit"])

        provenance_ok, provenance_problems = _provenance_complete(result, registry)
        provenance_pass += int(provenance_ok)
        schema_ok = (
            VisualizationIntentIR.validate_serialized(
                result.intent_ir.model_dump(mode="json")
            ).canonical_json()
            == result.intent_ir.canonical_json()
        )
        schema_pass += int(schema_ok)
        replay_ids = {result.intent_ir.deterministic_id}
        replay_json = {result.intent_ir.canonical_json()}
        for _ in range(max(1, replay_count) - 1):
            replay = run_m1_m3_pipeline(
                query=case["query"],
                dataset_id=case["dataset_id"],
                data=df,
                aliases=case.get("aliases") or {},
                config=config,
            )
            replay_ids.add(replay.intent_ir.deterministic_id)
            replay_json.add(replay.intent_ir.canonical_json())
        deterministic_ok = len(replay_ids) == 1 and len(replay_json) == 1
        deterministic_pass += int(deterministic_ok)

        per_case.append(
            {
                "task_id": case["task_id"],
                "query": case["query"],
                "expected_fields": sorted(expected_fields),
                "predicted_required_fields": sorted(predicted_fields),
                "ranked_field_candidates": ranked_fields,
                "field_metrics": fmetrics,
                "expected_tasks": sorted(expected_tasks),
                "predicted_tasks": sorted(predicted_tasks),
                "primary_task": result.task_profile.primary_task,
                "primary_task_acceptable": primary_ok,
                "ambiguity_expected": ambiguity_expected,
                "ambiguity_predicted": ambiguity_predicted,
                "abstention_expected": abstention_expected,
                "abstention_predicted": abstention_predicted,
                "aggregation_expected": expected["aggregation"],
                "aggregation_predicted": aggregation_observed,
                "sort_expected": expected["sort"],
                "sort_predicted": sort_observed,
                "limit_expected": expected["limit"],
                "limit_predicted": result.task_profile.limit,
                "provenance_complete": provenance_ok,
                "provenance_problems": provenance_problems,
                "schema_valid": schema_ok,
                "deterministic_replay": deterministic_ok,
                "latency_ms": elapsed,
                "intent_id": result.intent_ir.deterministic_id,
            }
        )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n = len(payload["cases"])
    ambiguity_metrics = _prf(ambiguity_tp, ambiguity_fp, ambiguity_fn)
    return {
        "schema_version": "skillvis-v2-development-metrics-v0.1.0",
        "status": "development_only_not_formal_evidence",
        "development_set_version": payload["schema_version"],
        "development_set_hash": payload["content_sha256_without_self_field"],
        "config_version": config.config_version,
        "registry_hash": registry.sha256,
        "case_count": n,
        "field": {
            "raw": {"tp": field_tp, "fp": field_fp, "fn": field_fn},
            "micro": _prf(field_tp, field_fp, field_fn),
            "macro": {
                key: statistics.mean(item[key] for item in field_macro)
                for key in ("precision", "recall", "f1")
            },
            "candidate_recall_at_k": {
                str(k): statistics.mean(values) for k, values in topk_hits.items()
            },
        },
        "task": {
            "raw": {"tp": task_tp, "fp": task_fp, "fn": task_fn},
            "micro": _prf(task_tp, task_fp, task_fn),
            "macro": {
                key: statistics.mean(item[key] for item in task_macro)
                for key in ("precision", "recall", "f1")
            },
            "primary_acceptable_rate": primary_correct / n if n else 0.0,
        },
        "ambiguity": {
            "raw": {
                "tp": ambiguity_tp,
                "fp": ambiguity_fp,
                "fn": ambiguity_fn,
                "tn": ambiguity_tn,
            },
            **ambiguity_metrics,
            "accuracy": (ambiguity_tp + ambiguity_tn) / n if n else 0.0,
        },
        "abstention_correctness": abstention_correct / n if n else 0.0,
        "transforms": {
            "aggregation_accuracy": aggregation_correct / aggregation_total
            if aggregation_total
            else None,
            "aggregation_n": aggregation_total,
            "sort_accuracy": sort_correct / sort_total if sort_total else None,
            "sort_n": sort_total,
            "limit_accuracy": limit_correct / limit_total if limit_total else None,
            "limit_n": limit_total,
        },
        "provenance_completeness": provenance_pass / n if n else 0.0,
        "schema_validation_rate": schema_pass / n if n else 0.0,
        "deterministic_replay_rate": deterministic_pass / n if n else 0.0,
        "resource_observations": {
            "latency_ms_mean": statistics.mean(latencies_ms) if latencies_ms else None,
            "latency_ms_median": statistics.median(latencies_ms) if latencies_ms else None,
            "latency_ms_p95": sorted(latencies_ms)[
                min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))
            ]
            if latencies_ms
            else None,
            "python_peak_alloc_mb_for_batch": peak_bytes / (1024 * 1024),
            "measurement_note": (
                "In-process development observation using perf_counter and tracemalloc; "
                "not comparable to frozen E-S1 fresh-process RSS."
            ),
        },
        "per_case": per_case,
    }


def load_development_set(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
