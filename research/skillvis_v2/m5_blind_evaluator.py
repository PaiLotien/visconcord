#!/usr/bin/env python3
"""Independent scorer for frozen M5 formal component outputs."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from skillvis_v2.contracts import VisualizationCandidate
from skillvis_v2.validator_formal import (
    CandidateSetContext,
    FormalCandidateDecision,
    FormalValidationSetResult,
    FormalValidatorConfig,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/skillvis_v2/m5_blind"
DATA = DATA_DIR / "m5_blind_candidate_set_v1.json"
GOLD = DATA_DIR / "m5_blind_adjudicated_gold_v1.json"
FREEZE = DATA_DIR / "m5_blind_freeze_manifest_v1.json"
CONFIG = ROOT / "configs/skillvis_v2/validator_formal_v1.json"

EXPECTED = {
    "config": "9bb8031a12d573377de0ca018bb3be590794606adacd559ec03a73de981d9509",
    "data": "7619520de1d0526c8c97bab59eccad72a5fe4bea8d4385a326fb291e95ef1ef0",
    "gold": "49e0de7885ffd374d9086302742c224d29cc7caa669ddcd984aae72e3f59c054",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("invalid Wilson count")
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "estimate": None,
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "method": "Wilson score",
        }
    if confidence != 0.95:
        raise ValueError("v1 freezes the Wilson interval at 95%")
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1 + (z * z / total)
    center = (estimate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            estimate * (1 - estimate) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "estimate": estimate,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "confidence": confidence,
        "method": "Wilson score",
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(
    values: list[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if not values:
        raise ValueError("bootstrap requires values")
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    return {
        "case_count": len(values),
        "estimate": sum(values) / len(values),
        "lower": _percentile(means, 0.025),
        "upper": _percentile(means, 0.975),
        "confidence": 0.95,
        "method": "case-level percentile bootstrap",
        "resamples": resamples,
        "seed": seed,
    }


def ndcg(order: list[str], relevance: dict[str, int]) -> float:
    if set(order) != set(relevance):
        raise ValueError("NDCG order/relevance identities differ")

    def dcg(items: list[str]) -> float:
        return sum(
            (2 ** relevance[item] - 1) / math.log2(index + 2)
            for index, item in enumerate(items)
        )

    ideal = sorted(order, key=lambda item: (-relevance[item], item))
    ideal_score = dcg(ideal)
    return dcg(order) / ideal_score if ideal_score else 1.0


def invalid_above_valid_inversions(
    order: list[str],
    relevance: dict[str, int],
) -> int:
    position = {candidate_id: index for index, candidate_id in enumerate(order)}
    invalid = [item for item, grade in relevance.items() if grade == 0]
    valid = [item for item, grade in relevance.items() if grade > 0]
    return sum(
        position[bad] < position[good] for bad in invalid for good in valid
    )


def validate_freeze_manifest() -> dict[str, Any]:
    if not FREEZE.exists():
        raise RuntimeError("M5 blind freeze manifest is missing")
    sidecar = FREEZE.with_suffix(FREEZE.suffix + ".sha256")
    if not sidecar.exists():
        raise RuntimeError("M5 blind freeze manifest sidecar is missing")
    expected_manifest_hash = sidecar.read_text(encoding="utf-8").split()[0]
    actual_manifest_hash = sha256_file(FREEZE)
    if actual_manifest_hash != expected_manifest_hash:
        raise RuntimeError("M5 blind freeze manifest sidecar mismatch")
    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_BEFORE_FIRST_M5_BLIND_EXECUTION":
        raise RuntimeError("M5 blind protocol is not frozen")
    errors = []
    for name, record in manifest["frozen_assets"].items():
        path = ROOT / record["path"]
        if not path.exists():
            errors.append({"asset": name, "error": "missing"})
            continue
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if (
            actual_hash != record["sha256"]
            or actual_size != record["size_bytes"]
        ):
            errors.append(
                {
                    "asset": name,
                    "error": "identity_mismatch",
                    "expected_hash": record["sha256"],
                    "actual_hash": actual_hash,
                    "expected_size": record["size_bytes"],
                    "actual_size": actual_size,
                }
            )
    if errors:
        raise RuntimeError(f"M5 blind frozen asset drift: {errors}")
    if sha256_file(CONFIG) != EXPECTED["config"]:
        raise RuntimeError("formal config identity drift")
    if sha256_file(DATA) != EXPECTED["data"]:
        raise RuntimeError("blind data identity drift")
    if sha256_file(GOLD) != EXPECTED["gold"]:
        raise RuntimeError("blind gold identity drift")
    return manifest


def _candidate_audit(
    result: Any,
    candidate: VisualizationCandidate,
    config: FormalValidatorConfig,
) -> dict[str, bool]:
    checks = {
        "candidate_binding": (
            result.candidate_id == candidate.candidate_id
            and result.candidate_signature == candidate.deterministic_signature
            and result.candidate_input_hash == candidate.canonical_sha256()
        ),
        "original_score_preserved": result.original_score == candidate.total_score,
        "original_rank_preserved": result.original_rank == candidate.rank,
        "score_overlay_valid": result.validated_score
        == round(
            max(0.0, result.original_score - result.validation_penalty), 6
        ),
        "trace_complete": (
            len(result.trace) == 18
            and len({trace.rule_id for trace in result.trace}) == 18
            and all(
                trace.source_ids
                and trace.evidence
                and trace.trace_signature
                for trace in result.trace
            )
        ),
        "issue_audit_complete": all(
            issue.violated_rule_ids
            and issue.evidence
            and issue.source_ids
            and issue.confidence > 0
            and issue.repair_proposal.rule_id
            in issue.violated_rule_ids
            and not issue.repair_applied
            and not issue.repair_proposal.applied
            for issue in result.issues
        ),
        "source_resolvable": all(
            source in config.source_registry
            for issue in result.issues
            for source in issue.source_ids
        ),
        "repair_unapplied": (
            not result.repair_applied
            and all(not proposal.applied for proposal in result.repair_proposal)
        ),
        "signature_complete": bool(result.deterministic_signature),
    }
    return checks


def evaluate_saved_outputs(output_root: Path) -> dict[str, Any]:
    freeze = validate_freeze_manifest()
    config = FormalValidatorConfig.model_validate(
        json.loads(CONFIG.read_text(encoding="utf-8"))
    )
    data = json.loads(DATA.read_text(encoding="utf-8"))
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    gold_by_case = {record["case_id"]: record for record in gold["records"]}
    config_rules = {rule.rule_id: rule for rule in config.rules}
    hard_rule_ids = {
        rule.rule_id
        for rule in config.rules
        if rule.allowed_action.value == "REJECT"
    }
    soft_rule_ids = set(config_rules) - hard_rule_ids

    hard_tp = hard_fn = 0
    soft_tp = soft_fn = soft_fp = 0
    hard_by_rule_expected: Counter[str] = Counter()
    hard_by_rule_detected: Counter[str] = Counter()
    false_safe = 0
    expected_rejectable = 0
    false_rejection = 0
    expected_nonrejectable = 0
    unsupported_expected = 0
    unsupported_pass_through = 0
    perceptual_hard_rejection = 0
    action_correct = 0
    candidate_total = 0
    audit_pass = audit_total = 0
    provenance_pass = provenance_total = 0
    deterministic_cases = 0
    byte_deterministic_cases = 0
    mutation_count = 0
    repair_tp = repair_fp = repair_fn = 0
    unsafe_repair = 0
    repair_applied_violations = 0
    decision_counts: Counter[str] = Counter()
    observed_rule_counts: Counter[str] = Counter()
    expected_rule_counts: Counter[str] = Counter()
    ranking_records = []
    ranking_deltas: list[float] = []
    error_records = []
    per_case = []

    for case in data["cases"]:
        case_id = case["case_id"]
        case_dir = output_root / "cases" / case_id
        context_path = case_dir / "context.json"
        repeat_paths = [
            case_dir / f"validation_repeat_{index}.json"
            for index in range(1, 4)
        ]
        process_path = case_dir / "process_log.json"
        if not context_path.exists() or not process_path.exists() or any(
            not path.exists() for path in repeat_paths
        ):
            raise RuntimeError(f"incomplete output artifacts for {case_id}")
        context_payload = json.loads(context_path.read_text(encoding="utf-8"))
        if context_payload != case["context"]:
            raise RuntimeError(f"saved context drift for {case_id}")
        context = CandidateSetContext.model_validate(context_payload)
        repeat_bytes = [path.read_bytes() for path in repeat_paths]
        repeat_payloads = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in repeat_paths
        ]
        repeats = [
            FormalValidationSetResult.model_validate(payload)
            for payload in repeat_payloads
        ]
        semantic_deterministic = len(
            {repeat.canonical_json() for repeat in repeats}
        ) == 1
        byte_deterministic = len(set(repeat_bytes)) == 1
        deterministic_cases += int(semantic_deterministic)
        byte_deterministic_cases += int(byte_deterministic)
        observed = repeats[0]
        mutation_count += observed.candidate_mutation_count
        result_by_candidate = {
            item.candidate_id: item for item in observed.candidate_results
        }
        candidate_by_id = {
            item.candidate_id: item for item in context.candidates
        }
        gold_record = gold_by_case[case_id]
        case_errors = []

        for expected in gold_record["candidate_gold"]:
            candidate_total += 1
            candidate_id = expected["candidate_id"]
            candidate_result = result_by_candidate[candidate_id]
            candidate = candidate_by_id[candidate_id]
            observed_rules = set(candidate_result.violated_rule_ids)
            expected_rules = set(expected["expected_rule_ids"])
            observed_hard = observed_rules & hard_rule_ids
            expected_hard = expected_rules & hard_rule_ids
            observed_soft = observed_rules & soft_rule_ids
            expected_soft = expected_rules & soft_rule_ids
            hard_tp += len(expected_hard & observed_hard)
            hard_fn += len(expected_hard - observed_hard)
            soft_tp += len(expected_soft & observed_soft)
            soft_fn += len(expected_soft - observed_soft)
            soft_fp += len(observed_soft - expected_soft)
            hard_by_rule_expected.update(expected_hard)
            hard_by_rule_detected.update(expected_hard & observed_hard)
            observed_rule_counts.update(observed_rules)
            expected_rule_counts.update(expected_rules)
            expected_reject = expected["expected_rejectability"]
            if expected_reject:
                expected_rejectable += 1
                false_safe += int(
                    candidate_result.decision
                    != FormalCandidateDecision.REJECT
                )
            else:
                expected_nonrejectable += 1
                false_rejection += int(
                    candidate_result.decision
                    == FormalCandidateDecision.REJECT
                )
            if expected_hard & {"R-VF-U001", "R-VF-U002"}:
                unsupported_expected += 1
                unsupported_pass_through += int(candidate_result.eligible)
            if (
                expected_soft
                and candidate_result.decision == FormalCandidateDecision.REJECT
                and not expected_hard
            ):
                perceptual_hard_rejection += 1
            action_ok = (
                candidate_result.decision.value in expected["allowed_actions"]
            )
            action_correct += int(action_ok)
            decision_counts.update((candidate_result.decision.value,))

            actual_repairs = [
                item.proposal_type for item in candidate_result.repair_proposal
            ]
            expected_repairs = expected["acceptable_repair_proposals"]
            actual_counter = Counter(actual_repairs)
            expected_counter = Counter(expected_repairs)
            repair_tp += sum(
                (actual_counter & expected_counter).values()
            )
            repair_fp += sum((actual_counter - expected_counter).values())
            repair_fn += sum((expected_counter - actual_counter).values())
            unsafe_repair += sum(
                proposal.rule_id not in observed_rules
                for proposal in candidate_result.repair_proposal
            )
            repair_applied_violations += int(candidate_result.repair_applied)
            repair_applied_violations += sum(
                proposal.applied for proposal in candidate_result.repair_proposal
            )

            checks = _candidate_audit(candidate_result, candidate, config)
            audit_pass += sum(checks.values())
            audit_total += len(checks)
            provenance_checks = {
                "candidate_sources": bool(candidate.source_ids),
                "candidate_evidence": bool(candidate.evidence_spans),
                "schema_provenance": bool(
                    context.schema_profile.applied_rule_ids
                )
                and all(
                    field.source_rule_ids
                    for field in context.schema_profile.fields
                ),
                "intent_provenance": bool(context.intent.provenance),
                "capability_provenance": bool(context.capability.provenance),
                "issue_sources": checks["source_resolvable"],
            }
            provenance_pass += sum(provenance_checks.values())
            provenance_total += len(provenance_checks)
            issue_exact = observed_rules == expected_rules
            if not issue_exact or not action_ok:
                details = {
                    "case_id": case_id,
                    "candidate_id": candidate_id,
                    "expected_rules": sorted(expected_rules),
                    "observed_rules": sorted(observed_rules),
                    "expected_actions": expected["allowed_actions"],
                    "observed_action": candidate_result.decision.value,
                }
                case_errors.append(details)
                error_records.append(details)

        expected_set_issues = set(gold_record["expected_set_issue_types"])
        observed_set_issues = {
            issue.issue_type for issue in observed.set_issues
        }
        if expected_set_issues != observed_set_issues:
            details = {
                "case_id": case_id,
                "scope": "candidate_set",
                "expected_issue_types": sorted(expected_set_issues),
                "observed_issue_types": sorted(observed_set_issues),
            }
            case_errors.append(details)
            error_records.append(details)

        if gold_record["ranking_relevance"]:
            relevance = {
                item["candidate_id"]: item["ranking_relevance"]
                for item in gold_record["candidate_gold"]
            }
            original = list(observed.original_candidate_order)
            validated = list(observed.validated_candidate_order)
            original_ndcg = ndcg(original, relevance)
            validated_ndcg = ndcg(validated, relevance)
            delta = validated_ndcg - original_ndcg
            ranking_deltas.append(delta)
            record = {
                "case_id": case_id,
                "relevance": relevance,
                "original_order": original,
                "validated_order": validated,
                "original_ndcg": original_ndcg,
                "validated_ndcg": validated_ndcg,
                "delta_ndcg": delta,
                "ranking_harm": delta < -1e-12,
                "original_invalid_above_valid_inversions": (
                    invalid_above_valid_inversions(original, relevance)
                ),
                "validated_invalid_above_valid_inversions": (
                    invalid_above_valid_inversions(validated, relevance)
                ),
                "original_top1_valid": relevance[original[0]] > 0,
                "validated_top1_valid": relevance[validated[0]] > 0,
            }
            ranking_records.append(record)
        per_case.append(
            {
                "case_id": case_id,
                "category": case["category"],
                "candidate_count": len(context.candidates),
                "semantic_deterministic": semantic_deterministic,
                "byte_deterministic": byte_deterministic,
                "candidate_mutation_count": observed.candidate_mutation_count,
                "set_decision": observed.overall_decision.value,
                "errors": case_errors,
            }
        )

    hard_total = hard_tp + hard_fn
    soft_expected_total = soft_tp + soft_fn
    soft_observed_total = soft_tp + soft_fp
    repair_total = repair_tp + repair_fp + repair_fn
    hard_family = {}
    for rule_id in sorted(hard_by_rule_expected):
        detected = hard_by_rule_detected[rule_id]
        expected_count = hard_by_rule_expected[rule_id]
        hard_family[rule_id] = wilson_interval(detected, expected_count)
    ranking_harm_count = sum(record["ranking_harm"] for record in ranking_records)
    original_inversions = sum(
        record["original_invalid_above_valid_inversions"]
        for record in ranking_records
    )
    validated_inversions = sum(
        record["validated_invalid_above_valid_inversions"]
        for record in ranking_records
    )
    delta_interval = bootstrap_mean_interval(
        ranking_deltas,
        resamples=int(config.statistics["bootstrap_resamples"]),
        seed=int(config.statistics["bootstrap_seed"]),
    )

    metrics = {
        "hard_error_detection_recall": wilson_interval(hard_tp, hard_total),
        "hard_family_recall": hard_family,
        "false_safe_rate": wilson_interval(false_safe, expected_rejectable),
        "false_rejection_rate": wilson_interval(
            false_rejection, expected_nonrejectable
        ),
        "unsupported_executable_pass_through": {
            **wilson_interval(
                unsupported_pass_through, unsupported_expected
            ),
            "count": unsupported_pass_through,
        },
        "soft_issue_recall": wilson_interval(
            soft_tp, soft_expected_total
        ),
        "soft_issue_precision": wilson_interval(
            soft_tp, soft_observed_total
        ),
        "action_correctness": wilson_interval(
            action_correct, candidate_total
        ),
        "perceptual_hard_rejection_count": perceptual_hard_rejection,
        "audit_completeness": wilson_interval(audit_pass, audit_total),
        "provenance_completeness": wilson_interval(
            provenance_pass, provenance_total
        ),
        "deterministic_replay": wilson_interval(
            deterministic_cases, len(data["cases"])
        ),
        "byte_deterministic_replay": wilson_interval(
            byte_deterministic_cases, len(data["cases"])
        ),
        "candidate_mutation_count": mutation_count,
        "repair_proposal_correctness": wilson_interval(
            repair_tp, repair_total
        ),
        "repair_counts": {
            "true_positive": repair_tp,
            "false_positive": repair_fp,
            "false_negative": repair_fn,
        },
        "unsafe_repair_proposal_count": unsafe_repair,
        "repair_applied_violation_count": repair_applied_violations,
        "decision_counts": dict(sorted(decision_counts.items())),
        "expected_rule_counts": dict(sorted(expected_rule_counts.items())),
        "observed_rule_counts": dict(sorted(observed_rule_counts.items())),
    }
    ranking = {
        "case_count": len(ranking_records),
        "mean_delta_ndcg": delta_interval,
        "ranking_harm_rate": wilson_interval(
            ranking_harm_count, len(ranking_records)
        ),
        "invalid_above_valid_inversions": {
            "original": original_inversions,
            "validated": validated_inversions,
            "delta": validated_inversions - original_inversions,
        },
        "top1_valid": {
            "original": sum(
                record["original_top1_valid"] for record in ranking_records
            ),
            "validated": sum(
                record["validated_top1_valid"] for record in ranking_records
            ),
            "total": len(ranking_records),
        },
        "per_case": ranking_records,
    }

    gates = config.gates
    family_minimum = min(
        (
            entry["estimate"]
            for entry in hard_family.values()
            if entry["estimate"] is not None
        ),
        default=1.0,
    )
    gate_results = {
        "hard_error_detection_recall": (
            metrics["hard_error_detection_recall"]["estimate"]
            >= gates["hard_error_detection_recall_min"]
        ),
        "hard_family_recall": family_minimum
        >= gates["hard_family_recall_min"],
        "false_safe_rate": metrics["false_safe_rate"]["estimate"]
        <= gates["false_safe_rate_max"],
        "false_rejection_rate": metrics["false_rejection_rate"]["estimate"]
        <= gates["false_rejection_rate_max"],
        "unsupported_executable_pass_through": unsupported_pass_through
        <= gates["unsupported_executable_pass_through_max"],
        "audit_completeness": metrics["audit_completeness"]["estimate"]
        >= gates["audit_completeness_min"],
        "provenance_completeness": metrics[
            "provenance_completeness"
        ]["estimate"]
        >= gates["provenance_completeness_min"],
        "deterministic_replay": metrics["deterministic_replay"]["estimate"]
        >= gates["deterministic_replay_min"],
        "candidate_mutation_count": mutation_count
        <= gates["candidate_mutation_count_max"],
        "ranking_harm_rate": ranking["ranking_harm_rate"]["estimate"]
        <= gates["ranking_harm_rate_max"],
        "mean_delta_ndcg": delta_interval["estimate"]
        >= gates["mean_delta_ndcg_min"],
        "repair_proposal_correctness": metrics[
            "repair_proposal_correctness"
        ]["estimate"]
        >= gates["repair_proposal_correctness_min"],
        "unsafe_repair_proposal_rate": (
            unsafe_repair / repair_total if repair_total else 0.0
        )
        <= gates["unsafe_repair_proposal_rate_max"],
    }
    structural_fail = any(
        (
            mutation_count > 0,
            repair_applied_violations > 0,
            perceptual_hard_rejection > 0,
        )
    )
    all_gates_pass = all(gate_results.values())
    independent_expert_gold = gold["annotation_workflow"][
        "independent_expert_gold_claim_allowed"
    ]
    if structural_fail:
        decision = "FAIL"
        reason = "Mutation/repair/perceptual permission boundary failed."
    elif not all_gates_pass:
        decision = "HOLD"
        reason = "One or more preregistered quantitative gates failed."
    elif not independent_expert_gold:
        decision = "PASS_WITH_LIMITATIONS"
        reason = (
            "All preregistered gates passed, but gold lacks independent "
            "human expert annotation; only pilot preparation is authorized."
        )
    else:
        decision = "PASS"
        reason = "All gates and independent expert-gold requirements passed."

    return {
        "schema_version": "skillvis-v2-m5-blind-evaluation-v1.0.0",
        "status": "COMPONENT_BLIND_VALIDATION_SCORED",
        "scope": {
            "component_level_only": True,
            "end_to_end_benchmark": False,
            "system_comparison": False,
            "llm_invoked": False,
            "paid_api_invoked": False,
            "independent_expert_gold": independent_expert_gold,
        },
        "input_identity": {
            "config_sha256": EXPECTED["config"],
            "data_sha256": EXPECTED["data"],
            "gold_sha256": EXPECTED["gold"],
            "freeze_manifest_sha256": sha256_file(FREEZE),
            "freeze_asset_count": len(freeze["frozen_assets"]),
        },
        "execution": {
            "case_count": len(data["cases"]),
            "candidate_count": candidate_total,
            "replays_per_case": 3,
            "validator_invocations": len(data["cases"]) * 3,
            "error_record_count": len(error_records),
        },
        "metrics": metrics,
        "ranking": ranking,
        "gates": {
            "results": gate_results,
            "passed": sum(gate_results.values()),
            "total": len(gate_results),
            "all_pass": all_gates_pass,
        },
        "transition": {
            "decision": decision,
            "reason": reason,
            "end_to_end_formal_benchmark_authorized": False,
            "end_to_end_pilot_preparation_authorized": (
                decision in {"PASS", "PASS_WITH_LIMITATIONS"}
            ),
        },
        "error_records": error_records,
        "per_case": per_case,
    }


__all__ = [
    "bootstrap_mean_interval",
    "evaluate_saved_outputs",
    "invalid_above_valid_inversions",
    "ndcg",
    "validate_freeze_manifest",
    "wilson_interval",
    "write_json",
]
