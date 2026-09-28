"""Deterministic, audit-first M5 Validator prototype.

The prototype validates one immutable ``VisualizationCandidate`` at a time. It
detects issues and derives a ranking overlay, but never changes, repairs, or
deletes the input candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from .contracts import CandidateChartType, FrozenContract, VisualizationCandidate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULE_REGISTRY_PATH = (
    ROOT / "data/skillvis_v2/m5_prototype/validator_rules_v0.json"
)


class ValidationSeverity(str, Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationRuleAction(str, Enum):
    WARN = "WARN"
    REJECT = "REJECT"


class ValidationDecision(str, Enum):
    ACCEPT = "ACCEPT"
    WARN = "WARN"
    REJECT = "REJECT"


class ValidationIssueType(str, Enum):
    FIELD_ROLE_MISMATCH = "FIELD_ROLE_MISMATCH"
    ENCODING_INCOMPATIBILITY = "ENCODING_INCOMPATIBILITY"
    TASK_CHART_MISMATCH = "TASK_CHART_MISMATCH"
    AGGREGATION_ERROR = "AGGREGATION_ERROR"
    UNSUPPORTED_DESIGN = "UNSUPPORTED_DESIGN"
    PERCEPTUAL_EFFECTIVENESS_WARNING = "PERCEPTUAL_EFFECTIVENESS_WARNING"
    HIGH_CARDINALITY_WARNING = "HIGH_CARDINALITY_WARNING"
    ENCODING_OVERLOAD_WARNING = "ENCODING_OVERLOAD_WARNING"


class ValidatorRule(FrozenContract):
    rule_id: str
    name: str
    source: tuple[str, ...]
    condition: str
    severity: ValidationSeverity
    action: ValidationRuleAction
    evidence: str
    example: str
    penalty: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_rule(self) -> "ValidatorRule":
        if not all(
            (
                self.rule_id,
                self.name,
                self.source,
                self.condition,
                self.evidence,
                self.example,
            )
        ):
            raise ValueError("validator rules require complete audit metadata")
        if self.action == ValidationRuleAction.WARN and (
            self.severity != ValidationSeverity.WARNING
        ):
            raise ValueError("WARN rules must use WARNING severity")
        if self.action == ValidationRuleAction.REJECT and (
            self.severity == ValidationSeverity.WARNING
        ):
            raise ValueError("REJECT rules cannot use WARNING severity")
        return self


class ValidatorRuleRegistry(FrozenContract):
    schema_version: str
    registry_version: str
    status: str
    rules: tuple[ValidatorRule, ...]

    @model_validator(mode="after")
    def validate_registry(self) -> "ValidatorRuleRegistry":
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("validator rule IDs must be unique")
        if len(ids) < 10:
            raise ValueError("M5 prototype requires at least ten rules")
        return self


class ValidationIssue(FrozenContract):
    issue_id: str
    candidate_id: str
    issue_type: ValidationIssueType
    severity: ValidationSeverity
    evidence: dict[str, Any]
    violated_rules: tuple[str, ...]
    source: tuple[str, ...]
    suggestion: str
    repair_action: None = None
    confidence: float = Field(ge=0.0, le=1.0)
    decision: ValidationRuleAction
    penalty: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_audit_fields(self) -> "ValidationIssue":
        if not self.evidence or not self.violated_rules or not self.source:
            raise ValueError("validation issues require evidence, rule, and source")
        if self.repair_action is not None:
            raise ValueError("the M5 prototype cannot emit automatic repairs")
        return self


class ValidationRuleTrace(FrozenContract):
    rule_id: str
    name: str
    source: tuple[str, ...]
    condition: str
    severity: ValidationSeverity
    action: ValidationRuleAction
    triggered: bool
    evidence: dict[str, Any]
    trace_signature: str


class CandidateValidationResult(FrozenContract):
    schema_version: str = "skillvis-v2-candidate-validation-result-v0.1.0"
    candidate_id: str
    candidate_signature: str
    candidate_input_hash: str
    registry_version: str
    registry_hash: str
    original_score: float = Field(ge=0.0, le=1.0)
    validation_penalty: float = Field(ge=0.0)
    validated_score: float = Field(ge=0.0, le=1.0)
    issues: tuple[ValidationIssue, ...]
    trace: tuple[ValidationRuleTrace, ...]
    rules_evaluated: int = Field(ge=0)
    decision: ValidationDecision
    eligible: bool
    deterministic_signature: str

    @model_validator(mode="after")
    def validate_decision_invariants(self) -> "CandidateValidationResult":
        if self.rules_evaluated != len(self.trace):
            raise ValueError("rules_evaluated must equal trace length")
        triggered_ids = {item.rule_id for item in self.trace if item.triggered}
        issue_rule_ids = {
            rule_id for issue in self.issues for rule_id in issue.violated_rules
        }
        if triggered_ids != issue_rule_ids:
            raise ValueError("triggered trace and issue rules must match")
        has_reject = any(
            issue.decision == ValidationRuleAction.REJECT for issue in self.issues
        )
        has_warning = any(
            issue.decision == ValidationRuleAction.WARN for issue in self.issues
        )
        expected_decision = (
            ValidationDecision.REJECT
            if has_reject
            else ValidationDecision.WARN
            if has_warning
            else ValidationDecision.ACCEPT
        )
        if self.decision != expected_decision:
            raise ValueError("result decision does not match emitted issues")
        if self.eligible != (self.decision != ValidationDecision.REJECT):
            raise ValueError("only REJECT results are ineligible")
        expected_score = round(
            max(0.0, self.original_score - self.validation_penalty), 6
        )
        if abs(self.validated_score - expected_score) > 1e-9:
            raise ValueError("validated_score must be a derived ranking overlay")
        if not self.deterministic_signature:
            raise ValueError("result deterministic signature is required")
        return self


_RULE_TO_ISSUE = {
    "R-VP-F001": ValidationIssueType.FIELD_ROLE_MISMATCH,
    "R-VP-F002": ValidationIssueType.FIELD_ROLE_MISMATCH,
    "R-VP-E001": ValidationIssueType.ENCODING_INCOMPATIBILITY,
    "R-VP-E002": ValidationIssueType.ENCODING_INCOMPATIBILITY,
    "R-VP-T001": ValidationIssueType.TASK_CHART_MISMATCH,
    "R-VP-A001": ValidationIssueType.AGGREGATION_ERROR,
    "R-VP-A002": ValidationIssueType.AGGREGATION_ERROR,
    "R-VP-U001": ValidationIssueType.UNSUPPORTED_DESIGN,
    "R-VP-P001": ValidationIssueType.PERCEPTUAL_EFFECTIVENESS_WARNING,
    "R-VP-P002": ValidationIssueType.HIGH_CARDINALITY_WARNING,
    "R-VP-P003": ValidationIssueType.ENCODING_OVERLOAD_WARNING,
}

_SUGGESTIONS = {
    ValidationIssueType.FIELD_ROLE_MISMATCH: (
        "Review the field-role assignment against the field semantic type."
    ),
    ValidationIssueType.ENCODING_INCOMPATIBILITY: (
        "Review the chart mark and channel semantic-type requirements."
    ),
    ValidationIssueType.TASK_CHART_MISMATCH: (
        "Review the candidate chart family against the analytical task."
    ),
    ValidationIssueType.AGGREGATION_ERROR: (
        "Review the declared aggregate, transform, and aggregate encodings."
    ),
    ValidationIssueType.UNSUPPORTED_DESIGN: (
        "Keep the candidate audit-only until the design becomes executable."
    ),
    ValidationIssueType.PERCEPTUAL_EFFECTIVENESS_WARNING: (
        "Consider a position- or length-based composition encoding."
    ),
    ValidationIssueType.HIGH_CARDINALITY_WARNING: (
        "Consider filtering, grouping, or a lower-cardinality encoding."
    ),
    ValidationIssueType.ENCODING_OVERLOAD_WARNING: (
        "Consider reducing simultaneous visual channels."
    ),
}

_EXPECTED_MARK = {
    CandidateChartType.BAR.value: "bar",
    CandidateChartType.GROUPED_BAR.value: "bar",
    CandidateChartType.DOT_PLOT.value: "point",
    CandidateChartType.HORIZONTAL_BAR.value: "bar",
    CandidateChartType.TOP_K_BAR.value: "bar",
    CandidateChartType.LINE.value: "line",
    CandidateChartType.MULTI_LINE.value: "line",
    CandidateChartType.SCATTER.value: "point",
    CandidateChartType.GROUPED_SCATTER.value: "point",
    CandidateChartType.HISTOGRAM.value: "bar",
    CandidateChartType.BOX_PLOT.value: "boxplot",
    CandidateChartType.STACKED_BAR.value: "bar",
    CandidateChartType.NORMALIZED_STACKED_BAR.value: "bar",
    CandidateChartType.PIE.value: "arc",
    CandidateChartType.RANKED_BAR.value: "bar",
    CandidateChartType.HIGHLIGHTED_EXTREMUM.value: "bar",
}

_TASK_CHARTS = {
    "comparison": {
        "bar",
        "grouped_bar",
        "dot_plot",
        "horizontal_bar",
        "stacked_bar",
        "normalized_stacked_bar",
    },
    "ranking": {"bar", "horizontal_bar", "ranked_bar", "top_k_bar"},
    "trend": {"line", "multi_line"},
    "correlation": {"scatter", "grouped_scatter"},
    "distribution": {"histogram", "box_plot"},
    "composition": {"stacked_bar", "normalized_stacked_bar", "pie"},
    "extremum": {"highlighted_extremum", "ranked_bar", "top_k_bar"},
    "aggregation": {"bar", "dot_plot", "line"},
    "filtering": set(_EXPECTED_MARK),
    "sorting": {"bar", "horizontal_bar", "ranked_bar", "top_k_bar"},
    "temporal_analysis": {"line", "multi_line"},
    "unresolved": set(),
}

_CARDINALITY_PATTERN = re.compile(
    r"^observed_cardinality:(?P<field>[^=]+)=(?P<count>\d+)$"
)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rule_registry(
    path: Path = DEFAULT_RULE_REGISTRY_PATH,
) -> tuple[ValidatorRuleRegistry, str]:
    """Load and validate the versioned rule registry with its file hash."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    registry = ValidatorRuleRegistry.model_validate(payload)
    known = {rule.rule_id for rule in registry.rules}
    if known != set(_RULE_TO_ISSUE):
        missing = sorted(set(_RULE_TO_ISSUE) - known)
        unexpected = sorted(known - set(_RULE_TO_ISSUE))
        raise ValueError(
            f"registry/implementation mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return registry, _sha256_file(path)


def _encoding_by_channel(
    candidate: VisualizationCandidate, channel: str
) -> Any | None:
    return next(
        (encoding for encoding in candidate.encodings if encoding.channel == channel),
        None,
    )


def _check_rule(
    rule_id: str, candidate: VisualizationCandidate
) -> dict[str, Any] | None:
    if rule_id == "R-VP-F001":
        mismatches = [
            {"field": encoding.field, "semantic_type": encoding.semantic_type}
            for encoding in candidate.encodings
            if encoding.field in candidate.fields.measure
            and encoding.semantic_type != "quantitative"
        ]
        return {"measure_role_mismatches": mismatches} if mismatches else None

    if rule_id == "R-VP-F002":
        mismatches = [
            {"field": encoding.field, "semantic_type": encoding.semantic_type}
            for encoding in candidate.encodings
            if encoding.field in candidate.fields.temporal
            and encoding.semantic_type != "temporal"
        ]
        return {"temporal_role_mismatches": mismatches} if mismatches else None

    if rule_id == "R-VP-E001":
        chart = candidate.chart_type.value
        expected = _EXPECTED_MARK[chart]
        if candidate.mark_type != expected:
            return {
                "chart_type": chart,
                "observed_mark": candidate.mark_type,
                "expected_mark": expected,
            }
        return None

    if rule_id == "R-VP-E002":
        if candidate.chart_type.value not in {"scatter", "grouped_scatter"}:
            return None
        axes = {
            channel: _encoding_by_channel(candidate, channel)
            for channel in ("x", "y")
        }
        bad_axes = {
            channel: (
                None
                if encoding is None
                else {
                    "field": encoding.field,
                    "semantic_type": encoding.semantic_type,
                }
            )
            for channel, encoding in axes.items()
            if encoding is None or encoding.semantic_type != "quantitative"
        }
        return {"invalid_axes": bad_axes} if bad_axes else None

    if rule_id == "R-VP-T001":
        chart = candidate.chart_type.value
        allowed = sorted(_TASK_CHARTS[candidate.task_alignment])
        if chart not in allowed:
            return {
                "task": candidate.task_alignment,
                "chart_type": chart,
                "allowed_chart_types": allowed,
            }
        return None

    if rule_id == "R-VP-A001":
        scoped_tasks = {"comparison", "ranking", "composition", "extremum"}
        if (
            candidate.task_alignment in scoped_tasks
            and len(candidate.fields.dimension) > 1
            and candidate.aggregation is None
        ):
            return {
                "task": candidate.task_alignment,
                "dimensions": list(candidate.fields.dimension),
                "aggregation": None,
            }
        return None

    if rule_id == "R-VP-A002":
        aggregation = candidate.aggregation
        if aggregation is None:
            return None
        matching_transforms = [
            transform
            for transform in candidate.transforms
            if transform.transform_type == "aggregate"
            and transform.operation == aggregation.operation
            and transform.field_candidates == aggregation.field_candidates
        ]
        encoding_conflicts = [
            {
                "channel": encoding.channel,
                "field": encoding.field,
                "encoding_aggregate": encoding.aggregate,
                "declared_aggregate": aggregation.operation,
            }
            for encoding in candidate.encodings
            if encoding.aggregate is not None
            and encoding.aggregate != aggregation.operation
        ]
        if len(matching_transforms) != 1 or encoding_conflicts:
            return {
                "declared_aggregation": aggregation.model_dump(mode="json"),
                "matching_transform_count": len(matching_transforms),
                "encoding_conflicts": encoding_conflicts,
            }
        return None

    if rule_id == "R-VP-U001":
        if (
            not candidate.executable
            or candidate.audit_only
            or candidate.unsupported_features
            or candidate.rejection_reasons
        ):
            return {
                "executable": candidate.executable,
                "audit_only": candidate.audit_only,
                "unsupported_features": list(candidate.unsupported_features),
                "rejection_reasons": list(candidate.rejection_reasons),
            }
        return None

    if rule_id == "R-VP-P001":
        if candidate.chart_type.value == "pie":
            return {
                "chart_type": "pie",
                "perceptual_channels": ["angle", "area"],
            }
        return None

    if rule_id == "R-VP-P002":
        cardinalities: dict[str, int] = {}
        for assumption in candidate.chart_compatibility_assumptions:
            match = _CARDINALITY_PATTERN.fullmatch(assumption)
            if match:
                cardinalities[match.group("field")] = int(match.group("count"))
        violations = []
        thresholds = {"color": 12, "row": 8, "column": 8}
        if candidate.chart_type.value == "pie":
            thresholds["color"] = 6
        for encoding in candidate.encodings:
            if (
                encoding.field in cardinalities
                and encoding.channel in thresholds
                and cardinalities[encoding.field] > thresholds[encoding.channel]
            ):
                violations.append(
                    {
                        "field": encoding.field,
                        "channel": encoding.channel,
                        "observed_cardinality": cardinalities[encoding.field],
                        "threshold": thresholds[encoding.channel],
                    }
                )
        return {"high_cardinality_encodings": violations} if violations else None

    if rule_id == "R-VP-P003":
        auxiliary = [
            encoding.channel
            for encoding in candidate.encodings
            if encoding.channel in {"color", "size", "row", "column", "detail"}
        ]
        if len(candidate.encodings) > 3 or len(auxiliary) > 2:
            return {
                "encoding_count": len(candidate.encodings),
                "auxiliary_channels": auxiliary,
            }
        return None

    raise KeyError(f"no validator implementation for {rule_id}")


def _issue_for(
    candidate: VisualizationCandidate,
    rule: ValidatorRule,
    evidence: dict[str, Any],
) -> ValidationIssue:
    issue_type = _RULE_TO_ISSUE[rule.rule_id]
    issue_seed = {
        "candidate_id": candidate.candidate_id,
        "rule_id": rule.rule_id,
        "evidence": evidence,
    }
    return ValidationIssue(
        issue_id=f"VI-{_canonical_hash(issue_seed)[:16]}",
        candidate_id=candidate.candidate_id,
        issue_type=issue_type,
        severity=rule.severity,
        evidence=evidence,
        violated_rules=(rule.rule_id,),
        source=rule.source,
        suggestion=_SUGGESTIONS[issue_type],
        repair_action=None,
        confidence=1.0,
        decision=rule.action,
        penalty=rule.penalty,
    )


def validate_candidate(
    candidate: VisualizationCandidate,
    *,
    registry: ValidatorRuleRegistry | None = None,
    registry_hash: str | None = None,
) -> CandidateValidationResult:
    """Validate without mutating ``candidate`` and preserve a full rule trace."""

    if (registry is None) != (registry_hash is None):
        raise ValueError("registry and registry_hash must be supplied together")
    if registry is None:
        registry, registry_hash = load_rule_registry()
    assert registry_hash is not None

    candidate_hash = candidate.canonical_sha256()
    issues: list[ValidationIssue] = []
    traces: list[ValidationRuleTrace] = []
    for rule in registry.rules:
        evidence = _check_rule(rule.rule_id, candidate)
        triggered = evidence is not None
        evidence_payload = evidence or {"status": "not_triggered"}
        trace_seed = {
            "candidate_id": candidate.candidate_id,
            "rule_id": rule.rule_id,
            "triggered": triggered,
            "evidence": evidence_payload,
        }
        traces.append(
            ValidationRuleTrace(
                rule_id=rule.rule_id,
                name=rule.name,
                source=rule.source,
                condition=rule.condition,
                severity=rule.severity,
                action=rule.action,
                triggered=triggered,
                evidence=evidence_payload,
                trace_signature=_canonical_hash(trace_seed),
            )
        )
        if triggered:
            issues.append(_issue_for(candidate, rule, evidence_payload))

    has_reject = any(
        issue.decision == ValidationRuleAction.REJECT for issue in issues
    )
    has_warning = any(
        issue.decision == ValidationRuleAction.WARN for issue in issues
    )
    decision = (
        ValidationDecision.REJECT
        if has_reject
        else ValidationDecision.WARN
        if has_warning
        else ValidationDecision.ACCEPT
    )
    penalty = round(sum(issue.penalty for issue in issues), 6)
    validated_score = round(max(0.0, candidate.total_score - penalty), 6)
    draft = {
        "candidate_id": candidate.candidate_id,
        "candidate_signature": candidate.deterministic_signature,
        "candidate_input_hash": candidate_hash,
        "registry_version": registry.registry_version,
        "registry_hash": registry_hash,
        "original_score": candidate.total_score,
        "validation_penalty": penalty,
        "validated_score": validated_score,
        "issues": [issue.model_dump(mode="json") for issue in issues],
        "trace": [trace.model_dump(mode="json") for trace in traces],
        "decision": decision.value,
        "eligible": decision != ValidationDecision.REJECT,
    }
    return CandidateValidationResult(
        candidate_id=candidate.candidate_id,
        candidate_signature=candidate.deterministic_signature,
        candidate_input_hash=candidate_hash,
        registry_version=registry.registry_version,
        registry_hash=registry_hash,
        original_score=candidate.total_score,
        validation_penalty=penalty,
        validated_score=validated_score,
        issues=tuple(issues),
        trace=tuple(traces),
        rules_evaluated=len(traces),
        decision=decision,
        eligible=decision != ValidationDecision.REJECT,
        deterministic_signature=_canonical_hash(draft),
    )


__all__ = [
    "CandidateValidationResult",
    "ValidationDecision",
    "ValidationIssue",
    "ValidationIssueType",
    "ValidationRuleAction",
    "ValidationRuleTrace",
    "ValidationSeverity",
    "ValidatorRule",
    "ValidatorRuleRegistry",
    "load_rule_registry",
    "validate_candidate",
]
