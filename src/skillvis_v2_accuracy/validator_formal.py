"""Context-aware, deterministic M5 formal Validator.

This module consumes only frozen public contracts. It does not reparse the
query, call M4 private templates, mutate candidates, or apply repair proposals.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .contracts import (
    CapabilityAssessment,
    CandidateTransform,
    FrozenContract,
    SchemaProfile,
    VisualizationCandidate,
    VisualizationIntentIR,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL_CONFIG_PATH = (
    ROOT / "configs/skillvis_v2/validator_formal_v1.json"
)


class FormalSeverity(str, Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class FormalRuleAction(str, Enum):
    WARN = "WARN"
    DEMOTE = "DEMOTE"
    REJECT = "REJECT"


class FormalCandidateDecision(str, Enum):
    ACCEPT = "ACCEPT"
    WARN = "WARN"
    DEMOTE = "DEMOTE"
    REJECT = "REJECT"


class FormalSetDecision(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    ANSWERABLE_WITH_WARNINGS = "ANSWERABLE_WITH_WARNINGS"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"


class FormalValidatorRule(FrozenContract):
    rule_id: str
    source_id: tuple[str, ...]
    level: Literal["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
    issue_type: str
    scope: str
    precondition: str
    evidence_requirement: str
    severity: FormalSeverity
    allowed_action: FormalRuleAction
    penalty_key: str
    repair_proposal: str
    repair_target_stage: str
    failure_example: str
    limitation: str

    @model_validator(mode="after")
    def validate_permission(self) -> "FormalValidatorRule":
        if self.severity == FormalSeverity.WARNING:
            if self.allowed_action == FormalRuleAction.REJECT:
                raise ValueError("WARNING rule cannot REJECT")
        elif self.allowed_action != FormalRuleAction.REJECT:
            raise ValueError("ERROR/CRITICAL formal rules must REJECT")
        if not self.source_id:
            raise ValueError("formal rule requires source provenance")
        return self


class FormalValidatorConfig(FrozenContract):
    schema_version: str
    config_version: str
    status: str
    authorized_scope: dict[str, Any]
    source_registry: dict[str, str]
    severity_taxonomy: dict[str, str]
    permission_matrix: dict[str, Any]
    field_role_matrix: dict[str, tuple[str, ...]]
    encoding_type_matrix: dict[str, tuple[str, ...]]
    task_chart_compatibility: dict[str, tuple[str, ...]]
    chart_mark_compatibility: dict[str, str]
    required_channels: dict[str, tuple[str, ...]]
    aggregation_data_grain_policy: dict[str, Any]
    cardinality_thresholds: dict[str, Any]
    penalty_weights: dict[str, float]
    tie_breaking_policy: dict[str, Any]
    ranking_relevance: dict[str, Any]
    rules: tuple[FormalValidatorRule, ...]
    gates: dict[str, float]
    statistics: dict[str, Any]
    transition_policy: dict[str, str]

    @model_validator(mode="after")
    def validate_registry(self) -> "FormalValidatorConfig":
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("formal Validator rule IDs must be unique")
        if self.status != "FROZEN_BEFORE_M5_BLIND_EXECUTION":
            raise ValueError("formal Validator config must be frozen")
        if self.permission_matrix.get("perceptual_hard_reject_forbidden") is not True:
            raise ValueError("perceptual hard rejection must be forbidden")
        for rule in self.rules:
            if rule.penalty_key not in self.penalty_weights:
                raise ValueError(f"unknown penalty key {rule.penalty_key}")
            missing_sources = set(rule.source_id) - set(self.source_registry)
            if missing_sources:
                raise ValueError(
                    f"unregistered sources for {rule.rule_id}: "
                    f"{sorted(missing_sources)}"
                )
        return self

    def rule(self, rule_id: str) -> FormalValidatorRule:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        raise KeyError(rule_id)


class CandidateSetContext(FrozenContract):
    schema_version: str = "skillvis-v2-m5-candidate-set-context-v1.0.0"
    context_id: str
    candidate_set_id: str
    source_id: str
    schema_profile: SchemaProfile = Field(alias="schema")
    intent: VisualizationIntentIR
    capability: CapabilityAssessment
    candidates: tuple[VisualizationCandidate, ...]
    candidate_set_signature: str

    @model_validator(mode="after")
    def validate_minimum_contract(self) -> "CandidateSetContext":
        if not self.candidates:
            raise ValueError("formal validation context requires candidates")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        ranks = [candidate.rank for candidate in self.candidates]
        if len(ranks) != len(set(ranks)) or min(ranks) < 1:
            raise ValueError("candidate original ranks must be unique and positive")
        if not self.candidate_set_signature:
            raise ValueError("candidate set signature is required")
        return self


class RepairProposal(FrozenContract):
    proposal_type: str
    target_stage: str
    rule_id: str
    description: str
    applied: Literal[False] = False


class FormalValidationIssue(FrozenContract):
    issue_id: str
    scope: Literal["context", "candidate", "candidate_set"]
    candidate_id: str | None
    issue_type: str
    level: Literal["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
    severity: FormalSeverity
    violated_rule_ids: tuple[str, ...]
    evidence: dict[str, Any]
    source_ids: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    decision: FormalRuleAction
    repair_proposal: RepairProposal
    repair_applied: Literal[False] = False
    penalty: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_issue(self) -> "FormalValidationIssue":
        if not self.violated_rule_ids or not self.evidence or not self.source_ids:
            raise ValueError("formal issues require rule, evidence, and source")
        if self.repair_applied or self.repair_proposal.applied:
            raise ValueError("M5 formal Validator cannot apply repairs")
        if self.severity == FormalSeverity.WARNING:
            if self.decision == FormalRuleAction.REJECT:
                raise ValueError("perceptual/set WARNING cannot hard reject")
        elif self.decision != FormalRuleAction.REJECT:
            raise ValueError("hard issue must use REJECT")
        return self


class FormalRuleTrace(FrozenContract):
    rule_id: str
    scope: Literal["candidate", "candidate_set"]
    candidate_id: str | None
    triggered: bool
    evidence: dict[str, Any]
    source_ids: tuple[str, ...]
    trace_signature: str


class FormalCandidateValidationResult(FrozenContract):
    schema_version: str = "skillvis-v2-m5-candidate-result-v1.0.0"
    candidate_id: str
    candidate_signature: str
    candidate_input_hash: str
    issue_type: tuple[str, ...]
    severity: tuple[FormalSeverity, ...]
    violated_rule_ids: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    source_ids: tuple[str, ...]
    confidence: tuple[float, ...]
    decision: FormalCandidateDecision
    original_score: float = Field(ge=0.0, le=1.0)
    validation_penalty: float = Field(ge=0.0)
    validated_score: float = Field(ge=0.0, le=1.0)
    original_rank: int = Field(ge=1)
    validated_rank: int = Field(ge=1)
    eligible: bool
    issues: tuple[FormalValidationIssue, ...]
    trace: tuple[FormalRuleTrace, ...]
    repair_proposal: tuple[RepairProposal, ...]
    repair_applied: Literal[False] = False
    deterministic_signature: str

    @model_validator(mode="after")
    def validate_overlay(self) -> "FormalCandidateValidationResult":
        if self.repair_applied or any(item.applied for item in self.repair_proposal):
            raise ValueError("formal Validator repair proposals must be unapplied")
        expected_score = round(
            max(0.0, self.original_score - self.validation_penalty), 6
        )
        if abs(self.validated_score - expected_score) > 1e-9:
            raise ValueError("validated_score is not the frozen overlay")
        if self.eligible != (self.decision != FormalCandidateDecision.REJECT):
            raise ValueError("only REJECT candidates are ineligible")
        if tuple(issue.issue_type for issue in self.issues) != self.issue_type:
            raise ValueError("issue_type projection is incomplete")
        if tuple(issue.severity for issue in self.issues) != self.severity:
            raise ValueError("severity projection is incomplete")
        expected_rules = tuple(
            rule_id for issue in self.issues for rule_id in issue.violated_rule_ids
        )
        if expected_rules != self.violated_rule_ids:
            raise ValueError("violated rule projection is incomplete")
        if not self.deterministic_signature:
            raise ValueError("candidate validation signature is required")
        return self


class FormalValidationSetResult(FrozenContract):
    schema_version: str = "skillvis-v2-m5-validation-set-result-v1.0.0"
    context_id: str
    candidate_set_id: str
    config_version: str
    config_sha256: str
    schema_sha256: str
    intent_sha256: str
    capability_sha256: str
    candidate_set_input_sha256: str
    candidate_results: tuple[FormalCandidateValidationResult, ...]
    set_issues: tuple[FormalValidationIssue, ...]
    set_trace: tuple[FormalRuleTrace, ...]
    original_candidate_order: tuple[str, ...]
    validated_candidate_order: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    overall_decision: FormalSetDecision
    candidate_mutation_count: int = Field(ge=0)
    repair_applied: Literal[False] = False
    deterministic_signature: str

    @model_validator(mode="after")
    def validate_set_result(self) -> "FormalValidationSetResult":
        result_ids = {item.candidate_id for item in self.candidate_results}
        if result_ids != set(self.original_candidate_order):
            raise ValueError("formal result must retain every original candidate")
        if result_ids != set(self.validated_candidate_order):
            raise ValueError("validated order must retain every candidate")
        if self.repair_applied:
            raise ValueError("formal M5 cannot apply repairs")
        expected_rejected = tuple(
            item.candidate_id
            for item in self.candidate_results
            if item.decision == FormalCandidateDecision.REJECT
        )
        if set(expected_rejected) != set(self.rejected_candidate_ids):
            raise ValueError("rejected candidate projection is incomplete")
        if not self.deterministic_signature:
            raise ValueError("set deterministic signature is required")
        return self


_CANDIDATE_RULE_IDS = (
    "R-VF-C001",
    "R-VF-F001",
    "R-VF-F002",
    "R-VF-F003",
    "R-VF-E001",
    "R-VF-E002",
    "R-VF-E003",
    "R-VF-E004",
    "R-VF-T001",
    "R-VF-A001",
    "R-VF-A002",
    "R-VF-A003",
    "R-VF-A004",
    "R-VF-U001",
    "R-VF-U002",
    "R-VF-P001",
    "R-VF-P002",
    "R-VF-P003",
)
_SET_RULE_IDS = ("R-VF-S001", "R-VF-S002")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_formal_config(
    path: Path = DEFAULT_FORMAL_CONFIG_PATH,
) -> tuple[FormalValidatorConfig, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = FormalValidatorConfig.model_validate(payload)
    expected = {*_CANDIDATE_RULE_IDS, *_SET_RULE_IDS}
    observed = {rule.rule_id for rule in config.rules}
    if observed != expected:
        raise ValueError(
            f"formal registry mismatch missing={sorted(expected-observed)} "
            f"unexpected={sorted(observed-expected)}"
        )
    return config, sha256_file(path)


def candidate_set_signature(candidates: tuple[VisualizationCandidate, ...]) -> str:
    return _canonical_hash(
        [
            {
                "candidate_id": item.candidate_id,
                "rank": item.rank,
                "signature": item.deterministic_signature,
                "input_hash": item.canonical_sha256(),
            }
            for item in sorted(candidates, key=lambda candidate: candidate.rank)
        ]
    )


def _schema_fields(schema: SchemaProfile) -> dict[str, Any]:
    return {field.field_name: field for field in schema.fields}


def _intent_roles(intent: VisualizationIntentIR) -> dict[str, set[str]]:
    return {
        "measure": {item.field_name for item in intent.measures},
        "dimension": {item.field_name for item in intent.dimensions},
        "temporal": {item.field_name for item in intent.temporal_fields},
        "grouping": {item.field_name for item in intent.grouping_fields},
    }


def _derived_field_types(candidate: VisualizationCandidate) -> dict[str, str]:
    """Return fields explicitly introduced by typed candidate transforms."""

    derived: dict[str, str] = {}
    for transform in candidate.transforms:
        if transform.transform_type != "fold" or not isinstance(transform.value, dict):
            continue
        key_field = transform.value.get("key_field")
        value_field = transform.value.get("value_field")
        if isinstance(key_field, str) and key_field:
            derived[key_field] = "categorical"
        if isinstance(value_field, str) and value_field:
            derived[value_field] = "quantitative"
    return derived


def _candidate_referenced_fields(candidate: VisualizationCandidate) -> set[str]:
    role_fields = {
        field
        for values in candidate.fields.model_dump(mode="python").values()
        for field in values
    }
    encoding_fields = {
        item.field for item in candidate.encodings if item.field is not None
    }
    transform_fields = {
        field
        for transform in (
            *candidate.transforms,
            *candidate.filters,
            *((candidate.aggregation,) if candidate.aggregation else ()),
            *((candidate.sort,) if candidate.sort else ()),
        )
        for field in transform.field_candidates
    }
    return (role_fields | encoding_fields | transform_fields) - set(
        _derived_field_types(candidate)
    )


def _encoding_map(candidate: VisualizationCandidate) -> dict[str, Any]:
    return {item.channel: item for item in candidate.encodings}


def _transform_signature(transform: Any) -> tuple[Any, ...]:
    return (
        transform.transform_type,
        transform.operation,
        tuple(transform.field_candidates),
        json.dumps(transform.value, ensure_ascii=False, sort_keys=True),
        transform.negated,
    )


def _same_aggregation(left: Any, right: Any) -> bool:
    return (
        left is not None
        and right is not None
        and left.operation == right.operation
        and tuple(left.field_candidates) == tuple(right.field_candidates)
    )


def _candidate_rule_evidence(
    rule_id: str,
    context: CandidateSetContext,
    candidate: VisualizationCandidate,
    config: FormalValidatorConfig,
) -> dict[str, Any] | None:
    schema_fields = _schema_fields(context.schema_profile)
    intent_roles = _intent_roles(context.intent)

    if rule_id == "R-VF-C001":
        problems: list[str] = []
        if context.schema_profile.dataset_id != context.intent.dataset_id:
            problems.append("schema_intent_dataset_id_mismatch")
        if context.capability.canonical_sha256() != (
            context.intent.capability_assessment.canonical_sha256()
        ):
            problems.append("capability_intent_binding_mismatch")
        if context.candidate_set_signature != candidate_set_signature(
            context.candidates
        ):
            problems.append("candidate_set_signature_mismatch")
        if problems:
            return {
                "problems": problems,
                "schema_dataset_id": context.schema_profile.dataset_id,
                "intent_dataset_id": context.intent.dataset_id,
                "observed_candidate_set_signature": context.candidate_set_signature,
                "expected_candidate_set_signature": candidate_set_signature(
                    context.candidates
                ),
            }
        return None

    if rule_id == "R-VF-F001":
        referenced = _candidate_referenced_fields(candidate)
        missing = sorted(referenced - set(schema_fields))
        return {
            "missing_fields": missing,
            "schema_fields": sorted(schema_fields),
        } if missing else None

    if rule_id == "R-VF-F002":
        mismatches = []
        for field in candidate.fields.measure:
            profile = schema_fields.get(field)
            if profile is None:
                continue
            if (
                profile.semantic_type
                not in config.field_role_matrix["measure"]
                or field not in intent_roles["measure"]
            ):
                mismatches.append(
                    {
                        "field": field,
                        "schema_type": profile.semantic_type,
                        "intent_measure_fields": sorted(intent_roles["measure"]),
                    }
                )
        return {"measure_role_mismatches": mismatches} if mismatches else None

    if rule_id == "R-VF-F003":
        mismatches = []
        candidate_roles = candidate.fields.model_dump(mode="python")
        role_expectations = {
            "dimension": intent_roles["dimension"] | intent_roles["grouping"],
            "temporal": intent_roles["temporal"],
            "grouping": intent_roles["grouping"] | intent_roles["dimension"],
            "color": intent_roles["grouping"] | intent_roles["dimension"],
            "size": intent_roles["measure"],
            "facet": intent_roles["grouping"] | intent_roles["dimension"],
        }
        for role, expected_fields in role_expectations.items():
            allowed_types = set(config.field_role_matrix[role])
            for field in candidate_roles[role]:
                profile = schema_fields.get(field)
                if profile is None:
                    continue
                if (
                    profile.semantic_type not in allowed_types
                    or field not in expected_fields
                ):
                    mismatches.append(
                        {
                            "role": role,
                            "field": field,
                            "schema_type": profile.semantic_type,
                            "allowed_types": sorted(allowed_types),
                            "intent_fields": sorted(expected_fields),
                        }
                    )
        return {"role_mismatches": mismatches} if mismatches else None

    if rule_id == "R-VF-E001":
        mismatches = []
        derived_types = _derived_field_types(candidate)
        for encoding in candidate.encodings:
            if encoding.field is None:
                continue
            if encoding.field in schema_fields:
                schema_type = schema_fields[encoding.field].semantic_type
            elif encoding.field in derived_types:
                schema_type = derived_types[encoding.field]
            else:
                # R-VF-F001 owns undeclared-field rejection.
                continue
            allowed = set(config.encoding_type_matrix[encoding.channel])
            if encoding.semantic_type != schema_type or schema_type not in allowed:
                mismatches.append(
                    {
                        "channel": encoding.channel,
                        "field": encoding.field,
                        "candidate_type": encoding.semantic_type,
                        "schema_type": schema_type,
                        "allowed_types": sorted(allowed),
                    }
                )
        return {"encoding_type_mismatches": mismatches} if mismatches else None

    if rule_id == "R-VF-E002":
        chart = candidate.chart_type.value
        expected = config.chart_mark_compatibility.get(chart)
        if expected is None or candidate.mark_type != expected:
            return {
                "chart_type": chart,
                "observed_mark": candidate.mark_type,
                "expected_mark": expected,
            }
        return None

    if rule_id == "R-VF-E003":
        chart = candidate.chart_type.value
        required = set(config.required_channels.get(chart, ()))
        observed = {encoding.channel for encoding in candidate.encodings}
        missing = sorted(required - observed)
        return {
            "chart_type": chart,
            "required_channels": sorted(required),
            "observed_channels": sorted(observed),
            "missing_channels": missing,
        } if missing else None

    if rule_id == "R-VF-E004":
        if candidate.chart_type.value not in {"scatter", "grouped_scatter"}:
            return None
        encodings = _encoding_map(candidate)
        invalid_axes = []
        for channel in ("x", "y"):
            encoding = encodings.get(channel)
            profile = (
                schema_fields.get(encoding.field)
                if encoding is not None and encoding.field is not None
                else None
            )
            if profile is None or profile.semantic_type != "quantitative":
                invalid_axes.append(
                    {
                        "channel": channel,
                        "field": encoding.field if encoding else None,
                        "schema_type": profile.semantic_type if profile else None,
                    }
                )
        return {"invalid_axes": invalid_axes} if invalid_axes else None

    if rule_id == "R-VF-T001":
        task = candidate.task_alignment
        chart = candidate.chart_type.value
        intent_tasks = {
            item.task for item in context.intent.task_candidates
        } | {context.intent.primary_task}
        allowed = set(config.task_chart_compatibility.get(task, ()))
        if task not in intent_tasks or chart not in allowed:
            return {
                "task": task,
                "chart_type": chart,
                "intent_tasks": sorted(intent_tasks),
                "allowed_charts": sorted(allowed),
            }
        return None

    if rule_id == "R-VF-A001":
        expected = context.intent.aggregation
        if expected is None:
            return None
        observed = candidate.aggregation
        if not _same_aggregation(expected, observed):
            return {
                "expected": expected.model_dump(mode="json"),
                "observed": (
                    observed.model_dump(mode="json") if observed else None
                ),
            }
        return None

    if rule_id == "R-VF-A002":
        aggregation = candidate.aggregation
        if aggregation is None:
            return None
        matching = [
            transform
            for transform in candidate.transforms
            if transform.transform_type == "aggregate"
            and _same_aggregation(aggregation, transform)
        ]
        conflicts = [
            {
                "channel": encoding.channel,
                "field": encoding.field,
                "encoding_aggregate": encoding.aggregate,
                "declared_operation": aggregation.operation,
            }
            for encoding in candidate.encodings
            if encoding.aggregate is not None
            and encoding.aggregate != aggregation.operation
        ]
        if len(matching) != 1 or conflicts:
            return {
                "declared": aggregation.model_dump(mode="json"),
                "matching_transform_count": len(matching),
                "encoding_conflicts": conflicts,
            }
        return None

    if rule_id == "R-VF-A003":
        expected_filters = {
            _transform_signature(item) for item in context.intent.filters
        }
        observed_filters = {
            _transform_signature(item) for item in candidate.filters
        }
        expected_sort = (
            _transform_signature(context.intent.sort)
            if context.intent.sort is not None
            else None
        )
        observed_sort = (
            _transform_signature(candidate.sort)
            if candidate.sort is not None
            else None
        )
        problems = []
        if not expected_filters <= observed_filters:
            problems.append("missing_filter")
        if expected_sort is not None and observed_sort != expected_sort:
            problems.append("sort_mismatch")
        if (
            context.intent.limit is not None
            and candidate.limit != context.intent.limit
        ):
            problems.append("limit_mismatch")
        if problems:
            return {
                "problems": problems,
                "expected_filters": sorted(map(str, expected_filters)),
                "observed_filters": sorted(map(str, observed_filters)),
                "expected_sort": str(expected_sort),
                "observed_sort": str(observed_sort),
                "expected_limit": context.intent.limit,
                "observed_limit": candidate.limit,
            }
        return None

    if rule_id == "R-VF-A004":
        chart_scope = set(
            config.aggregation_data_grain_policy[
                "duplicate_group_bound_scope"
            ]
        )
        if (
            candidate.chart_type.value not in chart_scope
            or candidate.aggregation is not None
        ):
            return None
        grouping_fields = tuple(
            dict.fromkeys(
                (
                    *candidate.fields.dimension,
                    *candidate.fields.temporal,
                    *candidate.fields.grouping,
                )
            )
        )
        if not grouping_fields or any(
            field not in schema_fields for field in grouping_fields
        ):
            return None
        group_capacity = math.prod(
            max(1, schema_fields[field].cardinality)
            for field in grouping_fields
        )
        if context.schema_profile.row_count > group_capacity:
            return {
                "row_count": context.schema_profile.row_count,
                "grouping_fields": list(grouping_fields),
                "group_cardinalities": {
                    field: schema_fields[field].cardinality
                    for field in grouping_fields
                },
                "maximum_group_capacity": group_capacity,
                "aggregation": None,
            }
        return None

    if rule_id == "R-VF-U001":
        chart = candidate.chart_type.value
        unsupported = (
            chart not in config.chart_mark_compatibility
            or not candidate.executable
            or candidate.audit_only
            or bool(candidate.unsupported_features)
            or bool(candidate.rejection_reasons)
        )
        if unsupported:
            return {
                "chart_type": chart,
                "executable": candidate.executable,
                "audit_only": candidate.audit_only,
                "unsupported_features": list(candidate.unsupported_features),
                "rejection_reasons": list(candidate.rejection_reasons),
            }
        return None

    if rule_id == "R-VF-U002":
        if (
            context.capability.abstention_required
            or context.intent.abstention_state.abstained
            or context.intent.unsupported_scope
        ):
            return {
                "unsupported_operations": list(
                    context.capability.unsupported_operations
                ),
                "risk_level": context.capability.risk_level,
                "capability_abstention": context.capability.abstention_required,
                "intent_abstention": context.intent.abstention_state.model_dump(
                    mode="json"
                ),
                "unsupported_scope": [
                    item.model_dump(mode="json")
                    for item in context.intent.unsupported_scope
                ],
            }
        return None

    if rule_id == "R-VF-P001":
        if candidate.chart_type.value == "pie":
            return {
                "chart_type": "pie",
                "task": candidate.task_alignment,
                "channels": sorted(
                    encoding.channel for encoding in candidate.encodings
                ),
                "preference_scope": "precise angle/area comparison",
            }
        return None

    if rule_id == "R-VF-P002":
        thresholds = config.cardinality_thresholds
        violations = []
        for encoding in candidate.encodings:
            if encoding.field is None or encoding.field not in schema_fields:
                continue
            cardinality = schema_fields[encoding.field].cardinality
            threshold = None
            if encoding.channel == "color":
                threshold = (
                    thresholds["pie_color_max"]
                    if candidate.chart_type.value == "pie"
                    else thresholds["series_max"]
                    if candidate.chart_type.value == "multi_line"
                    else thresholds["color_max"]
                )
            elif encoding.channel in {"row", "column"}:
                threshold = thresholds["facet_max"]
            if threshold is not None and cardinality > threshold:
                violations.append(
                    {
                        "field": encoding.field,
                        "channel": encoding.channel,
                        "cardinality": cardinality,
                        "threshold": threshold,
                    }
                )
        return {"high_cardinality_encodings": violations} if violations else None

    if rule_id == "R-VF-P003":
        maximum = int(config.cardinality_thresholds["encoding_count_max"])
        auxiliary_maximum = int(
            config.cardinality_thresholds["auxiliary_channel_count_max"]
        )
        auxiliary = [
            item.channel
            for item in candidate.encodings
            if item.channel in {"color", "size", "row", "column", "detail"}
        ]
        if (
            len(candidate.encodings) > maximum
            or len(auxiliary) > auxiliary_maximum
        ):
            return {
                "encoding_count": len(candidate.encodings),
                "encoding_count_max": maximum,
                "auxiliary_channels": auxiliary,
                "auxiliary_count_max": auxiliary_maximum,
            }
        return None

    raise KeyError(f"no formal candidate rule implementation for {rule_id}")


def _set_rule_evidence(
    rule_id: str,
    context: CandidateSetContext,
    decisions: dict[str, FormalCandidateDecision],
) -> dict[str, Any] | None:
    if rule_id == "R-VF-S001":
        by_signature: dict[str, list[str]] = {}
        for candidate in context.candidates:
            by_signature.setdefault(
                candidate.deterministic_signature, []
            ).append(candidate.candidate_id)
        duplicates = {
            signature: ids
            for signature, ids in by_signature.items()
            if len(ids) > 1
        }
        return {"duplicate_signatures": duplicates} if duplicates else None
    if rule_id == "R-VF-S002":
        if all(
            decision == FormalCandidateDecision.REJECT
            for decision in decisions.values()
        ):
            return {
                "candidate_decisions": {
                    key: value.value for key, value in decisions.items()
                }
            }
        return None
    raise KeyError(f"no formal set rule implementation for {rule_id}")


def _repair(rule: FormalValidatorRule) -> RepairProposal:
    return RepairProposal(
        proposal_type=rule.repair_proposal,
        target_stage=rule.repair_target_stage,
        rule_id=rule.rule_id,
        description=(
            f"Unapplied proposal for {rule.issue_type}; route to "
            f"{rule.repair_target_stage} under {rule.rule_id}."
        ),
        applied=False,
    )


def _issue(
    *,
    context: CandidateSetContext,
    candidate_id: str | None,
    scope: Literal["context", "candidate", "candidate_set"],
    rule: FormalValidatorRule,
    evidence: dict[str, Any],
    config: FormalValidatorConfig,
) -> FormalValidationIssue:
    seed = {
        "context_id": context.context_id,
        "candidate_id": candidate_id,
        "rule_id": rule.rule_id,
        "evidence": evidence,
    }
    return FormalValidationIssue(
        issue_id=f"VFI-{_canonical_hash(seed)[:18]}",
        scope=scope,
        candidate_id=candidate_id,
        issue_type=rule.issue_type,
        level=rule.level,
        severity=rule.severity,
        violated_rule_ids=(rule.rule_id,),
        evidence=evidence,
        source_ids=rule.source_id,
        confidence=(
            1.0
            if rule.severity in {FormalSeverity.ERROR, FormalSeverity.CRITICAL}
            else 0.85
        ),
        decision=rule.allowed_action,
        repair_proposal=_repair(rule),
        repair_applied=False,
        penalty=config.penalty_weights[rule.penalty_key],
    )


def _decision(issues: list[FormalValidationIssue]) -> FormalCandidateDecision:
    if any(issue.decision == FormalRuleAction.REJECT for issue in issues):
        return FormalCandidateDecision.REJECT
    if any(issue.decision == FormalRuleAction.DEMOTE for issue in issues):
        return FormalCandidateDecision.DEMOTE
    if any(issue.decision == FormalRuleAction.WARN for issue in issues):
        return FormalCandidateDecision.WARN
    return FormalCandidateDecision.ACCEPT


def validate_candidate_set(
    context: CandidateSetContext,
    *,
    config: FormalValidatorConfig | None = None,
    config_sha256: str | None = None,
) -> FormalValidationSetResult:
    """Validate a complete candidate set and return a reversible rank overlay."""

    if (config is None) != (config_sha256 is None):
        raise ValueError("config and config_sha256 must be supplied together")
    if config is None:
        config, config_sha256 = load_formal_config()
    assert config_sha256 is not None

    before = {
        candidate.candidate_id: candidate.canonical_json()
        for candidate in context.candidates
    }
    issues_by_candidate: dict[str, list[FormalValidationIssue]] = {}
    traces_by_candidate: dict[str, list[FormalRuleTrace]] = {}
    decisions: dict[str, FormalCandidateDecision] = {}
    penalties: dict[str, float] = {}
    validated_scores: dict[str, float] = {}

    for candidate in context.candidates:
        candidate_issues: list[FormalValidationIssue] = []
        candidate_traces: list[FormalRuleTrace] = []
        for rule_id in _CANDIDATE_RULE_IDS:
            rule = config.rule(rule_id)
            evidence = _candidate_rule_evidence(
                rule_id, context, candidate, config
            )
            triggered = evidence is not None
            evidence_payload = evidence or {"status": "not_triggered"}
            trace_seed = {
                "context_id": context.context_id,
                "candidate_id": candidate.candidate_id,
                "rule_id": rule_id,
                "triggered": triggered,
                "evidence": evidence_payload,
            }
            candidate_traces.append(
                FormalRuleTrace(
                    rule_id=rule_id,
                    scope="candidate",
                    candidate_id=candidate.candidate_id,
                    triggered=triggered,
                    evidence=evidence_payload,
                    source_ids=rule.source_id,
                    trace_signature=_canonical_hash(trace_seed),
                )
            )
            if triggered:
                candidate_issues.append(
                    _issue(
                        context=context,
                        candidate_id=candidate.candidate_id,
                        scope=(
                            "context"
                            if rule.level == "V0"
                            else "candidate"
                        ),
                        rule=rule,
                        evidence=evidence_payload,
                        config=config,
                    )
                )
        issues_by_candidate[candidate.candidate_id] = candidate_issues
        traces_by_candidate[candidate.candidate_id] = candidate_traces
        decisions[candidate.candidate_id] = _decision(candidate_issues)
        penalties[candidate.candidate_id] = round(
            sum(issue.penalty for issue in candidate_issues), 6
        )
        validated_scores[candidate.candidate_id] = round(
            max(0.0, candidate.total_score - penalties[candidate.candidate_id]),
            6,
        )

    set_issues: list[FormalValidationIssue] = []
    set_traces: list[FormalRuleTrace] = []
    for rule_id in _SET_RULE_IDS:
        rule = config.rule(rule_id)
        evidence = _set_rule_evidence(rule_id, context, decisions)
        triggered = evidence is not None
        evidence_payload = evidence or {"status": "not_triggered"}
        trace_seed = {
            "context_id": context.context_id,
            "rule_id": rule_id,
            "triggered": triggered,
            "evidence": evidence_payload,
        }
        set_traces.append(
            FormalRuleTrace(
                rule_id=rule_id,
                scope="candidate_set",
                candidate_id=None,
                triggered=triggered,
                evidence=evidence_payload,
                source_ids=rule.source_id,
                trace_signature=_canonical_hash(trace_seed),
            )
        )
        if triggered:
            set_issues.append(
                _issue(
                    context=context,
                    candidate_id=None,
                    scope="candidate_set",
                    rule=rule,
                    evidence=evidence_payload,
                    config=config,
                )
            )

    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in context.candidates
    }
    ordered_candidates = sorted(
        context.candidates,
        key=lambda candidate: (
            1
            if decisions[candidate.candidate_id]
            == FormalCandidateDecision.REJECT
            else 0,
            -validated_scores[candidate.candidate_id],
            candidate.rank,
            candidate.deterministic_signature,
        ),
    )
    validated_rank = {
        candidate.candidate_id: rank
        for rank, candidate in enumerate(ordered_candidates, start=1)
    }

    candidate_results: list[FormalCandidateValidationResult] = []
    for candidate in sorted(context.candidates, key=lambda item: item.rank):
        issues = issues_by_candidate[candidate.candidate_id]
        repairs = tuple(issue.repair_proposal for issue in issues)
        result_seed = {
            "candidate_id": candidate.candidate_id,
            "candidate_signature": candidate.deterministic_signature,
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "trace": [
                trace.model_dump(mode="json")
                for trace in traces_by_candidate[candidate.candidate_id]
            ],
            "original_score": candidate.total_score,
            "validation_penalty": penalties[candidate.candidate_id],
            "validated_score": validated_scores[candidate.candidate_id],
            "original_rank": candidate.rank,
            "validated_rank": validated_rank[candidate.candidate_id],
            "decision": decisions[candidate.candidate_id].value,
        }
        candidate_results.append(
            FormalCandidateValidationResult(
                candidate_id=candidate.candidate_id,
                candidate_signature=candidate.deterministic_signature,
                candidate_input_hash=candidate.canonical_sha256(),
                issue_type=tuple(issue.issue_type for issue in issues),
                severity=tuple(issue.severity for issue in issues),
                violated_rule_ids=tuple(
                    rule_id
                    for issue in issues
                    for rule_id in issue.violated_rule_ids
                ),
                evidence=tuple(issue.evidence for issue in issues),
                source_ids=tuple(
                    dict.fromkeys(
                        source
                        for issue in issues
                        for source in issue.source_ids
                    )
                ),
                confidence=tuple(issue.confidence for issue in issues),
                decision=decisions[candidate.candidate_id],
                original_score=candidate.total_score,
                validation_penalty=penalties[candidate.candidate_id],
                validated_score=validated_scores[candidate.candidate_id],
                original_rank=candidate.rank,
                validated_rank=validated_rank[candidate.candidate_id],
                eligible=(
                    decisions[candidate.candidate_id]
                    != FormalCandidateDecision.REJECT
                ),
                issues=tuple(issues),
                trace=tuple(traces_by_candidate[candidate.candidate_id]),
                repair_proposal=repairs,
                repair_applied=False,
                deterministic_signature=_canonical_hash(result_seed),
            )
        )

    mutation_count = sum(
        before[candidate.candidate_id] != candidate.canonical_json()
        for candidate in context.candidates
    )
    rejected = tuple(
        result.candidate_id
        for result in candidate_results
        if result.decision == FormalCandidateDecision.REJECT
    )
    if len(rejected) == len(candidate_results):
        overall = FormalSetDecision.NO_VALID_CANDIDATE
    elif set_issues or any(
        result.decision != FormalCandidateDecision.ACCEPT
        for result in candidate_results
    ):
        overall = FormalSetDecision.ANSWERABLE_WITH_WARNINGS
    else:
        overall = FormalSetDecision.ANSWERABLE
    original_order = tuple(
        candidate.candidate_id
        for candidate in sorted(context.candidates, key=lambda item: item.rank)
    )
    validated_order = tuple(
        candidate.candidate_id for candidate in ordered_candidates
    )
    candidate_set_input_sha = _canonical_hash(
        [candidates_by_id[item].model_dump(mode="json") for item in original_order]
    )
    set_seed = {
        "context_id": context.context_id,
        "config_sha256": config_sha256,
        "candidate_results": [
            item.model_dump(mode="json") for item in candidate_results
        ],
        "set_issues": [item.model_dump(mode="json") for item in set_issues],
        "set_trace": [item.model_dump(mode="json") for item in set_traces],
        "original_order": original_order,
        "validated_order": validated_order,
        "overall_decision": overall.value,
        "candidate_mutation_count": mutation_count,
    }
    return FormalValidationSetResult(
        context_id=context.context_id,
        candidate_set_id=context.candidate_set_id,
        config_version=config.config_version,
        config_sha256=config_sha256,
        schema_sha256=context.schema_profile.canonical_sha256(),
        intent_sha256=context.intent.canonical_sha256(),
        capability_sha256=context.capability.canonical_sha256(),
        candidate_set_input_sha256=candidate_set_input_sha,
        candidate_results=tuple(candidate_results),
        set_issues=tuple(set_issues),
        set_trace=tuple(set_traces),
        original_candidate_order=original_order,
        validated_candidate_order=validated_order,
        rejected_candidate_ids=rejected,
        overall_decision=overall,
        candidate_mutation_count=mutation_count,
        repair_applied=False,
        deterministic_signature=_canonical_hash(set_seed),
    )


__all__ = [
    "CandidateSetContext",
    "FormalCandidateDecision",
    "FormalCandidateValidationResult",
    "FormalRuleAction",
    "FormalSetDecision",
    "FormalSeverity",
    "FormalValidationIssue",
    "FormalValidationSetResult",
    "FormalValidatorConfig",
    "FormalValidatorRule",
    "RepairProposal",
    "candidate_set_signature",
    "load_formal_config",
    "validate_candidate_set",
]
