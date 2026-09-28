"""M4 Candidate Planner with hard field/transform obligation preservation."""

from __future__ import annotations

from typing import Any

from skillvis_v2_accuracy import candidate_planner as base
from skillvis_v2_accuracy.contracts import (
    AmbiguityStateType,
    AnalyticalTaskResult,
    CandidateDiversitySummary,
    CandidateEncoding,
    CandidateFields,
    CandidatePlanningResult,
    CapabilityAssessment,
    FieldResolutionResult,
    SchemaProfile,
    VisualizationCandidate,
    VisualizationIntentIR,
)
from skillvis_v2_accuracy.rules import RuleRegistry

from .config import SkillVISV2Config, load_m4_config
from .intent_compiler import VisualizationObligationSet, compile_intent_obligations


def _field_set(fields: CandidateFields) -> set[str]:
    return {
        *fields.measure,
        *fields.dimension,
        *fields.temporal,
        *fields.grouping,
        *fields.color,
        *fields.size,
        *fields.facet,
    }


def _close_row(
    row: dict[str, Any],
    obligations: VisualizationObligationSet,
    schema: SchemaProfile,
) -> dict[str, Any] | None:
    """Complete only role-compatible fields; reject rows with unclosable duties."""

    fields: CandidateFields = row["fields"]
    encodings = list(row["encodings"])
    update = fields.model_dump(mode="python")
    present = _field_set(fields)
    filter_only = {
        item.field_name for item in obligations.field_obligations if item.filter_only
    }
    for obligation in obligations.field_obligations:
        name = obligation.field_name
        if name in present:
            continue
        roles = set(obligation.permitted_roles)
        if obligation.filter_only:
            update["dimension"] = tuple(dict.fromkeys((*update["dimension"], name)))
            present.add(name)
            continue
        if "measure" in roles or "temporal" in roles:
            return None
        if roles & {"dimension", "geographic", "identifier", "grouping"}:
            update["dimension"] = tuple(dict.fromkeys((*update["dimension"], name)))
            update["grouping"] = tuple(dict.fromkeys((*update["grouping"], name)))
            channels = {item.channel for item in encodings}
            channel = "color" if "color" not in channels else "column"
            encodings.append(
                CandidateEncoding(
                    channel=channel,
                    field=name,
                    semantic_type=schema.field(name).semantic_type,
                )
            )
            if channel == "color":
                update["color"] = tuple(dict.fromkeys((*update["color"], name)))
            else:
                update["facet"] = tuple(dict.fromkeys((*update["facet"], name)))
            present.add(name)
            continue
        return None

    required_transforms = {
        (
            item.transform_type,
            item.operation,
            item.field_candidates,
            str(item.value),
        )
        for item in obligations.transform_obligations
    }
    observed_transforms = {
        (
            item.transform_type,
            item.operation,
            item.field_candidates,
            str(item.value),
        )
        for item in row["transforms"]
    }
    if not required_transforms <= observed_transforms:
        return None
    closed = dict(row)
    closed["fields"] = CandidateFields.model_validate(update)
    closed["encodings"] = tuple(encodings)
    closed["obligation_id"] = obligations.obligation_id
    closed["filter_only_fields"] = tuple(sorted(filter_only))
    return closed


def _decorate_candidate(
    candidate: VisualizationCandidate,
    obligations: VisualizationObligationSet,
    registry: RuleRegistry,
) -> VisualizationCandidate:
    rule_ids = registry.validate_ids(
        (*candidate.applied_rule_ids, "R-CP-010", "R-CP-011")
    )
    source_ids = tuple(
        sorted(
            {
                source_id
                for rule_id in rule_ids
                for source_id in registry.get(rule_id).source_ids
            }
        )
    )
    return candidate.model_copy(
        update={
            "applied_rule_ids": rule_ids,
            "source_ids": source_ids,
            "chart_compatibility_assumptions": (
                *candidate.chart_compatibility_assumptions,
                f"typed_obligation_set:{obligations.obligation_id}",
                "all required field and transform obligations were checked before ranking",
            ),
        }
    )


def plan_visualization_candidates(
    *,
    intent: VisualizationIntentIR,
    capability: CapabilityAssessment,
    schema: SchemaProfile,
    field_resolution: FieldResolutionResult,
    task_result: AnalyticalTaskResult,
    config: SkillVISV2Config | None = None,
    registry: RuleRegistry | None = None,
) -> CandidatePlanningResult:
    config = config or load_m4_config()
    registry = registry or RuleRegistry.from_config(config)
    settings = config.candidate_planner
    if settings is None:
        raise RuntimeError("M4 Candidate Planner settings are unavailable")
    if capability != intent.capability_assessment:
        raise ValueError("capability input must match VisualizationIntentIR")

    compiled = compile_intent_obligations(intent, task_result, registry)
    effective_intent = compiled.intent
    obligations = compiled.obligations
    schema_names = {field.field_name for field in schema.fields}
    required_names = {item.field_name for item in obligations.field_obligations}
    if not required_names <= schema_names or not required_names <= set(field_resolution.required_field_candidates):
        return base._empty_result(
            intent=intent,
            config=config,
            registry=registry,
            reason="inconsistent_upstream_field_contract",
            rejection_reasons=("obligation fields must exist in schema and required field candidates",),
        )
    if intent.abstention_state.abstained or capability.abstention_required:
        return base._empty_result(
            intent=intent,
            config=config,
            registry=registry,
            reason=intent.abstention_reason or intent.abstention_state.reason or "required_abstention",
            rejection_reasons=("upstream safety or semantic layer requires abstention",),
        )
    if intent.ambiguity_state.state in {
        AmbiguityStateType.MATERIAL_AMBIGUITY,
        AmbiguityStateType.MISSING_REQUIRED_ROLE,
        AmbiguityStateType.UNSUPPORTED_SCOPE,
    }:
        return base._empty_result(
            intent=intent,
            config=config,
            registry=registry,
            reason=f"blocked_ambiguity_state:{intent.ambiguity_state.state.value}",
            rejection_reasons=("typed ambiguity state forbids execution",),
        )

    filter_only = {
        item.field_name for item in obligations.field_obligations if item.filter_only
    }
    enumeration_intent = effective_intent.model_copy(
        update={
            "dimensions": tuple(ref for ref in effective_intent.dimensions if ref.field_name not in filter_only),
            "grouping_fields": tuple(ref for ref in effective_intent.grouping_fields if ref.field_name not in filter_only),
        }
    )
    rows = base._enumerate_templates(
        enumeration_intent, schema, compiled.task_result, config
    )
    rows, hint_rejections = base._explicit_hint_filter(rows, enumeration_intent)
    closed_rows = tuple(
        closed
        for row in rows
        if (closed := _close_row(row, obligations, schema)) is not None
    )
    if not closed_rows:
        return base._empty_result(
            intent=intent,
            config=config,
            registry=registry,
            reason="no_obligation_complete_candidate",
            rejection_reasons=hint_rejections or ("no template preserved every typed obligation",),
        )

    raw = tuple(
        _decorate_candidate(
            base._candidate_from_row(
                row,
                intent=effective_intent,
                schema=schema,
                task_result=compiled.task_result,
                config=config,
                registry=registry,
            ),
            obligations,
            registry,
        )
        for row in closed_rows
        if row["chart_type"] in settings.chart_vocabulary
    )
    deduped = base.deduplicate_candidates(raw)

    def stable_rank_key(item: VisualizationCandidate):
        channels = {encoding.channel for encoding in item.encodings}
        branch = 0 if "color" in channels else (1 if {"row", "column"} & channels else 0)
        return (-item.total_score, branch, item.deterministic_signature)

    ranked_source = sorted(deduped, key=stable_rank_key)[: settings.max_candidates]
    ranked = tuple(
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(ranked_source, start=1)
    )
    rule_ids = {
        rule_id for candidate in ranked for rule_id in candidate.applied_rule_ids
    }
    provenance = base._planner_provenance(registry, rule_ids, effective_intent)
    result_payload = {
        "intent_id": intent.deterministic_id,
        "obligation_id": obligations.obligation_id,
        "candidate_signatures": [item.deterministic_signature for item in ranked],
        "scores": [item.total_score for item in ranked],
        "ranks": [item.rank for item in ranked],
        "raw": len(raw),
        "dedup": len(deduped),
    }
    return CandidatePlanningResult(
        intent_id=intent.deterministic_id,
        config_version=config.config_version,
        registry_hash=registry.sha256,
        candidates=ranked,
        raw_candidate_count=len(raw),
        post_dedup_count=len(deduped),
        post_ranking_count=len(ranked),
        duplicate_count=len(raw) - len(deduped),
        diversity=base._diversity(ranked),
        abstained=False,
        abstention_reason=None,
        warnings=(("candidate_budget_applied",) if len(deduped) > settings.max_candidates else ()),
        rejection_reasons=(),
        provenance=provenance,
        deterministic_signature=base._canonical_hash(result_payload),
    )


deduplicate_candidates = base.deduplicate_candidates

__all__ = ["deduplicate_candidates", "plan_visualization_candidates"]
