"""Post-resolution exact-span dominance for the v1.1 joint resolver."""

from __future__ import annotations

from skillvis_v2_accuracy.config import SkillVISV2Config
from skillvis_v2_accuracy.contracts import FieldResolutionResult, QueryNormalizationResult, SchemaProfile
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_joint.config import JointSemanticSettings
from skillvis_v2_joint.field_resolution import (
    JointFieldArtifacts,
    run_joint_field_resolution,
)


_STRONG = {"exact", "normalized_lexical", "alias", "value"}


def _span_rows(candidate):
    return tuple(
        evidence.query_span
        for evidence in candidate.evidence
        if evidence.query_span is not None
    )


def run_joint_v2_field_resolution(
    *,
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    base: FieldResolutionResult,
    config: SkillVISV2Config,
    settings: JointSemanticSettings,
    registry: RuleRegistry,
) -> JointFieldArtifacts:
    artifacts = run_joint_field_resolution(
        normalized=normalized,
        schema=schema,
        base=base,
        config=config,
        settings=settings,
        registry=registry,
    )
    result = artifacts.result
    strong = tuple(
        candidate
        for candidate in result.candidate_fields
        if _STRONG & set(candidate.match_types)
    )
    suppressed: set[str] = set()
    for candidate in result.candidate_fields:
        if candidate.field_name not in result.required_field_candidates:
            continue
        candidate_spans = _span_rows(candidate)
        if not candidate_spans:
            continue
        if any(
            stronger.field_name != candidate.field_name
            and all(
                any(
                    strong_span.start <= weak_span.start
                    and weak_span.end <= strong_span.end
                    and (strong_span.start, strong_span.end)
                    != (weak_span.start, weak_span.end)
                    for strong_span in _span_rows(stronger)
                )
                for weak_span in candidate_spans
            )
            for stronger in strong
        ):
            suppressed.add(candidate.field_name)
    if not suppressed:
        return artifacts
    required = tuple(
        name for name in result.required_field_candidates if name not in suppressed
    )
    ambiguity = []
    for item in result.ambiguity:
        if item.dimension != "field":
            ambiguity.append(item)
            continue
        alternatives = tuple(name for name in item.alternatives if name in required)
        if len(alternatives) > 1:
            ambiguity.append(item.model_copy(update={"alternatives": alternatives}))
    updated = FieldResolutionResult.model_validate(
        result.model_copy(
            update={
                "required_field_candidates": required,
                "ambiguity": tuple(ambiguity),
                "abstention_reason": None if required else result.abstention_reason,
            }
        ).model_dump(mode="json")
    )
    warnings = tuple(
        sorted(
            {
                *artifacts.warnings,
                "v1.1 exact field span suppressed contained token-fragment obligations: "
                + ", ".join(sorted(suppressed)),
            }
        )
    )
    return JointFieldArtifacts(
        result=updated,
        schema_profiles=artifacts.schema_profiles,
        semantic_mentions=artifacts.semantic_mentions,
        warnings=warnings,
    )


__all__ = ["run_joint_v2_field_resolution"]
