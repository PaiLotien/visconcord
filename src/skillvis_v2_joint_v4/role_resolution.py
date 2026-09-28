"""Query-local field-role repair using explicit operator-to-field bindings."""

from __future__ import annotations

from skillvis_v2_accuracy.contracts import (
    AnalyticalTaskResult,
    FieldEvidence,
    FieldResolutionResult,
    QueryNormalizationResult,
    SchemaProfile,
)
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_joint_v2.task_resolution import _is_explicit_grouping_field


_STRONG_GROUNDING = {
    "exact",
    "normalized_lexical",
    "alias",
    "acronym_expansion",
    "semantic_concept",
    "schema_semantic_concept",
}


def resolve_operator_scoped_numeric_roles(
    *,
    normalized: QueryNormalizationResult,
    schema: SchemaProfile,
    fields: FieldResolutionResult,
    task: AnalyticalTaskResult,
    registry: RuleRegistry,
) -> tuple[SchemaProfile, FieldResolutionResult, tuple[str, ...]]:
    """Restore an aggregate operand to a measure without using a name list.

    The parent v1.1 layer may convert a low-cardinality quantitative field to a
    categorical view when any grouping cue is present.  This repair is narrower:
    the field must be physically numeric, explicitly grounded, directly bound by
    a parsed aggregate operation, and not locally marked as the group key.
    """

    aggregate_by_field: dict[str, list] = {}
    for operation in task.transforms:
        if operation.transform_type != "aggregate" or operation.negated:
            continue
        for field_name in operation.field_candidates:
            aggregate_by_field.setdefault(field_name, []).append(operation)

    candidates = {item.field_name: item for item in fields.candidate_fields}
    required = set(fields.required_field_candidates)
    promoted: dict[str, tuple] = {}
    schema_rows = []
    for profile in schema.fields:
        candidate = candidates.get(profile.field_name)
        operations = aggregate_by_field.get(profile.field_name, [])
        locally_grouped = bool(
            candidate
            and _is_explicit_grouping_field(
                normalized.original_query,
                candidate,
            )
        )
        can_promote = bool(
            profile.field_name in required
            and candidate
            and operations
            and profile.numeric_summary is not None
            and profile.semantic_type in {"categorical", "ordinal"}
            and (_STRONG_GROUNDING & set(candidate.match_types))
            and not locally_grouped
        )
        if not can_promote:
            schema_rows.append(profile)
            continue
        promoted[profile.field_name] = tuple(operations)
        schema_rows.append(
            profile.model_copy(
                update={
                    "semantic_type": "quantitative",
                    "role_candidates": ("measure",),
                    "source_rule_ids": registry.validate_ids(
                        (*profile.source_rule_ids, "R-FR-040")
                    ),
                    "warnings": tuple(
                        sorted(
                            {
                                *profile.warnings,
                                "query-local numeric aggregate operand restored to measure",
                            }
                        )
                    ),
                }
            )
        )

    if not promoted:
        return schema, fields, ()

    candidate_rows = []
    evidence_map = dict(fields.evidence)
    intended_map = dict(fields.intended_roles)
    match_map = dict(fields.match_types)
    traces = []
    for candidate in fields.candidate_fields:
        operations = promoted.get(candidate.field_name)
        if not operations:
            candidate_rows.append(candidate)
            continue
        evidence = list(candidate.evidence)
        previous_roles = tuple(candidate.intended_roles)
        operation_labels = []
        for operation in operations:
            operation_labels.append(operation.operation)
            span = operation.evidence_spans[0] if operation.evidence_spans else None
            evidence.append(
                FieldEvidence(
                    evidence_type="operator_scoped_role",
                    query_span=span,
                    value=f"{operation.operation}:measure:{candidate.field_name}",
                    rule_id="R-FR-040",
                    source_ids=registry.get("R-FR-040").source_ids,
                )
            )
        evidence = sorted(
            evidence,
            key=lambda item: (
                item.query_span.start if item.query_span else -1,
                item.query_span.end if item.query_span else -1,
                item.evidence_type,
                item.value,
                item.rule_id,
            ),
        )
        roles = tuple(
            sorted(
                {
                    *(role for role in previous_roles if role not in {"dimension", "grouping"}),
                    "measure",
                }
            )
        )
        match_types = tuple(sorted({*candidate.match_types, "operator_scoped_role"}))
        source_rule_ids = registry.validate_ids(
            (*candidate.source_rule_ids, "R-FR-040")
        )
        updated = candidate.model_copy(
            update={
                "evidence": tuple(evidence),
                "intended_roles": roles,
                "match_types": match_types,
                "source_rule_ids": source_rule_ids,
            }
        )
        candidate_rows.append(updated)
        evidence_map[candidate.field_name] = updated.evidence
        intended_map[candidate.field_name] = updated.intended_roles
        match_map[candidate.field_name] = updated.match_types
        traces.append(
            "R-FR-040 operator-scoped numeric role: "
            f"{candidate.field_name} {previous_roles}->measure "
            f"operations={','.join(sorted(set(operation_labels)))} local_grouping=false"
        )

    updated_schema = SchemaProfile.model_validate(
        schema.model_copy(update={"fields": tuple(schema_rows)}).model_dump(mode="json")
    )
    updated_fields = FieldResolutionResult.model_validate(
        fields.model_copy(
            update={
                "candidate_fields": tuple(candidate_rows),
                "evidence": evidence_map,
                "intended_roles": intended_map,
                "match_types": match_map,
                "applied_rule_ids": registry.validate_ids(
                    (*fields.applied_rule_ids, "R-FR-040")
                ),
            }
        ).model_dump(mode="json")
    )
    return updated_schema, updated_fields, tuple(sorted(traces))


__all__ = ["resolve_operator_scoped_numeric_roles"]
