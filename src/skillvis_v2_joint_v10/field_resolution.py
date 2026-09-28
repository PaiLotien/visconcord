"""Segment false exact field spans at typed operation boundaries."""
from __future__ import annotations
from skillvis_v2_accuracy.contracts import FieldResolutionResult, QueryNormalizationResult
from skillvis_v2_accuracy.rules import RuleRegistry
from skillvis_v2_joint_v6.evidence_strength import EvidenceStrength, classify_evidence


def _covered(span, owner)->bool: return owner.start<=span.start and span.end<=owner.end
def _composite(span, candidate_name, operations, exact_owners):
    matches=[]
    for operation in operations:
        for owner_name, owner in exact_owners:
            if owner_name==candidate_name: continue
            if span.start==operation.start and operation.end<=owner.start<=operation.end+2 and owner.start<span.end<=owner.end:
                matches.append((operation,owner_name,owner))
    return min(matches,key=lambda item:(item[0].start,item[1]),default=None)


def resolve_composite_operation_ownership(*,normalized:QueryNormalizationResult,fields:FieldResolutionResult,registry:RuleRegistry)->tuple[FieldResolutionResult,tuple[str,...]]:
    operations=tuple(normalized.aggregation_terms)
    if not operations:return fields,()
    exact_owners=tuple((candidate.field_name,evidence.query_span) for candidate in fields.candidate_fields for evidence in candidate.evidence if evidence.evidence_type=="exact" and evidence.query_span is not None)
    required=set(fields.required_field_candidates); suppressions={}; traces=[]
    for candidate in fields.candidate_fields:
        if candidate.field_name not in required:continue
        grounded=tuple(item for item in candidate.evidence if item.query_span is not None)
        if not grounded:continue
        owned=[]; details=[]
        for evidence in grounded:
            strength=classify_evidence(evidence.evidence_type); composite=_composite(evidence.query_span,candidate.field_name,operations,exact_owners) if evidence.evidence_type=="exact" else None
            weak_owned=strength not in {EvidenceStrength.EXPLICIT_GROUNDING,EvidenceStrength.SEMANTIC_GROUNDING} and any(_covered(evidence.query_span,operation) for operation in operations)
            if composite is not None:
                operation,owner_name,owner_span=composite; owned.append(True); details.append(f"exact={evidence.query_span.start}:{evidence.query_span.end}|op={operation.start}:{operation.end}|owner={owner_name}:{owner_span.start}:{owner_span.end}")
            elif weak_owned:
                owned.append(True); details.append(f"weak={evidence.query_span.start}:{evidence.query_span.end}|operation_owned")
            else:owned.append(False)
        if owned and all(owned): suppressions[candidate.field_name]=tuple(details)
    if not suppressions:return fields,()
    surviving=tuple(name for name in fields.required_field_candidates if name not in suppressions); ambiguity=[]
    for item in fields.ambiguity:
        if item.dimension!="field":ambiguity.append(item);continue
        alternatives=tuple(name for name in item.alternatives if name in surviving)
        if len(alternatives)>1:ambiguity.append(item.model_copy(update={"alternatives":alternatives}))
    updated=FieldResolutionResult.model_validate(fields.model_copy(update={"required_field_candidates":surviving,"ambiguity":tuple(ambiguity),"applied_rule_ids":registry.validate_ids((*fields.applied_rule_ids,"R-FR-100")),"abstention_reason":None if surviving else "no_field_candidate_after_composite_segmentation"}).model_dump(mode="json"))
    for name,detail in sorted(suppressions.items()):traces.append(f"R-FR-100 composite operation segmentation suppressed:{name}:"+";".join(detail))
    return updated,tuple(traces)
__all__=["resolve_composite_operation_ownership"]
