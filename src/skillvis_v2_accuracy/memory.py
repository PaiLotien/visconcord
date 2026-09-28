"""Typed, task-conditioned, Validator-bounded memory for SkillVIS v2.

The memory is intentionally deterministic and LLM-free.  It stores only
explicit user feedback over candidates that M5 did not reject.  Retrieval can
reuse an exact validated plan or add a bounded preference overlay; it cannot
change safety decisions, candidate contents, or Validator penalties.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import SchemaProfile, VisualizationCandidate, VisualizationIntentIR
from .validator_formal import (
    FormalCandidateDecision,
    FormalCandidateValidationResult,
    FormalValidationSetResult,
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


def _chart_family(chart_type: str) -> str:
    return {
        "grouped_bar": "bar",
        "horizontal_bar": "bar",
        "top_k_bar": "bar",
        "stacked_bar": "bar",
        "normalized_stacked_bar": "bar",
        "ranked_bar": "bar",
        "highlighted_extremum": "bar",
        "multi_line": "line",
        "grouped_scatter": "scatter",
    }.get(chart_type, chart_type)


class MemoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: str
    user_id: str
    task_signature: str
    intent_id: str
    schema_hash: str
    chart_type: str
    chart_family: str
    candidate_signature: str
    candidate_payload: dict[str, Any]
    selected_count: int = Field(default=0, ge=0)
    dismissed_count: int = Field(default=0, ge=0)
    last_event_index: int = Field(ge=1)
    validator_decision: Literal["ACCEPT", "DEMOTE"]
    validator_signature: str
    feedback_source: Literal["explicit_user"] = "explicit_user"
    provenance: tuple[str, ...]

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> "MemoryRecord":
        if self.selected_count + self.dismissed_count < 1:
            raise ValueError("memory record requires at least one feedback event")
        if not self.provenance:
            raise ValueError("memory record requires provenance")
        candidate = VisualizationCandidate.model_validate(self.candidate_payload)
        if candidate.deterministic_signature != self.candidate_signature:
            raise ValueError("candidate payload/signature mismatch")
        return self


class MemoryStore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["skillvis-v2-task-memory-v1.0.0"] = (
        "skillvis-v2-task-memory-v1.0.0"
    )
    event_counter: int = Field(default=0, ge=0)
    records: tuple[MemoryRecord, ...] = ()

    @model_validator(mode="after")
    def validate_store(self) -> "MemoryStore":
        ids = [record.entry_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("memory entry IDs must be unique")
        if any(record.last_event_index > self.event_counter for record in self.records):
            raise ValueError("record event index exceeds store event counter")
        return self


class MemoryCandidateOverlay(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    candidate_signature: str
    validator_decision: str
    original_rank: int = Field(ge=1)
    validated_score: float = Field(ge=0.0, le=1.0)
    selected_count: int = Field(ge=0)
    dismissed_count: int = Field(ge=0)
    preference_rate: float = Field(ge=0.0, le=1.0)
    memory_bonus: float = Field(ge=0.0, le=1.0)
    memory_score: float = Field(ge=0.0, le=1.0)
    memory_rank: int = Field(ge=1)
    eligible: bool
    evidence_entry_ids: tuple[str, ...]


class MemoryRankingResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["skillvis-v2-memory-ranking-v1.0.0"] = (
        "skillvis-v2-memory-ranking-v1.0.0"
    )
    user_id: str
    task_signature: str
    original_order: tuple[str, ...]
    memory_order: tuple[str, ...]
    overlays: tuple[MemoryCandidateOverlay, ...]
    candidate_mutation_count: Literal[0] = 0
    validator_override_count: Literal[0] = 0
    deterministic_signature: str


class ExactMemoryLookup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hit: bool
    reason: str
    entry_id: str | None = None
    candidate: VisualizationCandidate | None = None
    validator_signature: str | None = None


def task_signature(
    intent: VisualizationIntentIR,
    schema: SchemaProfile,
) -> str:
    profile = {field.field_name: field.semantic_type for field in schema.fields}
    payload = {
        "primary_task": intent.primary_task,
        "measure_types": sorted(profile[item.field_name] for item in intent.measures),
        "dimension_types": sorted(
            profile[item.field_name] for item in intent.dimensions
        ),
        "temporal_count": len(intent.temporal_fields),
        "transform_signature": sorted(
            (item.transform_type, item.operation)
            for item in intent.transforms
        ),
        "chart_hints": sorted(item.canonical for item in intent.chart_mentions),
        "ambiguity_state": intent.ambiguity_state.state.value,
        "unsupported_operations": sorted(
            intent.capability_assessment.unsupported_operations
        ),
    }
    return _canonical_hash(payload)


class TaskConditionedMemory:
    """Persistent memory with per-user isolation and deterministic eviction."""

    def __init__(self, path: str | Path, *, max_records: int = 1000) -> None:
        self.path = Path(path)
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.max_records = max_records

    def load(self) -> MemoryStore:
        if not self.path.exists():
            return MemoryStore()
        return MemoryStore.model_validate_json(self.path.read_text(encoding="utf-8"))

    def _save(self, store: MemoryStore) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            store.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def clear_user(self, user_id: str) -> int:
        store = self.load()
        kept = tuple(record for record in store.records if record.user_id != user_id)
        removed = len(store.records) - len(kept)
        if removed:
            self._save(store.model_copy(update={"records": kept}))
        return removed

    def record_feedback(
        self,
        *,
        user_id: str,
        intent: VisualizationIntentIR,
        schema: SchemaProfile,
        candidate: VisualizationCandidate,
        validation: FormalCandidateValidationResult,
        feedback: Literal["selected", "dismissed"],
    ) -> MemoryRecord:
        if not user_id.strip():
            raise ValueError("user_id is required for memory isolation")
        if validation.candidate_signature != candidate.deterministic_signature:
            raise ValueError("validation does not bind the supplied candidate")
        if validation.decision == FormalCandidateDecision.REJECT:
            raise ValueError("Validator-rejected candidates cannot enter memory")
        if intent.abstention_state.abstained:
            raise ValueError("abstained intents cannot enter memory")

        store = self.load()
        next_event = store.event_counter + 1
        signature = task_signature(intent, schema)
        schema_hash = schema.canonical_sha256()
        entry_id = "mem-" + _canonical_hash(
            {
                "user_id": user_id,
                "task_signature": signature,
                "intent_id": intent.deterministic_id,
                "schema_hash": schema_hash,
                "candidate_signature": candidate.deterministic_signature,
            }
        )[:24]
        by_id = {record.entry_id: record for record in store.records}
        existing = by_id.get(entry_id)
        selected = (existing.selected_count if existing else 0) + (
            1 if feedback == "selected" else 0
        )
        dismissed = (existing.dismissed_count if existing else 0) + (
            1 if feedback == "dismissed" else 0
        )
        record = MemoryRecord(
            entry_id=entry_id,
            user_id=user_id,
            task_signature=signature,
            intent_id=intent.deterministic_id,
            schema_hash=schema_hash,
            chart_type=candidate.chart_type.value,
            chart_family=_chart_family(candidate.chart_type.value),
            candidate_signature=candidate.deterministic_signature,
            candidate_payload=candidate.model_dump(mode="json"),
            selected_count=selected,
            dismissed_count=dismissed,
            last_event_index=next_event,
            validator_decision=validation.decision.value,
            validator_signature=validation.deterministic_signature,
            provenance=(
                "explicit_user_feedback",
                "m5_non_reject_gate",
                f"intent:{intent.deterministic_id}",
                f"candidate:{candidate.deterministic_signature}",
            ),
        )
        by_id[entry_id] = record
        ordered = sorted(
            by_id.values(),
            key=lambda item: (item.last_event_index, item.entry_id),
            reverse=True,
        )[: self.max_records]
        ordered.sort(key=lambda item: item.entry_id)
        self._save(
            MemoryStore(
                event_counter=next_event,
                records=tuple(ordered),
            )
        )
        return record

    def lookup_exact(
        self,
        *,
        user_id: str,
        intent: VisualizationIntentIR,
        schema: SchemaProfile,
    ) -> ExactMemoryLookup:
        if intent.abstention_state.abstained:
            return ExactMemoryLookup(hit=False, reason="upstream_abstention")
        schema_hash = schema.canonical_sha256()
        matches = [
            record
            for record in self.load().records
            if record.user_id == user_id
            and record.intent_id == intent.deterministic_id
            and record.schema_hash == schema_hash
            and record.selected_count > record.dismissed_count
        ]
        if not matches:
            return ExactMemoryLookup(hit=False, reason="no_exact_validated_plan")
        record = max(
            matches,
            key=lambda item: (
                item.selected_count - item.dismissed_count,
                item.last_event_index,
                item.entry_id,
            ),
        )
        return ExactMemoryLookup(
            hit=True,
            reason="exact_intent_schema_validated_plan",
            entry_id=record.entry_id,
            candidate=VisualizationCandidate.model_validate(record.candidate_payload),
            validator_signature=record.validator_signature,
        )

    def rank(
        self,
        *,
        user_id: str,
        intent: VisualizationIntentIR,
        schema: SchemaProfile,
        candidates: tuple[VisualizationCandidate, ...],
        validation: FormalValidationSetResult,
        max_bonus: float = 0.08,
    ) -> MemoryRankingResult:
        if not 0.0 <= max_bonus <= 0.25:
            raise ValueError("max_bonus must be between 0 and 0.25")
        candidate_by_id = {item.candidate_id: item for item in candidates}
        if set(candidate_by_id) != {
            item.candidate_id for item in validation.candidate_results
        }:
            raise ValueError("validation/candidate set mismatch")
        signature = task_signature(intent, schema)
        relevant = [
            record
            for record in self.load().records
            if record.user_id == user_id and record.task_signature == signature
        ]
        result_by_id = {
            item.candidate_id: item for item in validation.candidate_results
        }
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            result = result_by_id[candidate.candidate_id]
            family = _chart_family(candidate.chart_type.value)
            evidence = [
                record
                for record in relevant
                if record.chart_family == family
            ]
            selected = sum(record.selected_count for record in evidence)
            dismissed = sum(record.dismissed_count for record in evidence)
            total = selected + dismissed
            rate = selected / total if total else 0.0
            eligible = result.decision != FormalCandidateDecision.REJECT
            bonus = round(max_bonus * rate, 6) if eligible else 0.0
            score = round(min(1.0, result.validated_score + bonus), 6)
            rows.append(
                {
                    "candidate": candidate,
                    "validation": result,
                    "selected": selected,
                    "dismissed": dismissed,
                    "rate": rate,
                    "bonus": bonus,
                    "score": score,
                    "eligible": eligible,
                    "evidence_ids": tuple(sorted(record.entry_id for record in evidence)),
                }
            )
        ranked = sorted(
            rows,
            key=lambda row: (
                0 if row["eligible"] else 1,
                -row["score"],
                row["candidate"].rank,
                row["candidate"].deterministic_signature,
            ),
        )
        overlays = tuple(
            MemoryCandidateOverlay(
                candidate_id=row["candidate"].candidate_id,
                candidate_signature=row["candidate"].deterministic_signature,
                validator_decision=row["validation"].decision.value,
                original_rank=row["candidate"].rank,
                validated_score=row["validation"].validated_score,
                selected_count=row["selected"],
                dismissed_count=row["dismissed"],
                preference_rate=round(row["rate"], 6),
                memory_bonus=row["bonus"],
                memory_score=row["score"],
                memory_rank=index,
                eligible=row["eligible"],
                evidence_entry_ids=row["evidence_ids"],
            )
            for index, row in enumerate(ranked, start=1)
        )
        payload = {
            "user_id": user_id,
            "task_signature": signature,
            "original_order": [item.candidate_id for item in candidates],
            "memory_order": [item.candidate_id for item in overlays],
            "overlays": [item.model_dump(mode="json") for item in overlays],
            "candidate_mutation_count": 0,
            "validator_override_count": 0,
        }
        return MemoryRankingResult(
            user_id=user_id,
            task_signature=signature,
            original_order=tuple(item.candidate_id for item in candidates),
            memory_order=tuple(item.candidate_id for item in overlays),
            overlays=overlays,
            deterministic_signature=_canonical_hash(payload),
        )
