# Representative Paper-to-Rule Translation Cases v1

**Purpose:** Provide auditable examples of how a literature-grounded statement becomes a scoped, typed runtime rule without implying that the source directly authored the implementation.  
**Population:** 8 representative rules selected for stage, knowledge-family, authority, and source-mix coverage—not system performance.  
**Full population:** 107 rules in `appendix_knowledge_rule_registry_v1.md` and `knowledge_rule_taxonomy_v1.json`.

## Reading convention

Each case separates four identities:

1. **Source contribution:** what the cited work contributes at the source level;
2. **Operationalization:** the narrower runtime decision made by SkillVIS;
3. **Authority:** the strongest action the rule may take;
4. **Boundary evidence:** a counterexample, conflict policy, or test that prevents over-enforcement.

Source authority does not automatically prove translation correctness. The translation is made inspectable through locator, scope, authority, counterexample, tests, and version identity.

## Case 1 — Stable normalization without evidence loss

| Field | Value |
|---|---|
| Rule | `R-QN-003` |
| Runtime stage | M1 |
| Source(s) | `N1` NL4DV; `N7` nvBench 2.0 |
| Source locator | NL4DV Sections 4.1–4.2.4 and Fig. 4; nvBench 2.0 Sections 2–3 |
| Source contribution | NL-to-attribute/task/specification processing and explicit ambiguity; staged reasoning and multiple acceptable interpretations |
| Scoped rule | Equivalent Unicode, case, and whitespace forms share stable lexical evidence while original spans remain authoritative |
| Authority | `EVIDENCE_ONLY` |
| Why not stronger | Normalization may expose evidence but cannot bind a field, infer a task, or select a chart |
| Boundary tests | `test_case_whitespace_normalization`; `test_empty_query_failure` |

**Reviewer check:** The rule does not claim that NL4DV mandates SkillVIS’s exact Unicode implementation; it uses NL4DV/nvBench as evidence for preserving language evidence and ambiguity through staged processing.

## Case 2 — Temporal typing requires value-shape evidence

| Field | Value |
|---|---|
| Rule | `R-SP-004` |
| Runtime stage | M1 |
| Source(s) | `R10` Vega-Lite |
| Source locator | Vega-Lite Sections 3–5 |
| Source contribution | A typed grammar in which temporal, quantitative, nominal, and ordinal semantics affect transformations and encodings |
| Scoped rule | A field is temporal only with datetime dtype or date-like value evidence; a name token alone is weak evidence |
| Authority | `EVIDENCE_ONLY` |
| Counterexample | A numeric duration field named `processing_time` must not become a temporal axis only because its name contains `time` |
| Boundary tests | `test_temporal_dtype`; `test_temporal_string_values`; `test_year_name_alone_not_forced` |

**Reviewer check:** Vega-Lite supplies the typed semantic distinction; the value-profile threshold and evidence ordering are declared SkillVIS operational choices rather than attributed to Vega-Lite as empirical findings.

## Case 3 — Ranking is a target–score relation

| Field | Value |
|---|---|
| Rule | `R-FR-120` |
| Runtime stage | M2 |
| Source(s) | `D4` Brehmer–Munzner; `D5` Amar et al.; `N1` NL4DV |
| Source locator | Brehmer–Munzner Section 3/Fig. 1; Amar et al. Sections 3.1–4; NL4DV Sections 4.1–4.2.4/Fig. 4 |
| Source contribution | Analytical-task semantics and task sequences; low-level task vocabulary; joint extraction of attributes and analytical tasks |
| Scoped rule | In `rank X by aggregate(Y)`, an explicitly grounded numeric-coded X may be the entity dimension while Y is the score measure |
| Authority | `FIELD_BIND_OR_RANK` |
| Conflict policy | Do not rewrite roles without a grounded ranking target, a distinct linked score field, and an explicit aggregate |
| Boundary tests | `test_numeric_ranking_target_becomes_grouping_dimension`; `test_correlation_numeric_fields_are_not_converted`; `test_ranking_without_grounded_target_is_not_rewritten` |

**Reviewer check:** The rule is query-local. It does not globally relabel low-cardinality numeric fields as categorical.

## Case 4 — Underspecification remains unresolved

| Field | Value |
|---|---|
| Rule | `R-TP-003` |
| Runtime stage | M3 |
| Source(s) | `N7` nvBench 2.0 |
| Source locator | Sections 2–3 |
| Source contribution | Ambiguity-injected tasks, staged reasoning, and variable acceptable candidates |
| Scoped rule | An underspecified one- or multi-field request retains plausible task alternatives or an unresolved state instead of forcing a default |
| Authority | `TASK_OR_INTENT_MODEL` |
| Conflict policy | Explicit task evidence may resolve the alternatives; absent such evidence, the tie remains visible |
| Boundary tests | `test_bare_single_measure_ambiguity`; `test_bare_two_measure_ambiguity` |

**Reviewer check:** This rule prevents a high-coverage system from achieving apparent accuracy by silently choosing the most common task.

## Case 5 — Unsupported operations stop before planning

| Field | Value |
|---|---|
| Rule | `R-SA-004` |
| Runtime stage | M3.5 |
| Source(s) | `N7` nvBench 2.0 |
| Source locator | Sections 2–3 |
| Source contribution | Staged reasoning and explicit treatment of ambiguity in text-to-visualization evaluation |
| Scoped rule | A versioned unsupported governing operation forces semantic-safety abstention before M4 |
| Authority | `CAPABILITY_GATE` |
| Counterexample | `predict next year's sales` may not fall back to a descriptive line chart |
| Boundary tests | `test_unsupported_operation_forces_abstention`; `test_no_fallback_to_line_chart` |

**Reviewer check:** The taxonomy of unsupported operations is a declared SkillVIS capability boundary; the source does not independently prove every listed operation unsafe.

## Case 6 — Transform obligations survive candidate enumeration

| Field | Value |
|---|---|
| Rule | `R-CP-011` |
| Runtime stage | M4 |
| Source(s) | `D3` Nested Model; `D4` task typology; `N7` nvBench 2.0; `R10` Vega-Lite |
| Source locator | Nested Model Sections 2–4/Figs. 1–2; task typology Section 3/Fig. 1; nvBench 2.0 Sections 2–3; Vega-Lite Sections 3–5 |
| Source contribution | Stage-specific validation; task semantics; staged/multi-answer evaluation; typed transformations in a declarative grammar |
| Scoped rule | Filter, sort, time unit, aggregation, limit, and fold obligations are part of the candidate contract and survive ranking |
| Authority | `CONSTRUCT_OR_RANK` |
| Conflict policy | Explicit typed transforms dominate planner defaults; unresolved conflicts yield no executable candidate |
| Boundary tests | `test_time_unit_survives_planning`; `test_rank_sort_survives_planning`; `test_filter_survives_without_grouping` |

**Reviewer check:** The rule composes several prior ideas. It is literature-grounded synthesis, not a claim that any single source states the full rule verbatim.

## Case 7 — Perceptual evidence demotes rather than rejects

| Field | Value |
|---|---|
| Rule | `R-VF-P001` |
| Runtime stage | M5 |
| Source(s) | `D1` Mackinlay; `D6` Cleveland–McGill; `D7` Zeng–Battle |
| Source locator | Mackinlay Sections 5–7; Cleveland–McGill experiments/Figs. 20–21; Zeng–Battle Sections 3.1–4.3, Table 1, and Sections 6.1–6.3 |
| Source contribution | Expressiveness/effectiveness; empirical perceptual-accuracy orderings; limitations of translating perception knowledge into recommendation |
| Scoped rule | Pie angle/area comparison receives a traceable effectiveness warning and score penalty when precision matters |
| Authority | `DEMOTE` |
| Why not reject | A pie chart can remain a valid composition representation; relative perceptual effectiveness does not imply universal invalidity |
| Runtime tests | `test_validator_formal.py` exercises `PERCEPTUAL_EFFECTIVENESS_WARNING`; composition tests retain bounded pie candidates |

**Reviewer check:** This is the principal defense against converting “A is usually more effective than B” into “B is forbidden.”

## Case 8 — Protocol governance prevents safety bypass

| Field | Value |
|---|---|
| Rule | `R-VF-U002` |
| Runtime stage | M5 |
| Source mix | `LITERATURE_PLUS_PROJECT_PROTOCOL` |
| Source(s) | `D3` Nested Model; `SV2-PROTOCOL` SkillVIS typed-contract policy |
| Source locator | Nested Model Sections 2–4/Figs. 1–2; `m5_formal_protocol_v2_correction.md` Sections 4–8 |
| Source contribution | Upstream abstraction errors propagate; SkillVIS defines fail-closed permissions, provenance, and non-mutation |
| Scoped rule | A candidate cannot survive M5 when CapabilityAssessment or IntentIR requires abstention |
| Authority | `REJECT` |
| Counterexample | A syntactically valid chart for an unsupported prediction request must still be rejected |
| Runtime tests | `test_validator_formal.py::test_capability_and_context_rules` checks `SAFETY_BYPASS` behavior |

**Reviewer check:** The hard authority comes from the explicitly identified project safety policy, not from over-reading the Nested Model as a runtime authorization rule.

## Coverage summary

| Dimension | Covered values |
|---|---|
| Runtime stage | M1, M2, M3, M3.5, M4, M5 |
| Knowledge family | query normalization, schema profiling, field-role relations, task intent, capability safety, candidate planning, hard/soft validation |
| Authority | EVIDENCE_ONLY, FIELD_BIND_OR_RANK, TASK_OR_INTENT_MODEL, CAPABILITY_GATE, CONSTRUCT_OR_RANK, DEMOTE, REJECT |
| Source mix | literature-only; literature + project protocol |
| Boundary form | counterexample, conflict policy, abstention, non-mutation, test |

These eight cases are explanatory, not a statistical sample and not an external correctness evaluation. The full registry remains the reproducibility artifact.

