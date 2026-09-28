# SkillVIS v2 Knowledge Base Construction and Evolution

Status: paper-facing method asset  
Final method identity: `skillvis-v2-joint-semantic-precedence-v1.12.1`  
Machine-readable registry: `knowledge_source_registry_v1.json`

## 1. Why the knowledge base is part of the method

SkillVIS does not treat visualization knowledge as an unstructured prompt. Its
knowledge base is the set of versioned, executable, source-linked rules that
connects a natural-language request to typed intent, safe candidates, and
auditable validation.

This distinction matters:

- a bibliography says which work was read;
- a prompt says what an LLM was told;
- the SkillVIS knowledge base says which scoped statement became which
  deterministic rule, where that rule can act, what happens when it conflicts
  with another rule, and which trace proves that it fired.

The final M1--M4 runtime registry contains 87 rules. The frozen M5 Validator has
20 additional rules. The paper-facing registry therefore traces 107 executable
rules to 33 catalogued sources, including the project's own protocol rules.
Twenty-two sources contribute to at least one implemented rule; eleven are
surveyed or related-work sources only. These counts are reproducible by:

```bash
PYTHONPATH=src python3 research/skillvis_v2/paper_assets/build_knowledge_traceability.py
```

## 2. How visualization knowledge was collected

### 2.1 Search axes

The search was organized by missing decision type, not by paper popularity:

| Decision gap | Primary venues and source families | Representative sources |
|---|---|---|
| What constitutes a valid graphical design? | IEEE VIS/TVCG; visualization books | Mackinlay (D1), Grammar of Graphics (D2), Vega-Lite (R10) |
| How should analytical tasks be represented? | IEEE InfoVis/VIS | Brehmer and Munzner (D4), Amar et al. (D5), TaskVis (R5) |
| How should valid alternatives be enumerated and ranked? | IEEE VIS/TVCG; SIGMOD/HILDA | Draco (R1), Draco 2 (R2), CompassQL (R3), Voyager (R4) |
| Which encodings are perceptually preferable? | JASA; ACM CHI | Cleveland and McGill (D6), Zeng and Battle (D7) |
| How should natural language ground to data fields? | IEEE VIS/TVCG; ACL/EMNLP/AAAI | NL4DV (N1), RAT-SQL (S1), BRIDGE (S2), schema-linking analysis (S3), RESDSQL (S4) |
| How should ambiguity and multiple valid answers be represented? | IEEE VIS/TVCG; NeurIPS; HCI | NL4DV (N1), nvBench 2.0 (N7), CompassQL (R3), Data Formulator (R8/R9) |
| How should errors be localized and repaired? | IEEE VIS/TVCG; ACL/EMNLP | Munzner's Nested Model (D3), nvAgent (N4), LIDA (R7), execution/repair agents (E1--E4 in the survey) |

The primary-source order was:

1. peer-reviewed visualization theory and empirical perception work;
2. official papers and repositories for visualization systems;
3. adjacent schema-linking and semantic-parsing work;
4. project-specific engineering policies, marked as `SV2-PROTOCOL`.

Every catalogue entry is also assigned one collection method. This records how
the source entered the knowledge process; it does not automatically grant the
source executable authority.

| Collection method | Total sources | Implemented | Survey-only |
|---|---:|---:|---:|
| Theory-driven canonical search | 8 | 8 | 0 |
| System and artifact tracing | 17 | 8 | 9 |
| Benchmark and evaluation-gap search | 3 | 1 | 2 |
| Adjacent-domain transfer | 4 | 4 | 0 |
| Project-protocol elicitation | 1 | 1 | 0 |
| **Total** | **33** | **22** | **11** |

The source denominator (`N=33`) is separate from the executable-rule
denominator (`N=107`). Rule-to-source provenance is multi-valued and therefore
is not a part-to-whole distribution.

The detailed literature survey, authors, years, venues, URLs, and repository
links are recorded in `literature_landscape.md`. Any unresolved bibliographic
metadata remains explicitly marked `[VERIFY]`; it is not silently completed.

### 2.2 Inclusion and exclusion

A source was admitted to the knowledge catalogue when it supplied at least one
of the following:

- a visualization task vocabulary;
- a data/transform/mark/encoding grammar;
- an empirically grounded perceptual relationship;
- an explicit hard or soft constraint;
- a partial-specification or candidate-enumeration mechanism;
- an ambiguity, abstention, schema-linking, validation, or repair mechanism.

A source was **not** automatically converted into a rule. It remained
survey-only when the result depended on an LLM, a trained model, an interactive
UI advantage, a different task contract, or an experimental condition too
narrow to justify a deterministic rule.

Examples:

- Draco contributes the hard/soft constraint model and source-level
  provenance, but SkillVIS does not claim to invent constraint-based
  recommendation and does not need a direct Draco performance comparison.
- CompassQL contributes partial-spec enumeration and is eligible for a
  structured capability comparison, but it is not treated as an NL parser.
- graphical-perception findings become scoped warnings or preferences; they do
  not become universal hard chart rankings.
- schema-linking models motivate evidence types and stage separation; their
  learned neural architectures are not copied into deterministic rules.

## 3. From paper to executable knowledge

Each knowledge item passes a six-step translation gate.

### Step 1: Extract a knowledge statement

The statement must be narrower than the source's abstract claim. Example:

> A field/channel combination can be grammatically incompatible even when the
> overall chart type is allowed.

### Step 2: Define scope and counterexample

The author records:

- which stage may use the statement;
- what inputs are required;
- what observation would make it inapplicable;
- whether it is a hard constraint, soft preference, or diagnostic relation.

### Step 3: Encode a typed rule

M1--M4 rules contain:

- `rule_id`;
- `module`;
- `source_ids`;
- `scope`;
- `rationale`;
- `priority`;
- `failure_condition`;
- `conflict_policy`;
- `trace_fields`;
- named unit tests.

M5 rules additionally record issue type, severity, allowed action, evidence
requirements, repair proposal, target stage, example, and limitation.

### Step 4: Define conflict and permission boundaries

Rules do not share unlimited authority:

- M1 may normalize text but must preserve source spans.
- M2 may rank schema fields but must not select a chart.
- M3 may assign tasks and transforms but must preserve competing hypotheses.
- M3.5 may abstain for unsupported scope but must not turn an unsupported
  operation into a familiar chart.
- M4 may enumerate and rank candidates but must honor typed intent and safety.
- M5 may detect, reject, warn, demote, and propose a repair route; it cannot
  mutate a candidate or silently rewrite upstream intent.

### Step 5: Attach executable evidence

Every runtime decision can expose:

- the rule identifier;
- source identifiers;
- input and output fields;
- deterministic registry hash;
- unit-test name;
- candidate or validation trace.

The machine-readable registry resolves legacy source identifiers such as
`MACKINLAY1986` and `NL4DV2020` to canonical IDs such as `D1` and `N1`.

### Step 6: Freeze before independent evidence

Development failures may produce a new version, but never an in-place rewrite
of frozen evidence. A revision must receive a new registry/config identity, a
new development asset, and a new prospective qualification before formal
comparison.

## 4. What knowledge is currently encoded

Every executable rule receives exactly one primary category according to its
runtime responsibility:

| Knowledge family | Rules | Final modules | Representative behavior |
|---|---:|---|---|
| Query evidence and normalization | 6 | M1 | preserve chart, number, operator, span, and negation evidence |
| Schema and field semantics | 21 | M1/M2 | profile types and ground ordinary field evidence |
| Typed field-role relations | 15 | M2/M3 joint layer | bind measure, dimension, time, grouping, target, score, and operator ownership |
| Analytical task and intent | 25 | M3 | represent tasks, transforms, obligations, and competing hypotheses |
| Ambiguity and capability safety | 7 | M3.5 | distinguish lexical residue, material ambiguity, missing roles, and unsupported scope |
| Candidate construction and ranking | 13 | M4 | enumerate variable-length candidates, preserve obligations, rank, and deduplicate |
| Hard design validation | 15 | M5 | reject field, role, transform, encoding, task-chart, and safety-bypass violations |
| Soft perceptual and set audit | 5 | M5 | warn or demote cardinality, overload, duplicate, and exhaustion problems |
| **Total** | **107** | **M1–M5** | |

![SkillVIS rule-family and source-collection distributions](figures/knowledge_base_distributions_v1.png)

Selective Recovery is a wrapper around this deterministic core and is not
counted as an additional runtime knowledge family in the 107-rule inventory.

## 5. How benchmark failure changes the knowledge base

The revision process is a prospective spiral:

```text
frozen run
  -> typed failure taxonomy
  -> source-backed mechanism hypothesis
  -> new rule/config version
  -> development cases
  -> sealed holdout
  -> independent qualification
  -> only then a new formal comparison
```

The formal or held-out output is never used to rewrite its own gold. Failed
assets and stopped protocol attempts remain visible.

### 5.1 Failure-to-revision ledger

| Observed failure | Diagnosis | Knowledge used | New mechanism | Evidence boundary |
|---|---|---|---|---|
| Numeric field names containing time-like words were treated as temporal in correlation requests | lexical schema hint overpowered numeric value shape and query relation | Mackinlay expressiveness; typed task relations; NL4DV-style query evidence | v1.10 `R-SP-110`: correlation-scoped numeric/temporal guard | passed development, sealed holdout, and independent qualification-v5 lineage |
| Ranking phrases with a numeric-coded target were classified as correlation or comparison | target and score were not represented as a typed relation | Amar/Brehmer-Munzner tasks; NL4DV evidence; typed schema relations | v1.11 `R-FR-120` + `R-TP-130`: target-link-score closure | PCS-v6 evaluated v1.11; this evidence must not be relabeled v1.12.1 |
| “levels/tiers/strata” numeric categories remained measures | field role needed local discourse evidence | Grammar/role distinctions; task language | v1.12 `R-FR-140`: set-noun grouping role | implementation evidence; prospective incremental gain is not isolated |
| “highest to lowest average” produced both `max` and `mean` | ordering direction was conflated with aggregation operation | task/transform separation; partial-spec constraints | v1.12 `R-TP-140`: ordering direction vs aggregation precedence | v1.12 sealed holdout exposed additional timing failures and remained HOLD |
| generic grouped-superlative reasoning dominated a fully typed ranking relation | correct evidence existed but fired too late | task specificity and precedence | v1.12.1 `R-TP-141`: typed ranking pre-closure | 5/5 on new controlled ablation stratum |
| numeric duration under grouped min/max was misread as a temporal axis | weak temporal name cue overpowered value shape and explicit aggregate target | expressiveness, task, and NL query evidence | v1.12.1 `R-SP-150`: grouped extremum numeric-role guard | 5/5 on new controlled ablation stratum |
| correct Recovery bindings for `rank X by Y` still failed full M3 closure | generic `by=grouping` role overrode the numeric score relation | typed task/field roles and operator-specific precedence | development-only v1.12.2 `R-FR-151`: ranking score-role precedence | fixed both exposed failures and gave Qwen/DeepSeek 8/8 engineering revalidation, but a pre-frozen 60-case promotion qualification produced only two target improvements and four repeated-target false acceptances; candidate locked `FAIL` and excluded from final v1.12.1 |

### 5.2 Why the ledger is not benchmark tuning

The safeguards are:

- old formal inputs, outputs, gold, evaluator, and locks are immutable;
- a failed holdout remains failed and is not rerun after a patch;
- revised cases use new fields, values, and exact queries;
- exact-query and source overlap are audited before qualification;
- formal evidence names the exact evaluated version;
- the paper distinguishes controlled mechanism evidence from broad language
  generalization.

## 6. Existing knowledge versus SkillVIS contribution

### Existing visualization knowledge

SkillVIS does **not** claim the following as inventions:

- analytical task taxonomies;
- Grammar of Graphics or Vega-Lite-style typed specifications;
- perceptual effectiveness findings;
- chart recommendation constraints;
- hard/soft constraint separation;
- partial-spec enumeration;
- candidate ranking;
- schema linking;
- validation as a design activity.

### SkillVIS engineering and method contribution

The defensible contribution is their operational combination:

1. **Typed visualization skills.** Each stage has a bounded input/output contract
   rather than an informal prompt responsibility.
2. **Executable source traceability.** A rule connects source, scope, conflict
   policy, trace fields, tests, and runtime provenance.
3. **Stage-local authority.** Understanding, capability checking, candidate
   construction, validation, and recovery cannot silently override one another.
4. **Fail-closed back-routing.** M5 can name the upstream stage responsible for
   an error without mutating the candidate or bypassing safety.
5. **Failure-driven, version-isolated evolution.** Benchmark failures become
   typed knowledge gaps and new prospective rules while old evidence remains
   immutable.
6. **Selective LLM use.** Deterministic execution remains the default; a local
   LLM may supply only a bounded alias proposal, which must pass typed checks
   and the unchanged M5 Validator.
7. **Evidence-Grounded Visualization Rule Lifecycle.** Knowledge claims,
   translation assessments, typed rules, tests, admission, runtime evidence,
   revision, and deprecation are treated as one auditable lifecycle.

### 6.1 Evidence-Grounded Visualization Rule Lifecycle

The current system is a strict manually executed instance:

```text
decision gap
  -> source collection
  -> KnowledgeClaimIR
  -> TranslationAssessment
  -> RuleIR proposal
  -> dependency/conflict and counterexample tests
  -> human admission + version freeze
  -> runtime trace and failure taxonomy
  -> retain / revise under a new identity / deprecate
```

Future automation may retrieve papers, suggest claims/rules, inspect
dependencies, and generate counterexamples. It must not approve rules or turn
a soft finding into a hard rejection. The present paper does not implement or
evaluate a Rule Maintenance Assistant; it contributes the lifecycle framework
and its manual execution instance.

## 7. Reproducibility and audit checklist

- Source metadata and adoption state:
  `knowledge_source_registry_v1.json`.
- Complete rule taxonomy, source collection paths, and paper figure:
  `knowledge_rule_taxonomy_v1.json`,
  `knowledge_taxonomy_and_collection_method.md`, and
  `figures/knowledge_base_distributions_v1.png`.
- Complete 107-rule supplement:
  `appendix_knowledge_rule_registry_v1.md`.
- Source-level evidence locators for all 22 implemented sources:
  `knowledge_source_locator_registry_v1.json`.
- Full literature notes:
  `literature_landscape.md`.
- Final M1--M4 runtime registry hash:
  `d56dbfd2afdaa6688d197c2bec0853b4a1453066e9fbafe49e67acf1ab9c75f6`.
- Frozen M5 config path:
  `configs/skillvis_v2/validator_formal_v1.json`.
- Failure history:
  `joint_v110_v111_resolution_report.md`,
  `joint_v1121_semantic_hardening_report.md`,
  `joint_v1122_independent_qualification_report.md`, and
  `joint_v1122_posthoc_failure_analysis.md`.
- Mechanism evidence:
  `mechanism_ablation_report_v1.md`.
- Version/evidence separation:
  `final_candidate_version_note_v2.md`,
  `final_candidate_version_manifest_v2.json`.

## 8. Remaining human checks

The construction logic and runtime mapping are complete without external help.
DataTone, nvBench, Draco 2, CompassQL, and TaskVis metadata were rechecked
against primary or official paper pages by 2026-08-02. Before submission, a human reference-manager pass must still
normalize bibliography style, author-name formatting, DOI formatting, and the
distinction between archival venues, workshops, and preprints.

Source-level section/chapter/page locator coverage is complete for 22/22
implemented sources. This improves navigation but is not a semantic-fidelity
judgment. An independent knowledge-to-rule review is prepared but not yet executed. It
must evaluate source support, translation fidelity, rule scope, permission
level, and provenance sufficiency without seeing benchmark or Validator
outputs. Until that review is completed, the paper may claim traceability of
the registry, but not independent expert validation of its semantic fidelity.
