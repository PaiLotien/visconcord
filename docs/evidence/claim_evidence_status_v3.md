# SkillVIS Claim → Evidence Status v3

**Final method:** Joint v1.12.1  
**Evidence cutoff:** 2026-08-12  
**Narrative:** one primary construction-methodology contribution; M5 is a core
assurance module, not a parallel contribution.

## Core claim matrix

| Claim | Current evidence | Required scope language | Status |
|---|---|---|---|
| **C1 — Methodology.** Heterogeneous visualization knowledge can be retrieved, scoped, translated, permissioned, encapsulated, tested, and version-admitted as typed Visualization Skills. | 107 runtime rules, 33 sources, 245 rule–source links, five collection paths, source locators, rule contracts, positive/negative/conflict tests, version manifests, and one failed prospective admission. | This is a method and reference implementation, not 107 new theories and not proof that every translation is uniquely correct. | **SUPPORTED / PRIMARY** |
| **C2 — Controlled planning.** v1.12.1 produces more complete plans than NL4DV 4.1.0 on the frozen shared-capability slice. | PCS-v7: Complete Plan@1 30/30 vs 6/30; field F1 1.000 vs 0.622; task macro F1 0.867 vs 0.200; 80/80 parity/replay invocations pass. | Three UCI sources, five declared task families, 30 controlled tasks; no open-language or universal superiority claim. | **SUPPORTED / PRIMARY SYSTEM EVIDENCE** |
| **C3 — M5 assurance boundary.** A separately encapsulated Validator can check cross-stage typed obligations while preserving planner state and limiting its own authority. | Frozen public inputs; Detect/Rank/Repair/Reject permission contract; warning cannot reject; original and validated scores coexist; repair is `applied=false`; stage routes; full traces and hashes. Formal set: 80/80 hard recall, 0/80 false safe, 0/60 false reject, 10/10 valid-top1 after overlay, full audit/provenance, 13/13 gates. | Component-only evidence on 120 constructed cases / 140 candidates; 15 extra triggers; no open-world, user-interpretability, or end-to-end guarantee. | **SUPPORTED / CORE MODULE EVIDENCE** |
| **C4 — Existing-system boundary.** SkillVIS and CompassQL expose different structured completion gaps. | Both return any-conforming@5 in 28/35 frozen cells; SkillVIS gap is distribution, CompassQL gap is composition; CompassQL is faster on its native workload. | Conformance only; no subjective-quality winner and no claim that CompassQL is an NL system. | **SUPPORTED / SECONDARY** |
| **C5 — Bounded model authority.** An LLM can propose field evidence, but cannot select a chart, bypass safety/validation, or admit a new rule. | Recovery traces, token/cost accounting, full M1–M5 replay, provider replication on exposed cases, and failed v1.12.2 prospective qualification. | Mechanism and governance evidence only; no general Recovery gain or provider superiority. | **SUPPORTED / CONDITIONAL** |

## M5 novelty boundary

Existing work already provides hard/soft constraints, candidate enumeration and
ranking, schema validation, visualization linting, localized explanations, and
automatic fixes. M5 must not claim these primitives as new.

M5's defensible method difference is the **combination** of:

1. cross-stage obligations over schema, intent, capability, and candidates;
2. a separate post-M4 authority boundary;
3. contract-enforced Detect/Rank/Repair/Reject permissions;
4. immutable planner output plus reversible validation overlay;
5. `applied=false` repair routes to M2/M3/M3.5/M4;
6. fail-closed preservation of unsupported scope and abstention;
7. triggered/non-triggered traces with evidence, source, version, and hashes.

## Claims prohibited from the paper

- SkillVIS or M5 is the first visualization validator, linter, constraint
  engine, actionable diagnostic system, or visualization fixer.
- M5 is perfect or detects every possible visualization error.
- M5 component validation proves end-to-end or open-world system correctness.
- SkillVIS is generally superior to NL4DV or produces better charts than
  CompassQL.
- M5's traces are human-interpretable without a human-centered study.
- Recovery improves the final benchmark or DeepSeek is superior to Qwen.
- Memory, Quda, or automatic Rule Maintenance is an achieved contribution.

## Minimal main-paper evidence

1. knowledge-construction audit and representative source-to-rule cases;
2. PCS-v7 controlled SkillVIS × NL4DV result;
3. compact formal M5 component table plus ranking and 15-trigger error analysis;
4. CompassQL structured conformance boundary;
5. bounded Recovery and failed rule admission as governance cases.

Full rules, intervals, identities, hashes, per-family M5 results, and raw error
records belong in the supplement.

