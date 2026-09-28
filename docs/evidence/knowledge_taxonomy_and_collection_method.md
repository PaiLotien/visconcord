# SkillVIS Knowledge Taxonomy and Collection Methods

Method identity: `skillvis-v2-joint-semantic-precedence-v1.12.1`  
Schema: `skillvis-v2-knowledge-taxonomy-v1.0.0`  
Status: paper-facing deterministic taxonomy; no human-review outcome included

## 1. Units of analysis

The figures deliberately use two different denominators:

- **Rules (`N=107`):** executable M1–M5 rules, each assigned exactly one primary knowledge family.
- **Sources (`N=33`):** catalogued papers, books, artifacts, and the separately labelled project protocol, each assigned exactly one primary collection method.
- **Rule–source links:** multi-valued provenance edges; their total may exceed both denominators and is never interpreted as a percentage of rules or sources.

Rule count is an inventory measure, not a measure of scientific importance or novelty.

## 2. Rule categories

| Category | Count | Definition |
|---|---:|---|
| Query evidence & normalization | 6 | Preserve spans, chart mentions, operations, numbers, and negation without selecting a visualization. |
| Schema & field semantics | 21 | Profile physical/semantic types and resolve ordinary field evidence without manufacturing schema fields. |
| Typed field-role relations | 15 | Represent mention ownership, grouping, measure, temporal, target, score, and operator-scoped relations. |
| Analytical task & intent | 25 | Model tasks, transforms, obligations, decomposition, competing hypotheses, and typed intent assembly. |
| Ambiguity & capability safety | 7 | Separate lexical residue from material ambiguity and stop unsupported operations before planning. |
| Candidate construction & ranking | 13 | Enumerate variable-length typed candidates, preserve obligations, rank scoped alternatives, and deduplicate equivalent plans. |
| Hard design validation | 15 | Reject identity, field, role, transform, encoding, task-chart, unsupported-design, and safety-bypass violations. |
| Soft perceptual & set audit | 5 | Warn or demote for perceptual effectiveness, cardinality, overload, duplicates, and candidate-set exhaustion without automatic mutation. |

The primary category is determined by the rule's actual runtime responsibility, not by whichever cited source is most famous. M5 is separated into hard rejection and soft warning/demotion families so perceptual guidance is not presented as a universal validity constraint.

## 3. Collection methods

| Collection method | Sources | Implemented | Survey-only | Definition |
|---|---:|---:|---:|---|
| Theory-driven canonical search | 8 | 8 | 0 | Canonical visualization design, task, grammar, and perception sources used to establish the conceptual skeleton. |
| System & artifact tracing | 17 | 8 | 9 | Official system papers, documentation, repositories, examples, and tests inspected for executable mechanisms and limitations. |
| Benchmark & evaluation-gap search | 3 | 1 | 2 | Benchmark and evaluation sources selected to represent ambiguity, multi-answer behavior, validity, and evaluation contracts. |
| Adjacent-domain transfer | 4 | 4 | 0 | Schema-linking and semantic-parsing evidence transferred only after its visualization scope and authority were narrowed. |
| Project-protocol elicitation | 1 | 1 | 0 | SkillVIS-specific authority, immutability, fail-closed, and audit policies explicitly separated from prior visualization knowledge. |

Collection method describes how a source entered the catalogue; it does not by itself grant rule authority. Admission still requires a scoped statement, explicit limitations, a typed implementation, tests, and version-isolated qualification.

## 4. Source-to-rule lifecycle

```text
decision gap
  -> primary-source search or official artifact inspection
  -> inclusion/exclusion screening
  -> scoped knowledge statement
  -> typed rule + permission + conflict policy
  -> positive/negative tests and trace fields
  -> development + sealed holdout + independent qualification
  -> versioned registry
```

A frozen failure may open a new decision gap, but it cannot rewrite its own gold, protocol, or evaluated rule pack. A revision receives a new identity and new prospective evidence.

## 5. Figure

![SkillVIS knowledge-base distributions](figures/knowledge_base_distributions_v1.png)

The left panel counts executable rules by primary knowledge family. The right panel counts catalogued sources by collection method and separates sources that contribute implemented rules from survey-only sources.

## 6. Evidence boundary

This taxonomy establishes deterministic coverage and provenance structure. It does not establish that every paper-to-rule translation is semantically faithful. That claim remains contingent on the separate, result-blind external expert review.
