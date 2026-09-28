# PCS-v7: Final Joint v1.12.1 versus NL4DV

Status: `LOCKED_COMPLETED`  
Protocol: prospective controlled shared-capability comparison  
Final SkillVIS method: `skillvis-v2-joint-semantic-precedence-v1.12.1+pcs-v7r3`  
NL4DV: `4.1.0`, repository commit
`2581c9d75632d0ed451455c5588617e9f2f13cdb`

## 1. Result in one sentence

On this **30-task, three-UCI-source, pre-frozen controlled common-scope
comparison**, Joint v1.12.1 achieved Complete Plan@1 on 30/30 tasks versus 6/30
for NL4DV, with a paired source-cluster bootstrap difference of `+0.80`
(`95% CI [0.60, 1.00]`). Both systems used zero LLM calls, tokens, and API
cost, so this experiment compares deterministic visualization-planning
capability and local resources, not LLM cost.

This statement must not be generalized to unrestricted natural language, all
visualization tasks, or official benchmark performance.

## 2. Protocol integrity

### Pre-execution gate history

Two candidate builds were blocked before any system invocation:

1. r1 found two exact historical query overlaps.
2. retry2 found that the scanner conflated a zero-invocation aborted build with
   historical system exposure.

Both attempts remain preserved. retry3:

- used a new seed and output namespace;
- separately disclosed source reuse from the zero-invocation attempts;
- found zero historical system-exposure query, UCI-ID, or source-byte overlap;
- used disjoint qualification and formal UCI sources;
- froze tasks, gold, adapters, evaluator, schedules, system versions, and
  hashes before qualification.

### Qualification

The independent engineering qualification used UCI IDs 59, 110, 850, and 878:

- 20 tasks;
- 30 invocations including fixed three-run replay cases;
- all 15 predeclared gates passed;
- Complete Plan@1, field recall, chart hit@1, transform recall@1, coverage:
  `20/20`;
- worker, input parity, trace, provenance, and replay: `PASS`;
- LLM calls / tokens / cost: `0 / 0 / 0`.

The locked qualification then authorized only the already sealed formal
schedule.

### Formal set

- UCI sources: 189, 864, and 887;
- 30 tasks: 3 datasets × 5 task families × 2 paraphrases;
- task families: comparison, ranking, correlation, distribution, composition;
- 60 first-run quality observations;
- 20 additional fixed replay invocations;
- 80/80 processes completed successfully;
- equal natural-language query and table bytes were verified per system/task;
- deterministic replay and all artifact hashes passed.

Gold was derived before execution from explicit chart requests, predeclared
schema roles, and controlled templates. It is not human expert gold.

## 3. Main results

| Metric | Joint v1.12.1 | NL4DV | Difference | Paired cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Complete Plan@1 | 1.000 (30/30) | 0.200 (6/30) | +0.800 | [0.600, 1.000] |
| Complete Plan@3 | 1.000 | 0.200 | +0.800 | [0.600, 1.000] |
| Chart hit@1 | 1.000 | 0.733 | +0.267 | [0.200, 0.400] |
| Field macro F1 | 1.000 | 0.622 | +0.378 | [0.333, 0.400] |
| Task macro F1 | 0.867 | 0.200 | +0.667 | [0.467, 0.867] |
| Transform recall@1 | 1.000 | 0.333 | +0.667 | [0.600, 0.800] |
| Encoding-role satisfaction@1 | 1.000 | 0.467 | +0.533 | [0.400, 0.700] |
| Coverage | 1.000 | 0.733 | +0.267 | descriptive |
| Invalid / failed rate | 0 / 0 | 0 / 0 | 0 | descriptive |

The predeclared superiority rule was: the paired source-cluster bootstrap 95%
interval must exclude zero. The intervals above satisfy that rule on this
controlled slice. The bootstrap has only three source clusters, so uncertainty
outside those sources remains substantial even when the within-slice interval
excludes zero.

## 4. Task-family analysis

### Joint v1.12.1

| Task family | N | Complete Plan@1 | Chart hit@1 | Field F1 | Transform recall@1 | Task F1 |
|---|---:|---:|---:|---:|---:|---:|
| Comparison | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Ranking | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.333 |
| Correlation | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Distribution | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Composition | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The apparent tension between ranking Complete Plan@1 `1.000` and task F1
`0.333` is not hidden: the correct ranking candidate, fields, chart, and
transform were present, while the semantic layer retained additional entailed
task labels. This is a task-set precision issue, not a failed final plan.

### NL4DV

| Task family | N | Complete Plan@1 | Chart hit@1 | Field F1 | Transform recall@1 | Task F1 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Comparison | 6 | 0.000 | 1.000 | 0.667 | 0.000 | 0.000 | 1.000 |
| Ranking | 6 | 0.000 | 1.000 | 0.667 | 0.000 | 0.000 | 1.000 |
| Correlation | 6 | 0.333 | 1.000 | 0.778 | 1.000 | 0.333 | 1.000 |
| Distribution | 6 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| Composition | 6 | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 |

NL4DV often produced the requested chart family for comparison and ranking, but
did not satisfy the frozen task/field/transform contract. Composition requests
frequently emitted no candidate. These are protocol-specific observations, not
a claim that NL4DV is generally unusable.

## 5. Resources

| Resource metric (first runs) | Joint v1.12.1 median | NL4DV median | NL4DV / SkillVIS |
|---|---:|---:|---:|
| In-process wall time | 91.72 ms | 1035.25 ms | 11.29× |
| Whole subprocess wall time | 460.43 ms | 3564.22 ms | 7.74× |
| CPU time | 91.53 ms | 1032.63 ms | 11.28× |
| Peak RSS | 117.09 MB | 458.69 MB | 3.92× |
| LLM calls | 0 | 0 | — |
| Prompt/completion tokens | 0 / 0 | 0 / 0 | — |
| API cost | 0 | 0 | — |

These timings apply to the frozen local environment and single-task process
contract. They should be reported with the environment and should not be
presented as universal speed ratios.

## 6. What this experiment supports

Supported:

- the final v1.12.1 implementation can be evaluated under its own version
  identity rather than inheriting v1.11 evidence;
- on a predeclared shared capability slice, its typed field/task/transform
  planning produced more complete plans than NL4DV;
- both systems were deterministic and token-free in this comparison;
- SkillVIS retained complete rule provenance and M1--M5 artifacts;
- the local resource footprint was lower under the frozen process contract.

Not supported:

- arbitrary-language or all-domain superiority;
- official UCI benchmark performance;
- conclusions about LLM Recovery;
- conclusions about unsupported/safety tasks, because the formal set contains
  only supported common-scope cases;
- human preference or perceptual-quality superiority;
- independent expert validation of M5;
- non-inferiority, because no advisor-approved margin was frozen.

## 7. Locked evidence

- Protocol SHA256:
  `3ce80662da2d89db640ed9a265aacc2b1acbec70744070881ae41e71f76cd79d`.
- Qualification result lock:
  `outputs/skillvis_v2/pcs_v7_joint_v1121_retry3/skillvis-joint-v1121-qualification-pcs-v7-20260731-r3/qualification_result_lock_v1.json`.
- Formal result lock:
  `outputs/skillvis_v2/pcs_v7_joint_v1121_retry3/skillvis-joint-v1121-vs-nl4dv-pcs-v7-20260731-r3/formal_result_lock_v1.json`.
- Evaluation summary SHA256:
  `26b5144cff625421f4c6ea68eae3b38954cc15e7d2d1510208198816167e8116`.
- Paired comparison SHA256:
  `570749f3a24d8266da9572bd0fcad1c53508707d6cc3ac77ea8c6426757d562e`.
