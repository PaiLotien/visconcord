# M5 Pristine Formal Component Validation — 2026-08-12

Status: `COMPLETE / PASS / COMPONENT-LEVEL EVIDENCE ONLY`

## 1. Evidence identity and execution

Two independent non-author visualization researchers completed the frozen
annotation packets. A distinct non-author visualization researcher served as
the adjudicator role. The annotators agreed on every frozen core label for all
140 candidates; decision and hard-rejectability Cohen's kappa were both 1.000.
There were no substantive disagreements requiring item-level adjudication.

The human gold was frozen before any pristine Validator call. The advisor then
authorized exactly one component-validation run. The formal execution used 120
cases and 140 candidates, repeated each case three times, for 360 Validator
invocations. No LLM or paid API was invoked, and no candidate or gold record was
modified.

| Frozen/run asset | SHA256 |
|---|---|
| Candidate set | `279c913613af3bfc9c16b543bde9bf22c8b74770ec8a3abf99779e8319823b05` |
| Gold | `bd2d6dbc05e9c2067b9d0ff9d97a88c585e9861fcbcdd7841f9d23929002eb89` |
| Freeze manifest | `55770b321e65d1bd4619b21ee6ad70949af0553dcf67f5a233a748bd80c39945` |
| Validator config | `9bb8031a12d573377de0ca018bb3be590794606adacd559ec03a73de981d9509` |
| Advisor authorization | `7c842ba511a64ac1d83c71012184b2c6dfa859ad5536849c566fb1b635579404` |
| Run manifest | `8b38688cb4efb5fff83ab042fc3ea16154065458398d70ec64e7fecceaa91d49` |
| Evaluation summary | `18965b490102eed298e78da8f93b8716a4e930b2087887d112bf93133eb28d4d` |
| Artifact manifest | `d3b0b0b5b1667ff624d1dd00bf98724d7f9376f78c8d1b6a9c5011894f059870` |

The artifact manifest contains 726 records. All 726 pass recomputed SHA256 and
size validation; its own sidecar also passes. No `RUN_INVALIDATED.json` exists.

## 2. Primary and safety results

Intervals below are 95% Wilson intervals unless otherwise noted.

| Measure | Result | 95% interval / detail |
|---|---:|---:|
| Final action correctness | 140/140 = 1.000 | [0.973, 1.000] |
| Hard-error detection recall | 80/80 = 1.000 | [0.954, 1.000] |
| False-safe rate | 0/80 = 0.000 | [0.000, 0.046] |
| False-rejection rate | 0/60 = 0.000 | [0.000, 0.060] |
| Unsupported executable pass-through | 0/10 = 0.000 | [0.000, 0.278] |
| Soft-issue recall | 20/20 = 1.000 | [0.839, 1.000] |
| Soft-issue precision | 20/30 = 0.667 | [0.488, 0.808] |
| Repair-proposal correctness | 100/115 = 0.870 | [0.796, 0.919] |
| Unsafe repair proposals | 0 | — |
| Audit completeness | 1260/1260 = 1.000 | [0.997, 1.000] |
| Provenance completeness | 840/840 = 1.000 | [0.995, 1.000] |
| Deterministic replay | 120/120 = 1.000 | [0.969, 1.000] |
| Candidate mutation | 0 | — |

All 13 preregistered gates pass. The transition decision is `PASS` for the M5
component. The runner explicitly does not authorize an end-to-end formal
benchmark from this result.

## 3. Ranking overlay

Ten cases contain three candidates each. Before the Validator overlay, none of
the ten top-ranked candidates was valid. After the overlay, the valid candidate
is ranked first in 10/10 cases. Invalid-above-valid inversions fall from 20 to
0. Mean delta nDCG is +0.341 under the frozen 10,000-resample case-level
bootstrap procedure. Ranking harm is 0/10.

This demonstrates the behavior of the ranking overlay on the frozen ranking
cases; it is not evidence about arbitrary candidate generators.

## 4. Error analysis: passing is not perfection

The evaluator records 15 candidates whose final action is correct but whose
observed rule set contains an additional rule not present in human gold:

- five cases (`M5P-032`, `035`, `038`, `041`, `044`) correctly trigger
  `R-VF-F002` and `REJECT`, but additionally trigger `R-VF-F003`;
- ten ranking cases (`M5P-111`–`120`) correctly trigger `R-VF-E002` and
  `REJECT`, but additionally trigger the soft perceptual warning `R-VF-P001`.

These extra triggers do not change any final accept/warn/demote/reject action,
do not create a false-safe or false-rejection case, and do not produce an
unsafe repair. They nevertheless matter for diagnostic precision and explain
the 0.667 soft-issue precision and 0.870 repair-proposal correctness. The
frozen result must therefore not be described as “perfect.”

No post-result rule, gold, or evaluator change is permitted for this run. The
extra-trigger pattern is future maintenance evidence, not a reason to rewrite
the completed experiment.

## 5. Resource observations

Across 360 invocations, recorded in-process time totals 219.26 ms wall-clock
and 217.22 ms CPU; mean per invocation is 0.609 ms wall-clock and 0.603 ms CPU.
Maximum observed process RSS is about 125.5 MB. LLM calls, prompt tokens,
completion tokens, and monetary cost are all zero.

These are local component measurements, not cross-system latency claims.

## 6. Claim permission

Allowed:

> On a pre-execution-frozen 120-case/140-candidate component validation set
> labelled independently by two non-author visualization researchers, the M5
> Validator achieved 80/80 hard-error detection, 0/80 false-safe cases, 0/60
> false rejections, and complete audit and provenance records; it passed all 13
> preregistered gates. Fifteen candidates received an additional rule trigger,
> limiting soft-issue precision to 0.667 and repair-proposal correctness to
> 0.870.

Prohibited:

- “M5 is perfect” or “M5 detects every visualization error.”
- “The entire SkillVIS system is validated by this experiment.”
- “SkillVIS is generally superior to another system.”
- “The set represents unrestricted real-world visualization requests.”
- “Structural trace completeness proves human interpretability.”

## 7. Required disclosure

The cases are new and were not used to modify M5 before execution, but they are
constructed from a predeclared, system-neutral family of error scenarios rather
than sampled from unrestricted real use. This is independent human-gold
component evidence, not an end-to-end or open-world benchmark.
