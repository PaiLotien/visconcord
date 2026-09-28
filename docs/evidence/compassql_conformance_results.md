# SkillVIS v1.12.1 × CompassQL Structured Conformance Results

Status: `LOCKED_COMPLETED_CONFORMANCE_ONLY`  
Date: 2026-07-31  
Result lock SHA256:
`67563115e76a9782934432d48b7dbf6d33753c5419fead4905de70d7e48a190d`

## Outcome

The 90-invocation study completed with:

- 90/90 subprocesses completed;
- 70 first runs and 20 additional replay invocations;
- 70/70 first-run common-input parity checks passed;
- 10/10 three-run system/task replay groups were identical;
- artifact hash validation passed;
- zero LLM calls, tokens, and monetary cost for both systems.

This is a structured partial-specification **conformance** study. It has no
independent acceptable-visualization gold and does not authorize a quality,
accuracy, superiority, or non-inferiority claim.

## Common contract

Both systems received identical table bytes, selected fields, field roles,
analytical-task labels, fixed encoding/transform constraints, an unspecified
mark, and `top_k=5`. Neither received the natural-language query, an expected
chart, an acceptable visualization set, or the other system's output.

CompassQL received a native wildcard query. SkillVIS received a source-derived
typed `VisualizationIntentIR`. CompassQL has no native analytical-task slot;
SkillVIS IntentIR has no native hard channel-constraint slot. Both
representation losses are recorded rather than hidden.

## Results

| Measure | SkillVIS v1.12.1 | CompassQL 0.21.2 |
|---|---:|---:|
| Processes completed | 35/35 | 35/35 |
| Tasks with ≥1 native candidate | 28/35 | 28/35 |
| Any fully conforming candidate@5 | 28/35 | 28/35 |
| Total candidates | 47 | 84 |
| Candidate-count median | 1 | 3 |
| Conditional native-valid rate | 47/47 | 84/84 |
| Conditional field preservation | 47/47 | 84/84 |
| Conditional fixed-encoding preservation | 47/47 | 84/84 |
| Conditional required-transform preservation | 47/47 | 84/84 |
| Duplicate common projections | 0 | 0 |
| Median in-worker runtime | 23.24 ms | 9.68 ms |
| Median subprocess wall time | 349.55 ms | 85.41 ms |
| Median peak RSS | 115.98 MB | 77.38 MB |
| LLM calls / cost | 0 / 0 | 0 / 0 |

Candidate-level rates are conditional on a candidate being emitted. They must
not be confused with 35-task answer coverage.

## Complementary failure boundary

The equal 28/35 top-five conformance hides a clean task-family split:

- SkillVIS emitted no candidate on all seven `distribution` cells.
- CompassQL emitted no candidate on all seven `composition` cells.

Therefore the result is not “the systems are equivalent.” It shows that, under
this particular structured contract, they have complementary completion
boundaries:

- the current SkillVIS obligation compiler cannot close the supplied
  distribution specification in this adapter path;
- the pinned CompassQL query/configuration cannot complete the supplied
  composition/count specification.

No task was removed or rewritten after observing this behavior.

## Resource interpretation

CompassQL was faster and used less memory in this native structured-query
setting. That does not contradict the NL4DV comparison, where NL4DV performs
natural-language semantic parsing. It demonstrates why resource claims must be
attached to the exact interface and workload.

## Evaluator amendment

All 90 system invocations completed before evaluation. Frozen evaluator v1 then
raised a `KeyError` because duplicate-projection aggregation read
`projection_signature` from raw candidates instead of the already evaluated
candidate records. No evaluation result was written.

Before the first successful evaluation, amendment v1.1 froze a one-field
reference correction. Tasks, raw outputs, metrics, and thresholds were
unchanged. The amendment SHA256 is:

`b75081f4cd1780322e038de10aa377b008dc9ba128e6b9aa94368186f84648bd`

## Paper-safe statement

> On an exhaustive 35-cell structured partial-specification slice, SkillVIS
> v1.12.1 and CompassQL each produced at least one constraint-conforming
> top-five candidate for 28 cells, with complementary distribution and
> composition coverage gaps. CompassQL had lower runtime and memory on this
> native structured workload. The study evaluates conformance, not
> visualization quality.

## Artifacts

- Protocol: `docs/skillvis_v2/compassql_conformance_protocol_v1.md`
- Data and locks: `data/skillvis_v2/compassql_conformance_v1/`
- Implementation:
  `research/skillvis_v2/compassql_conformance_v1/`
- Outputs:
  `outputs/skillvis_v2/compassql_conformance_v1/skillvis-v1121-vs-compassql-conformance-v1-20260731/`
