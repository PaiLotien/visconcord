# Reproducibility levels

## Level 1 — packaged-artifact integrity

`python scripts/reproduce.py` verifies `release_manifest.json` and its SHA256
sidecar. This detects missing or modified files after packaging.

## Level 2 — deterministic method smoke replay

The script runs Joint v1.12.1 three times on the first frozen PCS-v7 task and
requires byte-stable candidate signatures. This validates import closure,
configs, data loading, deterministic execution, and the zero-LLM fast path.

## Level 3 — result reconstruction

- M5 metrics are recomputed from the 360 packaged validation outputs and frozen
  human-adjudicated gold, then compared with the archived formal summary.
- PCS-v7 and CompassQL values are validated from their publication-level frozen
  summaries and result locks.

The raw PCS-v7 and CompassQL invocation traces are not in this anonymous
candidate because they contain workstation-specific absolute paths. Their
protocol-lock hashes are retained in `provenance/`. A later public release may
either regenerate clean raw traces under a new reproduction identity or add a
documented path-redacted derivative. Neither operation may rewrite the identity
of the original formal run.

The original M5 preflight additionally verifies private annotator submissions,
participant-role declarations, and advisor authorization evidence. Those items
are excluded from the anonymous candidate. The public recomputation gate checks
the frozen candidate set, adjudicated gold, Validator config, and release
manifest before calling the unchanged metric formulas. This reproduces scores;
it does not reproduce the private authorization ceremony.

UCI source-metadata files are retained byte-for-byte so that dataset identities
continue to match the frozen task manifests. Public contact emails already
present in those UCI metadata records are counted separately by the anonymity
scanner; email addresses anywhere else remain a hard failure.

## Full fresh rerun

A fresh tool-comparison rerun requires separately installing NL4DV 4.1.0 at the
recorded commit and the CompassQL runtime. Such a rerun is a new reproduction
run, not the original formal experiment, and must receive a new manifest.

## Environment

- Python 3.11 or newer is recommended.
- Dependencies are intentionally limited to the packages in `requirements.txt`.
- No API key is required for the packaged deterministic and M5 checks.
