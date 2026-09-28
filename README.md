# VisConcord

This repository contains the public named code-and-data package for the
VisConcord paper, *Aligning Recommendation and Validation Rules with
Visualization Research Findings*. It packages the final deterministic method
`skillvis-v2-joint-semantic-precedence-v1.12.1`, its frozen data/configuration
assets, saved evaluation outputs, and the scripts needed to verify the package.

The repository is prepared for a public GitHub release under the name
`visconcord`. The code is released under the MIT License. Original project
data and documentation are released under CC BY 4.0 where the authors own the
rights; UCI-derived and other third-party material remains subject to its
original terms. See [`LICENSE`](LICENSE) and [`LICENSE-DATA.md`](LICENSE-DATA.md).

## What is included

- implementation closure for M1–M5 and the frozen configuration files;
- the 107-rule registry, source registry, taxonomy, and paper-to-rule assets;
- PCS-v7 task, dataset, and gold assets with publication-level summaries;
- CompassQL structured-conformance assets and summaries;
- the M5 frozen candidate set, adjudicated gold, validator configuration, and
  360 saved validator outputs;
- provenance records, an automated anonymization scan, a file manifest, and
  one-command verification.

## Reproduce the packaged checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

The command checks the manifest hashes, runs the deterministic v1.12.1 smoke
replay three times, checks the archived PCS-v7 and CompassQL headline values,
and recomputes M5 metrics from the saved outputs. It does not call an LLM or a
paid API and it does not rerun a formal benchmark. It writes a local
`verification_report.json`, which is ignored by Git.

For a fresh manifest after a deliberate package edit, run:

```bash
python scripts/build_manifest.py
```

Then run `python scripts/reproduce.py` again. The manifest intentionally excludes
`release_manifest.json`, its SHA256 sidecar, and generated
`verification_report.json`.

## Evidence boundary

PCS-v7 is a controlled 30-task, three-source, five-task-family comparison; it
is not open-language or universal-superiority evidence. CompassQL is reported
for structured conformance only. M5 is component-level evidence on 120
validation contexts, 140 candidates, and 360 validator replays (three replays
per context). The package reproduces the archived metrics from the saved
outputs; it does not reproduce the private authorization ceremony or claim a
fresh tool-comparison rerun.

## Citation and submission text

- [`CITATION.cff`](CITATION.cff) records the paper title, version, and author
  order from the manuscript.
- [`SUBMISSION_DECLARATION.md`](SUBMISSION_DECLARATION.md) contains ready-to-use
  data/code availability, competing-interest, funding, ethics/consent, and
  generative-AI wording. Replace only the repository URL, release tag, and DOI
  after the GitHub repository is created.
- [`GITHUB_UPLOAD_CHECKLIST.md`](GITHUB_UPLOAD_CHECKLIST.md) records the release
  decisions and the remaining mechanical publication steps.

## Current package status

The project is named `VisConcord`, the repository is prepared for public named
release, and the author-approved data release includes the packaged validation
outputs and adjudicated labels. The GitHub URL, release tag, and optional DOI
are intentionally left for the final repository creation step.
