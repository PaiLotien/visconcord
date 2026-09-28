# GitHub upload checklist

The author has confirmed a public named repository called `visconcord`, public
release of the packaged validation results and adjudicated labels, and the
author order used in the manuscript.

## Author decisions recorded

- [x] Public named repository: `visconcord`.
- [x] The packaged 120 validation contexts, 140 candidates, 360 validator
      replays, saved outputs, and adjudicated labels may be published.
- [x] No ethics approval or exemption is claimed for this release. Do not add an
      ethics-approval number.
- [x] Author order follows the paper and is recorded in `CITATION.cff`.
- [x] MIT License for project code.
- [x] CC BY 4.0 for original project data and documentation; UCI-derived and
      third-party assets retain their source terms.

## Remaining mechanical steps

- [x] Created the public GitHub repository at
      `https://github.com/PaiLotien/visconcord` and set its URL in the
      submission declaration.
- [x] First release tag selected: `v1.0.0`.
- [x] GitHub URL recorded in `CITATION.cff` and the submission declaration.
- [ ] Add an archival Zenodo/DOI identifier later if one is created.
- [ ] Confirm CRediT roles, competing-interest wording, and funding wording with
      every author; none are inferred from commits or file history.
- [ ] Run installation and `python scripts/reproduce.py` on a fresh machine.
- [ ] Inspect the complete staged file list before pushing.

## Local checks completed

- [x] Candidate copied into an isolated directory.
- [x] Python caches, `.pyc` files, and OS metadata removed.
- [x] No `.env` or obvious secret-bearing files copied.
- [x] Private authorization and annotator-submission packets remain excluded.
- [x] Deterministic replay and saved-output recomputation pass.
- [x] Manifest and SHA256 sidecar are present.
