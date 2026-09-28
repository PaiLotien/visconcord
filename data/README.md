# Packaged data

This directory contains the frozen task, dataset, gold-label, configuration,
and validation assets used by the packaged checks. The files are included so
that the deterministic smoke replay and the saved-output M5 recomputation can
run without downloading additional data.

Original project material is covered by CC BY 4.0 where the authors hold the
rights. UCI-derived datasets, source metadata, and other third-party assets
retain their original licenses and attribution requirements; see the root
`LICENSE-DATA.md`.

The public release includes the packaged validation outputs and adjudicated
labels. Private authorization packets and other excluded files are absent.
The public M5 verification recomputes metrics from the frozen candidate set,
adjudicated gold, validator configuration, and saved outputs; it does not
reproduce a private authorization ceremony.
