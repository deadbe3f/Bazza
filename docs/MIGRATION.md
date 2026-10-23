# Standalone Bazza repository

23 October 2026. Destination: `deadbe3f/Bazza`.

## Included

The refactored training/evaluation/model/feature-cache core, Python settings,
shared job launcher, source-specific data tools, plans, the full cached retrain
report, and tests. The two tested retrain compatibility fixes are included.
`MIGRATION_SOURCE.json` records the original source files and hashes.

The source was KneeJerk branch `feature/bazza-refactor`, HEAD `fcfa79e`, plus the
uncommitted retrain fixes and report. The source's local config had subsequently
acquired machine-specific paths; this repository restores the already-tested
portable config from `fcfa79e`. Local locations are in ignored `.env.local`.
The original source checkout was not edited or removed.

## Kept in KneeJerk

The complete research archive, unrelated Laya/Dory experiments, application/HTML
prototypes, and the historical run launchers. Only the import dependency closure
needed by parity tests is copied, byte-identically, to `tests/reference/legacy`.
It is test-only, not a second supported training implementation.

## Assets and exact resume

No weights, datasets, feature tensors or run directories are copied or committed.
Use the existing data/cache/model/run roots via `BAZZA_*` environment variables.
New runs write to new output directories. The old checkpoints, measurements and
source snapshots remain where they were. They are not silently rewritten for
this checkout, and exact historical resumes must use their original source.

Live Gemma uses the bundled standalone file `model/gemma.py`; no external
model-code path is configured. Cached training loads the feature-cache adapter
without constructing live Gemma.
The original model weight directory supplies the frozen letter prior unless a
separate letter-readout artifact is explicitly configured.

Groundbeef source tools under `scripts/data` retain their existing dataset/source
layout requirements and provenance ledgers; the move does not rebuild datasets.

## Migration checks

All 18 core CPU tests and 24 dataset tests passed from the standalone checkout.
A real Docker launch in plan-only mode read the existing data/cache identities
and resolved 96,000 training examples / 3,188 updates without training or live
backbone loading. Local settings remain ignored. No weights, raw data or old run
outputs are included in the Git import.

## Bundled backbone correction

The initial import retained an external live-backbone adapter. That adapter has
been removed. The standalone Gemma source is now included byte-for-byte as
`model/gemma.py` with its source notice. Its SHA-256 remains
`e6eef82665a55340ca62e248d7aa3502d196e45b5689ab2e754166c8b2cef2d0`, matching
the existing feature-bank implementation identity. No configuration field selects
external model code. Default weights are under `BAZZA_ROOT/weights`; machine-local
weight locations remain configurable without moving assets.
