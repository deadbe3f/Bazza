# Bazza implementation status

## Standalone repository — 23 October 2026

The production implementation was copied from KneeJerk's `feature/bazza-refactor`
worktree, including the compatibility fixes found by the 18 October retrain.
The original checkout and its uncommitted edits were preserved.
See [migration](MIGRATION.md) for scope and source hashes.

Implemented: root train/evaluate/demo commands; `Trainer.train()` and ordered event
callbacks; a separate model package; interchangeable live/cache backbones; Python
settings and explicit HF metadata export; full-state checkpoints and exact resume;
best-three retention; resumable evaluation journals; optional Overfit logging;
and foreground/detach/stop/resume/logs/status Docker controls.

## Real cached retrain

A fresh 96,000-example Groundbeef v0.5 run completed in 535.50 seconds including
initial/periodic validation, logging and checkpoints. The 80,019-example checkpoint
was selected on fixed validation: 69.90% pooled accuracy versus 69.85% previously.
Its external 13,499-row test macro was 68.06% versus 68.47% previously, using the
same panel identities and eight-dataset macro (two GSM8K views combined).
See [full results](research/BAZZA_REFACTOR_RETRAIN_20261018.md).

The real run exposed an FP32 probability-sum boundary check and historical test
records lacking training-only diagnostic tags. Both are fixed and covered by
regression tests. The final report was recovered from all saved predictions;
original failure markers were retained rather than rewritten as a clean process
exit. No targets, options, tokenization, or loss reduction were changed.

## Verification environment

NVIDIA RTX 3090; RunPodML-derived `kiln-transformers-cuda:5.2`, image
`sha256:ac19b35709b218010080711439c3791d0bf3041232535546bb6e161e6de21f21`;
PyTorch `2.8.0.dev20250319+cu128`, CUDA 12.8.

CPU tests cover exact historical head state keys/initialization/logits/CE/gradients,
optimizer and sample-schedule parity, callback ordering, interrupted full-state
training resume, public train/test/demo commands, interrupted prediction recovery,
Python settings and portable paths, live/cache miniature parity, and raw-final
normalization once. Tests use frozen originals under `tests/reference/legacy`;
production code never imports those fixtures.

## Remaining limits

- The live full-size 12B adapter has not received a real GPU parity run.
- Live frozen backbone forwards remain unpadded batch one; larger batches are
  supported in the cached decision head.
- Whole final-layer caching is supported; arbitrary cached prefixes and tap
  fusion remain planned.
- HF export currently produces metadata, not an AutoModel implementation or
  released weights.
- The hybrid System 1 / System 2 design remains a plan.
- Exact resume rejects changed source/config/runtime and moved asset identities;
  historical runs resume in their original checkout. No checkpoints are modified
  by this repository move.
