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

- Live inference uses the bundled standalone `model/gemma.py`; no external model
  repository is required. The complete live 12B path has not received a GPU parity
  run as part of this change.
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

## Standalone backbone

The standalone PyTorch Gemma implementation is included in `model/gemma.py`,
byte-identical to the source used for the stored feature banks. The live adapter
imports this local module lazily; cached training does not load it.
The default launcher image is `bazza-cuda:local`. On the development machine this
is an alias of the previously verified CUDA image, with identical binaries.

The standalone correction passes 21 core tests and 24 dataset tests. The new
checks instantiate and execute a reduced-size Gemma backbone through the local
loader, compare its unpadded outputs with cached states, reject source identity
mismatches, and confirm that the bundled file matches the feature-export source
hash. The container mounts only this repository; no external code checkout or
model weights are needed for these tests. This is not a full 12B GPU evaluation.

## Public inference API — 23 October 2026

`model/backbone.py` now contains only `BackboneOutput`, `FeatureCacheMixin`, and
the generic cache implementation. Gemma-specific loading, source checks, letter
readout extraction and final-layer selection are in `model/gemma_backbone.py`.
The standalone transformer file remains unchanged in `model/gemma.py`.

`SystemOne.from_checkpoint(checkpoint, backbone_path=...)` loads the trained head
and live backbone without a training selection or feature bank. `SystemOne`
accepts `state` and named `questions` with `type`, `instructions`, and `criteria`;
responses preserve question names and candidate keys. Supported types are choice,
boolean `noul`, and finite numeric `score`. Training and inference share the same
serializer. Tests compare tokens, option positions, order and labels against the
pre-refactor serializer, including Unicode and reordered candidates.

The demo accepts new requests from JSON and has no dependency on evaluate.py.
A Python loading example and editable JSON request are in `examples/`. The API is
in-process; an HTTP server is not included.

All 26 core/API tests and 24 dataset tests pass. The live check loaded checkpoint
`checkpoint-step-2661.pt` from the 96K refactor retrain and answered the two example
questions on the RTX 3090. It selected refund=false (0.9871985 model score) and
action=explain_policy (0.9989470). Loading and the two decisions took 38.72 seconds;
this includes checkpoint/backbone loading and weight verification, and is not a
steady-state inference latency measurement. The job exited 0.

The local `bazza-cuda:local` image now includes bitsandbytes 0.50.2; the initial
image alias lacked that dependency. Its Dockerfile installs the declared live
requirements. The successful image ID is
`sha256:5084e3ba29aaae6840e88eb7b96b744766ac10bbacf81a0a5df3d3c266f00edc`.
Local request/response artifacts are under `build/demo/systemone-20261023-live/`
and remain ignored. This functional check does not replace a full live/cache
12B numerical parity evaluation.
