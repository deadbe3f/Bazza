# Bazza refactor: full cached retrain

2026-10-18. Fresh head, one pass over the unchanged 96,000-example Groundbeef v0.5 selection. Frozen Gemma 4 12B IT NF4 features, raw final layer 48 with the original RMSNorm. No backbone training.

## Outcome

| Measurement | Previous trainer | Refactored trainer |
| --- | ---: | ---: |
| Selected 2,000-row validation accuracy | 69.85% | 69.90% |
| External 8-dataset macro accuracy | 68.47% | 68.06% |

The external panel contains 13,499 rows across nine tracks; the two GSM8K views are combined for the eight-dataset macro. Panel and manifest identities match the previous report. Test labels did not select the checkpoint.

| External track | Previous | Refactored |
| --- | ---: | ---: |
| gsm8k-4choice | 62.55% | 62.32% |
| gsm8k-10choice | 43.75% | 43.97% |
| when2call | 81.98% | 79.41% |
| mmlu-pro | 48.85% | 48.30% |
| arc-challenge | 92.49% | 92.41% |
| anli-r3 | 62.33% | 61.42% |
| winogrande | 71.03% | 71.43% |
| cruxeval-output-choice | 55.44% | 55.44% |
| ragtruth | 82.50% | 82.90% |

## Runtime and selection

- Training: 3,188 updates, 96,000 examples, 535.50 seconds (8.93 minutes). Includes initial/periodic full and retention validation, checkpoints and logging; excludes model/setup before the training timer and subsequent test.
- Selected checkpoint: `checkpoint-step-2661.pt` at 80,019 examples, using full-panel equal-track macro accuracy, CE tie-breaker. Best three retained.
- RTX 3090; `kiln-transformers-cuda:5.2`; image `sha256:ac19b35709b218010080711439c3791d0bf3041232535546bb6e161e6de21f21`; PyTorch `2.8.0.dev20250319+cu128`, CUDA 12.8.
- Same training settings: seed 20261013, batch 32/token budget 32768; Muon 0.003 plus auxiliary AdamW 3e-5, decay 0.01; sample WSD warmup 640, cooldown after 25K to 0.1 multiplier. Validation batch 64/token budget 65536.
- Logging: Overfit project **Bazza refactor**, run **refactor-retrain-20261018-fixed**; durable local JSONL.

## Compatibility fixes exposed by the real run

1. The first attempt stopped at 61,565 examples: a rounded teacher target sums to 0.9999, inside the 1e-4 input tolerance, but its FP32 sum crosses the boundary. Validation now uses `math.fsum` on supplied probabilities; target tensors and the loss are unchanged. Added regression coverage and checked all 96K supplied targets against the existing tolerance.
2. The corrected training completed, and all 13,499 predictions were journaled, but final test reporting failed because historical caches omit training-only category/semantic-class annotations. Metrics now use the existing domain/source when category is absent, and omit unavailable semantic-class diagnostics. The public train/automatic-test/evaluate/resume command tests now include historical-style test rows and pass.

The original failed attempt and original test exit marker (1) remain intact. The final report was rebuilt on CPU from every committed prediction, checking row IDs, complete batch membership, serialized input hashes, selection hash, and selected checkpoint hash. No inference was repeated and no labels/options/tokens changed. The reporting recovery records separate provenance; this is not represented as a clean exit of the original test process.

## Artifacts

- Settings: `build/retrain-refactor-20261018/train.py` and `test.py`.
- Failed initial attempt: `build/training/bazza/refactor-retrain-20261018/`.
- Completed training: `build/training/bazza/refactor-retrain-20261018-fixed/summary.json`.
- Original test predictions and failed report attempt: same run under `test/`.
- Recovered final report: same run under `test-reported/report.json`.
- Recovery provenance: same run under `report-recovery.json`.

This run checks cached training and reporting behavior. It does not validate the new live 12B backbone path or establish bitwise reproducibility across independent GPU runs. The observed 0.42-point macro difference is a single-run comparison, not a significance estimate.
