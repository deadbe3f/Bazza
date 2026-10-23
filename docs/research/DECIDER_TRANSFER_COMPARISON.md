# Bazza and Decider v1.1: completed transfer comparison

2026-10-11. Both runs completed **13,499 requests**, with the same question
and candidate membership across eight datasets / nine tracks. The two GSM8K
tracks share 1,319 underlying problems. This is our local panel comparison,
not the full Decision Index score.

## Identities and protocol

- Bazza: cooldown checkpoint **4736**, frozen Gemma 4 12B-IT NF4 backbone and
  2,301,185 trainable decision-path parameters. Checkpoint SHA256:
  `98b87b793800fd67afeb0de1370e44cdf0c905a221700912d76488b3833e52f9`.
- Decider: `perplexity-ai/pplx-decider-v1.1-27b`, revision
  `3b45dead91dfa6d95aad6b95764a606fab2bf7a6`, locally loaded in double-quantized
  NF4 with BF16 computation. The published model is a full backbone fine-tune.
- Panel: `datasets/evaluation-panel/test.jsonl`, SHA256
  `69a383d27a08b6a0dc090c1c082bb0f3626fbf1fa1d46fa767bf2696b49bc37f`.
- Native question/option order; each model's own serialization/readout.
  Batch one, maximum 8,192 input tokens, no input truncation. Decider uses its
  released temperature 1.0087417621345625 once; no calibration fitted here.
- Local reports and prediction hashes:
  `build/evaluations/{bazza,decider-v1.1}/text-decisions-v0.1/2026-10-11/`.
  Run plans retain source hashes, preprocessing and runtime details.

The models differ in size, backbone adaptation and training corpus. These
measurements do not isolate head architecture or prove training contamination
absent from either backbone/competitor.

## Completed accuracy results

| Track | Requests | Bazza 4736 | Decider v1.1 | Decider advantage, percentage points |
|---|---:|---:|---:|---:|
| When2Call | 3,652 | 34.12% | 74.70% | 40.58 |
| ANLI r3 | 1,200 | 53.92% | 70.17% | 16.25 |
| RAGTruth | 1,000 | 49.70% | 88.40% | 38.70 |
| GSM8K, 4 choices | 1,319 | 44.50% | 81.12% | 36.62 |
| GSM8K, 10 choices | 1,319 | 24.41% | 71.04% | 46.63 |
| MMLU-Pro | 2,000 | 27.50% | 68.15% | 40.65 |
| ARC-Challenge | 1,172 | 63.91% | 97.18% | 33.28 |
| WinoGrande | 1,267 | 66.77% | 92.34% | 25.57 |
| CRUXEval output choice | 570 | 53.16% | 85.26% | 32.11 |

RAGTruth's native hallucinated-class F1 is **57.34% Bazza / 83.33% Decider**.
Accuracy is shown above for consistency and prediction-bias diagnosis.
Do not average these tracks into a Decision Index number; task weights and
native metrics differ, and the GSM problems are represented twice.

## Which specific behaviours differ?

| Diagnostic | Bazza | Decider |
|---|---:|---:|
| When2Call: valid call recall | 49.96% | 91.04% |
| When2Call: clarification recall | 32.20% | 79.10% |
| When2Call: unavailable-tool/cannot-answer recall | 19.85% | 54.75% |
| When2Call: direct-response predictions (no gold A in this test) | 1,394 | 222 |
| ANLI: neutral recall | 17.91% | 67.66% |
| RAGTruth: supported-response recall | 24.69% | 92.24% |
| RAGTruth: hallucinated-response recall | 94.94% | 81.46% |

The supported-response result is particularly useful: Decider makes 50 false
hallucination flags, versus Bazza's 485, on the same 644 supported responses.
It balances the two evidence classes much better.

When2Call remains difficult even for Decider, especially unavailable tools:
it misses 586 / 1,295 such cases. The source policy and availability framing
still deserve careful conversion and label audits for our training view.

## Interpretation and v0.3 implications

Decider handles the evaluation format well. Bazza's broad transfer gap is
substantial despite its earlier competitiveness on Groundbeef's distribution.
The class-specific differences reinforce the training priorities already
identified in [Bazza's error analysis](BAZZA_TRANSFER_ERRORS.md): tool policy,
neutral evidence, supported paraphrases, arithmetic, knowledge and execution.

Proceed with the [v0.3 data plan](../plans/GROUNDBEEF_V03_PLAN.md), while keeping the
backbone/head limitations open. Full backbone tuning and 27B capacity could
account for part of the gap; more varied data alone is not a promised cure.
Use development groups for recipe decisions and preserve this external panel
for consistent comparisons, alongside an additional untouched transfer panel.
