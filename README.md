# Bazza

**Version: 0.1.0-alpha.1 — Experimental.** API and checkpoint formats may change.
The repository version is recorded in [VERSION](VERSION).

Bazza is a System 1 decision model implemented in PyTorch. It specializes a
frozen language model by training a separate decision head. Given a state and
typed questions, it scores the supplied answers without generating text.

The current model combines a frozen Gemma 4 12B IT backbone with a trainable
decision module: a bidirectional GRU, recurrent attention, and a gated pretrained
readout. The backbone implementation is included in this repository.

## Head-only adaptation

Training updates the decision head while leaving all pretrained backbone weights
unchanged. The head is stored separately and can train on reusable token features.

| Adaptation cost | Current model |
| --- | ---: |
| Trainable parameters | 30,117,964 (about 0.25% of 12B) |
| Backbone parameters updated | 0 |
| Head weight tensors, FP32 | 115.3 MiB |
| One pass over 96,000 cached examples, RTX 3090 | 8.93 minutes |

The training time includes validation, checkpointing, and logging. It excludes
feature extraction, model setup, and subsequent testing. Feature extraction is
an additional cost that can be reused across head-training experiments. The
weight size excludes optimizer state and checkpoint metadata.

## Setup

Use Python 3.10 or newer with a suitable PyTorch and CUDA installation. From the
repository root:

```bash
pip install -r requirements.txt
```

Inference requires a trained Bazza head checkpoint and matching backbone weights
and tokenizer. Replace the paths below with their local locations.

## Usage

```python
from systemone import SystemOne

model = SystemOne.from_checkpoint(
    "weights/bazza.pt",
    backbone_path="weights/Gemma4-12B-it",
    device="cuda",
)

result = model(
    state={
        "ticket": "Order #1182 arrived broken. The customer requests a refund.",
        "order_age_days": 40,
        "refund_policy_days": 30,
    },
    questions={
        "refund": {
            "type": "noul",
            "instructions": "Is the refund allowed under the policy?",
            "criteria": {
                "true": "The refund is allowed by the stated policy.",
                "false": "The refund is not allowed by the stated policy.",
            },
        },
        "action": {
            "type": "choice",
            "instructions": "Select the next action using the stated policy.",
            "criteria": {
                "refund_order": "Issue a refund because the policy permits it.",
                "explain_policy": "Explain that the order is outside the refund window.",
                "ask_order_age": "Ask the order age because it is unknown.",
            },
        },
    },
)

print(result["answers"]["refund"]["value"])          # boolean decision
print(result["answers"]["refund"]["noul"])           # probability assigned to true
print(result["answers"]["action"]["choice"])         # selected criterion key
print(result["answers"]["action"]["probabilities"])  # scores for all candidates
```

[decision.py](decision.py) is the runnable example. Set `BAZZA_CHECKPOINT` and
`BAZZA_BACKBONE_PATH` to your weights, then run `python decision.py`.

### Output types

| Type | Criteria | Result |
| --- | --- | --- |
| `choice` | Named candidate descriptions | Selected key and candidate probabilities |
| `noul` | Descriptions keyed by `"true"` and `"false"` | Probability of true and a boolean value |
| `score` | Descriptions keyed by numeric levels | Probability-weighted mean of those levels |

Each question uses 2–26 candidates and is scored separately. Returned
probabilities are softmax scores; they have not been calibrated. The current API
accepts text decisions with JSON-compatible state. Overlong inputs are rejected
rather than truncated.

For inference from a JSON request containing `state` and `questions`:

```bash
python demo.py --checkpoint weights/bazza.pt \
  --backbone-path weights/Gemma4-12B-it --request request.json
```

## Benchmarks

Accuracy (%) on the same 13,499 public-benchmark requests for every model.
MMLU-Pro uses a 2,000-example test subset. Bazza gain is relative to stock Gemma;
positive values indicate an improvement in percentage points.

| Benchmark | Examples | Stock Gemma 12B (NF4) | Bazza 12B (NF4) | Bazza gain (pp) | [Winnow 12B (Q8)](https://huggingface.co/EldanRing/Winnow-12B/tree/b6ac22b0d51b69b18200acacb3fbdd98073fffe8) | [PPLX-Decider v1.1 27B (NF4)](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/tree/3b45dead91dfa6d95aad6b95764a606fab2bf7a6) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| When2Call (test) | 3,652 | 63.34 | 79.41 | +16.07 | 83.19 | 74.70 |
| ARC-Challenge (test) | 1,172 | 90.27 | 92.41 | +2.13 | 94.88 | 97.18 |
| ANLI round 3 (test) | 1,200 | 58.25 | 61.42 | +3.17 | 62.17 | 70.17 |
| WinoGrande (validation) | 1,267 | 71.82 | 71.43 | -0.39 | 73.88 | 92.34 |
| MMLU-Pro (test subset) | 2,000 | 44.00 | 48.30 | +4.30 | 54.30 | 68.15 |
| GSM8K, 4 choices | 1,319 | 50.87 | 62.32 | +11.45 | 61.18 | 81.12 |
| GSM8K, 10 choices | 1,319 | 24.34 | 43.97 | +19.64 | 35.86 | 71.04 |
| CRUXEval output choice | 570 | 56.67 | 55.44 | -1.23 | 69.65 | 85.26 |
| RAGTruth classification | 1,000 | 76.90 | 82.90 | +6.00 | 82.60 | 88.40 |
| **Eight-dataset macro** | — | **62.36** | **68.06** | **+5.70** | **71.15** | **81.54** |

GSM8K and CRUXEval use adapted multiple-choice formats; RAGTruth uses binary
response-level classification accuracy. The macro weights eight datasets equally,
combining the two GSM8K views first; they share the same 1,319 source problems.

Bazza uses the frozen [Gemma 4 12B IT backbone](https://huggingface.co/google/gemma-4-12B-it/tree/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7)
and the head checkpoint trained on Groundbeef v0.5 at step 2,661 (80,019 training examples),
selected on validation. Stock Gemma uses its original answer-letter weights and
softcapping on the same cached features and prompts as Bazza. The head adds
**16.07 percentage points on When2Call** and **4.30 points on MMLU-Pro**; WinoGrande
drops by 0.39 points.

All runs use an RTX 3090 with no calibration fitted on the test panel. Winnow and
PPLX-Decider use their own prompts and readouts. These are decision-format accuracy
measurements, not official Decision Index scores or generative benchmark results.

## Training and evaluation

Groundbeef is our internally curated training dataset: a mixture of existing
source datasets with our preprocessing and data splits. It is not included in
this repository and may be released separately in the future.

Learning rates, batch sizes, data locations, and model settings are Python values
in [config.py](config.py). Alternative settings files export a `Config` instance
named `config`. Relative paths resolve against the settings file's directory;
the default settings also accept `BAZZA_*` storage roots from the environment.

Training requires a prepared data selection. With `backbone_mode="cache"`, stored
token representations replace the frozen backbone, allowing the decision module
to train without recomputing them.

```bash
python train.py --config config.py --output-dir outputs/train --plan-only
python train.py --config config.py --output-dir outputs/train

python evaluate.py --config test_config.py \
  --checkpoint outputs/train/checkpoint-step-N.pt \
  --output-dir outputs/test --split test
```

Training shows a progress bar and writes metrics and checkpoints to the output
directory. `--resume` restores optimizer, schedule, random state, callbacks, and
the data cursor for the same run. The latest checkpoint and best three validation
checkpoints are retained by default. Set `OVERFIT_URL` in the default settings to
publish metrics to Overfit & Chill.

## Implementation

- [model/](model/): standalone backbone, decision module, and feature-cache adapter.
- [trainer.py](trainer.py): training loop and named callback events.
- [callbacks.py](callbacks.py): checkpointing, early stopping, and logging.
- [data.py](data.py), [losses.py](losses.py), [metrics.py](metrics.py): serialization,
  batching, hard/soft-target loss, and evaluation metrics.

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) and
[third-party attributions](model/NOTICE.md).
