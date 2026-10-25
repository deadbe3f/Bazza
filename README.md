# Bazza

**Version: [0.5.0](VERSION)**

Bazza is a System 1 decision model implemented in PyTorch. It specializes a
frozen language model by training a separate decision head. Given a state and
typed questions, it scores the supplied answers without generating text.

The current model combines a frozen Gemma 4 12B IT backbone with a trainable
decision module: a bidirectional GRU, recurrent attention, and a gated pretrained
readout. The backbone implementation is included in this repository.

## Why a frozen backbone?

Freezing the backbone preserves the pretrained model while allowing decision
heads to be trained and replaced independently. Cached representations make head
training inexpensive and enable rapid architecture experiments. Longer term,
this provides a foundation for sharing one backbone between fast decisions and
autoregressive planning or generation.

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

The published head is available at [0xdeadbe3f/Bazza](https://huggingface.co/0xdeadbe3f/Bazza).
The loader downloads the head and its exact Gemma backbone revision into the
Hugging Face cache. The original BF16 backbone download is substantially larger
than the 115 MiB head; inference quantizes its linear weights to NF4 on CUDA.

## Usage

```python
from systemone import SystemOne

model = SystemOne.from_pretrained("0xdeadbe3f/Bazza", revision="0.5.0")

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

Use `backbone_path="/path/to/Gemma4-12B-it"` to reuse local backbone weights.
For offline loading, provide a local head release directory and backbone path,
with `local_files_only=True`. Loading uses this repository's PyTorch code;
Transformers `AutoModel` integration is not required or provided.

Existing training checkpoints can still be loaded with
`SystemOne.from_checkpoint("weights/bazza.pt", backbone_path="weights/Gemma4-12B-it")`.

[decision.py](decision.py) is the runnable local-checkpoint example. Set
`BAZZA_CHECKPOINT` and `BAZZA_BACKBONE_PATH` to your weights, then run
`python decision.py`.

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

Edit [config.py](config.py) to set the learning rates, batch sizes, data selection,
and model locations. Settings are Python values. Use `backbone_mode="live"` with
local backbone weights, or `backbone_mode="cache"` with prepared token features.
The data selection must contain the requested training, validation, or test splits.

Run directly from the repository root:

```bash
python train.py --config config.py --output-dir outputs/train

# Resume the same training run.
python train.py --config config.py --output-dir outputs/train --resume

# Evaluate a saved checkpoint using the configured test split.
python evaluate.py --config config.py \
  --checkpoint outputs/train/checkpoint-step-N.pt \
  --output-dir outputs/test --split test
```

Training displays a progress bar and saves metrics and checkpoints in the output
directory. Resume restores optimizer, schedule, random state, and data position.
The latest checkpoint and best three validation checkpoints are retained by
default. Set `OVERFIT_URL` to send metrics to Overfit & Chill.

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
