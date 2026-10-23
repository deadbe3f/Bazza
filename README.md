# Bazza

Bazza is a System 1 decision model implemented in PyTorch. A frozen
language-model backbone produces token representations. A trainable decision
module combines a bidirectional GRU, recurrent attention, and a gated pretrained
readout to score the supplied options. Cached token representations can replace
the backbone during training and evaluation.

## Inference example

Load a trained decision checkpoint and supply a state and named questions:

```python
from systemone import SystemOne

model = SystemOne.from_checkpoint(
    "weights/bazza.pt",  # trained decision head
    backbone_path="weights/Gemma4-12B-it",  # backbone weights and tokenizer
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

print(result["answers"]["refund"]["noul"])  # probability assigned to true
print(result["answers"]["refund"]["value"])  # boolean decision
print(result["answers"]["action"]["choice"])  # supplied candidate key
print(result["answers"]["action"]["probabilities"])
```

The response has an `answers` mapping keyed by question name. `choice` preserves
the supplied criterion key; `probabilities` maps every criterion key to its model
softmax score. Boolean (`noul`) answers also provide `noul`, the probability of
`true`, and a boolean `value`. Numeric `score` questions require numeric criterion
keys and return their probability-weighted mean in `score` and `value`.

Each question is serialized and scored separately, using the training prompt
format and candidate order. No training targets, feature bank, data selection, or
training configuration are required. Overlong inputs are rejected without
truncation. `model.model` exposes the underlying PyTorch `Bazza` instance.
This is a Python API using the System 1 request schema; no HTTP server is started.

The self-contained example in `decision.py` defines the state and
questions inline. Run it from the repository root with `python decision.py`,
with `BAZZA_CHECKPOINT` and `BAZZA_BACKBONE_PATH` set to your weights.
The CLI uses the same API with a JSON request file, such as
`examples/decision.json`:

```bash
python demo.py \
  --checkpoint /path/to/bazza.pt \
  --backbone-path /path/to/Gemma4-12B-it \
  --request examples/decision.json

# Docker execution, durable response and job controls:
bash run.sh demo --env-file .env.local \
  --checkpoint /path/to/bazza.pt \
  --backbone-path /path/to/Gemma4-12B-it \
  --request examples/decision.json \
  --output-dir /path/to/new-demo-output
```

The CLI prints the response JSON. With `--output-dir`, it also records the request,
response, input identity, timing and status. A completed `--resume` returns the
stored response without running inference again.

## Architecture

The model takes a serialized request containing context, a question, and candidate
answers. The decision module processes the backbone's token representations in
four stages:

1. Layer normalization and projection into the decision-module hidden dimension.
2. A bidirectional GRU with a gated residual connection over the token sequence.
3. A recurrent attention module initialized from the final context token and
   candidate-token representations.
4. A gated combination of decision logits and pretrained answer-letter logits,
   with a learned residual correction to the final token representation.

The default configuration uses a frozen Gemma 4 12B IT backbone, a hidden dimension
of 1,024, four attention heads, two low-level updates per high-level cycle, and
two high-level cycles. The number of candidate answers can vary from 2 to 26.
Training updates the decision module while keeping the backbone frozen.

`Gemma4` computes token representations from input tokens. `Gemma4FeatureCache`
reads stored representations and applies the final RMSNorm when the stored
features are unnormalized. Both implement the same `BackboneOutput` interface.

## Repository structure

```text
train.py             training entrypoint
trainer.py           training loop and event dispatch
callbacks.py         checkpointing, early stopping, and logging callbacks
evaluate.py          evaluation and resumable prediction storage
demo.py              load a model and answer a new JSON request
systemone.py         state/questions decision API
decision.py          self-contained model-loading and decision example
examples/            sample JSON request
data.py              input serialization, collation, and batching
losses.py            hard- and soft-target cross-entropy
metrics.py           accuracy, cross-entropy, and grouped metrics
optimizers.py        Muon and AdamW parameter groups
config.py            Python experiment settings
settings.py          Config dataclass and Hugging Face metadata export
configuration.py     configuration loading and model construction
logging_backend.py   local metrics and optional Overfit logging
utils.py             file I/O, hashing, and precision utilities
model/
  model.py           backbone and decision-module composition
  backbone.py        generic output contract and feature-cache machinery
  gemma_backbone.py  Gemma-specific loading, readout extraction and adapters
  gemma.py           standalone PyTorch Gemma backbone
  readers.py         token projection and bidirectional GRU
  reasoner.py        recurrent attention module
  readout.py         answer-letter readout, gating, and residual correction
features/            feature-bank readers
run.sh               Docker launcher and job controls
docker_job.sh        signal handling and execution logs
tests/               numerical, training, and command tests
  reference/         historical implementations used for comparison
```

## Dependencies

Use Python 3.10 or newer with a suitable PyTorch CPU or CUDA installation.
Install `requirements.txt` for training and evaluation, or
`requirements-live.txt` for live NF4 inference.

The Docker launcher defaults to the local RunPodML-derived image
`bazza-cuda:local`. Build it from an available RunPodML base image before launching a job:

```bash
docker build -t bazza-cuda:local .
# Alternative local CUDA base:
# docker build --build-arg BASE_IMAGE=your-cuda-image -t bazza-cuda:local .
```

The image installs the live-inference dependencies, including bitsandbytes.
The image name is configurable through `BAZZA_IMAGE`.

Cached training requires an encoded data selection, a compatible feature bank,
and pretrained answer-letter weights. The letter weights can be read from the
backbone checkpoint or supplied through `letter_readout_path`. Live inference uses the bundled standalone model in `model/gemma.py` and the
weights specified by `backbone_path`. The backbone does not import Transformers
or code from another repository.

## Configuration

Experiment settings are Python values in `config.py`. Alternative settings files
must export a `Config` instance named `config`:

```python
from pathlib import Path
from settings import Config

config = Config(
    data_selection=Path("/path/to/selection.json"),
    backbone_mode="cache",
    feature_format="layer",
    feature_cache_path=Path("/path/to/features"),
    letter_readout_path=Path("/path/to/letters.pt"),
    learning_rate=3e-5,
    muon_learning_rate=0.003,
    batch_size=32,
    epochs=1,
)
```

Relative paths resolve against the settings file's directory. The training code
constructs the model and optimizer from these values. Each run records the Python
settings, resolved configuration, and source hashes.

The default configuration obtains storage locations from environment variables:

| Variable | Launcher option | Default |
| --- | --- | --- |
| `BAZZA_ROOT` | `--root` | repository `build/` |
| `BAZZA_PROJECT_ROOT` | `--project-root` | repository parent |
| `BAZZA_DATA_ROOT` | `--data-root` | root `data/` |
| `BAZZA_CACHE_ROOT` | `--cache-root` | root `features/bazza/` |
| `BAZZA_MODEL_ROOT` | `--model-root` | root `weights/` |
| `BAZZA_RUN_ROOT` | `--run-root` | root `training/bazza/` |
| `BAZZA_SELECTION_PATH` | `--selection` | data root `groundbeef/v0.5/selection.json` |

Local settings can be placed in an ignored `.env.local` file and loaded explicitly
with `--env-file .env.local`. Command-line root options override that file, which
overrides inherited environment variables. Input directories are mounted
read-only; the selected output directory is writable.

Set `OVERFIT_URL` to enable Overfit logging. Metrics are also stored locally in
JSONL format. An unset URL disables remote logging.

## Training

Check the configuration and available inputs without training:

```bash
bash run.sh train \
  --config config.py \
  --env-file .env.local \
  --output-dir /path/to/new-run \
  --plan-only
```

Remove `--plan-only` to train with a tqdm progress bar. Add `--detach` to run in
the background. The launcher provides the following controls:

```bash
bash run.sh train --output-dir /path/to/run --status
bash run.sh train --output-dir /path/to/run --logs
bash run.sh train --output-dir /path/to/run --stop
bash run.sh train --config config.py --env-file .env.local \
  --output-dir /path/to/run --resume --detach
```

Resume restores model parameters, optimizer and scheduler state, random-number
state, callbacks, and the data cursor. It requires matching source, configuration,
data, and runtime identities. `--initialize-from /path/to/checkpoint.pt` starts a
new experiment from saved head weights without restoring optimizer state.

Checkpoints retain the latest state and the best three validation states by
default. Selection uses equal-track macro accuracy, with macro-track
cross-entropy as the tie-breaker. Setting `test_config_path` runs a separate test
panel on the selected checkpoint after training.

`Trainer` exposes `train()`, `train_batch()`, and `validate()`. Callbacks subscribe
to named events, including batch boundaries, forward and backward passes,
optimizer updates, validation, and checkpoint saves:

```python
trainer.on("batch_start", log_batch_start)
trainer.on("after_backward", inspect_gradients)
trainer.on("validation_end", stop_callback, key="early_stopping")
trainer.train(train_batches, validation_batches)
```

## Evaluation

```bash
bash run.sh evaluate \
  --config /path/to/test-settings.py \
  --env-file .env.local \
  --checkpoint /path/to/run/checkpoint-step-N.pt \
  --output-dir /path/to/new-test-output \
  --split test
```

Evaluation preserves candidate order and stores completed prediction batches in
a journal. Resume validates the stored input identities and continues from the
last committed batch. `demo.py` accepts raw requests through `--request`; it does not run evaluation.
Training, evaluation, and demo support the same launcher controls.

## Data and precision

A selection file contains backbone identity metadata and named splits of encoded
records. Each record includes token IDs, candidate positions, candidate order,
and a target distribution. `target_key` selects the stored target field; the
loss accepts hard labels represented as one-hot vectors and soft teacher targets.

Feature formats are per-example tensors (`pt`), raw layer shards (`layer`), and
multiple feature banks with explicit record routing (`routed`). Raw final-layer
features receive the stored final RMSNorm once, outside decision-module autocast.
The current cache adapter supports the final layer, numbered 48 for this backbone.

Live backbone execution uses unpadded single-example forwards. Decision-module
batches are padded and bounded by a token budget. The default decision-module
precision is BF16 autocast with FP32 GRU computation.

## Configuration export

```python
from config import config
from settings import export_to_hf_config

export_to_hf_config(config, "export/config.json")
```

This exports architecture metadata without local storage paths. Optional
`include_training=True` adds training hyperparameters. Weight export and
Transformers AutoModel registration are not implemented.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests cover numerical agreement with reference implementations, optimizer
updates, event ordering, checkpoint selection, training and evaluation resume,
configuration portability, and input serialization.

## Code style

Development tools are listed in `requirements-dev.txt`. Format and lint with:

```bash
python -m pip install -r requirements-dev.txt
ruff format .
ruff check .

# Check formatting without editing files:
ruff format --check .
```

`pyproject.toml` configures an 88-character formatter line length, sorted imports,
and basic PEP 8/error checks. Frozen reference implementations under
`tests/reference/` and local run artifacts are excluded. Active model code is
formatted; syntax-equivalence checks preserve compatibility with the original
Gemma feature-bank source identity when only formatting has changed. Full run
resume continues to require exact source identities.
