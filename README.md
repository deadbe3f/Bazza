# Bazza

Bazza is our System 1 decision model: a frozen backbone, token reader, recurrent
workspace, and variable-option decision/readout head. Groundbeef is the dataset.
Kiln owns the canonical readable PyTorch backbone implementations.

## Code layout

```text
train.py             configure and launch training
trainer.py           Trainer.train(), train_batch(), validate(), event dispatch
callbacks.py         checkpoints, stopping, logging and training observations
evaluate.py          fixed-panel evaluation with resumable prediction commits
demo.py              inspect an example through the same model/input path
data.py              named batches, unchanged prompt serialization and batching
losses.py            explicit native/teacher target CE and fitted weights
metrics.py           accuracy, CE and grouped reporting
optimizers.py        existing Muon/AdamW implementation and parameter assignment
config.py            editable Python values: paths, batch sizes, learning rates, etc.
settings.py          Config object and explicit export_to_hf_config() adapter
configuration.py     load settings, validate identities and assemble components
logging_backend.py   durable JSONL and optional Overfit publishing
utils.py             small IO, identity and precision helpers
model/
  model.py           Bazza and BazzaDecisionHead composition
  readers.py         recurrent decision interface and token readers
  reasoner.py        recurrent workspace and original small value decoder
  readout.py         pretrained letters, request gate and residual answer adapter
  backbone.py        Gemma4 / Gemma4FeatureCache, shared output and Kiln adapter
features/            CPU feature-bank IO and raw-shard readers
configs/             alternative Python settings files
scripts/
  run.sh             common Docker job controls
  lib/               signal forwarding and durable console/exit markers
  data/              source-specific dataset preparation and audits
tests/              focused parity, event, resume and command checks
  reference/         frozen historical code used only by parity tests
docs/plans/         future designs and agreed implementation plans
docs/research/      historical measurements and run notes
```

The production core never imports the historical test references. All current model math and
mixed-head state-dict names are retained. See [refactor status](docs/REFACTOR_STATUS.md)
for verification scope and the historical checkpoint/resume boundary.

This is the standalone model repository extracted from KneeJerk. The full research
archive and application prototype remain there. Weights, datasets, feature banks
and existing runs stay in their original storage locations; configure roots to
use them. See [migration notes](docs/MIGRATION.md) for provenance and scope.

## Environment

Use Python 3.10 or newer and install `requirements.txt` into your chosen PyTorch
environment. Groundbeef source preparation additionally uses
`requirements-data.txt`; live NF4 mode additionally uses `requirements-live.txt`.
The existing Docker launcher uses the local
RunPodML-derived `kiln-transformers-cuda:5.2` image; it must already be available.
The verified runtime is recorded in `docs/REFACTOR_STATUS.md`.

Cached training does not import or construct Kiln's live model. It needs an
existing feature bank, encoded selection, and either the original pretrained
letter rows (read from backbone weights) or an explicit `letter_readout_path`.
Live mode additionally needs a Kiln checkout through `kiln_model_path`. A fresh
clone does not download weights or prepared training data automatically.

## Python settings and model composition

Edit `config.py` to change ordinary values. It contains a flat Python `Config`
object, not model constructors or a nested JSON recipe:

```python
config = Config(
    data_selection=Path("/path/to/selection.json"),
    backbone_mode="cache",
    learning_rate=3e-5,
    muon_learning_rate=.003,
    batch_size=32,
    epochs=1,
    # ... data/cache locations and other settings
)
```

Model construction stays in the training code. `backbone_mode="live"` selects
`Gemma4`; `"cache"` selects `Gemma4FeatureCache`. Both go through
`Bazza.forward(batch)` and return the same `BackboneOutput` contract:

```python
backbone = Gemma4FeatureCache(directory=cache_path, selection=selection)
# Or: backbone = Gemma4(weights=weight_path, kiln_model_dir=kiln_path, selection=selection)
model = Bazza(backbone=backbone, head=head)
loader = BatchLoader(backbone.prepare_batch, device="cuda")
trainer = Trainer(model, optimizer, loss_fn, scheduler, prepare_batch=loader)
trainer.on("batch_start", log_batch_start)
trainer.on("validation_end", stop_callback, key="early_stopping")
trainer.train(train_batches, validation_batches)
```

The loader only collates and prefetches CPU inputs. Backbone execution and
normalization belong to the model; the trainer has no live/cache branch.
The cache class never constructs a live Gemma. `Bazza` checkpoints the head,
referencing the frozen backbone by run identity. Whole-backbone final-layer
replacement is implemented; cached prefixes and multi-tap fusion remain planned.

Callbacks are ordered functions/callable objects in an event-key registry.
`on()` subscribes; `emit()` dispatches. Unknown names are errors. Training and
validation produce batch/forward events with explicit stages. Stateful callbacks
have stable keys. Checkpoint requests commit after all handlers at a boundary.

Each run records `config-source.py`, `resolved-config.json`, and source/settings
hashes. Python is the input; JSON records are generated outputs.

```python
from config import config, export_to_hf_config

metadata = export_to_hf_config(config)
export_to_hf_config(config, "export/config.json")
```

The explicit HF adapter exports portable architecture metadata, with optional
`include_training=True`. Local data/cache paths are excluded. It does not create
weights, register an AutoModel, or provide from_pretrained support; a Transformers
model/config adapter is separate release work. See the
[HF custom-model contract](https://huggingface.co/docs/transformers/main/en/custom_models).

## Portable locations

The active configuration has no machine-specific paths. `config.py` derives
locations from environment variables; `run.sh` forwards and mounts those roots.

| Variable / launcher flag | Default |
| --- | --- |
| `BAZZA_ROOT` / `--root` | repository `build/` |
| `BAZZA_PROJECT_ROOT` / `--project-root` | repository parent, containing Kiln |
| `BAZZA_DATA_ROOT` / `--data-root` | root `data/` |
| `BAZZA_CACHE_ROOT` / `--cache-root` | root `features/bazza/` |
| `BAZZA_MODEL_ROOT` / `--model-root` | project root `Kiln/weights/` |
| `BAZZA_RUN_ROOT` / `--run-root` | root `training/bazza/` |
| `BAZZA_SELECTION_PATH` / `--selection` | data root `groundbeef/v0.5/selection.json` |

The selection is an existing prepared input; training does not create it.
The backbone source defaults to project root `Kiln/models/gemma4_12b_pytorch`.
Model/cache relative names remain editable Python values in `config.py`.
`OVERFIT_URL` is optional; an unset URL keeps logging local.

Set environment variables directly, pass root flags, or copy `.env.example`
to a Git-ignored `.env.local` and explicitly use `--env-file .env.local`.
That file contains trusted shell assignments and is not loaded implicitly.
Precedence is command-line roots, then an explicit env file, inherited environment,
and finally defaults. Direct Python entrypoints use exported environment values;
`--env-file` is a launcher option. Local resolved paths are recorded for run
reproducibility, but the HF config export excludes them.

The launcher mounts existing declared roots, the settings directory and the
selection's directory read-only. Only the requested output directory is writable.
It does not mount named local drives or create absent input roots. Put custom
settings' inputs/checkpoints under the configured roots so they are visible in
Docker. Historical research notes/snapshots retain their original provenance.

## Commands and job controls

Use the existing RunPodML-derived image (`kiln-transformers-cuda:5.2` by default).
The default Python settings describe the prepared-data and cache layout. They
do not download data, extract training features, or resume an old run automatically.
Set the roots to existing inputs before using it.

```bash
bash scripts/run.sh train \
  --config config.py \
  --env-file .env.local \
  --output-dir /absolute/path/to/a/NEW/run --plan-only

# Remove --plan-only to start, with foreground tqdm progress.
# Add --detach to let Docker own the job independently of the terminal.
bash scripts/run.sh train --output-dir /absolute/path/to/run --status
bash scripts/run.sh train --output-dir /absolute/path/to/run --logs
bash scripts/run.sh train --output-dir /absolute/path/to/run --stop
bash scripts/run.sh train --config /absolute/path/to/settings.py \
  --env-file .env.local --output-dir /absolute/path/to/run --resume --detach
```

Reuse the same roots/environment file when resuming.

`--resume` restores the same model, optimizer, sample-based schedule, RNG, data
cursor and callback state, and rejects changed source/data/config/runtime.
`--initialize-from /path/to/checkpoint.pt` starts a fresh experiment using old
weights; it does not restore optimizer or progress. Historical full-state runs
must resume in their original checkout/snapshot. No checkpoint is migrated in place.

Evaluation uses a configuration with the desired serialized selection and feature
bank, and a named split. It preserves option order and resumes committed batches
without duplicates. `demo` uses the same arguments plus `--example-id ID`.

```bash
bash scripts/run.sh evaluate --config /path/to/test-settings.py \
  --checkpoint /path/to/run/checkpoint-step-N.pt \
  --output-dir /path/to/NEW/test-output --split test
```

All three commands support the same launcher controls. Direct `python train.py`,
`evaluate.py` and `demo.py` remain usable inside the configured container. Durable
outputs include `status.json`, console/metrics logs, checkpoints or prediction
journals, reports, timing and an exit code. An unavailable dashboard does not
interrupt training. Metrics use `train/`, `validation/`, `validation-full/` and `test/`.

An optional `test_config_path=Path("/path/to/test-settings.py")`
setting runs the validation-selected best checkpoint after training. The
checkpoint is selected by fixed-panel equal-track macro accuracy, then CE;
keep the best three plus latest. Test scores never select weights.

## Inputs and precision

A selection JSON contains `identity` and named `splits` of encoded records.
The current Gemma serializer is unchanged; no choice shuffling or truncation
is introduced here. Default `target_key="target"` preserves the existing
native/teacher mixture. Alternative stored target fields must be explicitly
selected and present; there is no fallback that calls teacher scores ground truth.

Set `backbone_mode="cache"` with `feature_format="pt"`, `"layer"` or `"routed"`,
or `backbone_mode="live"`. Raw final features receive the original RMSNorm once
inside the cache backbone, outside head autocast. Cached batches are bound to
the cache that prepared them. Live forwards retain unpadded batch one; head
batches remain larger and length-budgeted, with bounded CPU prefetch. The cache
adapter reads existing banks only; it does not silently export new features.


## Verification

CPU checks in the established image compare the old and new implementations,
including exact keys, initialization, logits, CE, gradients, an optimizer update,
callback order, resumed training and partial prediction-journal recovery.
The miniature runs are correctness fixtures, not model-quality results.

```bash
# Inside the existing container, with this repository as the working directory:
python -m unittest discover -s tests -v
```

No weights, feature tensors, datasets or run output are committed. See
[plans](docs/plans/README.md) for the proposed shared-backbone hybrid with the
original autoregressive head. That hybrid is not implemented by this refactor.
