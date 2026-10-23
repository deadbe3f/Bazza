# Bazza refactor guide

Historical agreement. The implementation status and verification scope are
now recorded in [REFACTOR_STATUS](../REFACTOR_STATUS.md). The run recipes below
refer to October 5 and are not the current training defaults.

Agreed direction, 5 October 2026. This is the implementation plan for a new
worktree, rather than a migration of the actively running checkout.

## Names and purpose

- **KneeJerk:** the research repository and eventual agent application.
- **Bazza:** our System One model, including its backbone and decision path.
- **Groundbeef:** the versioned training/evaluation dataset.
- **Backbone:** standalone PyTorch model included in `model/gemma.py`.

Build a small research repository that makes the model and training loop easy
to read. The public training method is `Trainer.train()`. Use `evaluate.py`
for the evaluation command and `metrics.py` for metric calculations. Avoid a
general training framework, automatic experiment registries or replacing
functions at runtime to change trainer behaviour.

## Target layout

```text
train.py                    # Configure and launch one training run
evaluate.py                 # Evaluate a selected checkpoint
demo.py                     # Load a model and make interactive decisions
trainer.py                  # Trainer, training state and operation ordering
callbacks.py                # Callback contract and concrete callbacks
data.py                     # Dataset loading, input serialization and batching
losses.py                   # Native/teacher objectives and class weighting
metrics.py                  # Accuracy, CE, calibration and grouped reporting
utils.py                    # Small genuinely shared helpers
model/
  __init__.py               # Public Bazza and BazzaDecisionHead exports
  model.py                  # Complete model composition and forward path
configs/
  bazza_groundbeef_v02.json  # Explicit, reproducible run settings
tests/                      # Focused correctness and resume tests
scripts/
  data/                     # Dataset generation, download, audit and reporting
  experiments/              # Sweeps, comparisons and useful historical scripts
docs/                       # Architecture, protocols, results and source notes
```

Start the model package with one implementation file and its public exports.
Split out `heads.py`, `reasoner.py` or a backbone adapter only when the
component warrants its own file. Do not create a file per small class. Keep
`utils.py` small; a helper specific to data, metrics or checkpoints belongs
with that component. `tests/` contains code tests, not a second evaluation CLI.

Dataset preparation is a separate operation from training. Keep source-specific
adapters beside their build scripts; the runtime data loader consumes their
normalized, versioned records. The agent application remains a separate concern
from the training code.

## Model and data contracts

`from model import Bazza, BazzaDecisionHead` is the public model import.
`Bazza` composes the bundled backbone and decision head; `BazzaDecisionHead` is the
standalone head corresponding to the legacy `GemmaDecisionHead`.

Retain the current token serialization, final-token global seed, option-end
positions, masks, recurrent updates and shared scalar scoring. Options are a
variable supplied menu, not a fixed category vocabulary. Keep the base AR head
and its checkpoint intact; future planner/routing work must not require
discarding the original backbone. Vision, audio, LoRA and architectural changes
are separate experiments after the refactor.

Use named batch/output fields rather than positional tensor tuples. The data
layer owns serialization and collation. The model owns forward computation;
it does not create optimizers, read datasets or publish experiment logs.
Metrics and losses use the same masks and target conventions in training,
evaluation and demo inference.

Import the bundled standalone backbone through a local package import.
Do not mutate the global import path or resolve another repository's `model`
module as Bazza's `model` package, or vice versa.

Live frozen-backbone forwards remain the default. A cached-feature experiment,
if retained, uses an explicit alternate input path with its own feature identity;
it must not silently export all features before an ordinary training run.

Native labels and teacher distributions remain distinct. Configuration chooses
the target policy and specifies what happens when a source lacks that target.
Do not rename teacher probabilities as ground truth. Fit semantic weights on
training data only and reuse the fitted policy consistently in reports.

## Trainer contract

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler,
)
trainer.on("validation_end", early_stopping)
trainer.on("validation_end", checkpoint)
trainer.on("batch_end", log_batch)
trainer.train(train_loader, validation_loader)
```

`train.py` wires these concrete objects together from one configuration. The
trainer owns the loop, device/precision handling, gradients, optimizer and
scheduler ordering, validation timing, and shared state. It has explicit
`train_batch()` and `validate()` methods. Experiments can subclass
`train_batch()` when required; ordinary dataset or LR changes use configuration.

Maintain one serializable state with epoch, batch cursor, optimizer step,
examples seen, running aggregates and a stop request/reason. A batch is not
an optimizer step when using gradient accumulation. Loss scaling, accumulation,
gradient unscaling, clipping and scheduler advancement are visible in the loop.
Only count updates that actually occur, including under mixed precision.

The initial implementation must preserve today's one-update-per-batch behaviour.
Add accumulation support only with its own verification if it is needed. Do
not silently change training math while reorganizing the code.

## Callback contract and events

Use an event registry: `dict[str, list[Callable]]`. Event names are keys and
ordered lists of handlers are values, allowing multiple subscribers per event.
`trainer.on("batch_start", handler)` subscribes a function or callable object.
Inside the loop, `self.emit("batch_start", context)` dispatches the event.
Registration and dispatch are separate operations; do not call dispatch
`register_event()`. Use consistent lower-case snake-case names and reject
unknown names during registration so typos cannot silently disable logging.

```python
def log_batch_start(trainer, context):
    if context.stage == "train":
        print(trainer.state.examples_seen)


trainer.on("batch_start", log_batch_start)

# Inside the trainer, at the actual batch boundary:
self.emit("batch_start", context)
```

Handlers do not need to inherit a callback base class or implement unrelated
no-op methods. Plain functions suit stateless logging. Callable objects suit
stateful stopping/checkpoint components. Persistent handlers expose
`state_dict()` / `load_state_dict()` and have stable checkpoint keys; do not
use function memory addresses as their identity. Shared state used by several
event handlers is saved once under its explicit owner key.

Each handler receives the trainer plus a short-lived context containing the
event name and relevant stage, batch, outputs, loss and metrics. Fields that
do not apply are absent or clearly documented. Detach tensors retained for
logging; avoid retaining computation graphs between batches. An `off()` or
unsubscribe handle may remove a subscription. Iterate a snapshot of handlers,
so subscription changes during dispatch take effect on subsequent emissions.

Support these initial events:

| Event | Timing |
|---|---|
| `train_start`, `train_end` | Training lifecycle |
| `epoch_start`, `epoch_end` | Epoch boundaries |
| `batch_start`, `batch_end` | Every processed batch; context names the stage |
| `before_forward`, `after_forward` | Model forward boundaries |
| `before_backward`, `after_backward` | Training backward boundaries |
| `before_optimizer_step` | After accumulation/unscaling, before clipping and update |
| `after_optimizer_step` | After an optimizer update actually occurs |
| `validation_start`, `validation_end` | Whole validation pass; end supplies aggregated metrics |
| `exception` | Error notification before propagating the exception |

Forward and batch events also run during validation. Backward and optimizer
events run only during training. The stage is explicit, so logging callbacks
do not need different names for the same batch operation.

Dispatch in deterministic registration order. Run all relevant callbacks before
honouring a stop request. Persist the resulting early-stopping state together
with the checkpoint; define save timing explicitly so registration order cannot
produce stale resume counters. Exceptions propagate after cleanup; callbacks
must not hide failures or replace the loss and optimizer behind the trainer.

Initial concrete callbacks:

- **Checkpoint:** latest full state plus best three by the selected fixed-panel
  validation metric; publish atomically and prune only managed snapshots.
- **EarlyStopping:** patience, minimum examples and absolute improvement margin;
  its meaningful-improvement anchor is independent of exact best-model ranking.
- **ExperimentLogger:** durable local JSONL plus optional Overfit publishing.
  An unavailable dashboard must not interrupt training.
- **Progress:** tqdm batches/examples, current loss, LR, validation and phase.

Use explicit names for batch loss, running loss and validation loss. Preserve
raw and balanced metrics, plus source/task breakdowns. Report rule-case results
separately so their ease cannot obscure performance on other tasks.

## Initialization, resumption and reproducibility

Expose two distinct operations:

1. **Initialize weights:** load a prior head/model for a new experiment with
   fresh optimizer, scheduler, counters and stopping state.
2. **Resume:** restore the interrupted experiment's complete state and identity.

Checkpoint state includes model, optimizer, scheduler, precision scaler where
used, RNG state, sampler/order cursor, trainer state and every callback's state.
Reference frozen backbone weights by checkpoint/revision/hash instead of
duplicating them in every head checkpoint.

Record code revision, effective configuration, initialization hash, backbone
identity, dataset manifests and membership, serialization, fitted weight policy,
calibration split, hardware and container image. Reject incompatible resumption
before modifying outputs. Keep datasets, weights and run outputs outside Git.

Evaluate the initial model on fixed validation before its first update. Use
validation for selection, calibration for calibration, and test/OOD for final
reporting. Preserve the currently published split ownership and provenance;
do not regenerate the dataset during this code migration.

## Current behaviour to preserve

The current Bazza continuation starts from the selected Groundbeef v0.1 head
at optimizer step **2,816**, which saw **22,528** examples. Its full checkpoint
SHA-256 is:

```text
d2e8b1ab5c2181f5efc98eb1175302039a5735185e8e37f68145b124f294284f
```

The v0.2 release has 50K train, 2K validation, 1K calibration and 3K test
examples, plus the preserved original Open-Jev OOD partition. It includes 5K
locally authored rule cases; PrimeVul and Prometheus Feedback Collection are
excluded. Some training examples deliberately overlap v0.1, while heldout
IDs/groups remain protected. Keep this initialization history in comparisons.

The current launcher uses frozen NF4 Gemma, online backbone forwards,
batch eight, AdamW 1e-4, weight decay .01, warmup 64 updates, steady LR then
cosine decay over the final 20% of the planned one-pass budget to 1e-5.
It validates every 128 updates and retains three best snapshots plus latest.
Stopping counts eight checks without a 0.001 improvement in balanced validation
CE after at least 8K examples. Preserve these settings before testing changes.

## Migration sequence

### 1. Preserve the current experiment

Commit all required experiment source, scripts, provenance configurations,
summary reports, app mockup and this guide. Do not modify the actively running
trainer/model files. Existing names such as `gemma_decisions` and
`GemmaDecisionHead` remain in this snapshot for source-hash reproducibility.

### 2. Start a separate worktree

Create `feature/bazza-refactor` from that snapshot when requested. Keep the
active training checkout in place. Configure the new checkout to read immutable
datasets/backbone weights through explicit paths or read-only container mounts;
do not copy large assets into Git or rebuild splits. Give every new experiment
its own output directory.

### 3. Extract the readable model/data/loss/metrics path

Implement the named public interfaces and map old state-dict keys explicitly.
Preserve projection dimensions, serialization, masks, recurrent operation order
and loss reductions. Move normalized data access out of pilot/trainer code and
metric calculations out of experiment runners.

### 4. Implement Trainer and callbacks

Write one explicit loop and callback dispatcher. Extract checkpointing,
stopping, progress and logging from the loop. Remove runtime function replacement
and dataset-version-specific trainer wrappers. Implement full resumption and
new-stage weight initialization as separate paths.

### 5. Wire the entry points and organize experiments

Make `train.py`, `evaluate.py` and `demo.py` thin entry points using the same
model/data contracts. Express v0.1/v0.2 differences in configurations and data
manifests. Put sweeps and source-specific builders under `scripts/`; move
reports under `docs/`. Keep the original Git snapshot as the reference for
historical checkpoint audits, rather than duplicating each historical trainer
inside the new core. Retain only compatibility adapters that have an actual user.

### 6. Verify and publish the new interface

Required checks are:

- Fixed inputs: identical serialization, masks, logits, losses and trainable
  gradients against the committed reference within declared precision tolerances.
- Loaded checkpoints: explicit key mapping, exact parameter matching and
  preservation of trainable/frozen boundaries.
- Interrupted/resumed miniature training: identical next batch, LR, parameters,
  callback counters and selected checkpoints to an uninterrupted run.
- Callback order: batches versus updates, validation events, stopping and
  exception propagation; no stale state in saved checkpoints.
- Dataset membership and native/teacher policies: unchanged fixed panels,
  provenance and fitted weights.
- Short Docker end-to-end smoke: train, evaluate, demo, logs and resume without
  re-exporting features or requiring a functioning dashboard.

Once these gates pass, document the commands and start a separately identified
Bazza run. Do not rerun the full 50K experiment merely to check file movement.
Architecture, optimizer, schedule, rubric data and modality improvements are
follow-up experiments, with their own comparisons.
