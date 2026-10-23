# Python settings and interchangeable backbones

Agreed and implemented 18 October 2026 in the refactor worktree.

## Settings

`config.py` contains plain values in our flat `Config` object: learning rates,
batch sizes, data/model/cache locations, model dimensions, precision, schedules,
validation and logging settings. It does not construct trainers or models.
`train.py` and shared assembly helpers construct actual objects. An alternative
Python file must export `config = Config(...)`; relative paths resolve beside it.

Runs save the Python source, a generated resolved-settings JSON and its hash.
Resumption rejects changed settings or source; changing mode/config starts a
new output directory. Historical JSON-config runs remain reproducible using
the earlier source snapshot. JSON is no longer a training input in this version.

`export_to_hf_config(config, path=None, include_training=False)` explicitly
converts our object to portable HF-style config.json metadata. It excludes local
paths, never mutates training settings, and adds training values only when
requested. This is not an AutoConfig/AutoModel implementation or weight export.
See [Hugging Face's custom-model requirements](https://huggingface.co/docs/transformers/main/en/custom_models).
Other external schemas should get their own export adapter only when needed.

## Model interface

- `Gemma4`: live frozen backbone implemented in `model/gemma.py`; unpadded batch-one feature computation.
- `Gemma4FeatureCache`: an nn.Module using FeatureCacheMixin, no live model
  construction. Supports existing normalized per-example and raw/routed final
  layer banks. The cached raw final state receives the original RMSNorm once.
- Both accept a named DecisionBatch and return BackboneOutput with normalized
  final `hidden` and available raw `layer_states`.
- `Bazza(backbone, head)` always calls that backbone from forward. Loss,
  optimizer and callbacks use the same path. The trainer has no cache switch.
- CPU collation and bounded prefetch can use backbone.prepare_batch; disk IO
  remains outside the GPU update where possible. Direct non-prefetched model
  calls can load the same cached batch. Cross-cache prepared inputs are rejected.
- Frozen backbone identity is recorded; head-only checkpoint keys remain exact.
  The original AR head is retained in the live model, with hybrid routing still
  a separate planned operation.

## Scope and verification

Whole-backbone final-layer replacement is implemented first. Cached prefixes,
individual-layer substitution and multi-tap residual fusion require their own
forward contracts and checks; they are not silently enabled by selecting a tap.
A cached prefix can only represent the frozen computation through that boundary.

CPU checks cover exact live/cache logits and head gradients on a tiny encoder,
no live-weight load in cache mode, cache source mismatch rejection, final RMSNorm,
old-head/optimizer parity, Python settings and portable export, committed-progress
resumption and command workflows. No new full Gemma forward/training run was
started. Full 12B GPU parity remains a pre-release check.

## Portable roots

Active defaults derive root/project/data/cache/model/run locations from BAZZA_*
environment variables. The launcher accepts equivalent root flags and passes
them into Docker with read-only input mounts. Local paths belong in an explicitly
selected, Git-ignored .env.local; .env.example contains portable placeholders.
BAZZA_SELECTION_PATH can point at an existing selection independently of the
prepared-data layout. No feature files, selections or weights are relocated.
HF export continues to exclude all roots and local paths. Historical source and
run records keep their original provenance; release packaging must select current
code/artifacts rather than publishing those historical local records.

## Backbone separation and inference API — 23 October 2026

The generic output contract and cache classes are in `model/backbone.py`.
Gemma-specific adapters and loading are in `model/gemma_backbone.py`; the
standalone transformer remains in `model/gemma.py`. `systemone.py` provides the
state/questions/criteria Python API, independent of training configuration or
data selection. `demo.py` loads a model and processes a new JSON request, rather
than selecting an evaluation example. See the root README and `examples/`.
