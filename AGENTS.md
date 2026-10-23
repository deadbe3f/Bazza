# Bazza contributor notes

Bazza is a standalone System 1 decision model. Its backbone implementation is
`model/gemma.py`; live inference uses that local module through
`model/gemma_backbone.py`. `model/backbone.py` contains only the generic contract
and cache machinery. `systemone.py` exposes the raw-request inference API;
`demo.py` demonstrates model loading and inference, independently of evaluation. Do not introduce
external repository imports or a configurable external model-code path. Model
weights and datasets are external assets selected through storage roots.

Each comparison must name the exact backbone/checkpoint and revision, data
split, action schema, preprocessing, calibration split, hardware, and timing
scope. Do not treat a published benchmark number as a local measurement.
Compare on the same examples and action choices where possible, and record
any unmatched conditions.

Do not commit weights, datasets, screenshots from private environments, or
raw training outputs. Store these under ignored local paths. Use the
RunPodML-based Docker CUDA stack for GPU runs and record the image tag.

The model and repository are named Bazza; the dataset is Groundbeef. Follow docs/plans/BAZZA_REFACTOR.md for the agreed implementation: a small
Trainer with train(), explicit callbacks, metrics.py, and model/model.py.
Do not rewrite source files belonging to an active hash-pinned experiment.
Preserve historical checkpoints and source identities; incompatible changes need
a fresh run directory.

## Long-running job controls

All new or updated training, evaluation and data-preparation launchers must
support foreground progress by default, optional `--detach`, `--stop`, `--resume`,
`--logs` and `--status`. Resume must preserve exact committed progress and reject
incompatible inputs/configuration. Training restores optimizer, schedule, RNG
and data cursor; evaluation/data prep resumes completed requests/shards without
duplicates. Keep durable status, logs, timing and exit markers in a known job
directory. Docker owns detached job lifetime; do not use host nohup watchers.
Use `scripts/lib/docker_job.sh` for signal forwarding and durable console/exit
markers where applicable. Never silently overwrite previous run directories.

## Current organization

New code uses the public root entrypoints, `model/`, and `features/`. Do not
import `tests/reference/legacy` into the core. Those byte-identical source files are test-only comparison fixtures; old runs
resume in their original KneeJerk checkout. New plans go in `docs/plans/`;
measurements go in `docs/research/`. Use the shared Docker job launcher for new
long runs. The hybrid AR/decision design is a plan, not implemented behavior.

Training settings are authored as Python `Config` values in config.py (or
an alternative .py settings file). Model construction remains in training code.
Use explicit export adapters for external schemas, e.g. export_to_hf_config;
do not make an external config object the internal training contract. Name
backbone classes Gemma4 and Gemma4FeatureCache, without a _12B suffix.

Active config/launchers must use BAZZA_* roots rather than machine-specific
paths. Keep local overrides in an explicitly loaded, Git-ignored .env.local.
Forward declared roots into containers, mount inputs read-only, and never
create missing input roots as a side effect of Docker bind mounts.
