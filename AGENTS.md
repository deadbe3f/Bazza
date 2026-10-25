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

Keep the implementation small: Trainer with train(), explicit callbacks,
metrics.py, and model/model.py.
Do not rewrite source files belonging to an active hash-pinned experiment.
Preserve historical checkpoints and source identities; incompatible changes need
a fresh run directory.

## Repository boundaries

Public commands are train.py, evaluate.py, demo.py, and decision.py. Keep shell
launchers, Docker job lifecycle controls, monitoring, and storage orchestration
outside this model repository, in the parent project. Do not add run.sh or
similar wrappers here. Training and evaluation own their checkpoints, resume
state, metrics, and progress output; callers own process lifetime.

Research plans, personal experiment notes, dataset-construction scripts, and
one-off tools belong in the research project, not this public repository.
Document public usage and training examples in README.md.

## Current organization

New code uses the public root entrypoints, `model/`, and `features/`. Do not
import `tests/reference/legacy` into the core. Those byte-identical source files are test-only comparison fixtures; old runs
resume in their original research checkout. Keep research plans, experiment
reports, dataset construction, and one-off tools in the research repository,
outside this standalone model repo.

Training settings are authored as Python `Config` values in config.py (or
an alternative .py settings file). Model construction remains in training code.
Use explicit export adapters for external schemas, e.g. export_to_hf_config;
do not make an external config object the internal training contract. Name
backbone classes Gemma4 and Gemma4FeatureCache, without a _12B suffix.

Active configuration must use BAZZA_* roots rather than machine-specific
paths. Keep local overrides in an explicitly loaded, Git-ignored .env.local.
External container launchers should mount inputs read-only and never create
missing input roots as a side effect of Docker bind mounts.

Entrypoints expose `main(args)`. Parse CLI arguments inside the
`if __name__ == "__main__":` block, then call `main(args)`.

## Code style

Run `ruff format .` and `ruff check .` using `requirements-dev.txt`. Follow the
88-character formatter and import-order settings in `pyproject.toml`. Keep
`main(args)` entrypoints. Prefer named module functions or methods over nested
business helpers; keep closures only when they make scoped state clearer.
Do not format frozen `tests/reference/` fixtures. Model source formatting must
retain its executable syntax and compatibility with existing feature-bank IDs.

Separate logical stages within functions with a single blank line, especially
between setup, validation, model construction, execution, and reporting. Keep
closely related statements together. Ruff preserves these code paragraphs but
does not infer where they belong; review readability after formatting.

Keep entrypoint main(args) functions focused on orchestration. Extract distinct
stages such as input loading, identity checks, execution, and reporting into
named module functions. Apply the same approach to large methods when stages
have separate responsibilities; avoid one-line wrappers or nested business helpers.
