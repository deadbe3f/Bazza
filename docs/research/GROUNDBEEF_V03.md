# Groundbeef v0.3

Built 11 October 2026 for Bazza. The combined dataset lives in
`build/groundbeef/v0.3/`; raw sources and standalone conversions remain
separate and unchanged. Dataset files are excluded from Git.

## Training mixture

80,000 decisions: **64,000 from the new source views and 16,000 retained
from v0.2**. The mixture targets the failures in the
[transfer analysis](BAZZA_TRANSFER_ERRORS.md). No model training is launched
by the dataset workflow.

| Category | Decisions | Share | Sources |
|---|---:|---:|---|
| Tools | 23,000 | 28.75% | When2Call 6K, ToolACE 6K, ToolRet 6K, API-Bank 3K, retained Hermes 2K |
| Evidence and grounding | 17,000 | 21.25% | ANLI 10K, RAGTruth 6K, retained BoolQ 1K |
| Maths | 10,000 | 12.50% | GSM8K 6K unique problems, AQuA 3K, MathQA 1K |
| Knowledge | 6,000 | 7.50% | MMLU auxiliary training |
| Science | 4,000 | 5.00% | ARC Challenge/Easy, OpenBookQA and SciQ, 1K each |
| Commonsense | 2,000 | 2.50% | WinoGrande 700, HellaSwag 700, CommonsenseQA 600 |
| Code | 8,000 | 10.00% | CodeIO 5K unique programs, retained SWE-QA 3K |
| General typed decisions | 8,000 | 10.00% | Retained Open-Jev 5K, non-Open-Jev distill v3 2K, rule cases 1K |
| Human ordinal ratings | 2,000 | 2.50% | Retained HelpSteer2 |

Open-Jev comes from its original oracle targets. The distill replay contains
only previously audited `yuri_v3` examples, excluding copied Open-Jev rows.
Teacher probabilities are explicitly separate from native labels.

The original geometry corpus has 2,313 training rows but only 46 parent
cases. Its replay is capped at 180 decisions. The retained Open-Jev view
instead includes 1,872 workflow decisions, 920 customer decisions and
1,000 reasoning decisions, plus smaller game/spatial slices. The four
independent rule task families each contribute 250 rows. Random generated
tool IDs are reported as `compatible_tool`, rather than separate classes;
the actual IDs remain the model's supplied choices.

## Validation

| View | Decisions | Purpose |
|---|---:|---|
| Full validation | 4,000 | 3,700 new-source decisions and 300 previously monitored legacy anchors |
| Quick validation | 1,000 | Fixed nested subset, preserving full validation source proportions to rounding |
| Calibration | 500 | Separate parent groups; optional later probability calibration |

Validation is fixed, uses complete inputs and has no overlap with recorded
v0.1/v0.2 head training under the exact parent/input alias checks. Legacy
anchors have already been monitored in previous runs: report them separately
from the new-source slice. They are useful for measuring forgetting.

The new views use the pinned Decision Index layouts for GSM8K, ARC, ANLI,
WinoGrande, MMLU and RAGTruth on other split questions. Supplied MCQ candidate
order is preserved. RAGTruth supplies span annotations without a native
candidate menu; its fixed benchmark `true`/`false` layout explicitly maps
the standalone source indices with `[1, 0]`, preserving the answer labels.

Full validation includes both 4-choice and 10-choice presentations of the
same 200 held-out GSM8K problems. Training uses one presentation per problem,
with 3,000 problems in each view.

Important task differences remain visible:

- When2Call training provides two-response preferences; its benchmark has
  four response policies. No clarification/refusal labels are inferred from
  English response wording. ToolACE/API-Bank add variable-menu tool selection.
- MMLU auxiliary training has four choices and no subject labels; MMLU-Pro
  test has ten choices and named subjects. Keep their source identities.
- CodeIO provides JSON input/output choices; CRUXEval evaluates Python
  literal outputs. These are related execution tasks, with different formats.

Use unweighted category macro cross entropy plus per-task accuracy and
semantic class recall. Frequent quick checks can diagnose progress; full
validation should confirm checkpoint promotion or stopping. New-source and
legacy metrics should be reported separately. Balanced development metrics
are distinct from the natural benchmark test distribution.

## Labels, weighting and qualification

Selection uses source quotas, semantic class/task strata and parent caps.
Per-example loss weights initially remain one (`class_power=0` in the
dataset's `training_weights.json`), avoiding a second inverse-frequency
weighting pass. Validation losses are unweighted. The loss weighting support
can be used later if a measured imbalance warrants it.

ANLI is approximately balanced across entailment, neutral and contradiction.
RAGTruth training has 3,000 supported and 3,000 unsupported responses,
covering QA, summaries and data-to-text. Exact counts for every source,
task, semantic label and menu size are in `statistics.json` and the release's
`REPORT.md`.

All train/validation/calibration inputs fit 4,096 Gemma tokens, with no
truncation. The default source parent cap is four decisions; CodeIO and
distill use one per parent. SWE repository split ownership is retained,
with one variant per question rather than a four-row cap on an entire repo.
Training row order is mixed; there is no repeated-choice augmentation.

PrimeVul, Prometheus GPT-4 rubric data, vision and audio are excluded.
CLINC and Banking are omitted from this view because their requests appear
in protected ToolRet benchmark queries. HoVer lacks its supporting evidence;
ContractNLI is deferred instead of extracting answer-bearing spans from
long contracts. Removing a source here does not undo earlier checkpoint
exposure to it.

## Provenance and reproducibility

- `manifest.json`: immutable core files, hashes, exact recipe and exclusions.
- `membership.jsonl.gz`: each included source/parent ID, revision, original
  and exported row hash, input aliases, transformations and exposure flags.
- `sources.json`: source locks and complete parent/source manifests.
- `statistics.json`: source/category/task/class/menu counts and token lengths.
- `audit.json`: source answer/order checks, split ownership, old head exposure
  checks and preserved test hashes. Shared historical request aliases are
  listed explicitly, including two When2Call training rows whose complete
  tool/context inputs differ from older Hermes cases.
- `recipe_source/`: frozen converters, registry, scripts and recipe.
- `release.json`: report/source snapshot hashes tied to the core manifest.

The nine external test tracks / 13,499 requests remain unchanged. Full
native source benchmark holdouts are protected, including unlabelled or
rejected rows. The copied `test.jsonl.gz` and `ood.jsonl.gz` are byte-identical
v0.2 references already evaluated historically; they are not new blind tests.

The checks establish exact/group separation and approximate MCQ screening,
not semantic/paraphrase disjointness. Public examples may have appeared in
backbone or competitor training. Synthetic preference/call annotations and
source-provided execution outputs are not independently rejudged here.

After the [standalone source preparation](DATASET_SOURCE_UPLIFT.md), run:

```bash
bash scripts/build_groundbeef_v03.sh
```

This uses the existing Docker Python runtime on CPU. Repeating the command
verifies the frozen release. Historical assets are read from the existing
decision-baseline asset directory; they are not moved or overwritten.

## Training and validation

The separate v0.3 entrypoint initializes Bazza from the selected cooldown
checkpoint **4736** (`last.pt`, SHA-256
`98b87b793800fd67afeb0de1370e44cdf0c905a221700912d76488b3833e52f9`).
The backbone stays frozen and the head architecture is unchanged. New data
uses fresh AdamW state, a new row order and reset counters; repeating the
same command resumes this new experiment's complete state.

```bash
bash scripts/train_bazza_v03.sh --detach
```

The default output is
`build/training/bazza/groundbeef-v0.3/continue-4736-lr1e-5/`.
Use `--plan-only` to verify inputs without training, or omit `--detach` for
foreground tqdm progress. Dashboard project: **Bazza · Groundbeef v0.3**.
`console.log`, `status.json`, the run's `metrics.jsonl`, checkpoint index,
code snapshot and experiment identity persist locally.

- One pass over 80K examples; training batch eight, online backbone forwards.
- AdamW 1e-5, weight decay .01 for matrix parameters, gradient clip one.
- Warmup 64 updates, steady LR, cosine decay over the final 20% to 3e-6.
- Fixed quick 1K validation every 5K examples; full 4K every 10K and at the end.
  Full checks also produce the nested quick score without another forward.
- Cache frozen **validation token features only** when first needed: 1K at
  the 5K check, the remaining 3K at the 10K check. BF16 features require about
  11.76 GiB plus small file/index overhead. Later checks run the changing
  head directly over cached features, in length buckets up to 32 rows and
  32,768 padded tokens. Backbone exports remain unpadded batch one.
- Full category macro CE ranks the best three full-state checkpoints. Quick
  checks cannot change ranking or stopping patience. Rolling last state is
  saved every 125 updates (1K examples) and before every validation/export.
- Early stopping counts four full checks without a .001 category-CE improvement,
  starting at 20K examples. Every source/task/category/class and new-source vs
  legacy slice is logged separately.
- The untouched initialization is evaluated once on each validation tier
  when its features first become available, giving a starting-head reference
  without delaying the first training update for feature export.
- Separate 500-case calibration runs on the selected head after training.
  Test/OOD and the external panel remain deferred to a separate evaluation.

Cache and resume guards pin the backbone, tokenizer, quantization, prompt,
dataset membership, code, image and configuration. Backbone/LoRA changes need
new feature caches. The historical v0.2 trainers and checkpoints are unchanged.

Five focused Docker CPU checks pass: cached/live tensors, logits and gradients;
cache scope and identity/corruption guards; validation boundaries; bucket
membership/padding; category selection and exact interrupted/resumed training.
