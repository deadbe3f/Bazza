# Groundbeef v0.3: broader pool, targeted training view

Original plan, 2026-10-11. Groundbeef v0.3 is now built; the
[completed release guide](../research/GROUNDBEEF_V03.md) records the qualified mixture,
validation policy and provenance. Proposed allocations below are retained
as planning history. New model training and architecture changes have not
started. Model: Bazza. Repository: KneeJerk. Dataset: Groundbeef.

Related notes:

- [Specific transfer failures](../research/BAZZA_TRANSFER_ERRORS.md)
- [Completed Decider v1.1 comparison](../research/DECIDER_TRANSFER_COMPARISON.md)
- [Complete Decision Index source inventory](../research/DECISION_INDEX_DATA_SOURCES.md)
- [Architecture improvement options](ARCH_IMPROVEMENTS.md)

## Recommendation

Build an immutable, provenance-preserving **larger source pool**, establish
split ownership and remove overlap, then select a **lean, versioned training
view**. Start with a proposed **80,000 decisions**, one pass. Training shares
below are targets, not measured qualified row counts. If we choose a smaller
compute budget, preserve the priorities and scale the shares down.

A pool of roughly 250K–500K eligible records is a useful initial budget, not
a requirement to ingest everything. Several suitable training corpora already
contain hundreds of thousands of examples. Keep original records on disk;
do not precompute every backbone feature or launch a giant training run just
because the pool is large. Selection and deduplication are CPU work.

Fix group/split ownership **before** drawing training/validation/test views.
Randomly splitting the concatenated rows would allow related questions,
dialogue turns, tool registries and code variants to cross boundaries.

## What the evaluation says to train

The completed checkpoint-4736 transfer run gives concrete priorities:

| Failure | Evidence | New training coverage |
|---|---|---|
| Unsupported tool execution | When2Call cannot-answer recall 19.85%; 637 such cases become claimed direct successes | Unavailable/irrelevant tools, missing arguments, valid direct replies |
| Unknown treated as true/false | ANLI neutral recall 17.91%, versus 78.61% entailment | Three-way NLI, missing evidence, no implication from discovery time to event time |
| Supported language rejected | RAGTruth supported recall 24.69%; QA accuracy 28.44% | Supported paraphrases and summaries, task-by-label balance |
| Arithmetic/choice instability | GSM8K 44.50% with four choices, 24.41% with ten | Multistep maths, varying candidate counts, plausible numeric alternatives |
| Uneven subject coverage | Engineering/physics/chemistry/business approximately 18–20% | Broad training questions with more quantitative/science coverage |
| Software QA insufficient for execution | CRUXEval 53.16%; slicing 47.45% | Code + input → execution-verified output choices |

These are observed errors, not proof that a larger dataset will remove all
limitations. The frozen backbone, 256-wide projected memory and small head
may also constrain learning. Establish the data improvement with the existing
architecture before choosing the next targeted architecture change.

Decider v1.1 has now completed the same panel and leads on every track. Its
neutral recall is 67.66% and supported-response recall 92.24%, versus Bazza's
17.91% and 24.69%. This reinforces the behavioural priorities, while its 27B
size and full backbone tuning remain confounded with its training data.

## Proposed first training view

| Family | Decisions | Share | Initial source allocation |
|---|---:|---:|---|
| Tools, actions and intents | 24,000 | 30% | When2Call 7K; API-Bank 2K; ToolRet train 4K; xLAM or ToolACE 5K; SGD 2K; Banking/CLINC 3K; qualified existing Hermes 1K |
| Grounding and evidence | 16,000 | 20% | RAGTruth 6K; ANLI 6K; HoVer 2K; ContractNLI 2K |
| Maths | 10,000 | 12.5% | GSM8K main train 6K distinct problems; AQuA 4K |
| Knowledge and commonsense | 12,000 | 15% | MMLU auxiliary train 6K; ARC/OpenBookQA 3K; WinoGrande/HellaSwag/CommonsenseQA 3K |
| Code | 8,000 | 10% | Qualified SWE-QA 3K; executed CodeIO records 5K |
| General typed decisions | 8,000 | 10% | Open-Jev 5K; non-Open-Jev distill 2K; existing independent rules 1K |
| Human ordinal/preference decisions | 2,000 | 2.5% | HelpSteer2, with optional Humicroedit/New Yorker human comparisons |
| **Total** | **80,000** | **100%** | Counts finalized after qualification and overlap removal |

This reduces reliance on repeated Open-Jev/distill prompt families. Preserve
useful grading, finance, policy and spatial tasks within their retained shares;
avoid allowing a plentiful game/platformer class to dominate the total.

The view must report both decision count and distinct source-group count.
One long dialogue or many hypotheses about one document must not silently
supply most of a family's rows. A proposed cap is four decisions per source
group, with source-specific exceptions reported explicitly. GSM uses one
presentation per original problem in this first view.

**Conditional allocations:** HoVer/ContractNLI require whole applicable
context within the chosen token budget. CodeIO requires usable verified I/O.
When2Call needs an audited conversion. If qualification leaves a deficit,
replace it with new eligible groups from the same family, or lower the view
size; do not fill it with repeated presentations. ESCI is a useful optional
retrieval addition within the evidence/tool allocation, rather than an extra
unbounded pile of easy irrelevant-product negatives.

## Source conversion choices

### Tools and agent policy

[When2Call](https://huggingface.co/datasets/nvidia/When2Call) supplies SFT
conversations and preference pairs. Prefer direct chosen-versus-rejected
candidate decisions where the task distinction is meaningful; audit a sample
before assigning policy subtypes. For SFT positives, a rule-checked converter
can remove a required tool or mandatory argument to construct a labelled
capability/clarification case. Record the transformation and parent record.
Keep related variants together and select one initial training presentation.
Do not label natural-language responses using fragile refusal regexes.

Include genuine direct-response successes from qualified dialogue data as
well. A tentative tool-policy quota is approximately equal call/clarify/
unavailable/direct coverage **where labels are actually supported**. Preferences
and retrieval relevance are different tasks and retain separate strata.

[API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
has a separate training corpus. Retain schemas, supplied current time and
the dialogue prefix before the labelled call; exclude future messages and
tool results. Convert bracketed API calls into the same canonical typed JSON
used for candidate comparison, preserving values and argument types.

[ToolRet training](https://huggingface.co/datasets/mangopy/ToolRet-Training-20w)
is useful for matching a request to unfamiliar tools. Use judged binary
relevance per candidate or menus with a unique accepted answer. Multiple
positives cannot be forced into a single arbitrary correct label. A future
accepted-set loss can support those menus explicitly; it is not a prerequisite
for the first view. Do not treat unjudged candidates as negatives.

Use separately qualified [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
or [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) training for more
tool schemas and call arguments. Cross-source lineage is important: related
APIGen/xLAM, When2Call and ToolRet records must share group ownership.
The official BFCL evaluation release stays protected.

Selecting a complete call from candidates tests both tool and argument
discrimination. It does not give the current decision head the ability to
generate arbitrary new arguments; retain that scope distinction.

### Grounding and neutral evidence

Use [RAGTruth](https://github.com/ParticleMedia/RAGTruth) native training
response records and human span annotations; derive hallucinated = any span.
Select QA, summary and structured-data tasks with controlled positive/negative
shares. Include difficult supported paraphrases, not only corrupt responses.

Use [ANLI](https://github.com/facebookresearch/anli) train rounds with roughly
equal semantic labels and varied source rounds. ContractNLI adds longer
document evidence if context fits; HoVer adds multi-document verification.
Never shorten these inputs by selecting gold evidence spans or documents
based on the answer. Use source-supplied task context or label-blind retrieval
with a separately verified answer-preservation check. Reject examples that
cannot be represented correctly at the model's token budget.

### Maths, knowledge and code

- Create four-/ten-choice variants **only from GSM8K train**, splitting by
  original problem first. Choose one view per problem for this run. Distractors
  should be distinct, plausible and reproducible; avoid a pattern that reveals
  the gold through its offsets. AQuA adds existing five-choice maths.
- MMLU-Pro, GPQA, BBH, MuSR and CRUXEval are evaluation sources, not new
  training pools. Use MMLU auxiliary train, ARC/OpenBook and other verified
  native training splits. Exclude exact/near copies of **all** official test
  items, including questions inherited or revised across datasets.
- [CodeIO raw](https://huggingface.co/datasets/hkust-nlp/CodeIO-PyEdu-Reasoning-Raw)
  exposes code and executed input/output records. Use code + concrete input;
  stored outputs are targets, not prompt fields. Skip empty I/O. Read JSONL
  without Arrow's homogeneous-type assumption, canonicalize outputs while
  preserving Python types, and group by original code/transformation family.
  Re-execution audits must be isolated. Do not use generated descriptions or
  rationales as an unquestioned source of truth.

Existing source answers, ordinal ratings, rule-derived labels and teacher
distributions retain separate target provenance. No Prometheus GPT-4 rubric
data or PrimeVul is included. Vision/audio/RL remain subsequent versions.

## Data layout and provenance

Keep the current v3-compatible core keys: `id`, `kind`, `options`, `target`,
`state`, `question`, `domain`, `family`, `source`; retain the additive
`group_id`, `split`, `metadata` fields. The benchmark request format with
`questions`/`expected` is separate. Add versioned provenance alongside the
training records; do not hide ownership inside filenames alone or rename task
schema keys. Only state, question and ordered options enter the model input.

Required manifest/record information:

- Dataset/revision, original path/split/ID, source group and parent lineage.
- Licence, raw file checksum, normalized input/decision checksums.
- Transformation name/version, target origin, accepted answer(s), annotation
  or rule evidence; teacher identity/revision and probabilities separately.
- Task/domain, semantic class, menu size, token length, quality flags.
- Split owner and exclusion reason, selected view/version, selection seed.
- Historical exposure: seen by v0.1/v0.2 training or genuinely new to the head.

Retaining a row in the pool does not mean selecting it for training. Store raw
sources, normalization, split ledgers, quarantines and selected views under
ignored `build/groundbeef/v0.3/`. Keep weights/data/raw predictions out of Git.
Commit recipes, small manifests/reports and documentation when implementation
is reviewed. A public redistributable edition may need different membership
from the local research view because source terms differ.

## Split and overlap procedure

1. Register every previously protected Open-Jev/Groundbeef val/cal/test group,
   original Open-Jev OOD, fresh-600 cases, the current evaluation panel and
   complete upstream benchmark releases. Protect whole official heldouts,
   not only cases we sampled. Keep historical train exposure in the ledger.
2. Assign authoritative native split ownership; only native train is eligible
   for training. Benchmark-only corpora remain evaluation material.
3. Link exact duplicates, near copies and known derivation families across
   sources. Use IDs plus normalized code/question/context/registry hashes;
   text similarity alone does not establish independent ownership. Any linked
   heldout member protects the whole group. Quarantine opposing labels under
   equivalent context/policy rather than averaging them silently.
4. Reserve source-grouped development and calibration subsets from eligible
   material before quota selection. For new sources, use their official dev
   where available and appropriate; avoid consuming official test as dev.
5. Create a frozen fast-dev subset (~500) inside full-dev (~2K), calibration
   (~1K), and a fresh internal test (~3K), subject to available independent
   groups. These are additional to the 80K training target. Keep older panels
   intact so regressions and previous gains remain measurable.
6. Select the training view by family, semantic subtype, menu size and length;
   cap redundant siblings. Record realized counts and every exclusion.

Hold out selected tool domains/registry families and document/code sources
where feasible, rather than only random rows with familiar templates. Reserve
additional untouched groups for a fresh transfer assessment. The existing
external panel is still comparable, but now informs our plan and is no longer
an untouched assessment of v0.3 source selection.

## Balance, menus and objectives

- Balance training **semantic classes within tasks**: neutral/entailment/
  contradiction, supported/unsupported by source task, and the documented
  agent policy. Letter A means different things in different menus.
- Use quotas first. Start per-example loss weights at one; retain support for
  mild clipped inverse-square-root class corrections if realized imbalance
  requires them. Avoid combining aggressive oversampling and inverse-frequency
  weights, which would correct the same imbalance twice.
- Keep native evaluation prevalence; report accuracy, macro-F1, class recall,
  task/source scores and probability quality. Do not balance test artificially
  just to make the aggregate look better.
- Preserve original native MCQ order. Newly constructed training menus can
  assign positions once with a fixed seed and correct key mapping, balancing
  positions and menu sizes. No default per-epoch choice shuffling. Never
  permute order-dependent spatial, ordinal, numeric or “all the above” labels.
- Keep hard source SFT, soft teacher distillation, ordinal rating and preference
  objectives identifiable. The view must pin which target is used; teacher
  argmax is not an independent original label.

## Make the pool leaner without wasting compute

First remove redundancy, bad labels, missing inputs, equivalent options,
answer leakage and unrepresentable long contexts using CPU checks. Audit a
small stratified sample from each converter and each important class.

Use a small new development panel to identify missing behaviours, then replace
redundant training groups with qualified groups from those behaviours. High
loss alone is not a quality filter: ambiguous or incorrect labels can dominate
it. Keep easy anchors and ordinary positives as well as difficult contrasts.
Write every refined view as a new manifest; keep the pool immutable.

For the first training run, use the existing head and known low-LR/decay
recipe as the starting engineering baseline. Frequent inexpensive development
checks and retained best checkpoints are enough; a large factorial data/LR/
architecture sweep is unnecessary. Validation-only frozen feature caching
remains compatible with head-only training. Test once after development-based
selection, and preserve the result for future architecture comparisons.

## Build gates and next implementation

Before training:

1. Qualify source formats and the actual usable counts; finalize the 80K view
   or a smaller budget. Resolve missing parent IDs and ambiguous targets.
2. Build protected ownership/overlap ledgers, converters and selected manifests.
3. Report family × class × domain × menu-size counts, token distributions,
   newly exposed versus previously seen groups, and rejection reasons.
4. Verify gold keys/values, original ordering, no future dialogue/rationale
   leakage, no silent truncation, and no linked cross-split overlap.
5. Prepare a user-run training command with a new output directory, tqdm,
   local/Overfit logging and retained best checkpoints. Do not launch it here.

Then assess the current architecture. The saved architecture path is gated
intermediate projections → optional depth GRU, with token BiGRU or switchable
LoRA/QLoRA considered according to the remaining errors and memory budget.

## Research pins

Already resolved during this review; raw file hashes are also retained locally.
Pin remaining selected sources during construction rather than downloading
whatever happens to be latest.

| Source | Revision |
|---|---|
| When2Call | `0582f7749df63a96fdc3070932e83e72396ace53` |
| API-Bank Hub | `12e8158b7628c168f07e8f31fbbe3445e99f44cf` |
| ToolRet-Training-20w | `fdf5a317455b1e60785de7ba587496aa6cc878e4` |
| xLAM-60K | `26d14ebfe18b1f7b524bd39b404b50af5dc97866` |
| ToolACE | `6bda777c88d21e5a204703c1ee45597a8fa4f734` |
| ANLI | `8e4813d81f46d313dac7892e1c28076917cfcdf9` |
| MMLU | `c30699e8356da336a370243923dbaf21066bb9fe` |
| GSM8K | `740312add88f781978c0658806c59bc2815b9866` |
| AQuA | `33301c6a050c96af81f63cad5562cb5363e88971` |
| CodeIO raw PythonEdu | `413a78caa01e634318ef8aa0e84f5ff242f25807` |

Research evidence and fetch failures:
`build/research/groundbeef-v0.3/2026-10-11/`.
This proposal does not claim benchmark contamination is absent from the
pretrained backbone or competitors; it defines what we can control in our
own next training view.
