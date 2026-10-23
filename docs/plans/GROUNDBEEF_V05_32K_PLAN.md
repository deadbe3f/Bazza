# Groundbeef v0.5: 32K uplift and fresh-head training on the combined 96K

Status, 2026-10-17: audited 96K dataset and all new final-layer features
are complete. NVMe staging is in progress. The fresh training command is
prepared; the full run has not been started. See
`BAZZA_V05_TRAINING.md` for current commands and exact selection metrics.
The mixture below records the reviewed targets; achieved source counts are
frozen in `build/groundbeef/v0.5/recipe.json` (255 MBPP rows, with the shortfall
filled by additional verified CodeIO). The original v0.4 artifacts remain.


## Goal and starting point

User decision: mix the addition into the original training data and start the
decision head from its original fresh initialization. Do not initialize from
the control's trained step-2125 checkpoint. Keep the frozen NF4 Gemma-4-12B-it
backbone, tokenizer, request serialization and head architecture for this run.
Use the successful control recipe without another architecture or optimizer
sweep. The backbone is not being trained from scratch.

The main targets are CRUX-style code semantics and broad, harder MCQ knowledge.
Maintain coverage of maths, commonsense, evidence and tool decisions using fresh
examples from those families. All 32K selection slots target inputs/parents not
used in the 64K control. They are not all guaranteed unseen by every historical
pilot: the membership ledger must separately identify historical exposure.

## Proposed 32K addition to the 96K training mixture

Counts are selection targets, pending qualification and overlap checks. They
are not claims that all corresponding rows have already been converted.

| Family | Rows | Share | Source allocation |
| --- | ---: | ---: | --- |
| Knowledge and science | 10,000 | 31.25% | OpenScience 8,000; SciQ 1,000; curated MMLU auxiliary train 1,000 |
| Python execution | 8,000 | 25.00% | Unused verified CodeIO view 3,000; newly qualified programs from the already downloaded CodeIO raw pool 4,500; MBPP native train-derived I/O 500 |
| Maths | 4,000 | 12.50% | GSM8K 1,000 distinct parents; AQuA 1,500; MathQA 1,500 |
| Commonsense | 3,000 | 9.375% | WinoGrande 2,000; HellaSwag 500; CommonsenseQA 500 |
| Evidence and entailment | 3,000 | 9.375% | ANLI 2,000; RAGTruth 500; auxiliary PAWS 500 |
| Tool decisions | 2,000 | 6.25% | When2Call four-way 1,000; API-Bank 500; ToolRet 500 |
| Auxiliary language, ordinal and routing | 2,000 | 6.25% | WikiQA 500; STS-B 500; Decider custom 500; Decider routing 500 |
| **Total** | **32,000** | **100%** | **18K explicitly targets knowledge/code; 14K broadens or maintains other skills** |

The auxiliary supplement contributes 2,500 rows including PAWS (7.8125% of the
addition). Its routing and custom cases retain teacher-consistency provenance;
their labels are not independently established truth. Native WikiQA/PAWS and
human-score-derived STS-B remain distinguishable from those labels.

Most material is already local: 23,500 target slots draw from downloaded pools.
OpenScience (8,000) and MBPP (500 derived examples) need new source acquisition.
Local availability does not mean all necessary representations/features exist.

### Why these proportions

- Code has the largest local Winnow gap, 14.21 percentage points, and a clear
  program-distribution mismatch.
- MMLU-Pro has a 6.65-point gap, and the current nominal subject balance does
  not establish real advanced subject coverage.
- Maths and commonsense remain weak enough to deserve new data.
- When2Call, ARC and RAG are comparatively strong. Retain their task coverage
  without allocating another large quota to already successful formats.
- Do not add more legacy Open-Jev/Jev probabilities, SWE-QA, easy generated
  rules, news or sentiment rows in this addition. Their existing data remains
  archived. This choice is prioritization, not evidence those sources are harmful.

## Source and conversion details

### A. Knowledge: replace weak subject proxies with relevant questions

[NVIDIA OpenScience](https://huggingface.co/datasets/nvidia/OpenScience)
provides synthetic four- and ten-choice training questions across science and
humanities. Its card describes DeepSeek-R1 multiple-sample majority-vote answers
and upstream test decontamination. These are useful synthetic hard labels, not
human ground truth or Jev probability targets. Preserve answer provenance and
the source terms in the lock.

Select 7,000 native ten-choice and 1,000 native four-choice questions, one
presentation per question parent. Do not insert chain-of-thought, solutions or
answer-bearing generation metadata into model inputs. Preserve complete native
menus; do not manufacture ten choices by retrieving unrelated answers.

Desired coverage within those 8,000:

| Subject group | Target |
| --- | ---: |
| Physics | 1,500 |
| Chemistry | 1,500 |
| Engineering | 1,500 |
| Computer science | 1,000 |
| Law | 1,000 |
| Economics/business | 1,000 |
| Other knowledge | 500 |

First inspect the actual schema and available domain metadata. If native
subjects are absent, keep any reviewed derived classification explicitly
marked as derived; do not repeat the old keyword-balancing claim. These are
coverage targets, not established source counts. Reject ambiguous parsing,
multiple apparent correct answers or incompatible answer keys. A small initial
sample review checks whether the proposed source is suitable before a large
scan or feature export. Do not launch a new teacher-judging pipeline for this
iteration.

Add 1,000 unused [SciQ](https://huggingface.co/datasets/allenai/sciq) train rows
with their supplied distractors and 1,000 unused, reviewed closed-book MMLU
auxiliary questions. Exclude passage-comprehension examples from that last
knowledge quota. Preserve their actual unspecified-subject status unless a
reviewed annotation exists. SciQ support passages remain excluded for the
closed-book view. Keep science and knowledge subcategories in reports even
though the headline quota groups them.

OpenScience is a large release; stream a pinned candidate window and archive
selected records/offsets and hashes rather than downloading the entire stated
80GB corpus. The publisher's decontamination does not replace our own complete
held-out protections. If this source fails parsing/quality/overlap qualification,
revise the mix explicitly rather than backfilling with arbitrary numeric padding.

### B. Code: short programs and meaningful wrong answers

The local executed CodeIO view has 7,037 eligible source rows. After excluding
v0.4 control row IDs and parent groups, 3,930 remain; only 832 have at most 12
nonblank/noncomment lines and only 214 also have no imports. Those are counts
before the complete historical-protection audit. Simply selecting more from
that view would preserve much of the old distribution problem.

Use 3,000 unused rows from that verified view, repairing candidate types, and
qualify 4,500 new program parents from the much larger already downloaded
CodeIO raw file. This expands beyond the old 20K candidate selection.
Preparation found only 4,525 unused <=12-line candidates before execution and
development reservation, making the original 5,000-row short-program target
infeasible from that scan. Expand raw qualification to <=20 substantive lines
(11,110 candidates), taking <=12-line programs and string operations first.
Prioritize strings, collections, slicing, indexing, mutation, aliasing and short
control flow. Preserve some longer verified programs for breadth. Report the
achieved length/operation/import distribution rather than duplicating program
parents to force the original length target.

[MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) supplies
374 full-config training programs with tests. Target 500 derived execution
examples from those native training programs, capped at two inputs per parent.
This does not mean 500 distinct programs. Keep native validation, test and
prompt/few-shot ownership separate. Join sanitized/full identities so a second
configuration cannot reintroduce held-outs.

For every code example:

1. Use the current code/input/task schema and Python literal answer convention.
2. Execute the original function with its input in the restricted, disposable
   CPU worker and verify recorded expected output where supplied. Reject
   nondeterministic, unsupported or resource-exceeding programs.
3. Construct wrong outputs from defined semantic mistakes: off-by-one bounds,
   direction changes, first-versus-all replacement, ordering/deduplication,
   indexing or mutation mistakes. Execute supported mutated programs to obtain
   their outputs; retain only finite, deterministic outputs unequal to gold.
4. Serialize all candidates using the same type-preserving path. Do not make
   the gold the only tuple because wrong outputs passed through JSON.
5. Prefer plausible alternatives of the same output type, but allow an explicit
   type distinction where the code itself tests one. Use 2–6 distinct choices
   when available; do not force weak padding to fill a fixed menu size.

Group by normalized program/task identity across every input, mutation and
presentation. Count both rows and distinct programs. If raw qualification or
MBPP falls short, fill only from additional qualified raw CodeIO programs and
record the changed source count; do not silently replay old programs or claim
more distinct programs than exist. Do not use CRUX benchmark programs as data.

### C. Existing broad pools and auxiliary supplement

Use qualified train views under `datasets/decisions/*/v0.1/` and
`build/groundbeef/decider-supplement-v0.1/`. Their preselection pool counts include:

| Local pool | Qualified training rows before new exclusions |
| --- | ---: |
| MMLU auxiliary | 86,534 |
| SciQ | 11,613 |
| AQuA | 81,997 |
| MathQA | 24,454 |
| WinoGrande | 39,078 |
| ANLI | 93,409 |
| RAGTruth | 14,028 |
| API-Bank | 6,340 |
| ToolRet | 394,758 |

These are neither unused counts nor independent-parent counts. For example,
many ANLI decisions share a premise and ToolRet includes positive/negative
views of a query. Parent caps remain essential.

An initial row-ID/parent exclusion against v0.4 leaves 3,081 PAWS, 684 WikiQA,
2,361 STS-B, 3,943 custom and 4,970 routing examples in the auxiliary train
release. Further aliases/protections can reduce these counts. WikiQA's 500-row
quota has limited margin: use up to 250 additional unused PAWS and 250 STS-B
rows if it cannot be filled, preserving the overall auxiliary quota and
recording the source change.

When2Call's repaired full view has 8,324 train rows before exclusions. Select
1,000 unused eligible four-way examples with a balanced mix of its three native
positive policies. Do not invent native positive direct-answer labels; that
remains a separate source gap. API-Bank teaches tool selection and ToolRet
teaches relevance; neither is equivalent to a When2Call direct-answer positive.

Choose one GSM menu per parent: target 250 four-choice and 750 ten-choice
examples from 1,000 distinct unused training questions. Keep PAWS pairs, ANLI
premises, RAG prompts, tool queries and teacher multi-question states grouped.
Preserve native choice order. Balance meaningful labels and task coverage by
selection; do not apply inverse-frequency answer-letter weights as well.

## Membership and held-outs

Before selection, reserve source parent groups for new development. Enforce
existing native held-outs, all historical validation/calibration/test groups,
architecture-pilot exclusions and complete external benchmark source protection.
Exclude current control parents from the new training addition. Audit canonical
request/context aliases and near matches as in v0.4; do not claim that textual
screening proves semantic independence.

The membership ledger must include original source/revision/file/row hash,
native split, parent ID, transformations, label origin, candidate provenance,
raw/input hashes, token length, prior training exposure and selection reason.
Freeze a recipe and counts showing requested versus achieved quotas. Unfilled
quotas must be visible before export, not hidden by oversampling.

## Validation and training run

Combine the two training manifests without duplicating parent presentations.
The 64K component contributes two thirds and the 32K addition one third of the
nominal 96K pass. Deterministically shuffle rows across both components and
sources from the start; do not use a sequential 64K-then-32K curriculum. Preserve
the original answer-choice order within each row. Store component membership
and the actual combined source/category proportions in the release manifest.

### Fixed development panel

Build a 2,000-row development panel, outside the combined 96K training count:

- 500 held-out new knowledge questions, mainly native semantic ten-choice.
- 500 held-out code decisions from different program parents.
- 1,000 existing fixed development anchors: maths 200, science 200,
  commonsense 200, ANLI 150, RAG 150 and When2Call 100.

Existing anchors are already observed development material, not a new blind
test. Keep the old 4,500-row full validation as a retention check, including
ordinal/routing/language tasks not represented strongly in the new panel.
Report per-task accuracy and CE, and equal-task macro accuracy; combine paired
GSM views before the macro. Keep old and new panels separate in logging and
checkpoint metadata. Fix all panel membership before training.

### Single planned run

- Fresh decision-head initialization and fresh optimizer/scheduler state, using
  the control seed, in a new output directory. Keep the pretrained frozen
  readout/base components as in the control's fresh initialization.
- One pass through the combined 96K, mixing old and new data throughout.
- Same head architecture, optimizer assignment, weight decay, batch/token budget
  and loss as the control. Synthetic consensus rows have hard SFT targets and
  explicit provenance; add no new teacher-distribution matching objective.
- Use the control's peak rates: Muon 0.003 and auxiliary AdamW 3e-5,
  640-example warmup, hold until 25K, then cosine to 10% of peak at 96K.
  The cooldown endpoint follows the new total length; the earlier proposed
  lower-rate continuation schedule is superseded.
- Full development validation every 8K and final; quick diagnostics only if
  needed. Keep the best three checkpoints by equal-task macro accuracy, with
  macro-task CE as the tie-breaker. Store both metrics and all checkpoints.
- Evaluate old full validation at the same checks; flag approximately 2-point
  macro accuracy loss or a 3-point tools/grounding loss for review. These are
  practical review thresholds, not formal statistical significance tests.
- Run the existing 13,499-row external test once on the selected run's
  checkpoint. Compare nine tracks and eight-dataset macro with the control and
  Winnow on matching input identities. Do not choose a checkpoint using test.

No additional architecture or LR sweep is part of this proposal. Once features
are available, another recipe can reuse them cheaply if a concrete need arises.

### Cache and job controls

Export only final-layer features for this iteration, using the existing frozen
backbone and batch-one numerically validated path. Reuse the existing 64K
training features and old anchor/test caches when exact model/input hashes
match; use a combined indexed view rather than copying the old export. New
program menus or repaired candidate
text require new features; old features cannot simply be relabelled.

Store durable new feature shards on Bigly, stage the final-layer working set on
the Gamesly NVMe cache, and preserve append/resume compatibility metadata.
No earlier-layer export is needed for this data experiment. Measure new token
distribution and export throughput on a small initial chunk before giving a
wall-time estimate: the earlier seconds/minutes training timings excluded new
backbone feature extraction, which will dominate this addition.

All preparation/export/train/eval launchers must provide foreground progress,
`--detach`, `--stop`, `--resume`, `--logs`, `--status`, durable status/timing/exit
markers and exact compatible resume. TensorBoard/Overfit logging should use a
new named run with training, validation, validation-full and diagnostics
groups. Include the final selected-checkpoint test report in the run summary.

## Execution order after review

1. Inspect small OpenScience and MBPP samples; pin sources and verify converters.
2. Select candidate pools and freeze grouped development ownership.
3. Produce the 32K addition, repaired code menus and complete provenance ledger;
   freeze the combined 96K training manifest and mixed presentation order.
4. Check counts, overlap, labels/menus and representative examples; freeze hashes.
5. Export final-layer features and cache new validation features with resumable jobs.
6. Train the fresh head for one mixed 96K pass, then test the selected checkpoint.

The expected result is evidence about whether more appropriate data improves
the current frozen representation. It is not a promise that another 32K will
close Winnow's entire gap.
