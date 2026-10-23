# Standalone decision sources: v0.1

Prepared for Groundbeef v0.3, 11 October 2026. These are separate source
views, **not a combined Groundbeef release**. No training or feature export
is started by this workflow.

## Completed pool

23 pinned source packages downloaded: 136 files, 2,405,802,027 bytes.
The independent saved-file audit passed, including all GSM8K native answers,
split-local numeric distractors, ARC native answer labels/order, and complete
raw native held-outs including unlabelled test questions. Exact question and
parent overlap is zero between the qualified train, validation and test views.

Counts below use **one GSM8K view**, not both presentations of the same
problems. These are available pool sizes, not the intended training weights.

| Category | Training rows | Validation rows | Sources |
|---|---:|---:|---|
| Maths | 113,424 | 5,059 | GSM8K, AQuA, MathQA |
| Science | 19,816 | 2,362 | ARC Challenge/Easy, OpenBookQA, SciQ |
| Knowledge | 86,534 | 1,000 | MMLU auxiliary train |
| Commonsense | 87,021 | 2,500 | CommonsenseQA, WinoGrande, HellaSwag |
| Evidence | 114,611 | 4,527 | ANLI, RAGTruth, ContractNLI |
| Tools | 409,477 | 1,700 | When2Call preferences, API-Bank selection, ToolRet relevance |
| Code | 19,142 | 1,000 | Qualified CodeIO execution choices |
| **Total** | **850,025** | **18,148** | Before the lean Groundbeef selection |

Both GSM8K views separately contain 6,973 training and 500 validation problems.
ARC-Challenge contains 1,108 train / 296 validation after qualification, from
the original 1,119 / 299. ARC-Easy contains 2,236 / 570. All original source
files and existing evaluation sets are preserved.

The converter quarantined 33 groups of identical presentations with
contradictory native answers and excluded 19,789 close MCQ held-out matches.
Rejected rows and source-specific exclusion counts are recorded in manifests;
complete included-source membership and parent IDs are in the SQLite ledger.
ANLI and other document tasks have many decisions per context, so use the
reported distinct parent counts and per-parent caps when selecting the mix.

## Layout

- `scripts/datasets/sources.json`: public source revisions, URLs and SHA256s.
- `datasets/sources/<source>/v0.1/raw/`: original downloads, unchanged.
- `datasets/decisions/<source>/v0.1/{train,validation,test}.jsonl.gz`:
  independently stored normalized decisions and their manifest.
- `datasets/decisions/source-pool-v0.1/`: membership SQLite ledger, conflict
  and overlap reports, source totals and independent audit.
- `build/training-data-uplift/`: disposable conversion staging and logs.

Data remains ignored by Git. Code, source locks and this guide are versioned.
The existing GSM8K and 13,499-request evaluation datasets are preserved.

## Categories

Use a broad `domain` plus a specific `family`. Examples: `math` /
`arithmetic.word_problems`, `science` / `science.reasoning`, `commonsense` /
`commonsense.coreference`, `grounding` / `evidence.nli`, `tools` /
`tools.relevance`, `code` / `code.execution`.

Math and science can be aggregated for reporting, while retaining their
separate training quotas. General knowledge retains MMLU's subject where
provided; its auxiliary training split has no subjects, so those records are
explicitly `knowledge.unspecified`. Do not infer subject labels from answers.

## Adaptations

**GSM8K:** split original training problems first, reserving 500 by stable
SHA256 rank of the parent group. Generate separate four- and ten-choice
views within each split using the unmodified, pinned Decision Index
`answer-groups-v1` functions. Wrong numeric choices are answers to other
problems in that split. Original questions and gold answers are preserved;
rationales are never inputs. Both views share parent IDs; select ONE view
per original problem in the initial Groundbeef training pass. Existing test
files are referenced by hash and never regenerated here.

**Native MCQ:** ARC Challenge/Easy, OpenBookQA, AQuA, MathQA, MMLU,
WinoGrande, HellaSwag and CommonsenseQA retain their supplied choice order.
MathQA letter delimiters are parsed without splitting number commas. ARC
uses its actual answer labels, including numeric labels. SciQ has no native
presentation order: its four choices receive one deterministic presentation;
the supporting passage is omitted for closed-book science decisions.

**Evidence:** ANLI has native entailment/neutral/contradiction labels; group
by premise. RAGTruth uses human span annotations to label supported versus
unsupported responses; group all responses to a source prompt. The candidate
responses can be model-generated, but the labels are human annotations.
ContractNLI retains whole documents and human three-way labels, with all
hypotheses grouped by contract. Its full contexts need token-budget
qualification when selecting the eventual training mixture; no truncation is
performed in this source view.

**Tools:** When2Call's native chosen/rejected pairs become response ranking
decisions; its SFT conversations stay raw. This is not claimed to reproduce
its four-way evaluation task. ToolRet becomes binary relevance, with one
native positive and one native negative per query; multiple positives are
not incorrectly forced into a single-answer choice. API-Bank uses parseable
single calls whose target tool exists in the supplied registry; arguments
are not supervised in this view. Unparseable calls and missing schemas are
excluded. CLINC uses ten supplied candidates drawn from its 151 named
intents, explicitly an adapted candidate task rather than a full 151-class
benchmark.

**Code:** CodeIO uses one input/output presentation per source record. The
gold is the recorded executed output; alternatives are distinct recorded
outputs for other inputs to the same program. No downloaded code is
executed. Exclude missing I/O, fewer than four distinct outputs, oversized
options and obvious dynamic/external dependencies. The first view selects
20,000 program groups by stable hash. This favours variable-output programs
and does not independently verify the recorded outputs; retain that
limitation in future evaluations. Programs share an AST-based group after
removing comments/docstrings. Full CRUXEval source programs are protected.

## Split ownership and quality

Native train/validation/test ownership takes precedence over sampling.
Whole unlabelled native test files are protected too. If native validation
shares a question or parent with test, test owns it and that validation copy
is excluded from the qualified development view.
WinoGrande, HellaSwag and CommonsenseQA validation are benchmark-reserved;
new development sets come from training. MMLU auxiliary training is eligible;
its dev/validation/test and ALL MMLU-Pro test/validation questions are
protected. GSM8K, When2Call, ToolRet, API-Bank, CodeIO and RAGTruth receive
new validation groups from training where appropriate.

Native held-out question/context aliases and whole parent groups exclude
training copies. Prefer source-specific examples to repeated copies in
MMLU auxiliary training, AQuA to matching MathQA copies, and When2Call policy
preferences to matching generic ToolRet queries. Identical
presentations with contradictory native answers are quarantined together.
Reject duplicate/empty answer choices and unlabelled records for export.
The original files remain available to improve converters later.

A bottom-four trigram sketch plus Jaccard threshold 0.85 screens close MCQ
wording against held-outs, including newly derived development data. This is
an approximate audit, **not proof of semantic or paraphrase disjointness**.
Historical Groundbeef v0.2 held-outs receive exact complete-input protection;
its older source ancestry must also be considered when assembling v0.3.
Qualified train, validation and test views have disjoint exact question and
parent ownership. Original raw native files retain their original splits.

## Ready versus raw

ToolACE is downloaded but requires a further faithful conversation/tool-call
converter. HoVer claims and labels are downloaded; its supporting Wikipedia
contexts are not yet fetched, so those claims are not emitted as context-free
"evidence" training examples. MMLU-Pro and ToolRet-Queries downloads provide
benchmark protection, never training. Existing Open-Jev, distill, SWE-QA,
Banking77, Hermes, rules and human grading sources remain available for the
next combined selection. PrimeVul, GPT-4 Prometheus, vision and audio remain
outside this uplift.

CLINC's entire native training set is present in the published ToolRet query
benchmark. Under full benchmark protection it contributes no training rows;
its downloaded raw data and converted benchmark view remain available. Do
not silently relax this protection to fill an intent quota.

## Reproduce or inspect

```bash
cd /media/adrian/Fastly/Projects/ProbablyConverges/KneeJerk
bash scripts/prepare_training_sources.sh
```

Downloads resume and verify against their lock. Existing frozen output views
are audited; the converter refuses to overwrite an existing version. To
experiment with another recipe, pass a new `--output-root` directly to the
converter. The wrapper uses CPU-only `kiln-transformers-cuda:5.2`, Python
3.11 and isolated PyArrow 19.0.1. Optional `UPLIFT_PACKAGES`, `UPLIFT_HISTORY`
and `UPLIFT_IMAGE` allow another local layout.

The next step is a capped, balanced Groundbeef v0.3 view from qualified
source groups. Raw pool abundance is not a reason to train every row or let
tool relevance dominate the mixture. Retain separate target provenance for
hard labels, preferences, teacher probabilities and rules.
