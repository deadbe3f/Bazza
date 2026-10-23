# Bazza transfer errors: Groundbeef v0.3 targets

Analysis date: 2026-10-11. This examines saved predictions from a completed
evaluation; it does not run inference, choose a new checkpoint or alter labels.
Use it alongside the [v0.3 plan](../plans/GROUNDBEEF_V03_PLAN.md) and
[source inventory](DECISION_INDEX_DATA_SOURCES.md).

## Measurement identity

- Model: Bazza cooldown checkpoint **4736**, `last.pt`; frozen Gemma 4
  12B-IT NF4 backbone, 2,301,185 trainable decision-path parameters.
- Checkpoint SHA256:
  `98b87b793800fd67afeb0de1370e44cdf0c905a221700912d76488b3833e52f9`.
- Panel: `datasets/evaluation-panel/test.jsonl`, 13,499 requests across eight
  datasets and nine tracks. GSM8K's two tracks share the same 1,319 problems.
- Panel SHA256:
  `69a383d27a08b6a0dc090c1c082bb0f3626fbf1fa1d46fa767bf2696b49bc37f`.
- Native options and order, deterministic inference, maximum 8,192 input
  tokens. No truncated examples. Model selected before this panel evaluation.
- Local evidence: `build/evaluations/bazza/text-decisions-v0.1/2026-10-11/`.
  Derived counts, prediction hashes and source IDs:
  `build/research/groundbeef-v0.3/2026-10-11/bazza-error-analysis.json`.

Every prediction was joined to its native request by run ID and checked
against the native expected key, including boolean normalization. Counts below
use the full completed Bazza run. Decider subsequently completed all tracks;
see the [full matched comparison](DECIDER_TRANSFER_COMPARISON.md). The GSM
comparison is also included below.

## Overview

| Track | Correct / requests | Accuracy | Principal observed weakness |
|---|---:|---:|---|
| When2Call | 1,246 / 3,652 | 34.12% | Tool availability and missing information |
| ANLI r3 | 647 / 1,200 | 53.92% | Unsupported claims treated as true or false |
| RAGTruth | 497 / 1,000 | 49.70% | Supported QA and summaries rejected |
| GSM8K, 4 choices | 587 / 1,319 | 44.50% | Arithmetic and multistep constraints |
| GSM8K, 10 choices | 322 / 1,319 | 24.41% | Arithmetic plus candidate-set sensitivity |
| MMLU-Pro | 550 / 2,000 | 27.50% | Engineering, physics, chemistry, business |
| CRUXEval output choice | 303 / 570 | 53.16% | Precise Python execution semantics |
| ARC-Challenge | 749 / 1,172 | 63.91% | Maintain science reasoning coverage |
| WinoGrande | 846 / 1,267 | 66.77% | Maintain commonsense/coreference coverage |

RAGTruth's native Index metric is hallucinated-class **F1: 57.34%** here.
Accuracy is included to expose the prediction imbalance. Do not aggregate
these accuracies into a Decision Index score or count both GSM tracks as
independent problems.

## 1. Tool policy: claiming success without a capability

When2Call's options are direct response (A), tool call (B), clarification (C),
and inability to answer with the available tools (D). Rows below are gold;
columns are predictions.

| Gold | Direct response | Tool call | Clarification | Cannot answer | Recall |
|---|---:|---:|---:|---:|---:|
| Tool call | 457 | 647 | 141 | 50 | 49.96% |
| Clarification | 300 | 391 | 342 | 29 | 32.20% |
| Cannot answer | 637 | 156 | 245 | 257 | 19.85% |

- A is never the gold answer in this test, but receives **1,394 predictions**
  (38.17%). This is a property of the benchmark, not a reason to make direct
  responses invalid in our general agent.
- On all 1,295 cannot-answer rows, candidate B names an unavailable tool.
  Cannot-answer recall is 22.09% with no tools and 19.29% with irrelevant tools.
- **391 / 1,062** clarification cases instead become tool calls. Required
  argument availability needs explicit training coverage.
- Of 2,054 predictions with maximum probability at least 90%, **1,345 are
  wrong**. Calibration alone cannot fix these incorrect choices.

Example: a user requests a modified latte; the registry contains a food-editing
tool but no drink-editing tool. Bazza selects a response claiming the drink is
being prepared, with 98.87% probability. Reference case:
`candidates-v3:62:5ce19067-938e-41f5-acd5-0e8e66683f62`.

**Training target:** capability checks, unavailable relevant tools, irrelevant
registries, missing mandatory arguments, and legitimate direct responses.
Include full candidate responses/calls, rather than only four abstract action
names. Existing coarse dialogue routing does not cover this distinction well.
Make the policy explicit when a task requires using a tool: mental arithmetic
and refusal when a calculator is absent must not receive conflicting labels
under otherwise identical instructions.

## 2. ANLI: neutral is nearly missing

| Gold | Predicted entailment | Predicted neutral | Predicted contradiction | Recall |
|---|---:|---:|---:|---:|
| Entailment | 316 | 19 | 67 | 78.61% |
| Neutral | 135 | 72 | 195 | **17.91%** |
| Contradiction | 124 | 13 | 259 | 65.40% |

Gold classes are approximately balanced. Bazza predicts neutral only
**104 / 1,200 = 8.67%** of the time; macro-F1 is 49.88%.

Example: a report says a body was discovered on Sunday morning. It does not
give the time of death. Bazza entails the hypothesis that death occurred that
morning, where the source labels it neutral. Source ID:
`e999a32b-4835-4cd1-b100-8b0363476f5c`.

**Training target:** three-way evidence reasoning, especially unknown time,
identity, quantities, causes and relationships. Preserve the distinction
between absent evidence and evidence of the opposite. Use ANLI's training
rounds and document NLI, with semantic class quotas rather than weighting the
letter B across unrelated menus.

## 3. RAGTruth: a strong bias toward hallucination

| Gold | Predicted hallucinated | Predicted supported | Class recall |
|---|---:|---:|---:|
| Hallucinated | 338 | 18 | 94.94% |
| Supported | 485 | 159 | **24.69%** |

Hallucinations comprise 35.6% of the panel but 82.3% of predictions. Always
predicting supported would obtain 64.4% accuracy. Hallucination precision is
41.07%, despite high positive recall.

| Source task | Requests | Accuracy | Recall on supported responses |
|---|---:|---:|---:|
| QA | 334 | **28.44%** | **13.72%** |
| Summary | 330 | 43.64% | 25.10% |
| Data2txt | 336 | 76.79% | 49.17% |

This is not a uniform grounding failure: structured data-to-text is much
better than QA and summaries. The available length slices do not establish
long context as the cause.

**Training target:** supported paraphrases, compressed summaries, and supported
answers that do not copy the context verbatim; also entity/number near misses
and added unsupported claims. Balance **task × label**, not just the overall
hallucination rate. Use released human span annotations; do not substitute a
teacher's confidence or outside-world truth for support from the given context.

## 4. Maths: arithmetic plus unstable choice selection

For exactly the same GSM8K problems:

| Outcome | Problems |
|---|---:|
| Both 4-choice and 10-choice correct | 225 |
| 4-choice correct, 10-choice wrong | **362** |
| 4-choice wrong, 10-choice correct | 97 |
| Both wrong | 635 |

The numeric prediction stays the same on only 383 / 1,319 problems. With ten
choices, Bazza selects the last option 25.55% of the time, versus 9.33% gold
frequency; the first option receives 2.81%, versus 9.40% gold. This suggests
candidate/position sensitivity; it does **not** prove a particular architectural
cause or justify changing the benchmark's order.

The completed local competitor tracks use these same questions and options:

| Track | Bazza 4736 | Decider v1.1 27B NF4 |
|---|---:|---:|
| GSM8K, 4 choices | 44.50% | 81.12% (1,070 / 1,319) |
| GSM8K, 10 choices | 24.41% | 71.04% (937 / 1,319) |

Decider revision: `3b45dead91dfa6d95aad6b95764a606fab2bf7a6`; native
serialization/readout, local report under
`build/evaluations/decider-v1.1/text-decisions-v0.1/2026-10-11/`.
This is evidence that the choice protocol is usable by a decision model;
it does not isolate our architecture from model size or full backbone tuning.

Concrete misses: distributing a daily total of feed after two earlier meals
(source test item 4), and an alternating discount that applies to half a
purchase (item 5). The predictions ignore a constraint or combine quantities
incorrectly.

**Training target:** original GSM8K train and native MCQ maths, including rates,
units, percentages, discounts and multistep bookkeeping. Include varied menu
sizes and balanced positions in newly constructed training menus. Own splits
by original problem before creating any variants. No test questions enter the
training set; rationales stay outside the inference input.

## 5. Knowledge: the weak subjects are uneven

| MMLU-Pro subject | Correct / requests | Accuracy |
|---|---:|---:|
| Engineering | 29 / 161 | 18.01% |
| Physics | 39 / 216 | 18.06% |
| Chemistry | 35 / 188 | 18.62% |
| Business | 26 / 131 | 19.85% |
| Maths | 57 / 225 | 25.33% |
| Biology, for comparison | 59 / 119 | 49.58% |
| Computer science, for comparison | 27 / 68 | 39.71% |

Negatively worded questions score 23.72% (215 rows), versus 27.96% on the
remaining rows. This is a descriptive slice, not proof that negation explains
the subject gap. Some individual questions also need an ambiguity review;
do not silently relabel them based on our model.

**Training target:** broader science, quantitative reasoning and decision
instructions such as exceptions. MMLU-Pro itself has validation/test, **no
training split**. Use qualified upstream training material, and check MMLU's
auxiliary training pool against the entire MMLU-Pro test release because their
source lineage can overlap. A frozen backbone may also limit usable knowledge;
these results cannot separate that from head capacity or data coverage.

## 6. Code: executable semantics, beyond software QA

Slicing cases score 47.45% (137 rows), versus 54.97% without slicing. Loop
cases score 53.33%, versus 52.89% without loops: there is no evidence here
that loops alone are the principal difficulty.

Examples: CRUXEval `sample_105` requires Python title case for every word, but
Bazza chooses sentence case. `sample_112` mutates a list during iteration;
Bazza misses the resulting skipped elements. Both are execution questions,
which ordinary software-engineering multiple choice does not reliably teach.

**Training target:** code + concrete input → verified output candidates, with
string methods, slicing, indexing, mutation and short control flow. CodeIO's
raw executed I/O records are a candidate source; CRUXEval's released evaluation
functions are protected. Reject missing I/O and deduplicate by code family.

## What this establishes, and what remains open

The transfer gap contains identifiable task and class biases. This supports
targeting v0.3 at tool policy, evidence distinctions, maths and code execution.
It does not establish that the present head is sufficient, that the frozen
features are sufficient, or that more data will close the entire gap. Larger
fully fine-tuned competitors differ in capacity, adaptation and training data.

These external tests have now influenced our data plan. Keep them fixed for
comparability, but reserve additional untouched source/domain groups for a
later fresh transfer check. Use new development panels for training decisions.
No expensive architecture sweep is needed to make this dataset improvement.
