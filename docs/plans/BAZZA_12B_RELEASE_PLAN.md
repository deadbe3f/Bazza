# Bazza 12B v0.1 release plan

Prepared 18 October 2026. Proposed release target: Friday 23 October 2026,
Australia/Adelaide. This is a review plan; no publishing or new training is
started by this document.

## Release purpose

Ship an inspectable, locally runnable text decision model with a frozen Gemma
4 12B IT backbone and a separately trainable mixed decision head. The practical
offer is supplied-action selection, booleans and ordinal grading, readable
PyTorch code, small head artifacts and a reproducible adaptation workflow.

The current evidence supports a usable research release. A leaderboard lead,
general reasoning capability, multimodal inference and an accepted fast native
runtime have not been established for this checkpoint. A small retrainable head
is useful engineering, but the release must demonstrate that utility rather
than relying on the architecture description alone.

## What the results say

On the same 13,499-request external panel, eight-dataset macro accuracy is:

| System | Accuracy |
|---|---:|
| Latest Bazza 12B, validation-selected checkpoint from the 96K run | 68.47% |
| Original Qwen3.8 27B Q4_K_S, one-letter completion | 68.60% |
| Winnow 12B Q8 | 71.15% |
| Decider v1.1 27B NF4 | 81.54% |

Bazza's GSM8K 4/10-choice scores are 62.55/43.75%, versus Winnow's
61.18/35.86%. Winnow leads on most other tracks. The results do not support
an overall accuracy advantage. Different quantization, prompts, training and
runtimes prevent interpreting these as controlled architecture measurements.

Published generative scores and immediate decision scores measure different
capabilities. An independent original-BF16 Gemma evaluation reports 50.0%
MMLU-Pro under no-thinking loglikelihood scoring, numerically near Bazza's
48.85%; Google's published generative score is 77.2%. These are different
protocols and do not establish how much of our gap is due to precision, head,
data or missing intermediate computation.

Sources: [Google model card](https://ai.google.dev/gemma/docs/core/model_card_4),
[original-BF16 control evaluation](https://huggingface.co/DreamFast/Gemma4-12b-it-abliterlitics).

## Critical path

### Monday 19 October — freeze and recover the release inputs

1. Identify the existing validation-selected checkpoint from the latest 96K
   run; record exact hash, architecture, backbone/tokenizer revision and NF4
   configuration. Keep the 64K checkpoint as a reproducible reference.
2. Confirm the current locations of checkpoints, data manifests and backups.
   Gamesly and Bigly3 paths were unavailable in this planning session. Resolve
   asset access before promising a build; do not silently fall back to an old
   checkpoint because a drive is absent.
3. Commit the necessary source and written evidence in their owning repos.
   Current KneeJerk has extensive uncommitted research work. Stage deliberately
   and preserve unrelated work; do not publish the workspace wholesale.
4. Assemble one release inference path using the accepted PyTorch backbone and
   mixed head. Preserve exact prompt construction and candidate mapping.

Deliverable: a pinned candidate and an explicit list of files needed to load it.

### Tuesday 20 October — establish the useful comparison

1. Score the original Gemma answer-letter readout on the same existing test
   features if the bank is available. This provides a much better local
   backbone-only control than subtracting a published generative score.
2. Use the existing accepted reports for Bazza/Winnow/Decider/Qwen accuracy.
   Verify matching IDs, questions, candidate meanings/order and panel hashes.
3. Run one small, fixed latency/memory panel with the actual release candidate
   and Winnow on the same machine. Use live, uncached requests. Report prompt
   lengths, precision, load time, median/p95 request latency and peak VRAM;
   separate internal compute and service overhead where measurable.
4. Demonstrate two useful supplied-action tasks: tool/action selection and
   a policy/grounding decision. Label authored demo examples as demonstrations,
   separate from held-out evaluation. Include a straightforward adaptation
   example showing the real feature-extraction cost and subsequent head-training
   time separately.

Deliverable: an honest accuracy/cost comparison and a concrete reason to try it.
Do not spend the week on a broad hyperparameter sweep.

### Wednesday 21 October — make the release easy to use

- Export inference-only head weights in safetensors, with a small configuration
  and model card. Exclude optimizer state and cached datasets/features.
- Reference the exact original backbone instead of duplicating its entire
  weights in the head artifact. State clearly that it is required for inference.
- Provide a small importable `decide(state, questions)` interface, one CLI and
  one thin HTTP example using the same serializer/model path. Preserve choice
  keys and boolean/ordinal semantics. Document actual candidate/context limits.
- Supply a pinned working Docker recipe and a small runnable training example.
  Use the current validated dependency stack for this release.
- Package the files required for inference cleanly; keep historical experiment
  launchers outside the public quickstart. The full trainer/repo refactor is a
  separate workstream.
- Document supported hardware, batch-one behavior and the exact measured
  precision. Describe confidence as model scores unless calibration is measured.

Deliverable: someone can download the head, point it at the backbone and run a
decision without reconstructing our workstation directory layout.

### Thursday 22 October — verify the exact downloadable artifact

- Load from a fresh directory/container with no hidden worktree dependencies.
- Compare packaged logits and candidate mappings against the selected research
  checkpoint on fixed fixtures, including variable candidate counts and masks.
- Complete one final test-panel run for the packaged artifact. No checkpoint or
  hyperparameter selection on test results. Reuse existing feature fixtures for
  development checks; run live inference where checking the deployed path.
- Verify fresh-request behavior and input-limit handling. State unsupported
  batching behavior clearly.
- Finish the model card: data provenance/split ownership, teacher-vs-native
  targets, evaluation methods, exposure caveats, known weaknesses and examples.
- Read back the release weight hashes and verify the backup.

Deliverable: reproducible release metrics and a tested installation path.

### Friday 23 October — publish the reviewed v0.1

Publish the head, inference source, pinned setup, model card and result tables.
Keep the public quickstart small. Identify the work as an experimental decision
model with a reproducible adaptation recipe if speed/accuracy comparisons do
not establish an advantage over existing alternatives. Freeze the release tag
and preserve its exact artifacts.

## Runtime boundary

The existing C++/CUDA prototype targets the older 2.3M-parameter head at step
4736, not the current roughly30M mixed head. Its full-model numerical audit
remains open. The CPU implementation has much stronger evidence for the v0.4
mixed head, but head-only parity does not validate the newest full deployment.

Do not silently substitute those ports for the release candidate. Ship the
accepted PyTorch implementation unless the exact new checkpoint/runtime passes
its own checks in time. The native runtime can follow independently.

## Internal reasoning research after the release candidate is stable

Dory's looped backbone, TRM-style workspace recurrence and latent-token feedback
are related directions but different compute paths. Bazza already has a
recurrent head; added cycles alone do not establish that it learned an iterative
algorithm. Dory's small pilot also does not settle the general question.

Use the existing small feature bank for one bounded next experiment:

1. Keep the encoder, data, head width and parameter count fixed.
2. Train shared-weight workspace refinement to use additional iterations,
   revisiting token/candidate memory, with intermediate supervision or a sampled
   iteration count. Maintain the existing head as the control.
3. Measure 1/2/4/8-step validation quality and head latency; prioritize math,
   entailment and code, while watching tools/grounding retention.
4. Continue only if more internal computation produces a useful quality/cost
   tradeoff. Increasing inference iterations on the old checkpoint alone is
   not a test of a model trained for that behavior.
5. If the feature-based experiment stalls, test a small backbone adapter as the
   next distinct intervention. It requires fresh backbone forwards and cannot
   be learned from immutable final-layer caches alone.

This is a follow-up research branch, not a dependency of v0.1. Avoid a new
backbone, another full dataset export, or speculative architecture redesign on
the release critical path.

Research references: [TRM](https://arxiv.org/abs/2510.04871),
[Coconut](https://arxiv.org/abs/2412.06769).
