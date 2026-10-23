# Bazza architecture improvements

Agreed next steps, 2026-10-11. These are proposals, not implemented models or
measured improvements. The first priority is the Groundbeef v0.3 dataset plan.
Preserve the existing checkpoint, original autoregressive path, and active
evaluation sources. Architecture work starts separately after the data review.

## Starting point

Bazza uses frozen causal Gemma 4 12B-IT token features, a trainable 3840-to-256
projection, attention over candidate states and token memory, and a global
GRUCell across internal refinement cycles. That GRU does not scan the input
tokens and does not retain memory across requests. The active decision path
has 2,301,185 parameters. Decider v1.1 adapts its 27B backbone through a full
fine-tune; model size, data and backbone training all confound the comparison.

## Options to test

### 1. Gated connections from intermediate backbone layers

- Collect selected token representations during the existing backbone pass.
  Start with outputs after blocks 12, 24 and 36 of the 48-block model, plus the
  existing final normalized output. These locations are a starting proposal.
- Normalize each intermediate representation, project to the decision width,
  and add its gated contribution to the existing final projection.
- Begin with learned scalar gates; an input/schema-conditioned gate is a
  subsequent option. Gate inputs must be available at inference.
- Start new contributions at zero while initializing their projections so
  their gates can learn. Retain the current final branch and checkpoint.
- Use the fused memory consistently for candidate seeds, the global-state
  seed and cross-attention. Preserve token positions and padding masks.
- A bias-free 3840-to-32-to-256 projection has 131,072 weights; three have
  393,216. Full dense projections are another option if low rank restricts
  learning. The resulting 256-wide memory is still a compression bottleneck.
- Frozen intermediate features can be cached, but the old final-layer-only
  cache cannot supply these branches. Do not retain all backbone layers.

### 2. GRU over layer depth, alongside the backbone

- Feed the aligned projected features into a shared GRUCell as the selected
  layers become available. Each token has its own 256-wide recurrent state;
  process batch times tokens in parallel. Recurrence is across three or four
  tapped depths, not thousands of token positions or the candidate list.
- Combine the accumulated state with the existing final-layer memory through
  a gate. Cross-token and cross-candidate interaction remains in the decision
  module's attention. This does not change causal backbone attention.
- Three rank-32 projections plus one 256-input/256-hidden GRUCell add 787,968
  parameters before normalization and output gating.
- A forward depth recurrence can run after layer 12 while blocks 13-24 run,
  then after layer 24 alongside blocks 25-36, and so on. The backbone must not
  depend on the side state. Updates on the side remain sequential.
- Implement a clear single-stream reference first. Optional CUDA stream
  overlap requires producer events, state dependencies, a final join, and
  correct tensor lifetimes/autograd handling. Measure full-request latency;
  shared GPU resources may eliminate the expected benefit.
- A backward recurrence across depth waits for the deepest tap. It can be
  tested for quality but cannot fully overlap with unfinished backbone work.

### 3. Optional bidirectional GRU over token memory

- Add a small BiGRU after projection to contextualize token memory in both
  directions, then retain the existing candidate/refinement module initially.
- This is a different recurrence axis from the layer-depth GRU and the current
  global GRUCell. Do not scan the candidate menu as an arbitrary sequence.
- Existing cross-attention already reads the full token memory. A BiGRU adds
  a learned reader; quality and long-prompt latency both need measurement.

### 4. Backbone LoRA / QLoRA for the decision path

- Keep original backbone weights frozen; package unmerged adapter weights
  with the decision module. Enable them for decisions and disable them for
  the original autoregressive chat/planning path.
- Start with rank 8 or 16 on a small selected upper-layer set, subject to a
  memory profile. Inspect actual projections: this Gemma port has no separate
  value projection in its global-attention blocks.
- Backbone adaptation requires gradients through the adapted computation;
  the present blanket no-grad feature path is insufficient. A frozen prefix
  and adapted tail may keep the experiment within the 24 GB card.
- Final-layer or intermediate feature caches become stale when the backbone
  adapters change. Head-only and post-backbone feature adapters retain the
  ability to use frozen feature caches.
- Standalone factorized projections use the same BA form as LoRA. Literal
  LoRA is an update to an existing frozen projection, W_frozen + BA; record
  which formulation is actually used.

## Order of work

1. Review and build a broader, provenance-preserving Groundbeef v0.3 view.
2. Establish the current head on that view and fixed validation panels.
3. Try a modest intermediate-layer reader: gated projections, then the
   depth-GRU option if justified. Keep the recurrent decision core initially.
4. Consider targeted backbone adapters and/or the token BiGRU based on the
   remaining failures, memory budget and latency. Avoid a large exhaustive
   architecture sweep.

Save exact configuration, initialization/parent checkpoint, source hashes,
data membership, raw predictions, quality metrics and latency scope for each
actual run. Preserve the existing external test panel; choose models and
data refinements using development panels rather than repeatedly selecting
against that test panel.

## Sources

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [ELMo layer mixtures](https://aclanthology.org/N18-1202/)
- [PyTorch CUDA stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams)
- [Decider v1.1 model card](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
