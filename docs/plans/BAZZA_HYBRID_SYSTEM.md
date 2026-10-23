# Bazza: shared-backbone System 1 / System 2

Discussion captured 18 October 2026. Proposed experiment, not implemented.
Complete the repository refactor before starting this work. This supplements
our 12B release plan; it does not authorize a new training run or publication.

## Goal

Keep Bazza's fast structured decisions and its separately trainable head, while
retaining the original backbone and full autoregressive language-model head.
Use that same model to reply, reason, or revise a plan when a fast decision is
insufficient. Longer-term targets include agents, CUA, vision and audio.

The current 12B backbone is Gemma4-12B IT with NF4 linears/BF16 compute in our
PyTorch path. Its original language-model weights are retained. Instruction
tuning does not prevent using the original generative path for chat/reasoning.
Current results do not establish an accuracy advantage over Winnow, or a
particular benefit from this proposed hybrid.

## Data flow

```text
request + environment/context + supplied action schema
                         |
                  shared backbone
                         |
                 Bazza decision head
                         |
             +-----------+--------------+
             |           |              |
         execute      reply_to_user    deliberate
             |           |              |
           tool       original AR     original AR
                        head            head
                         |              |
                    stream reply    analysis / plan
                                        |
                               append bounded context
                                        |
                               fresh decision prefill
                                        |
                               Bazza decision head
```

A text-valued output field can describe a generative operation in the public
schema. The classifier selects that operation; it does not itself emit arbitrary
text. The original AR path generates the value. Do not imply arbitrary tool
arguments or text become available merely by adding a schema type.

The first implementation uses explicit text feedback. Feeding trainable latent
states directly into the AR path would require an additional learned interface
and a separate experiment. Existing answer-projection residuals are not already
such an interface.

## Minimal experiment

1. **Fast baseline:** current head, unchanged inputs and checkpoint.
2. **Always deliberate:** original AR head produces a bounded analysis, then
   the decision head selects from the original menu using that added context.
3. **Routed:** a simple explicit policy chooses fast or deliberate. First use
   a fixed policy or validation-selected threshold; train a route later only
   if deliberate mode demonstrates useful improvements.
4. **Reply demonstration:** `reply_to_user` invokes the original AR head with
   the original request, relevant state, and selected response instruction.

Use one small, fixed existing held-out panel containing tools, inference, math,
knowledge and code. Keep options/targets identical across modes. Select settings
on validation; inspect test once after selection. Report the modes separately.
Do not export a new full training cache or switch backbone as part of this test.

Explicit reasoning changes the serialized prompt and therefore its features.
Do not reuse cached old hidden states as if they represented the augmented
prompt. Initial hybrid decisions need live backbone forwards.

## Resource and runtime contract

- One backbone resident in memory, two output paths. Extra weights mainly
  belong to the small head; generation still adds KV-cache/activation cost.
- Preserve audited causal backbone semantics and original LM projection.
- Start serially: decide, optionally generate, then decide again. Concurrent
  continuous planning and CPU/GPU placement are later optimizations.
- Start with one deliberation round and a fixed generation-token budget.
- KV reuse is an optimization only when prefix token IDs, position semantics,
  attention masks, quantization and runtime match exactly. A selected action
  does not automatically make the two prompts share a reusable cache.
- Generated analysis is fallible context, not a replacement target or a reason
  to change the supplied action menu silently.
- Route overrides, repeated thought and final actions must be inspectable.

## Measures and decision

Measure per-track accuracy, decision changes and regressions, end-to-end
latency (including the second prefill), generated tokens, peak VRAM, and the
fraction routed to deliberation. Compare reply quality separately from label
selection. Distinguish compute spent on reasoning from head inference time.

Proceed if the shared model produces a useful quality/latency tradeoff or an
agent capability we cannot demonstrate with the decision-only interface. If it
does not, retain the fast model and document the result without expanding the
release scope.

## Later work

- Train a supplied-schema route such as `execute`, `reply`, `deliberate`,
  `ask_for_clarification`; account for routing supervision and errors.
- Learn iterative workspace refinement with intermediate supervision and
  variable iteration counts; merely increasing the existing cycles has not
  demonstrated a general improvement in our experiments.
- Optional backbone LoRA/QLoRA, separated from the original chat path.
- Distill successful planner trajectories back into the fast decision head.
- Continuous planning, explicit memory, smaller reply model, and alternative
  CPU/GPU placement; multimodal observation and action loops.
