"""The existing native/teacher soft-target CE, with explicit target policies."""
import torch


class DecisionLoss:
    def __init__(self, weighted=False):
        self.weighted = weighted

    def __call__(self, logits, batch):
        if batch.targets is None:
            raise ValueError("Training requires explicit targets")
        # Keep the original reduction: mask logits, softmax, per-row CE, mean.
        losses = -(batch.targets * logits.float().log_softmax(-1)).sum(-1)
        loss = losses.mean()
        if self.weighted:
            weights = logits.new_tensor([row.get("loss_weight", 1.) for row in batch.rows])
            if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("Invalid fitted training weights")
            loss = (losses * weights).sum() / weights.sum()
        return loss, losses
