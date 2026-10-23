"""Training-only observations; no extra model forwards or held-out selection."""

from collections import defaultdict, deque


class TrainingMetrics:
    def __init__(self, tail_rows=1000):
        self.tail = deque(maxlen=tail_rows)
        self.categories = defaultdict(lambda: [0, 0., 0.])
        self.rows = 0
        self.ce = self.hits = 0.
        self.batches = self.clipped_batches = 0
        self.maximum_gradient_norm = 0.

    def forward(self, trainer, context):
        losses = context["per_example_loss"].detach().float().cpu().tolist()
        logits, targets = context["outputs"].detach(), context["targets"]
        selected = targets.gather(1, logits.argmax(-1, keepdim=True)).squeeze(-1)
        hits = (selected >= targets.max(-1).values - 1e-6).float().cpu().tolist()
        for row, ce, hit in zip(context["rows"], losses, hits):
            self.rows += 1
            self.ce += ce
            self.hits += hit
            self.tail.append((ce, hit))
            category = self.categories[row["category"]]
            category[0] += 1
            category[1] += ce
            category[2] += hit

    def batch(self, trainer, context):
        norm = context["metrics"]["gradient_norm"]
        self.batches += 1
        self.clipped_batches += norm > 1.
        self.maximum_gradient_norm = max(self.maximum_gradient_norm, norm)

    def report(self):
        return {"training_rows": self.rows, "training_online_ce": self.ce / self.rows,
                "training_online_accuracy": self.hits / self.rows,
                "training_tail_rows": len(self.tail),
                "training_tail_ce": sum(ce for ce, _ in self.tail) / len(self.tail),
                "training_tail_accuracy": sum(hit for _, hit in self.tail) / len(self.tail),
                "gradient_clipped_fraction": self.clipped_batches / self.batches,
                "maximum_gradient_norm": self.maximum_gradient_norm,
                "by_category": {key: {"rows": count, "cross_entropy": ce / count, "accuracy": hits / count}
                                for key, (count, ce, hits) in sorted(self.categories.items())}}
