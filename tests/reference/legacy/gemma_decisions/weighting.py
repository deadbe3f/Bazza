"""Train-only semantic class weighting for variable option schemas and soft targets."""
from collections import Counter, defaultdict
import math

import torch

from .decision_metadata import fields, label, schema_key


def fit_weights(records, policy=None):
    unknown = set(policy or {}) - {"class_power", "class_cap", "sources", "tasks"}
    if unknown:
        raise ValueError(f"Unknown weighting options: {sorted(unknown)}")
    policy = {"class_power": .5, "class_cap": 2., "sources": {}, "tasks": {}, **(policy or {})}
    if not 0 <= policy["class_power"] <= 1 or not 1 <= policy["class_cap"] <= 4:
        raise ValueError("Class power must be in [0,1], cap in [1,4]")
    for scope in ("sources", "tasks"):
        if any(v <= 0 or not math.isfinite(v) for v in policy[scope].values()):
            raise ValueError("Category loss multipliers must be finite and positive")
    supported_sources = {fields(r)["dataset_source"] for r in records}
    supported_tasks = {fields(r)["task_key"] for r in records}
    if set(policy["sources"]) - supported_sources or set(policy["tasks"]) - supported_tasks:
        raise ValueError("Weighting override names an absent training source/task")
    counts = defaultdict(Counter)
    for record in records:
        meta = fields(record)
        if meta["dataset_source"] == "swe_qa":
            continue  # Dynamic answers are not universal A/B/C/D semantic classes.
        for option, mass in zip(meta["options"], record["target"]):
            counts[schema_key(record)][label(option)] += mass
    schemas = {}
    for key, mass in sorted(counts.items()):
        maximum, total = max(mass.values()), sum(mass.values())
        raw = {option: min(policy["class_cap"], (maximum / n) ** policy["class_power"]) if n > 0 else 1.
               for option, n in mass.items()}
        normalizer = sum(mass[option] * w for option, w in raw.items()) / total
        schemas[key] = {"target_mass": dict(mass), "weights": {k: w / normalizer for k,w in raw.items()},
                        "zero_support_classes": [k for k,v in mass.items() if v == 0]}
    result = {"format_version": 1, "policy": policy, "schemas": schemas,
              "fit_rows": len(records), "source_shares": {k: n / len(records) for k,n in Counter(fields(r)["dataset_source"] for r in records).items()},
              "definition": "mean of row_weight * sum(class_weight * target * negative_log_probability); targets never renormalized"}
    # Preserve source presentation mass by default. Explicit category multipliers are
    # normalized once over the entire training set, not independently per minibatch.
    result["category_normalizer"] = sum(category_weight(r, result, normalize=False) for r in records) / len(records)
    return result


def category_weight(record, fitted, normalize=True):
    meta = fields(record)
    policy = fitted["policy"]
    value = meta["loss_weight"] * policy["sources"].get(meta["dataset_source"], 1.) * policy["tasks"].get(meta["task_key"], 1.)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Row loss weight must be finite and positive")
    return value / fitted["category_normalizer"] if normalize else value


def weighted_loss(logits, targets, records, fitted):
    # Padded classes have -inf logits; zero times -inf is NaN. Mask their
    # log probabilities before multiplying, without removing genuine soft labels.
    logp = logits.float().log_softmax(-1)
    logp = logp.masked_fill(targets == 0, 0)
    if fitted is None:
        return -(targets * logp).sum(-1).mean()
    weights = torch.ones_like(targets)
    for i, record in enumerate(records):
        lookup = fitted["schemas"].get(schema_key(record), {}).get("weights", {})
        meta = fields(record)
        weights[i, :len(meta["options"])] = torch.tensor([lookup.get(label(o), 1.) for o in meta["options"]], device=targets.device)
        weights[i] *= category_weight(record, fitted)
    return -(weights * targets * logp).sum(-1).mean()
