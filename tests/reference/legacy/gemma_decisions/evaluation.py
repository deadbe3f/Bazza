"""Unweighted evaluation by source, task and semantic class; never fit weights here."""
from collections import Counter, defaultdict

import torch

from .decision_metadata import fields, label, schema_key


def breakdown(logits, records, temperature=1., source_shares=None):
    sources, tasks, classes = defaultdict(list), defaultdict(list), defaultdict(lambda: [0., 0.])
    vulnerabilities = Counter()
    for logit, row in zip(logits, records):
        meta = fields(row)
        target = torch.tensor(row["target"])
        logp = (logit.float() / temperature).log_softmax(-1)
        predicted = int(logp.argmax())
        winners = (target >= target.max() - 1e-6).nonzero().flatten().tolist()
        correct = float(predicted in winners)
        ce = float(-(target * logp.masked_fill(target == 0, 0)).sum())
        sources[meta["dataset_source"]].append((correct, ce))
        tasks[meta["task_key"]].append((correct, ce))
        if meta["options"] and meta["dataset_source"] != "swe_qa":
            for i in winners:
                key = (meta["task_key"], schema_key(row), label(meta["options"][i]))
                classes[key][0] += 1 / len(winners)
                classes[key][1] += correct / len(winners)
        if meta["dataset_source"] == "primevul":
            negative = meta["options"].index("No vulnerability annotated")
            positive = float(target[negative]) < .5
            predicted_positive = predicted != negative
            vulnerabilities["positive_rows" if positive else "negative_rows"] += 1
            vulnerabilities["tp" if positive and predicted_positive else "fn" if positive else "fp" if predicted_positive else "tn"] += 1
            if positive:
                vulnerabilities["type_correct"] += correct
    def aggregate(values):
        return {"rows": len(values), "accuracy": sum(v[0] for v in values) / len(values),
                "cross_entropy": sum(v[1] for v in values) / len(values)}
    source_report = {k: aggregate(v) for k,v in sorted(sources.items())}
    task_report = {k: aggregate(v) for k,v in sorted(tasks.items())}
    class_report = defaultdict(dict)
    task_recalls = defaultdict(list)
    for (task, schema, name), (support, hits) in sorted(classes.items()):
        class_report[schema][name] = {"accepted_target_support": support, "recall": hits / support}
        task_recalls[task].append(hits / support)
    for task, recalls in task_recalls.items():
        task_report[task]["macro_class_recall"] = sum(recalls) / len(recalls)
        task_report[task]["supported_schema_classes"] = len(recalls)
    for source in source_report:
        entries = [v for task,v in task_report.items() if task.startswith(source + "/")]
        source_report[source]["macro_task_accuracy"] = sum(v["accuracy"] for v in entries) / len(entries)
        source_report[source]["macro_task_cross_entropy"] = sum(v["cross_entropy"] for v in entries) / len(entries)
    shares = source_shares or {k: 1 / len(source_report) for k in source_report}
    denominator = sum(shares.get(k, 0.) for k in source_report)
    if denominator <= 0:
        raise ValueError("No declared source weights cover evaluation records")
    result = {"by_dataset_source": source_report, "by_task": task_report, "by_schema_class": dict(class_report),
              "selection_cross_entropy": sum(shares.get(k, 0.) * v["macro_task_cross_entropy"] for k,v in source_report.items()) / denominator,
              "macro_task_accuracy": sum(v["accuracy"] for v in task_report.values()) / len(task_report),
              "selection_definition": "declared source-weighted mean of unweighted per-task cross entropy; observed task classes reported separately"}
    if vulnerabilities:
        positives, negatives = vulnerabilities["positive_rows"], vulnerabilities["negative_rows"]
        result["vulnerability"] = {**dict(vulnerabilities),
            "recall": vulnerabilities["tp"] / max(1, positives),
            "false_positive_rate": vulnerabilities["fp"] / max(1, negatives),
            "type_accuracy_on_vulnerable": vulnerabilities["type_correct"] / max(1, positives)}
    return result


def log_values(score):
    output = {"validation/selection_cross_entropy": score["selection_cross_entropy"],
              "validation/macro_task_accuracy": score["macro_task_accuracy"]}
    for source, row in score["by_dataset_source"].items():
        for key in ("accuracy", "cross_entropy", "macro_task_accuracy", "macro_task_cross_entropy"):
            output[f"validation/source/{source}/{key}"] = row[key]
    for task, row in score["by_task"].items():
        for key in ("accuracy", "cross_entropy", "macro_class_recall"):
            if key in row:
                output[f"validation/task/{task}/{key}"] = row[key]
    for key in ("recall", "false_positive_rate", "type_accuracy_on_vulnerable"):
        if key in score.get("vulnerability", {}):
            output["validation/vulnerability/" + key] = score["vulnerability"][key]
    return output
