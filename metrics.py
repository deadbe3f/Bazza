"""Unweighted and grouped evaluation; targets are never relabelled as truth."""
from collections import Counter, defaultdict, deque
import torch
from data import fields, label, schema_key

def metrics(logits, rows):
    categories, sources = defaultdict(list), defaultdict(list)
    for logit, row in zip(logits, rows):
        target = torch.tensor(row["target"])
        ce = float(-(target * logit.float().log_softmax(-1)).sum())
        hit = float(target[int(logit.argmax())] >= target.max() - 1e-6)
        # Older external panels carry a domain/source but no training-category tag.
        categories[row.get("category", row.get("domain", row["dataset_source"]))].append((hit, ce))
        sources[row["dataset_source"]].append((hit, ce))
    def aggregate(values):
        return {"rows": len(values), "accuracy": sum(v[0] for v in values) / len(values),
                "cross_entropy": sum(v[1] for v in values) / len(values)}
    result = aggregate([item for values in categories.values() for item in values])
    result["by_category"] = {k: aggregate(v) for k, v in sorted(categories.items())}
    result["by_source"] = {k: aggregate(v) for k, v in sorted(sources.items())}
    result["macro_category_cross_entropy"] = sum(v["cross_entropy"] for v in result["by_category"].values()) / len(categories)
    return result

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

def score(logits,rows):
    if any(not torch.isfinite(v).all() for v in logits):raise FloatingPointError('Non-finite validation logits')
    result={**metrics(logits,rows),**breakdown(logits,rows)}
    grouped=defaultdict(list)
    for v,r in zip(logits,rows):
        if 'semantic_class' in r:
            grouped[(r['dataset_source'],r['semantic_class'])].append((v,r))
    result['by_semantic_class']={s+'/'+c:metrics([v for v,_ in rr],[r for _,r in rr]) for (s,c),rr in grouped.items()}
    return result

def track(row):
    source=row['dataset_source']
    if source.startswith('gsm8k-'):return 'gsm8k'
    if source.startswith('openscience-'):return 'openscience'
    return source


def selection_metrics(logits,rows):
    groups=defaultdict(list)
    for logit,row in zip(logits,rows):
        target=torch.tensor(row['target']);value=logit.float()
        correct=float(target[int(value.argmax())]>=target.max()-1e-6)
        ce=float(-(target*value.log_softmax(-1)).sum())
        groups[track(row)].append((correct,ce))
    tracks={k:{'rows':len(v),'accuracy':sum(a for a,_ in v)/len(v),'cross_entropy':sum(c for _,c in v)/len(v)} for k,v in sorted(groups.items())}
    return {'macro_track_accuracy':sum(v['accuracy'] for v in tracks.values())/len(tracks),
            'macro_track_cross_entropy':sum(v['cross_entropy'] for v in tracks.values())/len(tracks),
            'by_selection_track':tracks,
            'checkpoint_selection_definition':'Equal dataset/task tracks; GSM 4/10 views combined, OpenScience 4/10 combined; CE tie-breaker. Subtasks remain diagnostic.'}


def checkpoint_key(row):return (-row['macro_track_accuracy'],row['macro_track_cross_entropy'],row['step'])

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
