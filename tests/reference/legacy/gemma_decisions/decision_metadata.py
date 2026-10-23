"""Semantic task identifiers; these fields never enter the model prompt."""
import hashlib
import json
import unicodedata


def label(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def fields(row):
    meta = row.get("metadata", {})
    source = meta.get("dataset_source", row.get("dataset_source", row["source"]))
    task = meta.get("task_type", row.get("task_type", row["source"] + "/" + row["kind"]))
    schema = meta.get("balance_schema", row.get("balance_schema"))
    if schema is None:
        schema = hashlib.sha256(json.dumps([row.get("question", ""),
                    sorted(label(o) for o in row.get("options", []))], sort_keys=True).encode()).hexdigest()
    return {"dataset_source": source, "task_type": task,
            "task_key": source + "/" + task, "balance_schema": schema,
            "domain": row.get("domain", row["source"]),
            "options": [str(o) for o in row.get("options", [])],
            "loss_weight": float(meta.get("loss_weight", row.get("loss_weight", 1.0))),
            "pair_id": meta.get("upstream_paired_index", row.get("pair_id"))}


def schema_key(row):
    data = fields(row)
    return data["task_key"] + "/" + data["balance_schema"]
