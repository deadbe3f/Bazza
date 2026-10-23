"""Named batches, fixed Gemma serialization and deterministic batching."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata
import torch


@dataclass
class DecisionBatch:
    rows: list
    attention_mask: torch.Tensor
    option_positions: torch.Tensor
    option_mask: torch.Tensor
    targets: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    input_ids: torch.Tensor | None = None
    split: str = "train"
    feature_space: str | None = None
    feature_source: str | None = None

    def inputs(self):
        return dict(hidden=self.hidden, attention_mask=self.attention_mask,
                    option_positions=self.option_positions, option_mask=self.option_mask)

    def to(self, device):
        return DecisionBatch(self.rows, **{key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in vars(self).items() if key != "rows"})


def collate_records(rows, hidden=None, *, target_key="target"):
    """Pad only the head input. No truncation, option permutation or relabelling."""
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(row["ids"]) for row in rows)
    choices = max(len(row["positions"]) for row in rows)
    ids = torch.zeros(len(rows), length, dtype=torch.long)
    mask = torch.zeros_like(ids)
    positions = torch.zeros(len(rows), choices, dtype=torch.long)
    option_mask = torch.zeros_like(positions, dtype=torch.bool)
    targets = torch.zeros(len(rows), choices)
    memory = None
    if hidden is not None:
        memory = hidden[0].new_zeros(len(rows), length, hidden[0].shape[-1])
    has_targets = [target_key in row for row in rows]
    if any(has_targets) and not all(has_targets):
        raise ValueError("Mixed missing/present targets; choose an explicit target policy")
    for i, row in enumerate(rows):
        n, k = len(row["ids"]), len(row["positions"])
        if n < 1 or not 2 <= k <= 26 or any(p < 0 or p >= n for p in row["positions"]):
            raise ValueError("Invalid token/option positions: " + row["id"])
        ids[i, :n] = torch.tensor(row["ids"])
        mask[i, :n] = 1
        positions[i, :k] = torch.tensor(row["positions"])
        option_mask[i, :k] = True
        if all(has_targets):
            value = torch.tensor(row[target_key], dtype=torch.float32)
            # Validate the supplied probabilities before FP32 conversion moves a
            # rounded sum (e.g. 0.9999) across the acceptance boundary.
            if value.shape != (k,) or not torch.isfinite(value).all() or (value < 0).any() or abs(math.fsum(row[target_key]) - 1) > 1e-4:
                raise ValueError("Invalid target distribution: " + row["id"])
            targets[i, :k] = value
        if memory is not None:
            if hidden[i].shape != (n, memory.shape[-1]):
                raise ValueError("Cached hidden shape differs: " + row["id"])
            memory[i, :n].copy_(hidden[i])
    return DecisionBatch(rows, mask, positions, option_mask, targets if all(has_targets) else None,
                         memory, ids)


def load_selection(path):
    selection = json.loads(Path(path).read_text())
    splits = selection["splits"]
    for name, rows in splits.items():
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate example IDs in " + name)
    train_ids = {r["id"] for r in splits.get("train", [])}
    train_groups = {r["group_id"] for r in splits.get("train", []) if r.get("group_id")}
    for name, rows in splits.items():
        if name == "train":
            continue
        if train_ids & {r["id"] for r in rows} or train_groups & {r["group_id"] for r in rows if r.get("group_id")}:
            raise ValueError("Training/held-out membership overlap in " + name)
    return selection
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

def encode_input(row, tokenizer, max_length, order=None):
    options = row["options"]
    if order is None:
        order = list(range(len(options)))
    if not 2 <= len(order) <= 26:
        return None
    state = row["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, sort_keys=True)
    prompt = "<bos><|turn>user\nChoose the best answer. Reply with its letter only.\n"
    prompt += "Context:\n" + state + "\nQuestion:\n" + row["question"] + "\nOptions:\n"
    ends = []
    for index, original in enumerate(order):
        prompt += chr(65 + index) + ". " + str(options[original])
        ends.append(len(prompt))
        prompt += "\n"
    prompt += "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    if len(encoded.ids) > max_length:
        return None  # No silently truncated states, questions or choices.
    positions = []
    for end in ends:
        eligible = [i for i, (a, b) in enumerate(encoded.offsets) if 0 < b <= end]
        if not eligible:
            raise ValueError("Cannot locate option's final token")
        positions.append(eligible[-1])
    return {"id": row["id"], "group_id": row["group_id"], "source": row["source"],
            "kind": row["kind"], "ids": encoded.ids, "positions": positions,
            "order": order,
            **{**fields(row), "options": [str(options[i]) for i in order]}}

def encode(row, tokenizer, max_length, order=None):
    result = encode_input(row, tokenizer, max_length, order)
    if result is None:
        return None
    target = [float(row['target'][i]) for i in result['order']]
    if len(target) != len(result['order']) or min(target) < 0 or abs(sum(target) - 1) > 1e-4:
        raise ValueError('Invalid target distribution')
    result['target'] = target
    return result


def length_batches(records, max_rows, token_budget):
    """Bucket by length, bounding padded tokens without dropping long rows."""
    batch = []
    for row in sorted(records, key=lambda r: (len(r["ids"]), r["id"])):
        if batch and (len(batch) >= max_rows or
                      (len(batch) + 1) * len(row["ids"]) > token_budget):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch

def training_batches(rows, batch_size, token_budget, seed):
    """Shuffle examples, bucket locally by length, then shuffle the batches."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    batches = []
    for start in range(0, len(rows), batch_size * 16):
        batches.extend(length_batches(rows[start:start + batch_size * 16], batch_size, token_budget))
    rng.shuffle(batches)
    return batches


class BatchLoader:
    """Normal CPU collation/prefetch; it never runs a backbone or touches CUDA in its worker."""
    def __init__(self, collate_fn, device='cuda', prefetch_batches=8):
        self.collate_fn,self.device,self.depth=collate_fn,device,prefetch_batches
        if prefetch_batches<1:raise ValueError('prefetch_batches must be positive')

    def __call__(self, rows, split):
        return self.collate_fn(rows,split).to(self.device)

    def iter_batches(self, batches, start, split):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='batch-loader') as pool:
            pending=deque();cursor=start
            def submit():
                nonlocal cursor
                if cursor<len(batches):
                    pending.append((cursor,pool.submit(self.collate_fn,batches[cursor],split)));cursor+=1
            try:
                for _ in range(self.depth):submit()
                while pending:
                    index,future=pending.popleft();batch=future.result();submit()
                    yield index,batch.to(self.device)
            finally:
                for _,future in pending:future.cancel()
