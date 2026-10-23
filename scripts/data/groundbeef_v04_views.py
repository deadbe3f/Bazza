"""Standalone v0.4 source repairs; native parents keep their split ownership."""

import copy
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import canonical, digest, dumps, file_hash, jsonl
from groundbeef import decorate, save, state_object, write_rows
from python_execution_worker import check as check_program
from when2call_candidates import added_candidate, plain_call
from when2call_candidates import classify as pilot_classify

VERSION = "groundbeef-v0.4-views"
SEED = 20261013
POLICIES = ["direct", "tool_call", "request_for_info", "cannot_answer"]


def policy(text):
    result = pilot_classify(text)
    if result != "direct":
        return result
    if re.search(
        r"not within my capabilities|not to (generate|perform)|doesn't correspond to a ZIP code",
        text,
        re.I,
    ):
        return "cannot_answer"
    if (
        re.match(r"(for (how many|which)|to which)\b", text, re.I)
        or "may I know" in text
    ):
        return "request_for_info"
    return result


def when_row(raw):
    r = decorate(raw, "when2call", raw["split"])
    r["state"] = state_object(r["state"])
    messages = r["state"]["messages"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("not a single user turn")
    pp = [policy(t) for t in r["options"]]
    if len(set(pp)) != 2:
        raise ValueError(
            "same-action preference pair; preserve for separate preference task"
        )
    chosen = r["target"].index(1.0)
    if pp[chosen] == "direct":
        raise ValueError("unreviewed preferred direct response")
    r["state"] = {"tools": r["state"]["tools"], "question": messages[0]["content"]}
    candidates = {p: plain_call(t) for p, t in zip(pp, r["options"])}
    origins = {
        p: {
            "origin": "native_preference",
            "native_option_index": i,
            "text_sha256": digest(t),
        }
        for i, (p, t) in enumerate(zip(pp, r["options"]))
    }
    for p in POLICIES:
        if p not in candidates:
            candidates[p], origins[p] = added_candidate(p, r)
    r["options"] = [candidates[p] for p in POLICIES]
    r["target"] = [float(p == pp[chosen]) for p in POLICIES]
    r["target_native"] = list(r["target"])
    r["metadata"].update(
        dataset_source="when2call-fourway",
        category="tools",
        task_type="tools.response_selection",
        subcategory="tools.response_selection",
        semantic_class=pp[chosen],
        input_transform=VERSION + "/when2call-fourway",
        original_row_sha256=digest(raw),
        original_options=raw["options"],
        original_target=raw["target"],
        candidate_origins=[origins[p] for p in POLICIES],
        action_order=POLICIES,
        policy_annotation="auditable English response rules; native preference gold preserved",
        synthetic_negative_limit="template negatives, not recovered NVIDIA MCQ responses",
    )
    r["family"] = "tools.response_selection"
    assert r["options"][r["target"].index(1.0)] == plain_call(raw["options"][chosen])
    return r


REFERENCE = re.compile(
    r"\b(all|none) of (the |these |those )?(above|following)|\b(both|either) [A-D]\b|\b(option|choice|answer) [A-D]\b|\b[A-D] (and|or) [A-D]\b",
    re.I,
)


def topic(row):
    text = " ".join([str(state_object(row["state"])), *row["options"]]).lower()
    groups = {
        "chemistry": [
            "chemical",
            "molecule",
            "molar",
            "covalent",
            "ionic",
            "electron configuration",
            "oxidation",
            "organic compound",
        ],
        "physics": [
            "momentum",
            "electric field",
            "magnetic",
            "velocity",
            "acceleration",
            "photon",
            "wavelength",
            "thermodynamic",
        ],
        "engineering": [
            "circuit",
            "transistor",
            "amplifier",
            "impedance",
            "structural",
            "shear stress",
            "tensile",
            "torque",
        ],
        "quantitative": [
            "probability",
            "derivative",
            "integral",
            "matrix",
            "equation",
            "logarithm",
            "polynomial",
            "variance",
        ],
        "law": [
            "statute",
            "prosecution",
            "court",
            "liability",
            "plaintiff",
            "defendant",
            "contract",
            "constitutional",
        ],
        "business": [
            "investment",
            "accounting",
            "revenue",
            "interest rate",
            "market",
            "profit",
            "monetary",
            "demand",
        ],
        "computing": [
            "algorithm",
            "database",
            "software",
            "programming",
            "computer",
            "binary search",
            "operating system",
        ],
        "biology_health": [
            "cell",
            "organism",
            "protein",
            "dna",
            "enzyme",
            "patient",
            "disease",
            "gene",
        ],
    }
    counts = {k: sum(w in text for w in ws) for k, ws in groups.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else "other_unspecified"


def unsafe_menu(row):
    return any(REFERENCE.search(str(o)) for o in row["options"]) or bool(
        re.search(
            r"\b(which of (these|the following)|select|choose).*(false|incorrect|not true|least|except)\b",
            str(row["state"]),
            re.I,
        )
    )


def make_knowledge(source_root, output):
    # Only training-owned negative options enter the distractor bank. Validation
    # question content cannot enter training through candidate retrieval.
    candidates = list(jsonl(source_root / "mmlu/v0.1/train.jsonl.gz"))
    excluded = set(
        json.loads(
            Path("build/groundbeef/architecture-v0.1/exclusions.json").read_text()
        )["blocked_training_ids"]
    )
    # Donors come only from the previously qualified v0.3 training remainder,
    # so old native development/test and the architecture experiment ledger
    # cannot sneak into training through new distractor text.
    donor_ids = {
        r["id"]
        for r in jsonl(
            Path("build/groundbeef/architecture-v0.1/remaining-train.jsonl.gz")
        )
        if r["metadata"]["dataset_source"] == "mmlu"
    }
    eligible = [
        r
        for r in candidates
        if r["id"] in donor_ids
        and r["id"] not in excluded
        and not unsafe_menu(r)
        and len(str(r["state"])) < 3500
    ]
    stop = set(
        "the a an to of for from with and in which what is are was were be on at as this that it by or not all".split()
    )

    def terms(text):
        return Counter(
            t
            for t in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(t) > 2 and t not in stop
        )

    vectors = [terms(r["state"]) for r in eligible]
    tags = [topic(r) for r in eligible]
    df = Counter(t for v in vectors for t in v)
    idf = {t: math.log((len(vectors) + 1) / (n + 1)) + 1 for t, n in df.items()}
    norms = [math.sqrt(sum((n * idf[t]) ** 2 for t, n in v.items())) for v in vectors]
    postings = defaultdict(list)
    for i, v in enumerate(vectors):
        for t in v:
            postings[t].append(i)

    def retrieve(raw):
        q = terms(raw["state"])
        votes = Counter()
        tag = topic(raw)
        for t in sorted((t for t in q if t in idf), key=lambda t: (-idf[t], t))[:12]:
            for i in postings[t]:
                if tags[i] == tag:
                    votes[i] += idf[t]
        candidates = [i for i, _ in votes.most_common(128)]
        if not candidates:
            candidates = [i for i, r in enumerate(eligible) if tags[i] == tag][:128]

        def similarity(i):
            return sum(
                n * vectors[i].get(t, 0) * idf.get(t, 0) ** 2 for t, n in q.items()
            ) / max(norms[i], 1e-6)

        return sorted(candidates, key=lambda i: (-similarity(i), eligible[i]["id"]))[
            :32
        ]

    by_topic = defaultdict(list)
    for r in eligible:
        by_topic[topic(r)].append(r)
    manifests = {}
    for split in ["train", "validation"]:
        all_original = (
            candidates
            if split == "train"
            else list(jsonl(source_root / "mmlu/v0.1/validation.jsonl.gz"))
        )
        native = []
        for raw in all_original:
            base = decorate(raw, "mmlu", split)
            base["metadata"].update(
                selection_topic=topic(raw),
                selection_topic_basis="keyword heuristic; not a native subject label",
            )
            native.append(base)
        # Build a bounded 12K candidate pool, balanced by heuristic topic before
        # expensive retrieval. This is an explicit source view, not all MMLU.
        bins = defaultdict(list)
        for raw in all_original:
            if raw["id"] not in excluded and not unsafe_menu(raw):
                bins[topic(raw)].append(raw)
        for rows in bins.values():
            rows.sort(key=lambda r: digest([SEED, r["id"]]))
        original = []
        active = list(sorted(bins))
        while active and len(original) < 12000:
            for name in list(active):
                if bins[name]:
                    original.append(bins[name].pop())
                else:
                    active.remove(name)
        expanded, rejected = [], Counter()
        for start in range(0, len(original), 128):
            batch = original[start : start + 128]
            indexes = [retrieve(r) for r in batch]
            for raw, inds in zip(batch, indexes):
                base = decorate(raw, "mmlu", split)
                base["metadata"].update(
                    selection_topic=topic(raw),
                    selection_topic_basis="keyword heuristic; not a native subject label",
                )
                if unsafe_menu(raw):
                    rejected["option-relative or negative-stem semantics"] += 1
                    continue
                options = list(base["options"])
                provenance = []
                seen = {canonical(t) for t in options}
                # Retrieve wrong options for similar *other* training questions.
                # Record the source parent for each; no source gold becomes a
                # distractor. Exclude type/length outliers and relative options.
                gold = options[base["target"].index(1.0)]
                lengths = sorted(len(str(o)) for o in options)
                upper = min(1200, max(100, lengths[-1] * 3))
                for idx in inds:
                    other = eligible[int(idx)]
                    if other["group_id"] == raw["group_id"]:
                        continue
                    if tags[int(idx)] != topic(raw):
                        continue
                    for i, text in enumerate(other["options"]):
                        if (
                            other["target"][i] == 1.0
                            or canonical(text) in seen
                            or REFERENCE.search(text)
                        ):
                            continue
                        if not 1 <= len(text) <= upper:
                            continue
                        # Avoid obvious near-equivalent strings/numeric forms.
                        a, b = (
                            re.sub(r"\W+", "", str(gold).lower()),
                            re.sub(r"\W+", "", text.lower()),
                        )
                        if a == b or len(a) > 4 and (a in b or b in a):
                            continue
                        options.append(text)
                        seen.add(canonical(text))
                        provenance.append(
                            {
                                "source_id": other["id"],
                                "source_group": other["group_id"],
                                "source_split": "train",
                                "native_wrong_option_index": i,
                                "text_sha256": digest(text),
                            }
                        )
                        if len(options) == 10:
                            break
                    if len(options) == 10:
                        break
                if len(options) != 10:
                    rejected["fewer than six qualified retrieved negatives"] += 1
                    continue
                r = copy.deepcopy(base)
                order = list(range(10))
                # One deterministic presentation, not repeated shuffle augmentation.
                __import__("random").Random(f"{SEED}:{r['id']}:larger-menu").shuffle(
                    order
                )
                r["options"] = [options[i] for i in order]
                old_gold = base["target"].index(1.0)
                r["target"] = [float(i == old_gold) for i in order]
                r["target_native"] = list(r["target"])
                r["metadata"].update(
                    dataset_source="mmlu-tenchoice",
                    input_transform=VERSION + "/mmlu-tenchoice",
                    original_options=raw["options"],
                    original_target=raw["target"],
                    original_row_sha256=digest(raw),
                    candidate_order=order,
                    added_candidate_sources=provenance,
                    distractor_method="retrieved native wrong options from similar training questions",
                    distractor_limit="source wrong options are not independently proved false for this new question",
                )
                expanded.append(r)
            if start % 8192 == 0:
                print(
                    f"MMLU {split}: {min(start + 128, len(original))}/{len(original)}",
                    flush=True,
                )
        root = output / "mmlu-menus-v0.1"
        root.mkdir(parents=True, exist_ok=True)
        manifests[split] = {
            "native": write_rows(root / (split + "-fourchoice.jsonl.gz"), native),
            "expanded": write_rows(root / (split + "-tenchoice.jsonl.gz"), expanded),
            "rejected": dict(rejected),
        }
    save(
        root / "manifest.json",
        {
            "version": VERSION,
            "source_manifest_sha256": file_hash(
                source_root / "mmlu/v0.1/manifest.json"
            ),
            "files": manifests,
            "builder_sha256": file_hash(__file__),
            "limits": "Heuristic topic tags and derived distractors; no recovered native subject labels or ten-choice MMLU-Pro training claim.",
        },
    )


def execute(row):
    state = state_object(row["state"])
    try:
        check_program(state["code"])
        args = {
            "code": state["code"],
            "function": state["function"],
            "input": state["input"],
        }
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("python_execution_worker.py")),
            ],
            input=dumps(args),
            text=True,
            capture_output=True,
            timeout=3,
            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        result = json.loads(proc.stdout)
        if not result["ok"]:
            return None, result["error"]
        gold = row["options"][row["target"].index(1.0)]
        if result["json"] != dumps(json.loads(gold)):
            return None, "executed output differs from recorded source gold"
        # Convert other source outputs to Python literal candidates. Gold uses
        # the executed type (e.g. tuple vs list), not guessed JSON type ancestry.
        options = [
            result["literal"] if t == 1.0 else repr(json.loads(o))
            for o, t in zip(row["options"], row["target"])
        ]
        if len(set(options)) != len(options) or any(len(o) > 2000 for o in options):
            return None, "invalid converted candidates"
        r = decorate(row, "codeio", row["split"])
        code = state["code"] + "\n\nf = " + state["function"] + "\n"
        r["state"] = {"code": code, "input": result["invocation"], "function": "f"}
        r["question"] = (
            "What is the exact Python return value of the supplied function call?"
        )
        r["options"] = options
        r["metadata"].update(
            dataset_source="codeio-python",
            input_transform=VERSION + "/python-literals",
            original_row_sha256=digest(row),
            original_options=row["options"],
            original_target=row["target"],
            label_basis="independently_executed_source_gold_exact_json_match",
            execution={
                "runtime": "CPython 3.11 restricted worker",
                "worker_sha256": file_hash(
                    Path(__file__).with_name("python_execution_worker.py")
                ),
                "literal": result["literal"],
                "json": result["json"],
                "input": result["invocation"],
            },
            distractor_method="other source-recorded outputs converted to Python literals; types not independently reconstructed",
            semantic_class="answer_position:" + str(row["target"].index(1.0)),
        )
        return r, None
    except (
        ValueError,
        KeyError,
        SyntaxError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as e:
        return None, type(e).__name__ + ":" + str(e)[:100]


def make_code(source_root, output):
    root = output / "codeio-python-v0.2"
    root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split in ["train", "validation"]:
        rows = list(jsonl(source_root / "codeio/v0.1" / (split + ".jsonl.gz")))
        chosen, rejected = [], Counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, (row, (result, error)) in enumerate(
                zip(rows, pool.map(execute, rows)), 1
            ):
                if result:
                    chosen.append(result)
                else:
                    rejected[error] += 1
                if i % 1024 == 0:
                    print(
                        f"CodeIO {split}: {i}/{len(rows)}, verified {len(chosen)}",
                        flush=True,
                    )
        manifests[split] = {
            "file": write_rows(root / (split + ".jsonl.gz"), chosen),
            "rejected": dict(rejected),
        }
    save(
        root / "manifest.json",
        {
            "version": VERSION,
            "source_manifest_sha256": file_hash(
                source_root / "codeio/v0.1/manifest.json"
            ),
            "files": manifests,
            "builder_sha256": file_hash(__file__),
            "limits": "Only short allowlisted programs; exact executed gold verification; source negative outputs are not independently executed.",
        },
    )


def make_when(source_root, output):
    root = output / "when2call-fourway-v0.2"
    root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    quarantine = []
    for split in ["train", "validation"]:
        chosen = []
        rejected = Counter()
        for raw in jsonl(source_root / "when2call/v0.1" / (split + ".jsonl.gz")):
            try:
                chosen.append(when_row(raw))
            except ValueError as e:
                rejected[str(e)] += 1
                quarantine.append(
                    {"id": raw["id"], "split": split, "reason": str(e), "original": raw}
                )
        manifests[split] = {
            "file": write_rows(root / (split + ".jsonl.gz"), chosen),
            "rejected": dict(rejected),
            "policies": dict(Counter(r["metadata"]["semantic_class"] for r in chosen)),
        }
    write_rows(root / "quarantine.jsonl.gz", quarantine)
    save(
        root / "manifest.json",
        {
            "version": VERSION,
            "source_manifest_sha256": file_hash(
                source_root / "when2call/v0.1/manifest.json"
            ),
            "files": manifests,
            "builder_sha256": file_hash(__file__),
            "scope": "Full standalone native preferences transformed; benchmark test not read; original source splits preserved.",
        },
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", choices=["when2call", "knowledge", "code"])
    parser.add_argument("--source-root", type=Path, default=Path("datasets/decisions"))
    parser.add_argument(
        "--output", type=Path, default=Path("build/groundbeef/v0.4-sources")
    )
    args = parser.parse_args()
    folder = {
        "when2call": "when2call-fourway-v0.2",
        "knowledge": "mmlu-menus-v0.1",
        "code": "codeio-python-v0.2",
    }[args.view]
    existing = args.output / folder / "manifest.json"
    if existing.exists():
        # Re-running preparation must not rewrite source views used by a run.
        def verify(value):
            if isinstance(value, dict):
                if isinstance(value.get("file"), str) and "sha256" in value:
                    path = existing.parent / value["file"]
                    if file_hash(path) != value["sha256"]:
                        raise ValueError("Existing source view changed: " + str(path))
                else:
                    for item in value.values():
                        verify(item)
            elif isinstance(value, list):
                for item in value:
                    verify(item)

        verify(json.loads(existing.read_text())["files"])
        print("Existing immutable source view verified:", existing.parent, flush=True)
        return
    {"when2call": make_when, "knowledge": make_knowledge, "code": make_code}[args.view](
        args.source_root, args.output
    )


if __name__ == "__main__":
    main()
