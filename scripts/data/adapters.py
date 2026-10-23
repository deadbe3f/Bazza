"""Faithful source-to-decision conversions; labels/rationales never become inputs."""

import ast
import json
import random
import re
import zipfile
from pathlib import Path

from common import (
    SEED,
    code_key,
    digest,
    dumps,
    jsonl,
    key,
    parquet,
    placed,
    row,
)

BENCHMARK_VALIDATION = {"commonsenseqa", "winogrande", "hellaswag"}
DERIVED_VALIDATION = {
    "gsm8k": 500,
    "commonsenseqa": 500,
    "winogrande": 1000,
    "hellaswag": 1000,
    "mmlu": 1000,
    "when2call": 500,
    "api-bank": 200,
    "toolret": 1000,
    "codeio": 1000,
    "ragtruth": 1000,
}


def reject(identity, reason):
    return {"_reject": reason, "_identity": str(identity)}


def error_reason(error):
    if isinstance(error, SyntaxError):
        return "invalid_syntax"
    if isinstance(error, KeyError):
        return "missing_required_field"
    if isinstance(error, IndexError):
        return "index_out_of_range"
    if "is not in list" in str(error):
        return "native_label_not_in_supplied_options"
    if "invalid literal for int()" in str(error):
        return "unlabelled"
    return str(error)


def native_split(name, path):
    filename = Path(path).name
    if "auxiliary_train" in filename:
        return "train"
    if "train" in filename:
        return "train"
    if "validation" in filename or "dev" in filename:
        return (
            "test" if name in BENCHMARK_VALIDATION or name == "mmlu" else "validation"
        )
    return "test"


def labelled_options(items):
    return [re.sub(r"^\s*[A-Ea-e]\s*[).]\s*", "", text).strip() for text in items]


def mathqa_options(text):
    # Commas inside a number are not delimiters. Only the next letter marker is.
    matches = list(re.finditer(r"(?:^|,\s*)([a-e])\s*\)\s*", text, re.I))
    if [m.group(1).lower() for m in matches] != list("abcde"):
        raise ValueError("unparseable_choice_labels")
    return [
        text[
            m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)
        ].strip()
        for i, m in enumerate(matches)
    ]


def mcq(source, raw, identity, split, filename, intent_names=None):
    adapter = source["adapter"]
    meta = {"source_file": filename, "native_split": Path(filename).stem.split("-")[0]}
    if adapter == "gsm":
        from decimal import Decimal

        gold = Decimal(raw["answer"].split("####")[-1].strip().replace(",", ""))
        if not gold.is_finite():
            raise ValueError("invalid_numeric_gold")
        result = row(
            source,
            f"{split}:{identity}",
            split,
            raw["question"],
            ["pending", "conversion"],
            0,
            "arithmetic.word_problems",
            gold_numeric=format(gold.normalize(), "f"),
            **meta,
        )
        result["metadata"]["native_id"] = str(identity)
        result["_gsm"] = True
        return result
    if adapter in {"arc", "openbook", "csqa"}:
        stem = raw.get("question", raw.get("question_stem"))
        options = raw["choices"]["text"]
        correct = raw["choices"]["label"].index(raw["answerKey"])
        family = {
            "arc": "science.reasoning",
            "openbook": "science.facts_and_reasoning",
            "csqa": "commonsense.concepts",
        }[adapter]
        identity = raw["id"]
    elif adapter == "aqua":
        stem, options, correct = (
            raw["question"],
            labelled_options(raw["options"]),
            ord(raw["correct"]) - 65,
        )
        family = "math.quantitative"
    elif adapter == "mathqa":
        stem, options, correct = (
            raw["Problem"],
            mathqa_options(raw["options"]),
            ord(raw["correct"]) - 97,
        )
        family = "math." + raw["category"]
    elif adapter == "sciq":
        stem = raw["question"]
        options, correct = placed(
            [
                raw["distractor1"],
                raw["distractor2"],
                raw["distractor3"],
                raw["correct_answer"],
            ],
            3,
            f"sciq:{identity}",
        )
        family = "science.facts_and_reasoning"
        meta["support_omitted"] = True
    elif adapter == "wino":
        stem = raw["sentence"]
        options, correct = [raw["option1"], raw["option2"]], int(raw["answer"]) - 1
        family = "commonsense.coreference"
    elif adapter == "hella":
        stem, options, correct = raw["ctx"], raw["endings"], int(raw["label"])
        family = "commonsense.event_completion"
        meta.update(activity=raw["activity_label"], group=key(raw["source_id"]))
    elif adapter == "mmlu":
        stem, options, correct = raw["question"], raw["choices"], raw["answer"]
        subject = raw["subject"] or "unspecified"
        family = "knowledge." + subject
        meta["native_subject"] = subject
    elif adapter == "anli":
        return row(
            source,
            raw["uid"],
            split,
            "What is the relationship of the hypothesis to the premise?",
            ["Entailment", "Neutral", "Contradiction"],
            raw["label"],
            "evidence.nli",
            state={"premise": raw["premise"], "hypothesis": raw["hypothesis"]},
            group=key(raw["premise"]),
            protection_text=raw["premise"] + "\n" + raw["hypothesis"],
            native_round=re.search(r"_r([123])", filename).group(1),
            **meta,
        )
    elif adapter == "clinc":
        correct = raw["intent"]
        if not 0 <= correct < len(intent_names):
            raise ValueError("unlabelled")
        negatives = [i for i in range(len(intent_names)) if i != correct]
        random.Random(f"{SEED}:clinc:{split}:{identity}").shuffle(negatives)
        options, correct = placed(
            [intent_names[correct].replace("_", " ")]
            + [intent_names[i].replace("_", " ") for i in negatives[:9]],
            0,
            f"clinc:{split}:{identity}",
        )
        return row(
            source,
            f"{split}:{identity}",
            split,
            "Select the intent of this request from the supplied candidates.",
            options,
            correct,
            "tools.intent",
            state=raw["text"],
            protection_text=raw["text"],
            semantic_gold=intent_names[raw["intent"]],
            candidate_scope="10 of 151 native intents",
            **meta,
        )
    else:
        raise ValueError(f"Unknown MCQ adapter: {adapter}")
    if not 0 <= correct < len(options):
        raise ValueError("unlabelled")
    return row(
        source, f"{filename}:{identity}", split, stem, options, correct, family, **meta
    )


def records(source, raw_root):
    name, adapter = source["name"], source["adapter"]
    root = raw_root / name / "v0.1/raw"
    if adapter in {
        "gsm",
        "arc",
        "openbook",
        "csqa",
        "aqua",
        "sciq",
        "wino",
        "hella",
        "mmlu",
        "anli",
        "clinc",
    }:
        for file in sorted(root.rglob("*.parquet")):
            split = native_split(name, file)
            names = None
            if adapter == "clinc":
                import pyarrow.parquet as pq

                names = json.loads(
                    pq.ParquetFile(file).schema_arrow.metadata[b"huggingface"]
                )["info"]["features"]["intent"]["names"]
            for index, raw in enumerate(parquet(file)):
                try:
                    yield mcq(
                        source, raw, index, split, str(file.relative_to(root)), names
                    )
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    yield reject(f"{file.name}:{index}", error_reason(error))
    elif adapter == "mathqa":
        with zipfile.ZipFile(root / "MathQA.zip") as archive:
            for filename in [
                "train.json",
                "dev.json",
                "test.json",
                "challenge_test.json",
            ]:
                split = (
                    "train"
                    if filename == "train.json"
                    else "validation"
                    if filename == "dev.json"
                    else "test"
                )
                for i, raw in enumerate(json.loads(archive.read(filename))):
                    try:
                        yield mcq(source, raw, f"{filename}:{i}", split, filename)
                    except (ValueError, KeyError, IndexError) as error:
                        yield reject(f"{filename}:{i}", error_reason(error))
    elif adapter == "when_pref":
        for i, raw in enumerate(jsonl(root / "train/when2call_train_pref.jsonl")):
            chosen, rejected = (
                raw["chosen_response"]["content"],
                raw["rejected_response"]["content"],
            )
            options, correct = placed([chosen, rejected], 0, f"when2call:{i}")
            messages = raw["messages"]
            query = "\n".join(m["content"] for m in messages if m["role"] == "user")
            yield row(
                source,
                i,
                "train",
                "Which supplied response better follows the request and available tool capabilities?",
                options,
                correct,
                "tools.response_preference",
                state={
                    "tools": [json.loads(t) for t in raw["tools"]],
                    "messages": messages,
                },
                group=key(query),
                protection_text=query,
                origin="native_preference",
                source_file="train/when2call_train_pref.jsonl",
            )
    elif adapter == "toolret":
        for file in sorted(root.rglob("*.parquet")):
            for raw in parquet(file):
                for positive in [True, False]:
                    candidates = raw["positive" if positive else "negative"]
                    if not candidates:
                        yield reject(raw["id"], "missing_positive_or_negative")
                        continue
                    tool = min(candidates, key=digest)
                    try:
                        tool = json.loads(tool)
                    except ValueError:
                        yield reject(raw["id"], "invalid_tool_json")
                        continue
                    yield row(
                        source,
                        f"{raw['id']}:{int(positive)}",
                        "train",
                        "Is this tool relevant to fulfilling the request?",
                        ["Not relevant", "Relevant"],
                        int(positive),
                        "tools.relevance",
                        state={"request": raw["query"], "tool": tool},
                        group=key(raw["query"]),
                        protection_text=raw["query"],
                        origin="source_relevance_label",
                        source_file=str(file.relative_to(root)),
                    )
    elif adapter == "api_bank":
        for file in sorted(root.glob("training-data/*-api-train.json")):
            for i, raw in enumerate(json.loads(file.read_text())):
                try:
                    text, tools = raw["input"].lstrip(), []
                    while text.startswith("{"):
                        tool, offset = json.JSONDecoder().raw_decode(text)
                        tools.append(tool)
                        text = text[offset:].lstrip()
                    tree = ast.parse(
                        raw["output"].split("API-Request:", 1)[1].strip(), mode="eval"
                    ).body
                    if (
                        not isinstance(tree, ast.List)
                        or len(tree.elts) != 1
                        or not isinstance(tree.elts[0], ast.Call)
                    ):
                        raise ValueError("not_a_single_call")
                    func = tree.elts[0].func
                    if not isinstance(func, ast.Name):
                        raise ValueError("unsupported_call_name")
                    names = [t["apiCode"] for t in tools]
                    if func.id not in names:
                        raise ValueError("target_tool_missing_from_supplied_schema")
                    correct = names.index(func.id)
                    options = [t["apiCode"] + ": " + t["description"] for t in tools]
                    options.append("No supplied tool can fulfill the next call.")
                    yield row(
                        source,
                        f"{file.name}:{i}",
                        "train",
                        "Select the tool for the next call.",
                        options,
                        correct,
                        "tools.selection",
                        state={
                            "tools": tools,
                            "dialogue_prefix": text,
                            "instruction": raw["instruction"],
                        },
                        group=key(text),
                        protection_text=text,
                        origin="native_api_call_name",
                        source_file=str(file.relative_to(root)),
                        arguments_not_supervised=True,
                    )
                except (ValueError, KeyError, SyntaxError, IndexError) as error:
                    yield reject(f"{file.name}:{i}", error_reason(error))
    elif adapter == "rag":
        sources = {r["source_id"]: r for r in jsonl(root / "dataset/source_info.jsonl")}
        for raw in jsonl(root / "dataset/response.jsonl"):
            info = sources[raw["source_id"]]
            yield row(
                source,
                raw["id"],
                raw["split"],
                "Does the response contain any claim unsupported by the supplied source?",
                ["Supported", "Contains unsupported claim"],
                int(bool(raw["labels"])),
                "evidence." + info["task_type"].lower(),
                state={
                    "task": info["task_type"],
                    "prompt": info["prompt"],
                    "response": raw["response"],
                },
                group=key(info["prompt"]),
                protection_text=info["prompt"],
                origin="human_span_annotation",
                responder_model=raw["model"],
                source_id=raw["source_id"],
                source_file="dataset/response.jsonl",
            )
    elif name == "contractnli":
        with zipfile.ZipFile(root / "contract-nli.zip") as archive:
            for filename, split in [
                ("train.json", "train"),
                ("dev.json", "validation"),
                ("test.json", "test"),
            ]:
                data = json.loads(archive.read("contract-nli/" + filename))
                for document in data["documents"]:
                    for label, definition in data["labels"].items():
                        choices = {
                            a["annotations"][label]["choice"]
                            for a in document["annotation_sets"]
                        }
                        if len(choices) != 1:
                            yield reject(
                                f"{document['id']}:{label}", "annotation_disagreement"
                            )
                            continue
                        correct = ["Entailment", "NotMentioned", "Contradiction"].index(
                            choices.pop()
                        )
                        yield row(
                            source,
                            f"{document['id']}:{label}",
                            split,
                            "Does the contract entail, omit, or contradict the hypothesis?",
                            ["Entailment", "Not mentioned", "Contradiction"],
                            correct,
                            "evidence.contract_nli",
                            state={
                                "contract": document["text"],
                                "hypothesis": definition["hypothesis"],
                            },
                            group=key(document["text"]),
                            protection_text=document["text"]
                            + "\n"
                            + definition["hypothesis"],
                            origin="human_contract_annotation",
                            hypothesis_id=label,
                            source_file="contract-nli.zip:" + filename,
                            full_context=True,
                        )
    elif adapter == "codeio":
        file = next(root.glob("*.jsonl"))
        for i, raw in enumerate(jsonl(file)):
            try:
                code = raw["refcode"]
                tree = ast.parse(code)
                forbidden = {
                    "random",
                    "time",
                    "datetime",
                    "os",
                    "sys",
                    "subprocess",
                    "requests",
                    "socket",
                }
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import) and any(
                        n.name.split(".")[0] in forbidden for n in node.names
                    ):
                        raise ValueError("external_or_nondeterministic_dependency")
                    if (
                        isinstance(node, ast.ImportFrom)
                        and (node.module or "").split(".")[0] in forbidden
                    ):
                        raise ValueError("external_or_nondeterministic_dependency")
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in {"eval", "exec", "input", "open"}
                    ):
                        raise ValueError("dynamic_or_external_io")
                pairs = raw["ios"]
                if not pairs:
                    raise ValueError("no_io")
                # One presentation per program; alternatives are source-executed
                # outputs for OTHER inputs to this very program, never invented.
                selected = int(digest(f"{SEED}:codeio:{i}")[:8], 16) % len(pairs)
                gold = dumps(pairs[selected]["output"])
                alternatives = sorted(
                    {dumps(p["output"]) for p in pairs} - {gold}, key=digest
                )
                if len(alternatives) < 3:
                    raise ValueError("fewer_than_four_distinct_observed_outputs")
                options, correct = placed([gold] + alternatives[:3], 0, f"codeio:{i}")
                if len(code) > 24000 or any(len(o) > 4000 for o in options):
                    raise ValueError("oversize_code_or_options")
                yield row(
                    source,
                    i,
                    "train",
                    "What JSON-serialized return value does this function produce for the supplied input?",
                    options,
                    correct,
                    "code.execution",
                    state={
                        "code": code,
                        "function": raw["funcname"],
                        "input": pairs[selected]["input"],
                    },
                    group=code_key(code),
                    protection_text=code,
                    origin="source_executed_io_not_reexecuted",
                    source_file=file.name,
                    original_category=raw.get("category"),
                    original_source=raw.get("source"),
                    original_meta=raw.get("meta"),
                    io_index=selected,
                    distractor_method="other_observed_outputs_same_program",
                )
            except (ValueError, KeyError, IndexError, SyntaxError, TypeError) as error:
                yield reject(i, error_reason(error))
    else:
        return
