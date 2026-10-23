#!/usr/bin/env python3
"""Save the audited release report and recipe sources; never rewrite data files."""

import argparse
import json
import shutil
from pathlib import Path

from common import REPO, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    args = p.parse_args()
    root = args.dataset
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    stats = json.loads((root / "statistics.json").read_text())
    recipe = json.loads((root / "recipe.json").read_text())
    if not audit["verified"]:
        raise ValueError("Package only independently audited releases")
    if (root / "release.json").exists():
        release = json.loads((root / "release.json").read_text())
        for relative, expected in release["sha256"].items():
            if file_hash(root / relative) != expected:
                raise ValueError("Changed release artifact: " + relative)
        print("Existing release package verified", flush=True)
        return
    snapshot = root / "recipe_source"
    snapshot.mkdir()
    files = list(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).with_name("sources.json"),
        Path(__file__).with_name("groundbeef_v03_recipe.json"),
        REPO / "scripts/build_groundbeef_v03.sh",
    ]
    for file in files:
        shutil.copyfile(file, snapshot / file.name)
    if file_hash(snapshot / "groundbeef.py") != manifest["converter_sha256"]:
        raise ValueError("Builder changed since the release was constructed")
    if file_hash(snapshot / "audit_groundbeef.py") != audit["audit_script_sha256"]:
        raise ValueError("Auditor changed since verification")
    lines = [
        "# Groundbeef v0.3",
        "",
        "Created and audited 11 October 2026. No training run started.",
        "",
        "80,000 training decisions: 64,000 from the new source views and 16,000 retained from v0.2. "
        f"There are {stats['train']['distinct_groups']:,} source parent groups. "
        "All 84,500 train/development/calibration inputs were retokenized and retain their source answer labels.",
        "",
        "## Training mixture",
        "",
        "| Category | Rows | Share |",
        "|---|---:|---:|",
    ]
    for category, count in sorted(stats["train"]["counts"]["category"].items()):
        lines.append(f"| {category} | {count:,} | {count / 800:.2f}% |")
    lines += [
        "",
        "## Sources",
        "",
        "| Source | Train | Full validation | Quick validation | Calibration |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, count in recipe["train"].items():
        actual = source.removeprefix("retained:")
        counts = [
            stats[s]["by_source"].get(actual, {}).get("rows", 0)
            for s in ["validation", "validation-quick", "calibration"]
        ]
        lines.append(
            f"| {source} | {count:,} | {counts[0]:,} | {counts[1]:,} | {counts[2]:,} |"
        )
    lines += [
        "",
        "## Validation and checkpoint selection",
        "",
        "- **Full validation: 4,000 decisions.** 3,700 new-source decisions plus 300 previously monitored legacy anchors. "
        "None overlap recorded v0.1/v0.2 head training by the audited parent/input aliases.",
        "- **Quick validation: 1,000 fixed decisions.** An exact subset of full validation, with source proportions preserved "
        "to rounding. Never resample it between checkpoints.",
        "- **Calibration: 500 decisions**, with separate source parents from training and validation.",
        "- Native option order is fixed. Only row order is mixed. GSM training uses 6,000 unique problems, "
        "one 4-choice or 10-choice view each. Full validation pairs both views on the same 200 held-out problems.",
        "- Use unweighted category macro CE for development, plus per-task semantic class recall and accuracy. "
        "Report the new-source and legacy slices separately. Quick checks are diagnostic; full validation should "
        "confirm checkpoint promotion or stopping. These recommendations are not changes to the running trainer.",
        "- Natural benchmark test frequencies are preserved. Balanced development scores should not be presented "
        "as the external benchmark aggregate.",
        "",
        "### Format alignment and remaining gaps",
        "",
        "GSM8K, ARC, ANLI, WinoGrande, RAGTruth and MMLU use the pinned Decision Index input layouts "
        "on other split questions. Supplied MCQ candidate order is preserved. RAGTruth span annotations "
        "have no native candidate menu: the fixed benchmark true/false schema uses the explicit "
        "standalone-to-release index mapping `[1,0]`, preserving every source answer label. "
        "The following task differences remain explicit:",
        "",
    ]
    lines += ["- " + text for text in manifest["validation"]["mismatches"]]
    lines += [
        "",
        "## Selected semantic label counts",
        "",
        "| Source / class | Train | Validation |",
        "|---|---:|---:|",
    ]
    for source in [
        "anli",
        "ragtruth",
        "when2call",
        "toolace",
        "toolret",
        "helpsteer2",
        "groundbeef_rules",
    ]:
        classes = set(stats["train"]["by_source"][source]["classes"]) | set(
            stats["validation"]["by_source"][source]["classes"]
        )
        for label in sorted(classes):
            counts = [
                stats[s]["by_source"][source]["classes"].get(label, 0)
                for s in ["train", "validation"]
            ]
            lines.append(f"| {source} / {label} | {counts[0]:,} | {counts[1]:,} |")
    lines += [
        "",
        "## Objectives and weights",
        "",
        "The active `target` retains the source answer labels, with the RAGTruth index mapping recorded. Native labels and Open-Jev oracle distributions have "
        "`target_native`; the 2,000 non-Open-Jev distill rows have `target_teacher`, with no invented native answer. "
        "Teacher argmax is only a diagnostic class label, never renamed ground truth.",
        "",
        "Mixture quotas and source-specific class sampling supply the main balance. "
        "`training_weights.json` starts with equal per-example loss weights (`class_power=0`), "
        "avoiding a second inverse-frequency weighting pass. Validation uses unweighted losses. "
        "Finite semantic classes are separated from random entity/tool IDs: rule tool selection uses "
        "`compatible_tool`, `no_compatible_tool`, and `ask_for_destination` for reporting, while retaining "
        "the real supplied tool names as model candidates. Each of the four rule task families has 250 training rows.",
        "",
        "## Provenance and audit",
        "",
        "- `membership.jsonl.gz`: all 84,500 selected identities, source/parent IDs, revisions, original and exported "
        "row hashes, input aliases, transformations, labels, token lengths and historical exposure flags.",
        "- `sources.json`: complete upstream locks and parent/source manifests; `recipe.json`: exact quotas and caps.",
        "- `statistics.json`: every source/task/class/menu count, distinct parents and token summaries.",
        "- `audit.json`: native target/order checks, split/alias checks, historical exposure checks and unchanged external tests.",
        "- `recipe_source/`: frozen converters, audit scripts, registry and wrapper; `release.json`: package hashes.",
        "- Two new-source When2Call training rows share request wording with older Hermes training cases. "
        "Their complete tools/context inputs differ; the matched historical IDs are explicitly recorded in the audit. "
        "Neither enters validation or calibration.",
        "- A default four-decision parent cap applies. CodeIO has one I/O per program, distill one decision per parent, "
        "and training GSM one presentation per problem. SWE repository ownership is preserved, "
        "with one selected variant per question instead of four rows per entire repository.",
        "",
        "Training tokens: median {p50:,}, p95 {p95:,}, maximum {maximum:,}; no input truncation. "
        "Model inputs contain only state, question and supplied options; targets, rationales, "
        "reference calls and source metadata stay outside the prompt.".format(
            **stats["train"]["tokens"]
        ),
        "",
        "## Preserved tests and excluded sources",
        "",
        "`test.jsonl.gz` and `ood.jsonl.gz` are byte-identical v0.2 reference sets, already measured historically. "
        "The nine external tracks / 13,499 requests in `datasets/evaluation-panel` remain unchanged and are not "
        "used as validation. Full native benchmark holdouts, not just that sampled panel, are protected.",
        "",
        "PrimeVul, Prometheus GPT-4 rubric data, vision and audio are excluded. CLINC/Banking are omitted from "
        "this view because their requests occur in the protected ToolRet benchmark. HoVer lacks supplied evidence; "
        "ContractNLI is deferred rather than extracting answer-bearing spans from long contracts. "
        "Excluding data from this release does not undo exposure in an earlier checkpoint.",
        "",
        "Public sources may occur in backbone or competitor training. Exact/group and approximate MCQ screening "
        "do not prove semantic or paraphrase disjointness. Native synthetic preference/call labels and "
        "source-provided CodeIO execution outputs are not independently rejudged here.",
        "",
        "## Reproduce",
        "",
        "Prepare the standalone sources first, then run from the repository:",
        "",
        "```bash",
        "bash scripts/build_groundbeef_v03.sh",
        "```",
        "",
        "The wrapper uses the existing Docker Python runtime on CPU. Re-running it verifies the frozen release. "
        "Dataset files and raw downloads stay outside Git. The v0.2 trainer intentionally remains version pinned; "
        "a v0.3 training entrypoint is the next task.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    paths = [
        root / "manifest.json",
        root / "audit.json",
        root / "REPORT.md",
        root / "statistics.json",
        root / "sources.json",
    ] + list(snapshot.iterdir())
    release = {
        "dataset": "groundbeef",
        "version": "0.3",
        "verified": True,
        "sha256": {
            str(path.relative_to(root)): file_hash(path) for path in sorted(paths)
        },
        "data_files_sha256": {
            entry["file"]: entry["sha256"] for entry in manifest["files"]
        },
    }
    (root / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    print("Groundbeef report and frozen recipe sources saved", flush=True)


if __name__ == "__main__":
    main()
