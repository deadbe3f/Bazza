#!/usr/bin/env python3
"""Freeze the supplemental release only after both saved-file audits pass."""

import argparse
import json
from pathlib import Path

from common import REPO, file_hash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=REPO / "build/groundbeef/decider-supplement-v0.1",
    )
    a = ap.parse_args()
    out = a.dataset
    audit = json.loads((out / "audit.json").read_text())
    labels = json.loads((out / "label_audit.json").read_text())
    assert audit["verified"] and labels["verified"]
    assert labels["checked_every_exported_target"] == sum(audit["rows"].values())
    hashes = {
        str(p.relative_to(out)): file_hash(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
        and p.name != "release.json"
        and not p.name.endswith(("-wal", "-shm"))
    }
    release = {
        "dataset": "groundbeef-decider-supplement",
        "version": "0.1",
        "verified": True,
        "rows": audit["rows"],
        "sha256": hashes,
        "label_audit_script_sha256": labels["audit_script_sha256"],
        "packaging_script_sha256": file_hash(__file__),
        "purpose": "Additive local research data; original Groundbeef releases are unchanged.",
    }
    path = out / "release.json"
    if path.exists():
        assert json.loads(path.read_text()) == release, "Frozen release differs"
    else:
        path.write_text(json.dumps(release, indent=2) + "\n")
    print("Verified frozen release:", out, release["rows"])


if __name__ == "__main__":
    main()
