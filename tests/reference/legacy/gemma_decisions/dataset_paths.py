"""Resolve original Open-Jev or our versioned mixture without falsifying provenance."""
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "validation", "calibration", "test", "ood")
OPEN_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def resolve_dataset(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dataset") not in {"KneeJerk decisions", "groundbeef"} or manifest.get("version") not in {"0.1", "0.2"}:
            raise ValueError("Unsupported decision dataset manifest")
        expected = {Path(item["file"]).name: item for item in manifest["files"]
                    if Path(item["file"]).name in {s + ".jsonl.gz" for s in SPLITS}}
        paths = {split: directory / (split + ".jsonl.gz") for split in SPLITS}
        for split, path in paths.items():
            if path.name not in expected or sha256(path) != expected[path.name]["sha256"]:
                raise ValueError(f"Dataset split differs from its manifest: {split}")
        prefix = ("groundbeef-v" if manifest["dataset"] == "groundbeef" else "kneejerk-decisions-v") + manifest["version"] + ":"
        return paths, prefix + sha256(manifest_path)
    if (directory / "revision.txt").read_text().strip() != OPEN_REVISION:
        raise ValueError("Open-Jev revision mismatch")
    paths = {split: directory / "raw/release-v2-redistributable" / (split + ".jsonl.gz") for split in SPLITS}
    return paths, OPEN_REVISION
