"""Sample immutable OpenScience files across their byte extent, not topic prefixes."""

import argparse
import gzip
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from groundbeef_v05_fetch import SOURCES, url
from v05_job import Job, run, save, sha

FILES = {"OS-Q2.5-32B-10.jsonl": 64, "OS-Q3-235B-4.jsonl": 16}
WINDOW_BYTES = 8 << 20


def get(address, start, end):
    req = urllib.request.Request(
        address + f"?groundbeef-range={start}-{end}",
        headers={"Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        if response.status != 206 or not response.headers.get(
            "Content-Range", ""
        ).startswith(f"bytes {start}-"):
            raise ValueError("Source must honor byte ranges")
        return response.read(end - start + 1), response.headers["Content-Range"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    s = SOURCES["openscience"]
    cfg = {
        "source": s,
        "files": FILES,
        "window_bytes": WINDOW_BYTES,
        "code": {
            p.name: sha(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("v05_job.py"),
                Path(__file__).with_name("groundbeef_v05_fetch.py"),
            ]
        },
    }
    job = Job(a.output, "broaden", cfg, a.resume)

    def work():
        entries = []
        for filename, count in FILES.items():
            job.check()
            address = url(s, filename)
            _, header = get(address, 0, 0)
            size = int(header.split("/")[-1])
            root = (
                a.output / "raw" / "openscience-broad" / filename.removesuffix(".jsonl")
            )
            root.mkdir(parents=True, exist_ok=True)
            requests = [
                (i, min(size - WINDOW_BYTES, (size * (2 * i + 1)) // (2 * count)))
                for i in range(count)
            ]

            def fetch(item):
                i, start = item
                name = f"window-{i:03d}.jsonl.gz"
                receipt = root / (name + ".json")
                if receipt.exists():
                    m = json.loads(receipt.read_text())
                    if sha(root / name) != m["sha256"]:
                        raise ValueError("Corrupt committed broad window")
                    return m
                data, content_range = get(
                    address, start, min(size - 1, start + WINDOW_BYTES - 1)
                )
                # Range begins and ends inside arbitrary records; keep complete lines only.
                begin = data.find(b"\n") + 1
                end = data.rfind(b"\n")
                rows = []
                for raw in data[begin:end].splitlines():
                    rows.append(json.loads(raw))
                tmp = root / (name + ".tmp")
                with gzip.open(tmp, "wt") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                tmp.replace(root / name)
                m = {
                    **s,
                    "source_file": filename,
                    "file": name,
                    "start": start,
                    "complete_start": start + begin,
                    "content_range": content_range,
                    "rows": len(rows),
                    "sha256": sha(root / name),
                }
                save(receipt, m)
                return m

            with ThreadPoolExecutor(max_workers=4) as pool:
                for m in pool.map(fetch, requests):
                    entries.append(m)
                    job.update(
                        source=filename,
                        windows=len(entries),
                        rows=sum(x["rows"] for x in entries),
                    )
                    print(
                        f"{filename}: committed {m['file']}, {m['rows']} rows",
                        flush=True,
                    )
                    job.check()
        save(a.output / "broad-source-lock.json", {**cfg, "windows": entries})
        job.complete(rows=sum(x["rows"] for x in entries), windows=len(entries))

    return run(job, work)


if __name__ == "__main__":
    raise SystemExit(main())
