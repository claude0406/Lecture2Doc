"""Resumable, bounded parallel HTTP-range download of an official AccessMath video."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess

DEFAULT_URL = "https://www.cs.rit.edu/~accessmath/am_videos/lectures/Lecture_01/AccessMath_lecture_01_part_1.mp4"


def download(url, output, workers=4):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = subprocess.check_output(["curl", "-sS", "-L", "--fail", "--max-time", "60", "-I", url], text=True)
    lengths = [int(line.split(":", 1)[1]) for line in headers.splitlines() if line.lower().startswith("content-length:")]
    if not lengths:
        raise RuntimeError("Server did not supply Content-Length")
    size = lengths[-1]
    if output.is_file() and output.stat().st_size == size:
        print(f"Already downloaded: {output}", flush=True)
    else:
        parts = output.with_suffix(output.suffix+".parts")
        parts.mkdir(exist_ok=True)
        chunk = 16*1024**2
        ranges = [(i, start, min(size-1, start+chunk-1)) for i, start in enumerate(range(0, size, chunk))]
        def fetch(item):
            i, start, end = item
            path = parts / f"{i:05d}"
            if not path.is_file() or path.stat().st_size != end-start+1:
                subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "3", "--max-time", "300",
                                "-r", f"{start}-{end}", "-o", str(path), url], check=True)
            if path.stat().st_size != end-start+1:
                raise RuntimeError(f"Range length mismatch: {path}")
            return path
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = [pool.submit(fetch, item) for item in ranges]
            for n, task in enumerate(as_completed(tasks), 1):
                task.result()
                print(f"Downloaded {n}/{len(tasks)} chunks", flush=True)
        temp = output.with_suffix(output.suffix+".tmp")
        with temp.open("wb") as dest:
            for i, start, end in ranges:
                with (parts / f"{i:05d}").open("rb") as src:
                    while data := src.read(1024**2):
                        dest.write(data)
        temp.replace(output)
    h = hashlib.sha256()
    with output.open("rb") as f:
        while data := f.read(1024**2):
            h.update(data)
    output.with_suffix(output.suffix+".provenance.json").write_text(json.dumps(dict(
        url=url, bytes=size, sha256=h.hexdigest(), response_headers=headers), indent=2))
    print(f"Complete: {output} ({size} bytes)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", type=Path, default=Path("datasets/accessmath/AccessMath_lecture_01_part_1.mp4"))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    download(a.url, a.out, a.workers)
