"""Atomic manifests; immutable content-addressed stage directories."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

VERSION = "2026-11-01.3"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key(value) -> str:
    return hashlib.sha256(json.dumps([VERSION, value], sort_keys=True).encode()).hexdigest()[:20]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def stage(root, name, fingerprint, compute, force=False):
    suffix = f"-rerun-{time.time_ns()}" if force else ""
    folder = Path(root) / "cache" / name / (key(fingerprint)+suffix)
    manifest = folder / "stage.json"
    if manifest.exists():
        return read(manifest)
    folder.mkdir(parents=True, exist_ok=True)
    result = compute(folder)
    write(manifest, result)
    return result
