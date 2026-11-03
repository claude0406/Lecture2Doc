"""Run pinned upstream functions, not their whole CLI/model stacks.

Fetched sources stay in research/upstream, excluded from the application package.
--fetch restores these exact sources (network needed once).
"""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import urllib.request
import os

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import write

SOURCES = {
    "hwanyong/analysis-video": ("c0fc97080d3febd35f938e39a26fac25e31ef35f", ["LICENSE", "packages/core/src/analysis_video/detect/events.py", "packages/core/src/analysis_video/detect/signals.py", "packages/core/src/analysis_video/align.py"]),
    "eric-lf02/lecture-capture": ("9b8b4004888a0d6812f3e39608e45eb9db7a8642", ["LICENSE", "lecture_capture.ipynb"]),
    "drpwchen/lecture-to-notes": ("cc06141939ce56def13a15410d8d07e861028d94", ["LICENSE", "scripts/extract_slides.py"]),
    "kdavila/lecturemath": ("245d58a20393166addacb9ece75326ef3455b16f", ["LICENSE", "ACCESS2021_release/AccessMath/preprocessing/content/keyframe_extractor.py"]),
}
HERE = Path(__file__).parent


def functions(path, env):
    tree = ast.parse(path.read_text())
    # Keep original function bodies and literal module constants; omit package imports.
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) or
            (isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), env)
    return env


def decoded(video):
    with av.open(str(video)) as c:
        for f in c.decode(video=0):
            yield float(f.time), cv2.resize(f.to_ndarray(format="gray"), (320, 180)).astype("float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--inputs", type=Path, default=Path("experiments/validated/inputs"))
    args = parser.parse_args()
    if args.fetch:
        for repo, (rev, paths) in SOURCES.items():
            for path in paths:
                dest = HERE / "upstream" / repo.split("/")[1] / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{repo}/{rev}/{path}", dest)
    base = HERE / "upstream/analysis-video/packages/core/src/analysis_video"
    media = types.SimpleNamespace(get_fps=lambda _: 6., decode_gray_frames=decoded)
    overlay = types.SimpleNamespace(crop=lambda a, band: a)
    signals = functions(base / "detect/signals.py", dict(np=np, Path=Path, media=media, overlay=overlay))
    events = functions(base / "detect/events.py", dict(np=np))
    align = functions(base / "align.py", dict(bisect_right=__import__('bisect').bisect_right))
    capture = json.loads((HERE / "upstream/lecture-capture/lecture_capture.ipynb").read_text())
    from skimage.metrics import structural_similarity
    env = dict(np=np, cv2=cv2, ssim=structural_similarity)
    for cell in capture["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = ''.join(cell["source"])
        if 'def frames_are_different' in code:
            tree = ast.parse(code)
            exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), 'lecture_capture.ipynb', 'exec'), env)
    slides = functions(HERE / "upstream/lecture-to-notes/scripts/extract_slides.py",
                       dict(os=os, sys=sys, subprocess=subprocess, fmt_hms=lambda s: f"{s:.2f}"))
    output = {}
    for video in sorted(args.inputs.glob("*.mp4")):
        measured = signals["measure"](video, (0, 1), anchor_threshold=.02,
                                       rate_threshold=.0015, cut_area_threshold=.002)
        found = events["find"](measured, anchor_threshold=.02, rate_threshold=.0015, cut_area_threshold=.002)
        samples = list(decoded(video))
        # Original capture policy, using its default 20s cadence: for a 12s clip only
        # the initial observation is scanned. Also measure the original SSIM decision.
        a, b = samples[0][1].astype('uint8'), samples[-1][1].astype('uint8')
        diff, score, fraction = env['frames_are_different'](a, b, .93, .18)
        dest = HERE / "runs" / video.stem
        dest.mkdir(parents=True, exist_ok=True)
        extracted = slides['extract_frames'](str(video), str(dest), 15, 60)
        unique = slides['dedup_frames'](extracted, str(dest), 15)
        output[video.stem] = dict(analysis_video_event_count=len(found),
                                 analysis_video_event_times=[e['time'] for e in found],
                                 lecture_capture_default_scan_times=[t for t, _ in samples if abs(t%20)<1e-5],
                                 lecture_capture_first_last_ssim=float(score),
                                 lecture_capture_first_last_change_fraction=float(fraction),
                                 lecture_capture_first_last_is_change=bool(diff),
                                 lecture_to_notes_extracted=len(extracted), lecture_to_notes_unique=len(unique))
    write(HERE / "benchmark.json", dict(scope="Original selected functions with IO adapters; not complete upstream applications", results=output,
                                      sources={repo: rev for repo, (rev, _) in SOURCES.items()}))


if __name__ == '__main__':
    main()
