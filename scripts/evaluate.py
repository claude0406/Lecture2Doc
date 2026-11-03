"""Separate execution results from visual quality; produce inspectable comparisons."""
import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lecture2doc.pipeline import process
from lecture2doc.storage import write
from lecture2doc.vision import Config
from fixtures import generate, render


def compare(root, manifest, name=None):
    frames = {f["id"]: f for f in manifest["source_frames"]}
    rows, quality = [], []
    for s, r in zip(manifest["board_states"], manifest["reconstructed_boards"]):
        first = cv2.imread(str(root / frames[s["source_frame_ids"][0]]["image"]))
        selected = frames[s["selected_frame_id"]]
        a = cv2.imread(str(root / selected["image"]))
        b = cv2.imread(str(root / r["image"]))
        truth = render(name, selected["time"])[1] if name else a
        q = dict(state=s["id"], start=s["start"], end=s["end"], status=r["status"], flags=r["quality_flags"])
        if name:
            # Compare bright chalk pixels only; avoid rewarding board background.
            ink = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY) > 190
            if ink.mean() > .5:  # whiteboard cut: dark ink
                ink = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY) < 80
            for tag, image in (("selected", a), ("reconstruction", b)):
                close = np.max(np.abs(image.astype(float)-truth), axis=2) < 35
                q[tag+"_ink_recall"] = round(float(close[ink].mean()), 4) if ink.any() else 1.0
            labels = {render(name, frames[fid]["time"])[2] for fid in s["source_frame_ids"]}
            q["state_truth_labels"] = sorted(labels)
            q["mixed_truth_states"] = len(labels) > 1
        quality.append(q)
        canvas = Image.new("RGB", (1280, 230), "#f5f7f2")
        d = ImageDraw.Draw(canvas)
        for col, (image, label) in enumerate(zip((first, a, b, truth), ("FIRST OBSERVATION", "A: SELECTED ORIGINAL", "B: RESULT", "GROUND TRUTH" if name else "SELECTED REFERENCE"))):
            rgb = cv2.cvtColor(cv2.resize(image, (320, 180)), cv2.COLOR_BGR2RGB)
            canvas.paste(Image.fromarray(rgb), (col*320, 28))
            d.text((col*320+5, 7), label, fill="#19312e")
        d.text((6, 211), f'{s["id"]} {s["start"]:.2f}-{s["end"]:.2f}s | {r["status"]}', fill="#19312e")
        rows.append(canvas)
    # Paginate to prevent enormous images for long real lectures.
    paths = []
    for page in range(0, len(rows), 12):
        part = rows[page:page+12]
        sheet = Image.new("RGB", (1280, 230*len(part)), "white")
        for i, row in enumerate(part):
            sheet.paste(row, (0, 230*i))
        dest = root / f"comparison-{page//12+1}.jpg"
        sheet.save(dest, quality=90)
        paths.append(dest.name)
    write(root / "quality.json", quality)
    return dict(states=len(rows), comparisons=paths, quality=quality)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("experiments"))
    p.add_argument("--record", type=Path, help="기존 실제 영상 분석 결과의 비교 이미지 생성 (정답 없음)")
    args = p.parse_args()
    if args.record:
        from lecture2doc.storage import read
        result = compare(args.record, read(args.record / "manifest.json"))
        write(args.record / "comparison-summary.json", result)
        return
    videos = generate(args.root / "inputs")
    report = {}
    for name, video in videos.items():
        folder = args.root / name
        print(f"Evaluating {name}", flush=True)
        m = process(video, folder, Config(), subtitles=args.root / "inputs/example.srt")
        report[name] = compare(folder, m, name)
        report[name]["elapsed_seconds"] = m["elapsed_seconds"]
    write(args.root / "report.json", report)
    html = '<!doctype html><meta charset="utf-8"><title>Lecture2Doc quality comparison</title><style>body{font:16px system-ui;margin:30px;background:#f5f7f2}img{max-width:100%}section{margin:40px 0}</style><h1>Execution ≠ preservation quality</h1><p>FIRST OBSERVATION / A: SELECTED / B: RESULT / GROUND TRUTH</p>'
    for name, item in report.items():
        html += f'<section><h2>{name}: {item["states"]} states</h2><a href="{name}/quality.json">Quality measurements</a>'
        html += ''.join(f'<img src="{name}/{path}" loading="lazy">' for path in item["comparisons"])+"</section>"
    (args.root / "index.html").write_text(html)


if __name__ == "__main__":
    main()
