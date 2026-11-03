"""Reproduce the declared AccessMath first-file baseline and board-type comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lecture2doc.pipeline import process
from lecture2doc.storage import digest, read, write
from lecture2doc.vision import Config
from accessmath_evaluate import evaluate
from accessmath_fetch import download, DEFAULT_URL

ANNOTATIONS_URL = "https://www.cs.rit.edu/~rlaz/files/tangent-v/AccessMath_ICDAR_2017_data.zip"


def extract_annotations(archive, output):
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            relative = Path(info.filename).relative_to("AccessMath_ICDAR_2017_data")
            if ("AccessMath2015_lecture_01/" not in info.filename
                and not info.filename.endswith(("db_AccessMath2015.xml", "gpl-3.0.txt", "lectures/info.txt", "AccessMath2015_lecture_01.xml"))):
                continue
            dest = output / relative
            if not dest.resolve().is_relative_to(output.resolve()):
                raise ValueError("Unsafe archive path")
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(info))


def hardlink_or_copy(src, dst):
    if Path(dst).exists():
        return str(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


def run_if_needed(video, folder, config):
    manifest = folder / "manifest.json"
    if manifest.is_file():
        previous = read(manifest)
        if (previous["source"]["sha256"] == digest(video)
            and json.dumps(previous["config"], sort_keys=True) == json.dumps(config.json(), sort_keys=True)):
            print("Using completed run; preserving its original elapsed time", flush=True)
            return previous
    return process(video, folder, config, progress=lambda s: print(s, flush=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("datasets/accessmath"))
    p.add_argument("--out", type=Path, default=Path("experiments/accessmath"))
    p.add_argument("--download", action="store_true", help="Fetch official video and annotation zip if needed")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    video = a.data / "AccessMath_lecture_01_part_1.mp4"
    archive = a.data / "AccessMath_ICDAR_2017_data.zip"
    if a.download:
        download(DEFAULT_URL, video)
        download(ANNOTATIONS_URL, archive)
    if not video.is_file() or not archive.is_file():
        p.error("Missing input; run with --download to fetch official files")
    extract_annotations(archive, a.data / "annotations")
    gt = a.data / "annotations/data/annotations/AccessMath2015_lecture_01"
    code_hashes = {str(f): digest(f) for f in sorted(Path("lecture2doc").glob("*.py"))}
    frozen = a.out / "baseline-code.json"
    if frozen.exists() and read(frozen) != code_hashes:
        raise ValueError("Application code differs from frozen baseline; use a new output directory")
    write(frozen, code_hashes)
    base = a.out / "lecture01_part1"
    print("=== frozen default auto baseline ===", flush=True)
    baseline = run_if_needed(video, base, Config())
    evaluate(base, gt, a.out / "evaluation")
    # Reuse immutable original PNGs via hardlinks (fallback: copy across filesystems).
    # These keep identical relative paths under each record, so the viewer's path
    # containment checks continue to work; no symlink outside a record is needed.
    white = a.out / "lecture01_part1_whiteboard"
    white.mkdir(parents=True, exist_ok=True)
    hardlink_or_copy(base / baseline["source"]["path"], white / baseline["source"]["path"])
    shutil.copy2(base / "source.json", white / "source.json")
    shutil.copytree(base / "cache/decode", white / "cache/decode", dirs_exist_ok=True,
                    copy_function=hardlink_or_copy)
    print("=== same algorithm, board=whiteboard only ===", flush=True)
    configured = run_if_needed(white / baseline["source"]["path"], white, Config(board="whiteboard"))
    evaluate(white, gt, a.out / "evaluation-whiteboard")
    write(a.out / "dataset-provenance.json", dict(video_url=DEFAULT_URL, annotations_url=ANNOTATIONS_URL,
          video_sha256=digest(video), annotations_sha256=digest(archive),
          baseline_config=baseline["config"], configured_config=configured["config"],
          declared_scope="lecture_01 training split, official part 1 entire file; 3 of 7 annotated keyframes",
          data_not_used_for_training=True, runtime_algorithm_modified=False,
          speech_evaluation="not run; this experiment evaluates board extraction and reconstruction"))
    (a.out / "index.html").write_text('''<!doctype html><meta charset="utf-8"><title>AccessMath runs</title>
<style>body{font:17px/1.8 system-ui;max-width:850px;margin:60px auto;padding:20px;background:#f5f7f2;color:#19312e}a{color:#176b51}</style>
<h1>AccessMath · 고정 알고리즘 실행 비교</h1><p>Lecture 01 첫 파일 전체, 학습 분할. 전체 데이터셋 테스트 점수가 아닙니다.</p>
<ul><li><a href="evaluation/index.html">기본 자동 칠판 종류 판별 결과</a></li>
<li><a href="evaluation-whiteboard/index.html">화이트보드 명시 결과</a></li>
<li><a href="dataset-provenance.json">원본 출처·해시·설정</a></li></ul>
<p>서버 뷰어: lecture2doc serve --record experiments/accessmath/lecture01_part1_whiteboard</p>''', encoding="utf-8")


if __name__ == "__main__":
    main()
