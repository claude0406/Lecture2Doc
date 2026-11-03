"""Evaluate a frozen Lecture2Doc run against AccessMath keyframe annotations.

This is a diagnostic pixel/CC-coverage evaluation, NOT the official unique-CC
summary metric. All thresholds and temporal selection policies are recorded.
No GT masks are supplied to the application pipeline.
"""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import digest, read, write


def annotations(folder):
    r = ET.parse(folder / "segments.xml").getroot()
    keys = []
    for node in r.findall("./VideoKeyFrames/VideoKeyFrame"):
        idx = int(node.findtext("Index"))
        image = cv2.imread(str(folder / "keyframes" / f"{idx}.png"))
        binary = cv2.imread(str(folder / "binary" / f"{idx}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or binary is None:
            raise ValueError(f"Missing GT image {idx}")
        valid = np.ones(binary.shape, bool)
        h, w = binary.shape
        for obj in node.findall("./VideoObjects/VideoObject"):
            # These are annotated non-content/background regions, not our masks.
            x, y, rw, rh = (float(obj.findtext(k)) for k in ("X", "Y", "W", "H"))
            valid[max(0, int(y)):min(h, int(y+rh)), max(0, int(x)):min(w, int(x+rw))] = False
        keys.append(dict(index=idx, time=float(node.findtext("AbsTime"))/1000,
                         image=image, ink=(binary < 128) & valid, valid=valid))
    segments = [(int(n.findtext("Start")), int(n.findtext("End"))) for n in r.findall("./VideoSegments/VideoSegment")]
    # Convert annotated segment frame boundaries via the keyframe index/time relation.
    # AbsTime is authoritative for GT keyframe timestamps. Frame boundaries are only
    # used for diagnostic overlap candidate pools, not for choosing the at-time output.
    return keys, segments


def predicted_ink(image, shape, threshold):
    image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return contrast > threshold


def score(prediction, truth, valid, radius=3):
    p, g = prediction & valid, truth & valid
    kernel = np.ones((radius*2+1, radius*2+1), np.uint8)
    p_near = cv2.dilate(p.astype(np.uint8), kernel).astype(bool)
    g_near = cv2.dilate(g.astype(np.uint8), kernel).astype(bool)
    recall = float(p_near[g].mean()) if g.any() else None
    precision = float(g_near[p].mean()) if p.any() else 0.
    f1 = 2*recall*precision/(recall+precision) if recall and recall+precision else 0.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    hits = np.bincount(labels[p_near & g], minlength=n)[1:]
    eligible = areas >= 10
    component_recall = float((hits[eligible]/areas[eligible] >= .8).mean()) if eligible.any() else None
    return dict(precision=precision, recall=recall, f1=f1, gt_pixels=int(g.sum()),
                predicted_pixels=int(p.sum()), gt_components_area_ge_10=int(eligible.sum()),
                component_coverage_at_80pct=component_recall)


def overlay(image, prediction, truth, valid):
    im = cv2.resize(image, (truth.shape[1], truth.shape[0])).copy()
    k = np.ones((7, 7), np.uint8)
    pn = cv2.dilate((prediction & valid).astype(np.uint8), k).astype(bool)
    gn = cv2.dilate(truth.astype(np.uint8), k).astype(bool)
    im[truth & pn] = (70, 170, 50)       # matched GT ink: green
    im[truth & ~pn] = (40, 40, 235)      # missed GT ink: red
    im[prediction & valid & ~gn] = (220, 60, 210)  # extra detected ink: magenta
    return im


def evaluate(record, gt, output):
    output.mkdir(parents=True, exist_ok=True)
    m = read(record / "manifest.json")
    keys, gt_segments = annotations(gt)
    by_id = {f["id"]: f for f in m["source_frames"]}
    results = {r["board_state_id"]: r for r in m["reconstructed_boards"]}
    valid_keys = [k for k in keys if k["time"] < m["source"]["duration"]]
    rows = []
    for k in valid_keys:
        state = next((s for s in m["board_states"] if s["start"] <= k["time"] < s["end"]), None)
        if state is None:
            raise ValueError(f"No state covering annotated timestamp {k['time']}")
        sf = by_id[state["selected_frame_id"]]
        nearest = min(m["source_frames"], key=lambda f: abs(f["time"]-k["time"]))
        reconstruction = results[state["id"]]
        selected_image = cv2.imread(str(record / sf["image"]))
        restored_image = cv2.imread(str(record / reconstruction["image"]))
        nearest_image = cv2.imread(str(record / nearest["image"]))
        # Keep full-frame coordinates for cropped composite results.
        if restored_image.shape != selected_image.shape:
            full = selected_image.copy()
            H, W = full.shape[:2]
            x, y, w, h = m["config"]["roi"]
            box = (int(x*W), int(y*H), int((x+w)*W), int((y+h)*H))
            X, Y, X2, Y2 = box
            full[Y:Y2, X:X2] = cv2.resize(restored_image, (X2-X, Y2-Y))
            restored_image = full
        row = dict(gt_index=k["index"], gt_time=k["time"], state_id=state["id"],
                   selected_frame_id=sf["id"], selected_time=sf["time"],
                   selected_time_delta=sf["time"]-k["time"], nearest_source_time=nearest["time"],
                   reconstruction_status=reconstruction["status"],
                   warnings=reconstruction["quality_flags"], metrics={})
        for name, image in (("gt_raw_calibration", k["image"]), ("nearest_source", nearest_image),
                            ("selected_A", selected_image), ("result_B", restored_image)):
            row["metrics"][name] = {}
            for threshold in (10, 20, 30):
                pred = predicted_ink(image, k["ink"].shape, threshold)
                row["metrics"][name][str(threshold)] = score(pred, k["ink"], k["valid"])
        gray_gt = cv2.cvtColor(cv2.resize(k["image"], (960, 540)), cv2.COLOR_BGR2GRAY).astype("float32")
        gray_video = cv2.cvtColor(cv2.resize(nearest_image, (960, 540)), cv2.COLOR_BGR2GRAY).astype("float32")
        displacement, response = cv2.phaseCorrelate(gray_gt, gray_video)
        row["alignment_diagnostic_half_resolution"] = dict(dx=displacement[0], dy=displacement[1], response=response)
        panels = [k["image"], nearest_image, selected_image, restored_image,
                  cv2.cvtColor((~k["ink"]).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR),
                  overlay(restored_image, predicted_ink(restored_image, k["ink"].shape, 10), k["ink"], k["valid"])]
        labels = ["GT source", "Nearest observation", "A: selected original", "B: reconstruction/original", "GT handwriting mask", "B vs GT: green=hit red=miss purple=extra"]
        sheet = Image.new("RGB", (1440, 600), "#f5f7f2")
        draw = ImageDraw.Draw(sheet)
        for j, (im, label) in enumerate(zip(panels, labels)):
            x, y = (j % 3)*480, (j//3)*300
            sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(im, (480, 270)), cv2.COLOR_BGR2RGB)), (x, y+24))
            draw.text((x+5, y+5), label, fill="#19312e")
        name = f"gt-{k['index']}.jpg"
        sheet.save(output / name, quality=93)
        # Full resolution A/B and GT masks for manual inspection, not only thumbnails.
        for tag, image in (("A", selected_image), ("B", restored_image), ("GT", k["image"])):
            cv2.imwrite(str(output / f"gt-{k['index']}-{tag}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 97])
        row["comparison"] = name
        rows.append(row)
        print(f"GT {k['index']} @ {k['time']:.3f}s: A={row['metrics']['selected_A']['10']['recall']:.3f}, B={row['metrics']['result_B']['10']['recall']:.3f}", flush=True)
    aggregate = {}
    for threshold in ("10", "20", "30"):
        aggregate[threshold] = {}
        for label in ("gt_raw_calibration", "nearest_source", "selected_A", "result_B"):
            stats = [r["metrics"][label][threshold] for r in rows]
            aggregate[threshold][label] = {metric: float(np.mean([s[metric] for s in stats])) if stats else None
                                          for metric in ("precision", "recall", "f1", "component_coverage_at_80pct")}
    covered_gt_segments = [s for s in gt_segments if any(s[0] <= k["index"] <= s[1] for k in valid_keys)]
    seconds_per_frame = 1.0 / m["source"]["fps"]
    if any(abs(k["time"]-k["index"]*seconds_per_frame) > .001 for k in valid_keys):
        raise ValueError("First-part frame/time mapping does not match annotated AbsTime")
    gt_boundaries = sorted({s[1]*seconds_per_frame for s in gt_segments
                            if 0 < s[1]*seconds_per_frame < m["source"]["duration"]})
    boundaries = [s["start"] for s in m["board_states"]][1:]
    boundary_diagnostics = []
    for t in gt_boundaries:
        containing = next((s for s in m["board_states"] if s["start"] <= t < s["end"]), None)
        boundary_diagnostics.append(dict(gt_boundary_time=t,
            nearest_app_boundary_distance=min((abs(t-b) for b in boundaries), default=None),
            containing_state_id=containing["id"] if containing else None,
            containing_state_interval=[containing["start"],containing["end"]] if containing else None))
    protocol = dict(dataset="AccessMath ICDAR 2017 / AccessMath2015", lecture="lecture_01", split="training",
                    scope="Official first video file only, not all lecture parts or full dataset",
                    video_sha256=m["source"]["sha256"], annotation_xml_sha256=digest(gt / "segments.xml"),
                    source=m["source"], application_config=m["config"],
                    gt_keyframes_total=len(keys), gt_keyframes_in_scope=len(valid_keys),
                    gt_segments_with_keyframe_in_scope=covered_gt_segments,
                    prediction_policy="State containing each GT timestamp; selected original and corresponding B output. No oracle/best-GT selection.",
                    binarization="whiteboard grayscale black-hat elliptical 31x31; thresholds 10,20,30; no person mask",
                    scoring="diagnostic tolerant pixel precision/recall/F1 at native GT size; 3px radius; CC coverage >=80%, GT components >=10px",
                    excluded_regions="only non-content rectangles from official exported keyframe annotations",
                    official_unique_cc_metric=False, uses_gt_in_application=False,
                    counts=dict(source_frames=len(m["source_frames"]), states=len(m["board_states"]),
                                board_type_predictions=dict(Counter(x["board"] for x in m["analysis_metrics"])),
                                status=dict(Counter(r["status"] for r in m["reconstructed_boards"])),
                                boundary_reasons=dict(Counter(s["boundary_reason"] for s in m["board_states"])),
                                quality_flags=dict(Counter(f for r in m["reconstructed_boards"] for f in r["quality_flags"]))),
                    elapsed_seconds=m["elapsed_seconds"], aggregate_macro=aggregate, keyframes=rows,
                    gt_segment_boundary_diagnostics=boundary_diagnostics,
                    boundary_time_conversion="First-part FPS, checked against every in-scope annotated AbsTime within 1ms")
    write(output / "report.json", protocol)
    document = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>AccessMath 평가</title>
<style>body{font:15px/1.65 system-ui;margin:35px;background:#f5f7f2;color:#19312e}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccd5cc;padding:8px}section{margin:40px 0}a{color:#176b51}</style>
<h1>AccessMath Lecture 01 · 첫 번째 원본 파일 전체</h1><p>기존 알고리즘·기본 설정 고정. 학습 분할의 진단 실행이며 전체 데이터셋 또는 공식 unique-CC 벤치마크 점수가 아닙니다.</p>
<p>A/B는 각 정답 시각이 속한 상태의 결과를 선택합니다. 정답을 보고 가장 좋은 프레임을 고르지 않습니다. 세 가지 이진화 임계값과 GT 원본에 같은 이진화를 적용한 보정 결과를 report.json에 제공합니다.</p><a href="report.json">전체 프로토콜·결과 JSON</a>'''
    document += f'<p>원본 {m["source"]["width"]}×{m["source"]["height"]}, {m["source"]["duration"]:.2f}초 · 관측 {len(m["source_frames"])}개 · 상태 {len(m["board_states"])}개</p>'
    document += '<table><tr><th>GT 시각</th><th>선택 시각</th><th>B 상태</th><th>A recall / precision</th><th>B recall / precision</th><th>GT 자체 recall</th></tr>'
    for row in rows:
        a,b,g = (row['metrics'][tag]['10'] for tag in ('selected_A','result_B','gt_raw_calibration'))
        document += f'<tr><td>{row["gt_time"]:.2f}</td><td>{row["selected_time"]:.2f}</td><td>{row["reconstruction_status"]}</td><td>{a["recall"]:.3f} / {a["precision"]:.3f}</td><td>{b["recall"]:.3f} / {b["precision"]:.3f}</td><td>{g["recall"]:.3f}</td></tr>'
    document += '</table><p>표는 임계값 10. 붉은 정답 필기 누락과 보라색 추가 검출은 이진화 오차를 포함합니다. 원본 GT 자체의 회수율과 함께 해석해야 합니다.</p>'
    for row in rows:
        document += f'<section><h2>GT {row["gt_index"]} · {row["gt_time"]:.3f}s</h2><img src="{row["comparison"]}" loading="lazy"><p><a href="gt-{row["gt_index"]}-A.jpg">원본 A 확대</a> · <a href="gt-{row["gt_index"]}-B.jpg">결과 B 확대</a> · <a href="gt-{row["gt_index"]}-GT.jpg">정답 원본 확대</a></p></section>'
    (output / "index.html").write_text(document+'</html>', encoding="utf-8")
    return protocol


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record", type=Path, default=Path("experiments/accessmath/lecture01_part1"))
    p.add_argument("--gt", type=Path, default=Path("datasets/accessmath/annotations/data/annotations/AccessMath2015_lecture_01"))
    p.add_argument("--out", type=Path, default=Path("experiments/accessmath/evaluation"))
    a = p.parse_args()
    evaluate(a.record, a.gt, a.out)
