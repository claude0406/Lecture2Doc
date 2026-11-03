"""Export exact legacy ranking and masks without consulting ground-truth images."""
import argparse
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, default=Path("experiments/accessmath"))
    p.add_argument("--out", type=Path, default=Path("experiments/accessmath-improvement/diagnosis"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("legacy_vision", a.out.parent/"baseline-source/vision.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    for variant in ("lecture01_part1", "lecture01_part1_whiteboard"):
        root = a.baseline/variant
        m = read(root/"manifest.json")
        state = next(s for s in m["board_states"] if s["start"] <= 607.674 < s["end"])
        candidates = [dict(x, legacy_sort_key=[round(x["occlusion"], 3), -x["sharpness"], -x["time"]])
                      for x in m["analysis_metrics"] if x["id"] in state["source_frame_ids"]]
        candidates.sort(key=lambda x: x["legacy_sort_key"])
        for rank, c in enumerate(candidates, 1):
            c["rank"] = rank
        write(a.out/(variant+".json"), dict(state=state, candidates=candidates,
              selection_rule="lexicographic(round(global_foreground_fraction,3), -whole_image_Laplacian_variance, -time)",
              foreground_rule="entire connected color-departure component masked if any pixel has distanceTransform >5"))
        chosen = [min(candidates, key=lambda c:abs(c["time"]-607.607)), candidates[0],
                  min(candidates, key=lambda c:abs(c["time"]-663.1625))]
        by_id = {f["id"]:f for f in m["source_frames"]}
        sheet = Image.new("RGB", (1440, 3*300), "#f5f7f2")
        draw = ImageDraw.Draw(sheet)
        for row, c in enumerate(chosen):
            im = cv2.imread(str(root/by_id[c["id"]]["image"]))
            analysis = old.small(im, old.Config(**m["config"]))
            ink, mask, kind = old.features(analysis, m["config"]["board"])
            c["median_gray"] = float(np.median(cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)))
            overlay = analysis.copy()
            overlay[mask] = (.4*overlay[mask]+.6*np.array([30,30,240])).astype(np.uint8)
            binary = cv2.cvtColor(ink.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
            for col, panel in enumerate((analysis, overlay, binary)):
                sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(panel,(480,270)),cv2.COLOR_BGR2RGB)), (480*col,300*row+30))
            draw.text((6,row*300+7), f't={c["time"]:.4f}s rank={c["rank"]} mask={c["occlusion"]:.4f} sharp={c["sharpness"]:.1f} ink={c["ink_pixels"]} median={c["median_gray"]:.1f} {kind}',fill="#19312e")
        sheet.save(a.out/(variant+".jpg"),quality=94)
        write(a.out/(variant+".json"), dict(state=state,candidates=candidates,
              selection_rule="lexicographic(round(global_foreground_fraction,3), -whole_image_Laplacian_variance, -time)"))


if __name__ == "__main__":
    main()
