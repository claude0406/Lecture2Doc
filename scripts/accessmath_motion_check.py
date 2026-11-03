"""Static-region motion control for AccessMath; never used by the application."""
import argparse
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lecture2doc.storage import read, write
from lecture2doc.vision import Config, small


def check(record, output):
    manifest = read(record / "manifest.json")
    frames = manifest["source_frames"]
    config = Config(**manifest["config"])
    rows = []
    for i, metric in enumerate(manifest["analysis_metrics"]):
        if metric["event"] != "camera_motion" or i == 0:
            continue
        images = []
        for frame in (frames[i-1], frames[i]):
            image = small(cv2.imread(str(record / frame["image"])), config)
            images.append(cv2.cvtColor(image[:image.shape[0]//4], cv2.COLOR_BGR2GRAY).astype("float32"))
        shift, response = cv2.phaseCorrelate(*images)
        rows.append(dict(time=metric["time"], full_frame_shift=metric["shift"],
                         static_top_quarter_shift=[*shift, response]))
    write(output, dict(method="Phase correlation restricted to static top quarter; diagnostic only, not a hand-annotated motion ground truth or application input", events=rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record", type=Path, default=Path("experiments/accessmath/lecture01_part1"))
    p.add_argument("--out", type=Path, default=Path("experiments/accessmath/motion-diagnostic.json"))
    a = p.parse_args()
    check(a.record, a.out)
