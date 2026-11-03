"""Small known-answer checks for the external-dataset evaluation, not the app."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from accessmath_evaluate import annotations, score


def test_recall_and_precision_are_not_interchanged():
    truth = np.zeros((20, 20), bool)
    truth[5:9, 5:9] = True
    pred = truth.copy()
    pred[7:9, 5:9] = False
    metrics = score(pred, truth, np.ones_like(truth), radius=0)
    assert metrics["recall"] == .5
    assert metrics["precision"] == 1
    assert metrics["f1"] == pytest.approx(2/3)
    assert metrics["component_coverage_at_80pct"] == 0


def test_tolerance_and_excluded_regions():
    truth = np.zeros((20, 20), bool)
    truth[4:8, 4:8] = True
    pred = np.roll(truth, 1, axis=1)
    pred[15:19, 15:19] = True
    valid = np.ones_like(truth)
    valid[15:19, 15:19] = False
    metrics = score(pred, truth, valid, radius=1)
    assert metrics["recall"] == metrics["precision"] == 1
    assert metrics["predicted_pixels"] == 16


def test_gt_milliseconds_and_black_ink_convention(tmp_path):
    for name in ("keyframes", "binary"):
        (tmp_path/name).mkdir()
    binary = np.full((20, 20), 255, np.uint8)
    binary[5:9, 5:9] = 0
    cv2.imwrite(str(tmp_path/'binary/123.png'), binary)
    cv2.imwrite(str(tmp_path/'keyframes/123.png'), cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    (tmp_path/'segments.xml').write_text('<Annotations><VideoSegments><VideoSegment><Start>0</Start><End>200</End></VideoSegment></VideoSegments><VideoKeyFrames><VideoKeyFrame><Index>123</Index><AbsTime>4100.5</AbsTime><VideoObjects/></VideoKeyFrame></VideoKeyFrames></Annotations>')
    keys, segments = annotations(tmp_path)
    assert keys[0]['time'] == 4.1005
    assert keys[0]['ink'].sum() == 16
    assert segments == [(0, 200)]
