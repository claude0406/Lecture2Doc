"""Model-free fixed-camera board measurements. No generated/inpainted pixels."""
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Config:
    sample_seconds: float = 0.5
    analysis_width: int = 640
    board: str = "auto"
    roi: tuple = (0.0, 0.0, 1.0, 1.0)
    confirm: int = 2
    change_pixels: int = 35
    change_fraction: float = 0.001
    cut_fraction: float = 0.65
    tile_size: int = 32
    reconstruction: bool = True

    def validate(self):
        if not 0.05 <= self.sample_seconds <= 10:
            raise ValueError("sample_seconds 범위는 0.05~10초입니다")
        if not 160 <= self.analysis_width <= 1280 or not 1 <= self.confirm <= 10:
            raise ValueError("분석 해상도 또는 확인 횟수가 범위를 벗어났습니다")
        if self.board not in ("auto", "blackboard", "whiteboard"):
            raise ValueError("board는 auto/blackboard/whiteboard입니다")
        x, y, w, h = self.roi
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1 or y + h > 1:
            raise ValueError("ROI는 정규화된 x,y,w,h이며 영상 안에 있어야 합니다")
        return self

    def json(self):
        return asdict(self)


def crop(image, roi):
    h, w = image.shape[:2]
    x, y, rw, rh = roi
    return image[int(y*h):max(int((y+rh)*h), int(y*h)+1),
                 int(x*w):max(int((x+rw)*w), int(x*w)+1)]


def small(image, config):
    a = crop(image, config.roi)
    width = min(config.analysis_width, a.shape[1])
    return cv2.resize(a, (width, max(1, round(a.shape[0]*width/a.shape[1]))), interpolation=cv2.INTER_AREA)


def features(image, board="auto"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kind = ("whiteboard" if np.median(gray) > 140 else "blackboard") if board == "auto" else board
    # A thick color departure is foreground; thin components can be writing.
    bg = np.median(image.reshape(-1, 3), axis=0)
    departure = (np.max(np.abs(image.astype(float)-bg), axis=2) > 28).astype("uint8")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(departure)
    distance = cv2.distanceTransform(departure, cv2.DIST_L2, 3)
    thick_ids = np.unique(labels[distance > 5])
    foreground = np.isin(labels, thick_ids[thick_ids != 0]).astype("uint8")
    foreground = cv2.dilate(foreground, np.ones((9, 9), np.uint8)).astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    op = cv2.MORPH_BLACKHAT if kind == "whiteboard" else cv2.MORPH_TOPHAT
    contrast = cv2.morphologyEx(gray, op, kernel)
    ink = (contrast > 23) & ~foreground
    return ink, foreground, kind


def cut_score(a, b):
    delta = np.max(np.abs(a.astype(float)-b), axis=2)
    # Diffuse change across the image; a moving lecturer usually occupies a local region.
    changed = delta > 45
    cells = [c.mean() for row in np.array_split(changed, 4) for c in np.array_split(row, 4, axis=1)]
    return float(changed.mean()), sum(x > 0.4 for x in cells) / 16


def camera_shift(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype("float32")
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype("float32")
    (dx, dy), response = cv2.phaseCorrelate(ga, gb)
    # Only a strong whole-image translation is evidence of camera motion.
    return bool(response > 0.35 and (abs(dx) > 2 or abs(dy) > 2)), [float(dx), float(dy), float(response)]


def analyze(root, frames, config, duration):
    """Persistent observed stroke changes form state boundaries, with pre-change preservation.

    Occluded pixels never vote for erasure. Unconfirmed transitions still get their own
    conservative state, preventing speculative cross-state compositing.
    """
    states, metrics = [], []
    previous = None
    known = reference = streak = None
    pending_start = None
    state_start = scene = 0
    reason = "initial"
    current_flags = set()

    def close(end_index, end_time, why):
        nonlocal state_start, current_flags
        if end_index <= state_start:
            return
        ids = [f["id"] for f in frames[state_start:end_index]]
        # Prefer low foreground coverage, then sharpness. Latest breaks ties.
        candidates = metrics[state_start:end_index]
        best = min(candidates, key=lambda m: (round(m["occlusion"], 3), -m["sharpness"], -m["time"]))
        states.append(dict(id=f"b{len(states):05d}", scene=scene,
                           start=frames[state_start]["time"], end=end_time,
                           source_frame_ids=ids, selected_frame_id=best["id"],
                           selection_reason="least_foreground_then_sharpness",
                           boundary_reason=why, quality_flags=sorted(current_flags),
                           transcript_ids=[]))
        state_start = end_index
        current_flags = set()

    for i, f in enumerate(frames):
        raw = cv2.imread(str(root / f["image"]))
        image = small(raw, config)
        ink, foreground, kind = features(image, config.board)
        cut, distribution = cut_score(previous, image) if previous is not None else (0., 0.)
        moving, shift = camera_shift(previous, image) if previous is not None else (False, [0, 0, 1])
        scene_cut = cut > config.cut_fraction and distribution > .7
        m = dict(id=f["id"], time=f["time"], occlusion=float(foreground.mean()),
                 sharpness=float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()),
                 ink_pixels=int(ink.sum()), cut=cut, shift=shift, board=kind,
                 additions=0, removals=0, event="none")
        metrics.append(m)
        if scene_cut or moving:
            if pending_start is not None and pending_start > state_start:
                close(pending_start, frames[pending_start]["time"], "uncertain_transition")
                current_flags.add("unconfirmed_change")
            close(i, f["time"], "camera_motion" if moving and not scene_cut else "scene_cut")
            scene += 1
            known = reference = streak = None
            pending_start = None
            if moving and not scene_cut:
                current_flags.add("camera_motion_reconstruction_disabled")
            m["event"] = "scene_cut" if scene_cut else "camera_motion"
        if reference is None:
            reference = ink.copy()
            known = ~foreground
            streak = np.zeros(ink.shape, np.uint16)
        else:
            visible = ~foreground
            newly_known = visible & ~known
            reference[newly_known] = ink[newly_known]
            known |= visible
            # One analysis pixel of edge jitter is common after video compression.
            # Count genuinely new/disappeared strokes, rather than shifted boundaries.
            near_reference = cv2.dilate(reference.astype("uint8"), np.ones((3, 3), np.uint8)).astype(bool)
            near_ink = cv2.dilate(ink.astype("uint8"), np.ones((3, 3), np.uint8)).astype(bool)
            disagreement = visible & known & ((ink & ~near_reference) | (reference & ~near_ink))
            streak = np.where(disagreement, streak + 1, 0).astype(np.uint16)
            persistent = streak >= config.confirm
            adds = int((persistent & ink).sum())
            removes = int((persistent & ~ink).sum())
            m.update(additions=adds, removals=removes)
            threshold = max(config.change_pixels, int(ink.size * config.change_fraction))
            if disagreement.sum() >= threshold and pending_start is None:
                pending_start = i
            if adds + removes >= threshold:
                boundary = pending_start if pending_start is not None else i
                why = "erasure_or_revision" if removes >= threshold else "writing_progress"
                close(boundary, frames[boundary]["time"], why)
                reference[visible] = ink[visible]
                streak[:] = 0
                pending_start = None
                m["event"] = why
            elif pending_start is not None and not disagreement.any():
                # Short-lived writing is retained even if it never became persistent.
                close(pending_start, frames[pending_start]["time"], "transient_change")
                current_flags.add("unconfirmed_change")
                close(i, f["time"], "transient_return")
                pending_start = None
        if foreground.mean() > .03:
            current_flags.add("foreground_heuristic")
        previous = image
    if pending_start is not None and pending_start > state_start:
        close(pending_start, frames[pending_start]["time"], "unconfirmed_change")
        current_flags.add("unconfirmed_change")
    close(len(frames), duration, "end_of_video")
    return dict(states=states, metrics=metrics)
