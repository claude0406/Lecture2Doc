from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np

from .vision import crop, features, small


def reconstruct(root, folder, state, frames_by_id, config):
    selected = frames_by_id[state["selected_frame_id"]]
    base = crop(cv2.imread(str(root / selected["image"])), config.roi).copy()
    result = dict(id="r"+state["id"][1:], board_state_id=state["id"],
                  status="fallback", image=selected["image"],
                  source_frame_ids=[selected["id"]], provenance=None,
                  quality_flags=["model_free_masks_require_review"], coverage=None)
    if not config.reconstruction:
        result["quality_flags"] = ["reconstruction_disabled"]
        return result
    if any(x in state["quality_flags"] for x in ("unconfirmed_change", "camera_motion_reconstruction_disabled")):
        result["quality_flags"].append("unsafe_state_original_used")
        return result
    ids = state["source_frame_ids"]
    if len(ids) < 2:
        result["quality_flags"].append("insufficient_observations")
        return result
    # Bound RAM independently of lecture/state duration; always include selected frame.
    indices = np.linspace(0, len(ids)-1, min(48, len(ids))).astype(int)
    ids = list(dict.fromkeys([selected["id"]] + [ids[i] for i in indices]))
    observations = []
    for fid in ids:
        im = small(cv2.imread(str(root / frames_by_id[fid]["image"])), config)
        ink, mask, kind = features(im, config.board)
        observations.append((fid, ink, mask, im))
    shape = observations[0][1].shape
    # Conflict test compares only jointly visible pixels. Any substantial writing change
    # inside a state disables the entire composite, rather than guessing a common state.
    seen = np.zeros(shape, bool)
    remembered = np.zeros(shape, bool)
    for fid, ink, mask, im in observations:
        visible = ~mask
        shared = seen & visible
        kernel = np.ones((3, 3), np.uint8)
        near_ink = cv2.dilate(ink.astype("uint8"), kernel).astype(bool)
        near_memory = cv2.dilate(remembered.astype("uint8"), kernel).astype(bool)
        conflicts = shared & ((ink & ~near_memory) | (remembered & ~near_ink))
        if conflicts.sum() > max(15, shared.sum()*.003):
            result["quality_flags"].append("inconsistent_writing_original_used")
            return result
        remembered[visible] = ink[visible]
        seen |= visible
    output = base.copy()
    h, w = shape
    H, W = base.shape[:2]
    tiles, used, covered, total = [], set(), 0, 0
    selected_mask = observations[0][2]
    @lru_cache(maxsize=3)
    def source_crop(fid):
        return crop(cv2.imread(str(root / frames_by_id[fid]["image"])), config.roi)
    tile = config.tile_size
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y2, x2 = min(y+tile, h), min(x+tile, w)
            X, Y, X2, Y2 = round(x*W/w), round(y*H/h), round(x2*W/w), round(y2*H/h)
            total += (y2-y)*(x2-x)
            choices = []
            for fid, ink, mask, im in observations:
                # Reject any tile touched by foreground (dilated in features).
                if mask[y:y2, x:x2].any():
                    continue
                sharp = cv2.Laplacian(im[y:y2, x:x2], cv2.CV_64F).var()
                choices.append((float(sharp), fid))
            if len(choices) < 2:
                # Unobserved/one-off content stays in the selected original and is labelled.
                tiles.append(dict(x=X, y=Y, width=X2-X, height=Y2-Y, source_frame_id=selected["id"], verified=False))
                used.add(selected["id"])
                continue
            _, fid = max(choices)
            # Keep the original wherever it is already unobscured. A different codec
            # sample being marginally sharper is not evidence of a restoration benefit.
            if not selected_mask[y:y2, x:x2].any():
                fid = selected["id"]
            original = source_crop(fid)
            output[Y:Y2, X:X2] = original[Y:Y2, X:X2]
            covered += (y2-y)*(x2-x)
            used.add(fid)
            tiles.append(dict(x=X, y=Y, width=X2-X, height=Y2-Y,
                              source_frame_id=fid, verified=True))
    coverage = covered / max(1, total)
    result["coverage"] = coverage
    if coverage < 1:
        result["quality_flags"].append("unverified_regions_original_retained")
    if not covered or np.array_equal(output, base):
        result["quality_flags"].append("no_verified_improvement")
        return result
    path = folder / (state["id"]+".png")
    cv2.imwrite(str(path), output)
    from .storage import write
    provenance = path.with_suffix(".json")
    write(provenance, dict(coordinate_system="roi_pixels", roi=config.roi,
                           width=W, height=H, tiles=tiles))
    result.update(status="composited" if coverage == 1 else "partial", image=path.relative_to(root).as_posix(),
                  source_frame_ids=sorted(used), provenance=provenance.relative_to(root).as_posix())
    return result
