from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lecture2doc.pipeline import process
from lecture2doc.server import create_app
from lecture2doc.storage import read
from lecture2doc.transcript import load_subtitles
from lecture2doc.vision import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fixtures import FPS, generate, render


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory):
    folder = tmp_path_factory.mktemp("inputs")
    return generate(folder)


@pytest.mark.parametrize("name", ["progressive", "occlusion", "partial_erase", "rewrite", "cut", "motion_only", "long_hold", "never_visible"])
def test_temporal_integrity_and_state_purity(fixtures, tmp_path, name):
    m = process(fixtures[name], tmp_path / name)
    frames = {f["id"]: f for f in m["source_frames"]}
    assert m["board_states"]
    assert m["source"]["height"] == 720
    assert m["source_frames"][0]["time"] == 0
    ids = []
    for s, r in zip(m["board_states"], m["reconstructed_boards"]):
        assert s["start"] < s["end"]
        ids.extend(s["source_frame_ids"])
        assert s["selected_frame_id"] in s["source_frame_ids"]
        assert set(r["source_frame_ids"]) <= set(s["source_frame_ids"])
        assert {render(name, frames[f]["time"])[2] for f in s["source_frame_ids"]}.__len__() == 1
    assert ids == list(frames)  # No sample lost or assigned to two states.
    if name in ("motion_only", "long_hold", "occlusion", "never_visible"):
        assert len(m["board_states"]) <= 2
    if name == "cut":
        assert len({s["scene"] for s in m["board_states"]}) == 2
    if name == "never_visible":
        assert all(r["status"] != "composited" for r in m["reconstructed_boards"])


def test_cache_invalidation_and_stt_failure(fixtures, tmp_path):
    out = tmp_path / "job"
    a = process(fixtures["rewrite"], out, model="/nonexistent/model")
    assert a["transcript"]["status"] == "failed"
    decode = out / a["source_frames"][0]["image"]
    old = decode.stat().st_mtime_ns
    b = process(fixtures["rewrite"], out, Config(reconstruction=False))
    assert decode.stat().st_mtime_ns == old
    assert all(r["status"] == "fallback" for r in b["reconstructed_boards"])
    c = process(fixtures["rewrite"], out, Config(sample_seconds=1))
    assert c["source_frames"][0]["image"] != a["source_frames"][0]["image"]
    with pytest.raises(ValueError):
        process(fixtures["cut"], out)


def test_subtitle_preservation(tmp_path):
    path = tmp_path / "c.vtt"
    path.write_text('WEBVTT\n\ncue-1\n00:00:01.125 --> 00:00:03.250 align:start\n  한글 <b>원문</b>\n둘째 줄\n\n')
    s = load_subtitles(path)[0]
    assert s["start"] == 1.125 and s["end"] == 3.25
    assert s["text"] == '  한글 <b>원문</b>\n둘째 줄'
    path.write_text('1\n00:00:03,000 --> 00:00:01,000\n역전\n')
    with pytest.raises(ValueError):
        load_subtitles(path)


def test_total_runtime_includes_ocr_and_cached_stages(fixtures, tmp_path, monkeypatch):
    from lecture2doc import ocr
    def delayed_ocr(*args, **kwargs):
        time.sleep(.05)
        return dict(status="complete", pages=[], cache_hits=0)
    monkeypatch.setattr(ocr, "recognize_pages", delayed_ocr)
    out = tmp_path / "timed"
    first = process(fixtures["long_hold"], out, ocr_language="en")
    second = process(fixtures["long_hold"], out, ocr_language="en")
    assert first["timings"]["ocr"]["elapsed_seconds"] >= .045
    assert second["elapsed_seconds"] >= second["timings"]["ocr"]["elapsed_seconds"]
    assert not first["timings"]["decode"]["cache_hit"]
    assert second["timings"]["decode"]["cache_hit"]
    assert second["timings"]["pages"]["cache_hit"]
    assert read(out / "manifest.json")["elapsed_seconds"] == second["elapsed_seconds"]


def test_server_upload_seek_and_security(fixtures, tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.post("/api/jobs").status_code == 403
        with fixtures["rewrite"].open("rb") as video:
            r = client.post("/api/jobs", headers={"X-Lecture2Doc": "1"}, files={"video": ("class.mp4", video, "video/mp4")})
        assert r.status_code == 202
        job = r.json()["id"]
        for _ in range(300):
            status = client.get(f"/api/jobs/{job}").json()
            if status["status"] in ("failed", "complete"):
                break
            time.sleep(.05)
        assert status["status"] == "complete", status
        m = client.get(f"/api/jobs/{job}/manifest").json()
        assert m["playback"] != m["source"]["path"]  # MP4V fixture gets a browser proxy.
        response = client.get(f'/api/jobs/{job}/files/{m["source"]["path"]}', headers={"Range": "bytes=0-99"})
        assert response.status_code == 206 and len(response.content) == 100
        assert client.get(f"/api/jobs/{job}/files/status.json").status_code == 404
        assert client.get(f"/api/jobs/{job}/files/%2e%2e/secret").status_code == 404
        assert client.get("/", headers={"Host": "evil.example"}).status_code == 400


def test_observed_pixel_provenance(fixtures, tmp_path):
    m = process(fixtures["occlusion"], tmp_path / "o")
    by_id = {f["id"]: f for f in m["source_frames"]}
    composites = [r for r in m["reconstructed_boards"] if r["provenance"]]
    assert composites
    for r in composites:
        result = cv2.imread(str(tmp_path / "o" / r["image"]))
        for tile in read(tmp_path / "o" / r["provenance"])["tiles"]:
            original = cv2.imread(str(tmp_path / "o" / by_id[tile["source_frame_id"]]["image"]))
            x, y, w, h = (tile[k] for k in ("x", "y", "width", "height"))
            assert np.array_equal(result[y:y+h, x:x+w], original[y:y+h, x:x+w])


def test_video_failure_does_not_prevent_subtitle_record(fixtures, tmp_path, monkeypatch):
    from lecture2doc import pipeline
    def fail(*args, **kwargs):
        raise RuntimeError("injected analysis failure")
    monkeypatch.setattr(pipeline, "analyze", fail)
    subtitles = fixtures["rewrite"].parent / "example.srt"
    m = process(fixtures["rewrite"], tmp_path / "failed", subtitles=subtitles)
    assert m["transcript"]["status"] == "complete"
    assert m["visual_error"]
    assert len(m["board_states"]) == len(m["source_frames"])
    assert all(r["status"] == "fallback" for r in m["reconstructed_boards"])


def test_variable_frame_rate_pts(tmp_path):
    import av
    from fractions import Fraction
    from lecture2doc.media import extract
    path = tmp_path / "variable.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = 320, 180, "yuv420p"
        s.time_base = Fraction(1, 1000)
        for pts in (0, 100, 300, 400, 700, 800):
            f = av.VideoFrame.from_ndarray(np.zeros((180, 320, 3), np.uint8), format="bgr24")
            f.pts, f.time_base = pts, Fraction(1, 1000)
            for packet in s.encode(f):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    folder = tmp_path / "frames"
    folder.mkdir()
    frames = extract(path, tmp_path, folder, .05)
    assert [f["time"] for f in frames] == [0, .1, .3, .4, .7, .8]


def test_whiteboard_rewrite_and_non_destructive_rerun(tmp_path):
    path = tmp_path / "white.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    for i in range(12*FPS):
        writer.write(render("rewrite", i/FPS, white=True)[0])
    writer.release()
    folder = tmp_path / "out"
    a = process(path, folder, Config(board="whiteboard"))
    assert len(a["board_states"]) == 2
    old = folder / a["source_frames"][0]["image"]
    b = process(path, folder, Config(board="whiteboard"), rerun=["decode"])
    assert old.exists()
    assert b["source_frames"][0]["image"] != a["source_frames"][0]["image"]
