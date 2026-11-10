import base64
import json

import cv2
import numpy as np
import pytest

from lecture2doc import qwen_vision


def test_ollama_status_checks_minimum_version_and_model(monkeypatch):
    def fake_request(path, payload=None, timeout=15):
        if path == "/api/version":
            return {"version": "0.12.7"}
        if path == "/api/tags":
            return {"models": [{"name": "qwen3-vl:4b"}]}
        raise AssertionError(path)

    monkeypatch.setattr(qwen_vision, "_request_json", fake_request)
    status = qwen_vision.ollama_status()
    assert status["installed"] is True
    assert status["model"] == "qwen3-vl:4b"
    assert status["ollama_version"] == "0.12.7"


def test_ollama_status_rejects_old_version(monkeypatch):
    monkeypatch.setattr(
        qwen_vision, "_request_json",
        lambda path, payload=None, timeout=15: (
            {"version": "0.12.6"} if path == "/api/version" else {"models": []}
        ),
    )
    with pytest.raises(RuntimeError, match="이상"):
        qwen_vision.ollama_status()


def test_qwen_review_sends_board_image_and_preserves_ocr_hints(tmp_path, monkeypatch):
    image_path = tmp_path / "board.png"
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    image[20:100, 20:220] = (30, 90, 40)
    assert cv2.imwrite(str(image_path), image)

    monkeypatch.setattr(
        qwen_vision, "ollama_status",
        lambda model=None: {"installed": True, "model": model or "qwen3-vl:4b"},
    )
    captured = {}

    def fake_request(path, payload=None, timeout=15):
        captured["path"] = path
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {
            "model": "qwen3-vl:4b",
            "message": {
                "content": json.dumps({
                    "language": ["ko", "en", "math"],
                    "blocks": [{
                        "kind": "formula",
                        "text": "y'' + y' + 2y = 0",
                        "latex": "y'' + y' + 2y = 0",
                        "confidence": "medium",
                        "source_ocr_regions": [1],
                    }],
                    "unreadable_regions": ["오른쪽 아래 작은 글씨"],
                    "notes": [],
                }),
            },
        }

    monkeypatch.setattr(qwen_vision, "_request_json", fake_request)
    result = qwen_vision.review_board_image(
        image_path,
        ocr_regions=[{"order": 1, "text": "y''+y'+2y=0", "score": 0.62}],
    )

    payload = captured["payload"]
    image_data = payload["messages"][0]["images"][0]
    decoded = base64.b64decode(image_data)
    assert captured["path"] == "/api/chat"
    assert captured["timeout"] == 240
    assert payload["model"] == "qwen3-vl:4b"
    assert payload["stream"] is False
    assert "y''+y'+2y=0" in payload["messages"][0]["content"]
    assert decoded.startswith(b"\xff\xd8")
    assert result["status"] == "complete"
    assert result["local_only"] is True
    assert result["blocks"][0]["kind"] == "formula"
    assert result["unreadable_regions"] == ["오른쪽 아래 작은 글씨"]


def test_qwen_review_requires_model_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_vision, "ollama_status",
        lambda model=None: {"installed": False, "model": model or "qwen3-vl:4b"},
    )
    with pytest.raises(RuntimeError, match="qwen-models"):
        qwen_vision.review_board_image(tmp_path / "not-read.png")
