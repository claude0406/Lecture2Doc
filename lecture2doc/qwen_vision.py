"""Local Qwen vision review for OCR pages through the Ollama API."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import cv2

from .storage import digest

DEFAULT_QWEN_MODEL = "qwen3-vl:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
MIN_OLLAMA_VERSION = (0, 12, 7)
QWEN_REVIEW_VERSION = "2026-11-10.qwen-board-review-1"


def ollama_url() -> str:
    return os.environ.get("LECTURE2DOC_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def selected_model() -> str:
    return os.environ.get("LECTURE2DOC_QWEN_MODEL", DEFAULT_QWEN_MODEL).strip() or DEFAULT_QWEN_MODEL


def _request_json(path: str, payload: dict | None = None, timeout: int = 15) -> dict:
    url = ollama_url() + path
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(1500).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API 오류({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "Ollama에 연결할 수 없습니다. Ollama 앱을 설치하고 실행했는지 확인하세요."
        ) from exc


def ollama_status(model: str | None = None) -> dict:
    """Return local Ollama version and whether the requested model is installed."""
    info = _request_json("/api/version")
    version_text = str(info.get("version", ""))
    numbers = tuple(int(part) for part in re.findall(r"\d+", version_text)[:3])
    padded = numbers + (0,) * (3 - len(numbers))
    if padded < MIN_OLLAMA_VERSION:
        raise RuntimeError(
            f"Ollama {MIN_OLLAMA_VERSION[0]}.{MIN_OLLAMA_VERSION[1]}.{MIN_OLLAMA_VERSION[2]} 이상이 필요합니다. "
            f"현재 버전: {version_text or '확인 불가'}"
        )
    tags = _request_json("/api/tags")
    installed = {item.get("name") or item.get("model") for item in tags.get("models", [])}
    wanted = model or selected_model()
    return {
        "ollama_version": version_text,
        "model": wanted,
        "installed": wanted in installed or wanted.split(":")[0] in installed,
        "base_url": ollama_url(),
    }


def pull_model(model: str | None = None) -> dict:
    """Download a Qwen vision model into Ollama's managed local model store."""
    name = model or selected_model()
    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError(
            "Ollama 명령을 찾을 수 없습니다. https://ollama.com/download 에서 macOS용 Ollama를 설치하세요."
        )
    ollama_status(name)
    completed = subprocess.run([executable, "pull", name], check=False, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"Qwen 모델 다운로드 실패(exit={completed.returncode}). Ollama 앱과 네트워크를 확인하세요."
        )
    status = ollama_status(name)
    if not status["installed"]:
        raise RuntimeError(f"다운로드 명령은 끝났지만 {name} 모델이 설치 목록에 없습니다.")
    return status


def _image_as_base64(image_path: Path, max_side: int = 2200) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Qwen 이미지 읽기 실패: {image_path.name}")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError("Qwen 입력 이미지 인코딩 실패")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def review_board_image(
    image_path: Path,
    ocr_regions: list[dict] | None = None,
    model: str | None = None,
) -> dict:
    """Read a single board image and compare the visual evidence with local OCR text."""
    started = time.monotonic()
    image_path = Path(image_path).resolve()
    model_name = model or selected_model()
    status = ollama_status(model_name)
    if not status["installed"]:
        raise RuntimeError(
            f"Qwen 모델 {model_name}이 설치되어 있지 않습니다. "
            "터미널에서 lecture2doc qwen-models를 먼저 실행하세요."
        )

    ocr_hints = [
        {
            "region": int(item.get("order", index + 1)),
            "text": str(item.get("text", ""))[:180],
            "score": round(float(item.get("score", 0)), 3),
        }
        for index, item in enumerate(ocr_regions or [])
        if str(item.get("text", "")).strip()
    ][:80]

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "language": {"type": "array", "items": {"type": "string"}},
            "blocks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": ["title", "text", "formula", "list"]},
                        "text": {"type": "string"},
                        "latex": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "source_ocr_regions": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["kind", "text", "latex", "confidence", "source_ocr_regions"],
                },
            },
            "unreadable_regions": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["language", "blocks", "unreadable_regions", "notes"],
    }
    prompt = (
        "대학교 강의 칠판 이미지를 읽는 역할입니다. 첨부 이미지를 가장 중요한 근거로 사용하세요. "
        "OCR 후보는 참고용이며 틀릴 수 있으니 이미지와 대조해 수정하세요. "
        "한글, 영어, 숫자, 수학 기호와 수식을 원문 그대로 보존하고, 줄과 항목의 읽기 순서를 유지하세요. "
        "수식은 가능한 경우 LaTeX로 작성하고 일반 문장에는 latex를 빈 문자열로 두세요. "
        "읽을 수 없는 부분은 추측하지 말고 unreadable_regions에 위치나 특징을 적으세요. "
        "이미지에 없는 단어·공식·설명을 새로 만들지 마세요. "
        "OCR 후보의 번호는 source_ocr_regions에 기록하되, 확실히 연결할 수 없으면 빈 배열을 쓰세요. "
        "각 블록의 confidence는 시각적으로 판독 가능한 정도이며 실제 정답 확률로 가장하지 마세요. "
        "결과는 지정된 JSON 형식으로만 반환하세요.\n\n"
        "기존 OCR 후보:\n" + json.dumps(ocr_hints, ensure_ascii=False)
    )
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [_image_as_base64(image_path)],
        }],
        "format": schema,
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    response = _request_json("/api/chat", payload, timeout=240)
    content = (response.get("message") or {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Qwen 응답을 JSON으로 읽지 못했습니다.") from exc
    blocks = parsed.get("blocks")
    if not isinstance(blocks, list):
        raise RuntimeError("Qwen 응답에 blocks 목록이 없습니다.")
    text_lines = [str(item.get("text", "")).strip() for item in blocks if str(item.get("text", "")).strip()]
    return {
        "status": "complete",
        "version": QWEN_REVIEW_VERSION,
        "model": response.get("model") or model_name,
        "source_image": image_path.name,
        "source_sha256": digest(image_path),
        "language": parsed.get("language", []),
        "blocks": blocks,
        "text": "\n".join(text_lines),
        "unreadable_regions": parsed.get("unreadable_regions", []),
        "notes": parsed.get("notes", []),
        "ocr_hints": ocr_hints,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "local_only": True,
    }
