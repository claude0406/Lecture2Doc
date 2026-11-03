from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from lecture2doc.transcript import resolve_whisper_model


def test_shared_alias_and_explicit_folder(tmp_path, monkeypatch):
    shared = tmp_path / "models" / "whisper"
    tiny = shared / "tiny"
    tiny.mkdir(parents=True)
    (tiny / "model.bin").write_bytes(b"model")
    (tiny / "config.json").write_text("{}")
    monkeypatch.setenv("LECTURE2DOC_WHISPER_MODELS", str(shared))
    assert resolve_whisper_model("tiny") == tiny
    assert resolve_whisper_model(str(tiny)) == tiny
    with pytest.raises(FileNotFoundError):
        resolve_whisper_model("base")


def test_download_uses_shared_folder(tmp_path, monkeypatch):
    shared = tmp_path / "whisper"
    monkeypatch.setenv("LECTURE2DOC_WHISPER_MODELS", str(shared))
    calls = []
    def download(name, **kwargs):
        calls.append((name, kwargs))
        folder = Path(kwargs["output_dir"])
        folder.mkdir(parents=True)
        (folder / "model.bin").write_bytes(b"model")
        (folder / "config.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", SimpleNamespace(download_model=download))
    assert resolve_whisper_model("base", local_only=False) == shared / "base"
    assert calls[0][1]["output_dir"] == str(shared / "base")
    assert resolve_whisper_model("base", local_only=False) == shared / "base"
    assert len(calls) == 1


def test_missing_explicit_path_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_whisper_model(str(tmp_path / "missing"), local_only=False)
    with pytest.raises(ValueError):
        resolve_whisper_model("repo/../other", local_only=False)
