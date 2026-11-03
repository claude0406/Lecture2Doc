import math
import os
import re
from pathlib import Path

from . import media
from .storage import digest, read, stage


def whisper_models_dir():
    return Path(os.environ.get("LECTURE2DOC_WHISPER_MODELS", "models/whisper")).expanduser().resolve()


def resolve_whisper_model(model, local_only=True):
    """Resolve a short model name to the shared project model directory."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or str(model).startswith((".", "~")):
        raise FileNotFoundError(f"Whisper 모델 폴더가 없습니다: {path}")
    if not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", str(model)) or ".." in str(model).split("/"):
        raise ValueError("Whisper 모델 이름 또는 폴더를 확인하세요")
    target = whisper_models_dir() / str(model).replace("/", "--")
    if (target / "model.bin").is_file() and (target / "config.json").is_file():
        return target
    if local_only:
        raise FileNotFoundError(f"Whisper 모델이 준비되지 않았습니다: {model} ({target})")
    from faster_whisper.utils import download_model
    download_model(str(model), output_dir=str(target), cache_dir=str(whisper_models_dir()/".cache"))
    return target


def validate_segments(segments):
    output = []
    for i, segment in enumerate(segments):
        start, end = float(segment["start"]), float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"잘못된 자막 시간: {i+1}")
        output.append(dict(id=f"t{i:06d}", start=start, end=end, text=str(segment["text"])))
    return sorted(output, key=lambda s: s["start"])


def load_subtitles(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = read(path)
        return validate_segments(data["segments"] if isinstance(data, dict) else data)
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    # SRT and WebVTT, including cue IDs, multiline text, and VTT cue settings.
    stamp = r"(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})"
    pattern = re.compile(r"^"+stamp+r"\s+-->\s+"+stamp+r"[^\n]*\n(.*?)(?=\n\s*\n|\Z)", re.M | re.S)
    segments = []
    def seconds(groups):
        h, m, s, ms = groups
        return int(h or 0)*3600+int(m)*60+int(s)+int(ms)/1000
    for match in pattern.finditer(text):
        g = match.groups()
        segments.append(dict(start=seconds(g[:4]), end=seconds(g[4:8]), text=g[8].rstrip("\n")))
    if not segments:
        raise ValueError("SRT/VTT 자막 구간이 없습니다")
    return validate_segments(segments)


def transcribe(root, video, source_hash, subtitles=None, model=None, language=None, local_only=True, force=False):
    if subtitles:
        return stage(root, "transcript", ["subtitles", digest(subtitles)],
                     lambda folder: dict(status="complete", origin="subtitles", segments=load_subtitles(subtitles)), force=force)
    if not model:
        return dict(status="skipped", origin=None, segments=[], message="자막 또는 Whisper 모델을 지정하세요")
    model_path = resolve_whisper_model(model, local_only=local_only)
    model_files = {p.name: digest(p) for p in model_path.iterdir() if p.is_file()}
    model_key = [str(model), model_files]
    def run(folder):
        from faster_whisper import WhisperModel
        wav = stage(root, "audio", source_hash, lambda audio_folder: extract_audio(video, audio_folder, root))
        whisper = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=4,
                               local_files_only=True)
        segments, info = whisper.transcribe(str(root / wav["image"]), language=language,
                                            beam_size=5, vad_filter=True)
        data = validate_segments([dict(start=s.start, end=s.end, text=s.text) for s in segments])
        return dict(status="complete", origin="faster-whisper", model=str(model), language=info.language,
                    model_path=str(model_path), model_file_hashes=model_files,
                    language_probability=info.language_probability, segments=data)
    return stage(root, "transcript", [source_hash, model_key, language], run, force=force)


def extract_audio(video, folder, root):
    wav = folder / "audio.wav"
    media.audio(video, wav)
    return dict(image=wav.relative_to(root).as_posix())
