import subprocess
from pathlib import Path

import av
import cv2


def probe(path):
    with av.open(str(path)) as container:
        s = container.streams.video[0]
        duration = float(s.duration * s.time_base) if s.duration else float(container.duration or 0) / av.time_base
        if duration <= 0:
            raise ValueError("영상 길이를 확인할 수 없습니다")
        return dict(width=s.width, height=s.height, duration=duration,
                    fps=float(s.average_rate or 0), codec=s.codec_context.name,
                    audio=bool(container.streams.audio),
                    audio_codec=container.streams.audio[0].codec_context.name if container.streams.audio else None)


def browser_compatible(meta, suffix):
    if suffix in (".mp4", ".m4v", ".mov"):
        return meta["codec"] in ("h264", "av1") and meta.get("audio_codec") in (None, "aac", "mp3")
    if suffix == ".webm":
        return meta["codec"] in ("vp8", "vp9", "av1") and meta.get("audio_codec") in (None, "opus", "vorbis")
    return False


def extract(path, root, folder, sample_seconds):
    """Decode in order; sample using presentation timestamps, never frame_index/fps."""
    frames = []
    next_t = 0.0
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        for index, frame in enumerate(c.decode(stream)):
            if frame.time is None:
                raise ValueError("PTS 없는 영상: 타임스탬프를 신뢰할 수 없습니다")
            t = float(frame.time) - origin
            if t + 1e-6 < next_t:
                continue
            out = folder / f"frame-{len(frames):06d}.png"
            image = frame.to_ndarray(format="bgr24")
            if not cv2.imwrite(str(out), image):
                raise OSError(f"이미지 저장 실패: {out}")
            frames.append(dict(id=f"f{len(frames):06d}", time=round(t, 6),
                               pts=frame.pts, time_base=str(frame.time_base), decode_index=index,
                               image=out.relative_to(root).as_posix(), width=frame.width, height=frame.height))
            next_t = t + sample_seconds
    if not frames:
        raise ValueError("영상 프레임을 디코딩하지 못했습니다")
    return frames


def audio(path, output):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                    "-vn", "-ac", "1", "-ar", "16000", str(output)], check=True,
                   capture_output=True)


def playback(path, output):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                    "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-crf", "20",
                    "-preset", "fast", "-c:a", "aac", "-movflags", "+faststart", str(output)],
                   check=True, capture_output=True)
