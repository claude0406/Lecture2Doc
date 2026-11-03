# 로컬 모델

```text
models/
  ocr/            # C OCR 모델과 언어별 해시 목록
  whisper/
    tiny/         # 음성 전사 tiny
    base/         # 음성 전사 base
    index.json    # 정리한 모델의 원본 snapshot·파일 해시
```

Whisper는 웹 모델 칸에 `tiny` 또는 `base`, CLI에서 `--model base`로 지정합니다. 비워두면 전사를 생략합니다. 다른 모델도 처음 다운로드하면 `whisper/<이름>/`에 저장합니다.

기본 저장 위치는 프로젝트에서 실행할 때 이 폴더입니다. `LECTURE2DOC_WHISPER_MODELS`와 `LECTURE2DOC_OCR_MODELS`로 각각 바꿀 수 있습니다. 기존 실험 기록과 Hugging Face 캐시는 유지했습니다. 정리한 tiny/base는 외부 캐시에 의존하지 않는 실제 파일입니다.
