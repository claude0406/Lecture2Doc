# 로컬 모델

모델 가중치와 개인별 캐시는 저장소에 포함하지 않습니다. OCR 모델 해시 목록과 Whisper 스냅샷 인덱스는 재현을 위해 보존하고, 각 팀원은 필요한 모델 파일을 자신의 작업 공간에 준비합니다.

```text
models/
  ocr/            # C OCR 모델과 언어별 해시 목록
  whisper/
    tiny/         # 음성 전사 tiny
    base/         # 음성 전사 base
    index.json    # 정리한 모델의 원본 snapshot·파일 해시
```

Whisper 전사는 선택 기능입니다. 웹 모델 칸이나 CLI의 `--model`에 `tiny`, `base` 등 모델 이름을 입력합니다. 모델을 지정하지 않으면 전사를 생략합니다. 공개 모델을 허용된 첫 실행에서 내려받으면 `whisper/<이름>/`에 저장됩니다.

Whisper와 OCR의 기본 저장 위치는 실행한 현재 폴더와 관계없이 이 저장소의 `models/whisper/`와 `models/ocr/`입니다. `LECTURE2DOC_WHISPER_MODELS`와 `LECTURE2DOC_OCR_MODELS` 환경 변수로 각각 바꿀 수 있습니다. [OCR 모델 준비](../README.md#빠른-시작)를 참고하세요.
