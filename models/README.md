# 로컬 모델

모델 가중치는 저장소에 올리지 않습니다. 각 팀원은 필요한 모델을 저장소 안의 `models/` 폴더에 내려받습니다. 모델 파일은 `.gitignore` 대상이며, URL·버전·SHA-256 정보는 JSON 인덱스로 공유합니다.

```text
models/
  ocr/
    en.json       # 영문 모델 파일명·URL·체크섬
    ko.json       # 한글 모델 파일명·URL·체크섬
    *.onnx        # 실행 후 내려받는 OCR 가중치 (Git 제외)
  whisper/
    index.json    # tiny/base 스냅샷·파일 체크섬
    <model>/      # Whisper 가중치 (Git 제외)
```

## OCR

프로젝트 의존성을 설치한 뒤 필요한 언어를 준비합니다.

```bash
python -m pip install -e '.[ocr]'
lecture2doc ocr-models --language en
lecture2doc ocr-models --language ko
```

`en`은 영문, `ko`는 한글·영문 인식 모델을 받습니다. 파일은 `models/ocr/`에 저장되며 인식은 로컬에서 실행합니다. 기본 위치를 바꾸려면 `LECTURE2DOC_OCR_MODELS` 환경 변수를 지정합니다.

## Whisper 음성 전사

```bash
python -m pip install -e '.[stt]'
lecture2doc process lecture.mp4 --out data/my-lecture \
  --model base --language ko --allow-download
```

첫 실행 때 지정한 모델을 `models/whisper/<이름>/`에 내려받습니다. `--allow-download`를 생략하면 준비된 로컬 모델만 사용합니다. 웹에서는 전사 설정에서 모델 이름과 첫 다운로드 허용을 선택합니다. 기본 폴더를 바꾸려면 `LECTURE2DOC_WHISPER_MODELS`를 지정합니다.

기본 모델 위치는 실행한 현재 폴더와 관계없이 저장소 루트 기준입니다. JSON 인덱스에 기록된 모델 출처와 라이선스는 [제3자 구성 요소 안내](../THIRD_PARTY.md)에서 확인할 수 있습니다.
