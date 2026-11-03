# 참고 구현 조사 · 2026-11-01

README의 기능 설명과 실제 구현을 구분했습니다. 아래 커밋의 소스를 읽었고, 일부 순수 함수 및 FFmpeg 추출 함수는 합성 입력에서 실행했습니다. 저장소 전체 애플리케이션이나 모든 사전학습 모델을 실행한 것은 아닙니다.

## 조사 범위와 선택

| 프로젝트 | 고정 커밋 / 마지막 push (UTC) | 실제 확인한 코드 | 판단 |
|---|---|---|---|
| [analysis-video](https://github.com/hwanyong/analysis-video) | `c0fc97080d3febd35f938e39a26fac25e31ef35f` / 2026-09-10 | `detect/signals.py`, `detect/events.py`, `frames.py`, `align.py`, `debug_viz.py`, core pyproject | 가장 가까운 A단계 참고 구현. 측정/판단 분리, 사건 전후 상태 보존, 화면 단위 전사 연결을 설계에 반영. 교수자 가림·삭제 구분은 별도 구현 필요 |
| [lecture-capture](https://github.com/eric-lf02/lecture-capture) | `9b8b4004888a0d6812f3e39608e45eb9db7a8642` / 2026-08-05 | `lecture_capture.ipynb` 전체 코드 셀 | 간단한 Colab 노트북. 기본 20초 관측, 2회 연속 변화 확인, 최소 8초 간격은 짧은 판서 보존에 부적합. 참고용으로만 사용 |
| [lecture-to-notes](https://github.com/drpwchen/lecture-to-notes) | `cc06141939ce56def13a15410d8d07e861028d94` / 2026-10-13 | `scripts/extract_slides.py`, `scripts/export_web.py`, `scripts/transcribe_video.py`, 의존성 및 라이선스 | 웹 타임라인 manifest, 원본/파생물 분리, 양방향 탐색 구조 참고. 전체 노트 생성·Pandoc·다중 산출물 구조를 이식하지 않음 |
| [lecturemath / FCN-LectureNet](https://github.com/kdavila/lecturemath) | `245d58a20393166addacb9ece75326ef3455b16f` / 2024-01-03 | `keyframe_extractor.py`, `video_segmenter.py`, `FCN_lecturenet_binarizer.py` | 연결 요소 시간 구간, 삭제 구간, 겹치는 그룹 충돌 처리를 참고. 오래된 연구 파이프라인·추가 의존성과 모델 필요. 모델 및 전체 ST3D 파이프라인 실행하지 않음 |

위 4개 저장소는 조사 당시 모두 archive 상태가 아니었고 루트 LICENSE는 MIT였습니다. 마지막 push는 유지보수 품질이나 지원을 보증하지 않습니다. 실제 읽은 소스 사본과 LICENSE는 로컬 `research/upstream/`에 보존했습니다. 소스 사본은 앱 패키지에 들어가지 않으며 재현 스크립트로 복원할 수 있습니다.

### analysis-video: 충분한 범위와 부족한 범위

[`signals.py`](https://github.com/hwanyong/analysis-video/blob/c0fc97080d3febd35f938e39a26fac25e31ef35f/packages/core/src/analysis_video/detect/signals.py)는 앵커 대비 누적 차이, 직전 프레임 차이, 픽셀 변화 면적을 각각 측정하고 안정 시 앵커를 재설정합니다. 고정 오버레이 띠를 제외하는 전처리도 있습니다. [`events.py`](https://github.com/hwanyong/analysis-video/blob/c0fc97080d3febd35f938e39a26fac25e31ef35f/packages/core/src/analysis_video/detect/events.py)는 국소 피크들을 합치고 전후 안정 프레임을 제한된 시간 범위에서 찾습니다. 화면 시작과 화면 끝에서 점진적으로 늘어난 필기 보존을 명시적으로 고려합니다.

`frames.py`는 initial/screen-start/screen-end 후보를 만들고 화면 시작/끝이 달라졌는지 비교합니다. 현재 조사 커밋의 gate는 내용이 비었는지를 기준으로 후보를 탈락시키며, 일반 중복 gate는 제거되어 있습니다. 주석의 과거 중복 설명만 읽고 현재 기능으로 판단해서는 안 됩니다. 원본 추출과 읽기용 축소본을 구분하고 판정 근거·탈락 원본을 남깁니다. `debug_viz.py`는 측정값과 이벤트의 시각적 보고 기능을 제공합니다.

`align.py`는 이미지 시점 하나를 다음 이미지까지의 구간 시작으로 오해하지 않고 화면 구간을 구성합니다. 전사 구간을 최대 겹침 화면에 배정하는 함수가 있습니다. 본 앱은 원문 전사 전체를 별도로 저장하고 상태별로 겹치는 구간 ID를 참조합니다. 연결이 시간상 겹침이라는 의미를 UI에 명시합니다.

판서의 실제 추가/삭제와 교수자의 움직임은 밝기 차이만으로 구분되지 않습니다. 본 앱이 해당 프로젝트를 전체 기반으로 채택하지 않은 이유는 핵심 상태 정의와 보수적인 복원 조건이 별도로 필요하고, 노트/에이전트 중심 CLI까지 포함할 필요가 없기 때문입니다. 런타임 코드 복사 없이 자체 모듈로 구현했습니다.

### lecture-capture

HSV로 칠판 종류와 피부색 제외 영역을 추정하고 CLAHE 처리한 320×180 회색 이미지에서 SSIM을 비교합니다. `score < .93`과 `changed_fraction > .18`이 동시에 만족되어야 변화로 봅니다. 판서 중 작은 추가/삭제는 면적 임계 아래로 떨어질 수 있고, 교수가 넓게 가리면 판서 변화 없이도 참이 될 수 있습니다. 노트북의 Whisper 전사와 Gemini OCR 셀은 영상 캡처와 별도입니다. Gemini는 본 앱에 도입하지 않았습니다.

### lecture-to-notes

`extract_slides.py`는 기본 15초 간격 FFmpeg 추출과 중앙 crop 밝기 해시를 사용합니다. 이름은 pHash이지만 읽은 구현은 DCT 기반 pHash가 아니라 평균 밝기 비교 비트 해시입니다. 직전 항목과의 차이뿐 아니라 그룹 첫 항목과의 drift도 검사하고 그룹 마지막 이미지를 남깁니다. 이 방식은 점진 변화 사슬을 일부 줄이지만 짧은 중간 판서를 관측하지 못하는 문제는 남습니다.

`export_web.py`는 원본 데이터와 배포용 보기, 타임라인 JSON, 영상과 문서 블록의 시간 참조를 구분합니다. 실제 코드는 생성 결과 안전 표시와 영상 재생 코덱 처리까지 포함하는 큰 시스템입니다. 본 앱은 더 작은 3영역 뷰어와 별도 원본 근거 패널로 구현했습니다.

### lecturemath

`keyframe_extractor.py`의 `GenerateFromST3DForIntervals`는 연결 요소 그룹의 존재 기간과 겹침을 보고 충돌 그룹을 처리하며 시간 정보를 남깁니다. `video_segmenter.py`에는 이진 필기량과 삭제 구간, 연결 요소 충돌에 따른 분할 코드가 있습니다. FCN worker는 전달받은 모델로 binary/text/reconstruction 출력과 프레임 시각·인덱스를 저장합니다. 단순 temporal median과 달리 시간상 공존할 수 없는 요소를 고려해야 한다는 점이 중요합니다. 다만 본 앱은 해당 모델의 재현·평가를 완료하지 않았으며 논문 수준의 복원 성능을 주장하지 않습니다.

## 실제 함수 실행

```bash
python -m pip install -e '.[test,research]'
python research/benchmark_upstream.py --fetch --inputs experiments/validated/inputs
```

결과: [`research/benchmark.json`](../research/benchmark.json). 실행 범위를 다음처럼 제한했습니다.

- analysis-video: 원본 `signals.measure`와 `events.find`. PyAV 회색 디코딩 IO 어댑터(320×180, 오버레이 제거 없음)를 전달했습니다. `align.py` 함수 코드도 로드했지만 end-to-end CLI·GUI·STT 실행 결과는 아닙니다.
- lecture-capture: 원본 `frames_are_different` 함수를 첫/마지막 회색 프레임에 실행했습니다. 보드 mask/CLAHE를 포함한 전체 노트북은 실행하지 않았습니다. 기본 20초 관측 시각은 입력 길이로 계산했습니다. 캡처 정확도와 동등한 측정이 아닙니다.
- lecture-to-notes: 원본 `extract_frames`, `dedup_frames` 함수로 FFmpeg 15초 관측과 그룹화를 실행했습니다. viewer 및 전체 노트 생성은 실행하지 않았습니다.
- lecturemath: 코드 조사만 수행. 모델·연구 파이프라인 미실행.

| 합성 입력 | analysis-video 사건 수 | capture 기본 관측 수 | notes 15초 추출 수 |
|---|---:|---:|---:|
| progressive 12s | 3 | 1 | 1 |
| occlusion 12s | 5 | 1 | 1 |
| partial_erase 12s | 1 | 1 | 1 |
| rewrite 12s | 1 | 1 | 1 |
| cut 12s | 1 | 1 | 1 |
| motion_only 12s | 13 | 1 | 1 |
| long_hold 40s | 0 | 2 | 3 |
| never_visible 12s | 0 | 1 | 1 |

사건 수는 보존 품질 점수가 아닙니다. 이 실행에서는 움직임만으로 많은 사건이 생기는 경우와 간격 관측의 누락 가능성을 확인했습니다. 입력과 설정이 제한된 비교이므로 각 프로젝트의 일반 성능 우열로 해석하지 않습니다.

## 기반 구현과 연구

[faster-whisper](https://github.com/SYSTRAN/faster-whisper)의 공개 `WhisperModel` CPU/int8 API를 사용합니다. 실제 tiny 영어 모델 다운로드·추론을 실행했습니다. [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)는 별도 장면 검출 도구로 검토했으며 핵심 실행 의존성으로 도입하지 않았습니다.

[Visual Transcripts 저자 원고](https://people.csail.mit.edu/hishin/projects/visual_transcripts/paper.pdf)와 [MIT DSpace 서지·초록](https://dspace.mit.edu/entities/publication/af790137-b3ec-46a4-820b-eaa5d08a4750)은 시각적 필기 요소와 전사문을 시간 관계로 분석하고 탐색 가능한 노트로 배치하는 구조를 설명합니다. 본 구현은 이미지·전사·원본 영상의 시간 탐색만 구현하며 논문의 의미 연결이나 사용자 평가를 재현하지 않았습니다.

## 라이선스·재사용 비용

조사한 저장소의 코드를 앱에 복사하지 않았습니다. 연구 실행기는 원본 함수를 로컬 소스에서 로드하므로 재현 소스와 MIT LICENSE를 함께 유지합니다. 사용한 라이브러리의 라이선스는 각 배포본을 따릅니다. 핵심 앱에 AGPL 저장소는 도입하지 않았으며, 향후 분할/OCR 모델을 추가할 때는 코드와 가중치 라이선스를 각각 검토해야 합니다.

FFmpeg는 시스템 실행 파일을 사용합니다. 설치한 FFmpeg의 빌드 옵션 및 `libx264` 사용 여부에 따라 배포 조건이 달라질 수 있으므로 바이너리를 재배포할 때 별도로 확인해야 합니다. MIT 실제 영상과 그 파생 비교 이미지는 앱 코드와 다른 CC BY-NC-SA 조건을 따릅니다. [THIRD_PARTY.md](../THIRD_PARTY.md)에 출처를 구분했습니다.
