import argparse
from pathlib import Path

from .pipeline import process
from .vision import Config


def main():
    parser = argparse.ArgumentParser(description="로컬 판서 기록 및 근거 기반 복원")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="로컬 웹 앱")
    serve.add_argument("--data", type=Path, help="업로드 및 라이브러리 폴더 (기본값: 저장소의 data/)")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--record", type=Path, help="CLI로 분석한 결과 폴더를 웹 라이브러리에서 열기")
    models = sub.add_parser("ocr-models", help="로컬 OCR 모델 준비 (첫 다운로드)")
    models.add_argument("--language", choices=["en", "ko"], default="en")
    models.add_argument("--dir", type=Path)
    qwen = sub.add_parser("qwen-models", help="로컬 Qwen Vision 모델 다운로드 (Ollama 필요)")
    qwen.add_argument("--model", help="Qwen 모델 이름 (기본값: qwen3-vl:4b)")
    ocr = sub.add_parser("ocr", help="기존 A/B 판서에 C 텍스트 인식 추가")
    ocr.add_argument("record", type=Path)
    ocr.add_argument("--out", type=Path, help="기존 결과를 보존하고 새 폴더에 OCR 기록")
    ocr.add_argument("--language", choices=["en", "ko"], default="en")
    ocr.add_argument("--preprocessing", choices=["rgb", "adaptive"], default="rgb")
    ocr.add_argument("--rerun", action="store_true")
    run = sub.add_parser("process", help="영상 분석; 같은 출력 폴더로 재실행하면 단계 캐시 사용")
    run.add_argument("video", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--subtitles", type=Path)
    run.add_argument("--model", help="Whisper 모델 이름 또는 로컬 CTranslate2 모델 폴더")
    run.add_argument("--language", help="ko/en 등; 생략하면 자동 감지")
    run.add_argument("--allow-download", action="store_true", help="최초 Whisper 모델 다운로드 허용")
    run.add_argument("--sample-seconds", type=float, default=.5)
    run.add_argument("--roi", default="0,0,1,1", help="판서 영역 x,y,w,h (0~1)")
    run.add_argument("--board", choices=["auto", "blackboard", "whiteboard"], default="auto")
    run.add_argument("--no-reconstruct", action="store_true", help="B를 생략하고 A 원본만 제공")
    run.add_argument("--reconstruct", action="store_true", help="누적 판서 B 복원 (기본값)")
    run.add_argument("--make-playback", action="store_true", help="H.264/AAC 브라우저 재생 사본 생성")
    run.add_argument("--ocr", choices=["en", "ko", "off"], default="en", help="C OCR 언어; off는 A/B만 실행")
    run.add_argument("--rerun", choices=["decode", "analysis", "pages", "transcript", "reconstruction", "page-restoration", "playback", "ocr", "learning-analysis"],
                     action="append", default=[], help="지정 단계를 새 캐시 폴더에서 재실행 (반복 지정 가능)")
    args = parser.parse_args()
    if args.command == "qwen-models":
        from .qwen_vision import DEFAULT_QWEN_MODEL, pull_model
        try:
            result = pull_model(args.model or DEFAULT_QWEN_MODEL)
            print(f"Qwen 모델 준비 완료: {result['model']} · Ollama {result['ollama_version']} · 로컬 설치 확인")
        except Exception as exc:
            parser.exit(1, f"Qwen 모델 준비 실패: {exc}\n")
        return
    if args.command in ("ocr-models", "ocr"):
        from .ocr import OCRConfig, annotate_record, prepare_models
        try:
            if args.command == "ocr-models":
                result = prepare_models(args.language, args.dir)
                print(f"OCR 모델 준비 완료: {result['models']}")
            else:
                result = annotate_record(args.record, OCRConfig(language=args.language, preprocessing=args.preprocessing),
                                         args.out, print, args.rerun)
                print(f"C OCR 완료: {len(result['ocr']['pages'])}장 · {(args.out or args.record).resolve() / 'manifest.json'}")
                print(f"OCR {result['ocr']['elapsed_seconds']:.1f}초 · 캐시 {result['ocr']['cache_hits']}장")
        except Exception as exc:
            parser.exit(1, f"OCR 처리 실패: {exc}\n")
        return
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(args.data, args.record), host="127.0.0.1", port=args.port)
        return
    config = Config(sample_seconds=args.sample_seconds, board=args.board,
                    roi=tuple(float(x) for x in args.roi.split(",")), reconstruction=not args.no_reconstruct)
    try:
        m = process(args.video, args.out, config, args.subtitles, args.model, args.language,
                    args.allow_download, args.make_playback, progress=print, rerun=args.rerun,
                    ocr_language=None if args.ocr == "off" else args.ocr)
    except Exception as exc:
        parser.exit(1, f"처리 실패: {exc}\n")
    print(f"{args.out.resolve() / 'manifest.json'}\n누적 판서 {len(m['board_pages'])}개 (내부 관측 상태 {len(m['board_states'])}개) · 전사 {m['transcript']['status']}")
    print(f"전체 처리 {m['elapsed_seconds']:.1f}초")
    if "ocr" in m:
        print(f"C OCR: {m['ocr']['status']}")
        if m["ocr"]["status"] == "failed":
            parser.exit(1, f"A/B 저장 완료 · OCR 처리 실패: {m['ocr']['message']}\n")
    # Register CLI output in the web library only when it is under --data with a valid job ID.


if __name__ == "__main__":
    main()
