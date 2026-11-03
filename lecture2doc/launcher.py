"""One local entry point: fixed project paths, reload, and browser readiness."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser

from .server import code_revision

PROJECT = Path(__file__).resolve().parent.parent


def application():
    from .server import create_app
    return create_app(os.environ["LECTURE2DOC_SERVE_DATA"], os.environ.get("LECTURE2DOC_SERVE_RECORD") or None)


def server_info(url):
    try:
        with urllib.request.urlopen(url + "/api/app", timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError):
        return None


def same_server(info, data, record):
    return bool(info and info.get("app") == "lecture2doc" and info.get("project") == str(PROJECT)
                and info.get("data") == str(data) and info.get("record") == (str(record) if record else None)
                and info.get("code_revision") == code_revision())


def main():
    parser = argparse.ArgumentParser(description="Lecture2Doc 실행 · 최신 코드 자동 반영")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--no-browser", action="store_true", help="브라우저 자동 열기 생략")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("포트는 1~65535 사이입니다")
    os.chdir(PROJECT)
    data = args.data.expanduser().resolve()
    record = args.record.expanduser().resolve() if args.record else None
    if record and not (record / "manifest.json").is_file():
        parser.error("기록 폴더에 manifest.json이 없습니다")
    url = f"http://127.0.0.1:{args.port}"
    browser_url = f"{url}/?open={time.time_ns()}"
    lock_dir = PROJECT / ".cache" / "launcher"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / f"{args.port}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owns_lock = True
    except BlockingIOError:
        owns_lock = False
    with socket.socket() as sock:
        occupied = sock.connect_ex(("127.0.0.1", args.port)) == 0
    if occupied or not owns_lock:
        # A second launch opens the same app instead of creating another worker.
        for _ in range(30):
            info = server_info(url)
            if same_server(info, data, record):
                print(f"이미 실행 중입니다: {url}", flush=True)
                if not args.no_browser:
                    webbrowser.open(browser_url)
                return
            if occupied:
                break
            time.sleep(.1)
        parser.exit(1, f"{args.port} 포트에 다른 설정 또는 이전 서버가 실행 중입니다. 기존 실행 창에서 Ctrl+C로 종료한 뒤 serve.sh를 다시 실행하세요.\n")
    os.environ["LECTURE2DOC_SERVE_DATA"] = str(data)
    os.environ["LECTURE2DOC_SERVE_RECORD"] = str(record) if record else ""
    print(f"Lecture2Doc · V1\n주소: {url}\n코드 변경 자동 반영 · 종료: Ctrl+C", flush=True)
    if not args.no_browser:
        def open_when_ready():
            for _ in range(100):
                if same_server(server_info(url), data, record):
                    webbrowser.open(browser_url)
                    return
                time.sleep(.2)
            print(f"브라우저에서 {url} 을 열어주세요.", flush=True)
        threading.Thread(target=open_when_ready, daemon=True).start()
    import uvicorn
    # Import string + main guard allow Uvicorn's reload worker to start safely.
    uvicorn.run("lecture2doc.launcher:application", factory=True, host="127.0.0.1", port=args.port,
                reload=True, reload_dirs=[str(PROJECT / "lecture2doc")])
    lock.close()


if __name__ == "__main__":
    main()
