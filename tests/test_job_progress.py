import threading
import time

from fastapi.testclient import TestClient

from lecture2doc import server
from lecture2doc.storage import write


def test_elapsed_time_advances_between_stage_updates_and_freezes_on_finish(tmp_path, monkeypatch):
    started, finish = threading.Event(), threading.Event()
    def process(*args, **kwargs):
        kwargs["progress"]("판서 변화 분석")
        started.set()
        finish.wait(5)
        return {}
    monkeypatch.setattr(server, "process", process)
    with TestClient(server.create_app(tmp_path)) as client:
        try:
            r = client.post('/api/jobs', headers={'X-Lecture2Doc': '1'}, files={'video': ('test.mp4', b'fixture', 'video/mp4')})
            identity = r.json()['id']
            assert started.wait(2)
            first = client.get(f'/api/jobs/{identity}').json()
            assert first['message'] == '판서 변화 분석'
            time.sleep(.06)
            later = client.get(f'/api/jobs/{identity}').json()
            assert later['elapsed_seconds'] > first['elapsed_seconds']
            assert later['updated_at'] == first['updated_at']
            assert client.get('/api/jobs').json()[0]['elapsed_seconds'] >= later['elapsed_seconds']
        finally:
            finish.set()
        for _ in range(100):
            last = client.get(f'/api/jobs/{identity}').json()
            if last['status'] == 'complete':
                break
            time.sleep(.01)
        assert last['status'] == 'complete'
        assert last['finished_at'] >= last['started_at']
        time.sleep(.03)
        assert client.get(f'/api/jobs/{identity}').json()['elapsed_seconds'] == last['elapsed_seconds']


def test_completed_older_worker_uses_recorded_full_pipeline_time(tmp_path):
    identity = 'c'*24
    folder = tmp_path / identity
    write(folder/'status.json', dict(id=identity, status='complete', message='완료'))
    write(folder/'manifest.json', dict(elapsed_seconds=123.5, elapsed_scope='video_input_through_ocr_excluding_final_manifest_write'))
    with TestClient(server.create_app(tmp_path)) as client:
        assert client.get(f'/api/jobs/{identity}').json()['elapsed_seconds'] == 123.5
        assert client.get('/api/jobs').json()[0]['elapsed_seconds'] == 123.5
