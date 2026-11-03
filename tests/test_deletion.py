from concurrent.futures import Future

from fastapi.testclient import TestClient

from lecture2doc.server import create_app
from lecture2doc.storage import read, write
from pathlib import Path

JOB = "a" * 24
IMPORTED = "000000000000000000000001"
HEADERS = {"X-Lecture2Doc": "1"}


def record(folder):
    folder.mkdir(parents=True)
    (folder / "source.mp4").write_bytes(b"original-video")
    write(folder / "source.json", {"name": "Test lecture"})
    write(folder / "status.json", {"id": JOB, "status": "complete", "message": "완료"})
    write(folder / "manifest.json", dict(source=dict(name="Test lecture", path="source.mp4", sha256="original"),
        playback="source.mp4", source_frames=[], board_states=[], reconstructed_boards=[],
        board_pages=[dict(id=f"p{i}", start=i*10, end=(i+1)*10, selected_frame_id=f"f{i}") for i in range(3)]))


def test_lecture_trash_persistence_and_restore(tmp_path):
    folder = tmp_path / JOB
    record(folder)
    original = (folder / "manifest.json").read_bytes()
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.delete(f"/api/jobs/{JOB}").status_code == 403
        assert client.delete(f"/api/jobs/{JOB}", headers=HEADERS).status_code == 200
        assert client.get("/api/jobs").json() == []
        assert client.get(f"/api/jobs/{JOB}/manifest").status_code == 404
        assert client.get(f"/api/jobs/{JOB}/files/source.mp4").status_code == 404
        assert client.get("/api/trash").json()[0]["id"] == JOB
        assert (folder / "source.mp4").read_bytes() == b"original-video"
        assert (folder / "manifest.json").read_bytes() == original
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/jobs").json() == []
        assert client.post(f"/api/jobs/{JOB}/restore", headers=HEADERS).status_code == 200
        assert client.get("/api/jobs").json()[0]["id"] == JOB
        assert client.get("/api/trash").json() == []


def test_bulk_board_delete_empty_restore_and_original_unchanged(tmp_path):
    folder = tmp_path / JOB
    record(folder)
    original = (folder / "manifest.json").read_bytes()
    url = f"/api/jobs/{JOB}"
    m = read(folder / "manifest.json")
    m["ocr"] = {"pages": [{"board_page_id": f"p{i}", "text": str(i)} for i in range(3)]}
    write(folder / "manifest.json", m)
    original = (folder / "manifest.json").read_bytes()
    with TestClient(create_app(tmp_path)) as client:
        assert client.post(url+"/boards/delete", headers=HEADERS, json={"ids": []}).status_code == 422
        assert client.post(url+"/boards/delete", headers=HEADERS, json={"ids": ["p0", "bad"]}).status_code == 404
        assert len(client.get(url+"/manifest").json()["board_pages"]) == 3
        assert client.post(url+"/boards/delete", headers=HEADERS, json={"ids": ["p0", "p2"]}).status_code == 200
        m = client.get(url+"/manifest").json()
        assert [b["id"] for b in m["board_pages"]] == ["p1"]
        assert m["board_pages"][0]["display_number"] == 2
        assert [p["board_page_id"] for p in m["ocr"]["pages"]] == ["p1"]
        assert [b["id"] for b in m["deleted_boards"]] == ["p0", "p2"]
        client.post(url+"/boards/delete", headers=HEADERS, json={"ids": ["p1"]})
        assert client.get(url+"/manifest").json()["board_pages"] == []
        assert (folder / "manifest.json").read_bytes() == original
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(url+"/manifest").json()["board_pages"] == []
        client.post(url+"/boards/restore", headers=HEADERS, json={"ids": ["p0", "p1", "p2"]})
        assert len(client.get(url+"/manifest").json()["board_pages"]) == 3


def test_active_job_and_import_isolation(tmp_path):
    folder = tmp_path / JOB
    record(folder)
    app = create_app(tmp_path)
    future = Future()
    app.state.futures[JOB] = future
    with TestClient(app) as client:
        assert not client.get("/api/jobs").json()[0]["can_delete"]
        assert client.delete(f"/api/jobs/{JOB}", headers=HEADERS).status_code == 409
        assert client.post(f"/api/jobs/{JOB}/boards/delete", headers=HEADERS, json={"ids": ["p0"]}).status_code == 409
        future.set_result(None)
    imported = tmp_path / "experiment-a"
    other = tmp_path / "experiment-b"
    record(imported)
    record(other)
    library = tmp_path / "library"
    with TestClient(create_app(library, imported)) as client:
        client.delete(f"/api/jobs/{IMPORTED}", headers=HEADERS)
        assert client.get("/api/jobs").json() == []
        assert (imported / "manifest.json").exists()
    with TestClient(create_app(library, other)) as client:
        assert client.get("/api/jobs").json()[0]["id"] == IMPORTED
        assert client.get("/api/trash").json() == []


def test_reanalysis_does_not_hide_a_different_board(tmp_path):
    folder = tmp_path / JOB
    record(folder)
    with TestClient(create_app(tmp_path)) as client:
        client.post(f"/api/jobs/{JOB}/boards/delete", headers=HEADERS, json={"ids": ["p0"]})
        m = read(folder / "manifest.json")
        m["board_pages"][0]["selected_frame_id"] = "new-content"
        write(folder / "manifest.json", m)
        visible = client.get(f"/api/jobs/{JOB}/manifest").json()
        assert len(visible["board_pages"]) == 3
        assert visible["deleted_boards"] == []


def test_permanent_delete_requires_trash_and_confirmation(tmp_path):
    folder = tmp_path / JOB
    record(folder)
    other = tmp_path / ("b" * 24)
    record(other)
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.bin").write_bytes(b"shared")
    with TestClient(create_app(tmp_path)) as client:
        url = f"/api/jobs/{JOB}"
        assert client.post(url+"/purge", headers=HEADERS, json={"confirmed": True}).status_code == 409
        client.delete(url, headers=HEADERS)
        assert client.post(url+"/purge", json={"confirmed": True}).status_code == 403
        assert client.post(url+"/purge", headers=HEADERS, json={"confirmed": False}).status_code == 422
        assert folder.is_dir()
        assert client.post(url+"/purge", headers=HEADERS, json={"confirmed": True}).status_code == 200
        assert not folder.exists()
        assert other.is_dir() and (models / "model.bin").read_bytes() == b"shared"
        assert client.get("/api/trash").json() == []
        assert client.post(url+"/restore", headers=HEADERS).status_code == 404
        assert client.get(url+"/files/source.mp4").status_code == 404


def test_import_purge_removes_reference_without_deleting_external_files(tmp_path):
    external = tmp_path / "external"
    library = tmp_path / "library"
    record(external)
    with TestClient(create_app(library, external)) as client:
        url = f"/api/jobs/{IMPORTED}"
        client.delete(url, headers=HEADERS)
        assert client.get("/api/trash").json()[0]["purge_scope"] == "library_entry"
        assert client.post(url+"/purge", headers=HEADERS, json={"confirmed": True}).status_code == 200
        assert (external / "source.mp4").exists()
        assert client.get("/api/jobs").json() == []
    with TestClient(create_app(library, external)) as client:
        assert client.get("/api/jobs").json() == []
        assert client.get("/api/trash").json() == []
        assert client.post(f"/api/jobs/{IMPORTED}/restore", headers=HEADERS).status_code == 404


def test_purge_rejects_linked_folder(tmp_path):
    external = tmp_path / "external"
    record(external)
    library = tmp_path / "library"
    library.mkdir()
    (library / JOB).symlink_to(external, target_is_directory=True)
    with TestClient(create_app(library)) as client:
        url = f"/api/jobs/{JOB}"
        client.delete(url, headers=HEADERS)
        assert client.post(url+"/purge", headers=HEADERS, json={"confirmed": True}).status_code == 403
        assert (external / "source.mp4").exists()


def test_failed_purge_keeps_trash_entry_for_retry(tmp_path, monkeypatch):
    from lecture2doc import server
    folder = tmp_path / JOB
    record(folder)
    original = server.shutil.rmtree
    def fail(path):
        raise OSError("fixture failure")
    with TestClient(create_app(tmp_path)) as client:
        url = f"/api/jobs/{JOB}"
        client.delete(url, headers=HEADERS)
        monkeypatch.setattr(server.shutil, "rmtree", fail)
        assert client.post(url+"/purge", headers=HEADERS, json={"confirmed": True}).status_code == 500
        assert client.get("/api/trash").json()[0]["id"] == JOB
        monkeypatch.setattr(server.shutil, "rmtree", original)
        assert client.post(url+"/purge", headers=HEADERS, json={"confirmed": True}).status_code == 200
        assert not folder.exists()
