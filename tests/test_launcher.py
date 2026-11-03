from fastapi.testclient import TestClient

from lecture2doc.launcher import PROJECT, same_server
from lecture2doc.server import code_revision, create_app
import hashlib
from pathlib import Path


def test_running_server_identity_and_fresh_assets(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        info = client.get("/api/app").json()
        assert info["code_revision"] == code_revision()
        assert same_server(info, tmp_path, None)
        assert not same_server(info, tmp_path / "other-data", None)
        assert not same_server({**info, "code_revision": "old-code"}, tmp_path, None)
        assert not same_server({**info, "project": str(PROJECT / "other")}, tmp_path, None)
        for path in ["/", "/static/app.js", "/static/style.css", "/api/jobs", "/api/trash"]:
            assert client.get(path).headers["cache-control"] == "no-store"


def test_changed_script_has_new_url_even_without_server_restart(tmp_path, monkeypatch):
    from lecture2doc import server
    package = tmp_path / "package"
    static = package / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text('<script src="/static/app.js"></script><link href="/static/style.css">')
    (static / "app.js").write_text('old-script')
    (static / "style.css").write_text('body{}')
    monkeypatch.setattr(server, "__file__", str(package / "server.py"))
    with TestClient(server.create_app(tmp_path / "data")) as client:
        old = hashlib.sha256(b'old-script').hexdigest()[:16]
        assert f'/static/app.js?v={old}' in client.get('/').text
        (static / "app.js").write_text('new-delete-handlers')
        new = hashlib.sha256(b'new-delete-handlers').hexdigest()[:16]
        html = client.get('/').text
        assert f'/static/app.js?v={new}' in html and old not in html
        assert client.get(f'/static/app.js?v={new}').text == 'new-delete-handlers'
