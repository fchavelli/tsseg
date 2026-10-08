"""Tests of the reproduction download helper and of the marker setup.

These tests are fast and offline (a local HTTP server stands in for the
dataset source), so they run in CI, unlike the reproduction tests themselves.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import threading
import zipfile
from pathlib import Path

import pytest

from ._data import REGISTRY, DataFile, cache_dir, fetch

REF = "Test fixture"
LICENSE = "CC0-1.0"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class _LocalServer:
    """Serve ``root`` over HTTP on 127.0.0.1, counting the requests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests = 0
        server = self

        class Handler(_QuietHandler):
            def do_GET(self):  # noqa: N802 - name required by the base class
                server.requests += 1
                super().do_GET()

        self._httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(Handler, directory=str(root))
        )
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread.is_alive():
            self._httpd.shutdown()
            self._thread.join()
        self._httpd.server_close()


@pytest.fixture
def server(tmp_path, monkeypatch):
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    root = tmp_path / "served"
    root.mkdir()
    srv = _LocalServer(root)
    yield srv
    srv.stop()


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    path = tmp_path / "data"
    monkeypatch.setenv("TSSEG_DATA_DIR", str(path))
    return path / "reproduction"


def _publish(server, name: str, content: bytes) -> str:
    (server.root / name).write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _entry(server, name: str, sha256: str, **kwargs) -> DataFile:
    return DataFile(
        url=f"{server.url}/{name}",
        sha256=sha256,
        reference=REF,
        license=LICENSE,
        **kwargs,
    )


def test_cache_dir_follows_tsseg_data_dir(cache):
    assert cache_dir() == cache


def test_fetch_downloads_and_checks_digest(server, cache):
    content = b"t,x\n0,1.0\n1,2.0\n"
    sha = _publish(server, "series.csv", content)

    path = fetch(_entry(server, "series.csv", sha, fname="series.csv"))

    assert path == cache / "series.csv"
    assert path.read_bytes() == content
    assert server.requests == 1


def test_fetch_reuses_cache_without_network(server):
    sha = _publish(server, "series.csv", b"0,1,2\n")
    entry = _entry(server, "series.csv", sha)
    first = fetch(entry)
    server.stop()

    assert fetch(entry) == first
    assert server.requests == 1


def test_fetch_rejects_wrong_digest(server, cache):
    _publish(server, "series.csv", b"0,1,2\n")
    wrong = hashlib.sha256(b"something else").hexdigest()

    with pytest.raises(ValueError, match="(?i)hash"):
        fetch(_entry(server, "series.csv", wrong, fname="series.csv"))
    assert not (cache / "series.csv").exists()


def test_fetch_downloads_again_a_corrupted_cache(server):
    content = b"0,1,2\n"
    sha = _publish(server, "series.csv", content)
    entry = _entry(server, "series.csv", sha)
    path = fetch(entry)
    path.write_bytes(b"corrupted")

    assert fetch(entry).read_bytes() == content
    assert server.requests == 2


def test_fetch_unpacks_zip_archive(server, tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("bundle/a.csv", "1,2\n")
        zf.writestr("bundle/b.csv", "3,4\n")
    sha = _publish(server, "bundle.zip", archive.read_bytes())

    paths = fetch(_entry(server, "bundle.zip", sha, unpack=True))

    assert [p.name for p in paths] == ["a.csv", "b.csv"]
    assert [p.read_text() for p in paths] == ["1,2\n", "3,4\n"]


def test_fetch_skips_without_network(server):
    sha = _publish(server, "series.csv", b"0,1,2\n")
    entry = _entry(server, "series.csv", sha)
    server.stop()

    with pytest.raises(pytest.skip.Exception, match="no network"):
        fetch(entry)


def test_fetch_fails_on_http_error(server):
    requests = pytest.importorskip("requests")
    entry = _entry(server, "missing.csv", "0" * 64)

    with pytest.raises(requests.HTTPError):
        fetch(entry)


def test_fetch_unknown_name():
    with pytest.raises(KeyError, match="Unknown reproduction dataset"):
        fetch("no-such-dataset")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"sha256": "abc"}, "sha256"),
        ({"sha256": "A" * 64}, "sha256"),
        ({"url": "ftp://example.org/x.csv"}, "http"),
        ({"reference": " "}, "reference"),
        ({"license": ""}, "license"),
        ({"unpack": True}, "Cannot unpack"),
    ],
)
def test_data_file_validation(kwargs, match):
    fields = {
        "url": "https://example.org/x.csv",
        "sha256": "0" * 64,
        "reference": REF,
        "license": LICENSE,
    }
    fields.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DataFile(**fields)


def test_registry_entries_are_data_files():
    for name, entry in REGISTRY.items():
        assert isinstance(entry, DataFile), name
        assert entry.url.startswith("https://"), name


def test_reproduction_marker_is_declared_and_deselected_by_default(pytestconfig):
    markers = [line.split(":", 1)[0] for line in pytestconfig.getini("markers")]
    assert "reproduction" in markers

    addopts = pytestconfig.getini("addopts")
    assert addopts[addopts.index("-m") + 1] == "not reproduction"
