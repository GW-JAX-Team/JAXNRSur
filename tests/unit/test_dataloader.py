"""Unit tests for the surrogate data download in DataLoader.py (no network)."""

import h5py
import pytest
import requests

from jaxnrsur.DataLoader import load_data


class _FakeResponse:
    status_code = 200

    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error

    def iter_content(self, chunk_size):
        yield from self.chunks
        if self.error is not None:
            raise self.error


def test_broken_download_is_retried_on_next_load(tmp_path, monkeypatch):
    """A download that breaks midway must not leave a file that blocks the next load."""
    good = tmp_path / "good.h5"
    with h5py.File(good, "w") as f:
        f["x"] = [1, 2, 3]
    content = good.read_bytes()

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    broken = _FakeResponse([content[:100]], requests.exceptions.ChunkedEncodingError())
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: broken)
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        load_data("https://example.org/model.h5", "model.h5")

    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: _FakeResponse([content])
    )
    with load_data("https://example.org/model.h5", "model.h5") as f:
        assert f["x"][:].tolist() == [1, 2, 3]
