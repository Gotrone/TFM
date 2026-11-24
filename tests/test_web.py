import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cfdbench import leaderboard
from cfdbench.benchmark import run_benchmark
from cfdbench.web.app import create_app


@pytest.fixture()
def client(tiny_dataset, tmp_path, monkeypatch):
    import cfdbench.web.app as web

    monkeypatch.setattr(web, "load_dataset", lambda path: tiny_dataset)
    data = tmp_path / "data.npz"
    data.write_bytes(b"x")  # only needs to exist: loading is patched
    db = tmp_path / "lb.db"
    rows, _ = run_benchmark(tiny_dataset, ["mean", "linear"])
    leaderboard.add_entries(db, rows)
    return TestClient(create_app(data, db))


def _npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_health_and_index(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert "CFD Flow Prediction Benchmark" in client.get("/").text


def test_dataset_info_hides_test_truth(client):
    info = client.get("/api/dataset").json()
    assert info["splits"]["train"] == [40, 60, 80, 100, 120]
    assert info["test_inputs"]["test_interp"] == [50, 70, 90]
    train = np.load(io.BytesIO(client.get("/api/dataset/train").content))
    assert set(train.files) == {"reynolds", "fields", "mask", "test_interp_reynolds", "test_extrap_reynolds"}
    assert len(train["reynolds"]) == 5


def test_leaderboard_sorted_and_validated(client):
    d = client.get("/api/leaderboard?split=test_interp&metric=rel_l2").json()
    errs = [e["metrics"]["rel_l2"] for e in d["entries"]]
    assert errs == sorted(errs) and [e["rank"] for e in d["entries"]] == [1, 2]
    assert client.get("/api/leaderboard?split=train").status_code == 422
    assert client.get("/api/leaderboard?metric=bogus").status_code == 422


def test_submit_perfect_predictions_ranks_first(client, tiny_dataset):
    _, a = tiny_dataset.subset("test_interp")
    _, b = tiny_dataset.subset("test_extrap")
    r = client.post("/api/submit", data={"model": "oracle", "author": "me"}, files={"file": ("s.npz", _npz(test_interp=a, test_extrap=b))})
    assert r.status_code == 200 and r.json()["results"]["test_interp"]["rel_l2"] == 0
    top = client.get("/api/leaderboard?split=test_interp").json()["entries"][0]
    assert top["model"] == "oracle" and top["source"] == "submission"


def test_submit_rejects_bad_files(client, tiny_dataset):
    _, a = tiny_dataset.subset("test_interp")
    bad_name = client.post("/api/submit", data={"model": "<script>"}, files={"file": ("s.npz", _npz(test_interp=a))})
    assert bad_name.status_code == 422
    not_npz = client.post("/api/submit", data={"model": "m"}, files={"file": ("s.npz", b"garbage")})
    assert not_npz.status_code == 422
    missing = client.post("/api/submit", data={"model": "m"}, files={"file": ("s.npz", _npz(test_interp=a))})
    assert missing.status_code == 422
    wrong_shape = client.post("/api/submit", data={"model": "m"}, files={"file": ("s.npz", _npz(test_interp=a[:1], test_extrap=a[:1]))})
    assert wrong_shape.status_code == 422
    nan = a.copy()
    nan[0, 0, 0, 0] = np.nan
    r = client.post("/api/submit", data={"model": "m"}, files={"file": ("s.npz", _npz(test_interp=nan, test_extrap=nan))})
    assert r.status_code == 422


def test_api_key_required_when_configured(tiny_dataset, tmp_path, monkeypatch):
    import cfdbench.web.app as web

    monkeypatch.setattr(web, "load_dataset", lambda path: tiny_dataset)
    monkeypatch.setenv("CFDBENCH_API_KEY", "s3cret")
    (tmp_path / "d.npz").write_bytes(b"x")
    c = TestClient(create_app(tmp_path / "d.npz", tmp_path / "lb.db"))
    _, a = tiny_dataset.subset("test_interp")
    _, b = tiny_dataset.subset("test_extrap")
    files = {"file": ("s.npz", _npz(test_interp=a, test_extrap=b))}
    assert c.post("/api/submit", data={"model": "m"}, files=files).status_code == 401
    assert c.post("/api/submit", data={"model": "m"}, files=files, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/api/submit", data={"model": "m"}, files=files, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_rate_limit(tiny_dataset, tmp_path, monkeypatch):
    import cfdbench.web.app as web

    monkeypatch.setattr(web, "load_dataset", lambda path: tiny_dataset)
    monkeypatch.setenv("CFDBENCH_RATE_LIMIT", "2")
    (tmp_path / "d.npz").write_bytes(b"x")
    c = TestClient(create_app(tmp_path / "d.npz", tmp_path / "lb.db"))
    _, a = tiny_dataset.subset("test_interp")
    _, b = tiny_dataset.subset("test_extrap")
    files = {"file": ("s.npz", _npz(test_interp=a, test_extrap=b))}
    codes = [c.post("/api/submit", data={"model": f"m{i}"}, files=files).status_code for i in range(3)]
    assert codes == [200, 200, 429]
