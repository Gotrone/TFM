"""Web platform: dataset information/download, submission and leaderboard API.

Run with ``cfdbench serve`` or ``uvicorn cfdbench.web.app:app``. Configuration (environment):

* ``CFDBENCH_DATA``: path of the dataset ``.npz`` (default ``data/cylinder_flow.npz``)
* ``CFDBENCH_DB``: path of the SQLite leaderboard (default ``results/leaderboard.db``)
* ``CFDBENCH_API_KEY``: if set, submissions must send it in the ``X-API-Key`` header
* ``CFDBENCH_RATE_LIMIT``: max submissions per client IP and hour (default 10)
"""

from __future__ import annotations

import io
import os
import re
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import leaderboard
from ..benchmark import TEST_SPLITS, evaluate_submission, hidden_test_inputs
from ..dataset import DEFAULT_PATH, CFDDataset, load_dataset
from ..metrics import METRICS, RANKING_METRIC

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
NAME_RE = re.compile(r"^[\w .\-+]{1,60}$")


def create_app(data_path: str | Path | None = None, db_path: str | Path | None = None) -> FastAPI:
    data_path = Path(data_path or os.environ.get("CFDBENCH_DATA", DEFAULT_PATH))
    db_path = Path(db_path or os.environ.get("CFDBENCH_DB", "results/leaderboard.db"))
    app = FastAPI(title="CFD Flow Prediction Benchmark", version="1.0")
    cache: dict[str, CFDDataset] = {}
    api_key = os.environ.get("CFDBENCH_API_KEY", "")
    rate_limit = int(os.environ.get("CFDBENCH_RATE_LIMIT", "10"))
    recent: dict[str, deque] = defaultdict(deque)  # client -> timestamps of recent submissions

    def dataset() -> CFDDataset:
        if "ds" not in cache:
            if not data_path.exists():
                raise HTTPException(503, f"dataset not found at {data_path}; run `cfdbench generate`")
            cache["ds"] = load_dataset(data_path)
        return cache["ds"]

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/dataset")
    def dataset_info():
        ds = dataset()
        info = ds.info()
        info["test_inputs"] = {s: [int(r) for r in v] for s, v in hidden_test_inputs(ds).items()}
        info["submission_format"] = (
            "npz file with one float array per test split ("
            + ", ".join(TEST_SPLITS)
            + "), shape (N, 2, nx, ny), samples in ascending Reynolds order"
        )
        return info

    @app.get("/api/dataset/train")
    def download_train():
        """Training split only: the test ground truth is never exposed."""
        ds = dataset()
        re_train, f_train = ds.subset("train")
        buf = io.BytesIO()
        np.savez_compressed(
            buf, reynolds=re_train, fields=f_train, mask=ds.mask, **{f"{k}_reynolds": v for k, v in hidden_test_inputs(ds).items()}
        )
        return Response(
            buf.getvalue(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=cylinder_flow_train.npz"},
        )

    @app.get("/api/leaderboard")
    def get_leaderboard(split: str = "test_interp", metric: str = RANKING_METRIC):
        if split not in TEST_SPLITS:
            raise HTTPException(422, f"split must be one of {TEST_SPLITS}")
        if metric not in METRICS:
            raise HTTPException(422, f"metric must be one of {METRICS}")
        return {"split": split, "metric": metric, "entries": leaderboard.get_leaderboard(db_path, split, metric)}

    @app.post("/api/submit")
    async def submit(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(...),
        author: str = Form(""),
        x_api_key: str = Header(""),
    ):
        if api_key and not secrets.compare_digest(x_api_key, api_key):
            raise HTTPException(401, "missing or invalid API key (X-API-Key header)")
        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = recent[client]
        while window and now - window[0] > 3600:
            window.popleft()
        if len(window) >= rate_limit:
            raise HTTPException(429, f"rate limit: {rate_limit} submissions per hour")
        if not NAME_RE.match(model):
            raise HTTPException(422, "model name: 1-60 characters (letters, digits, space, . _ - +)")
        if author and not NAME_RE.match(author):
            raise HTTPException(422, "author: 1-60 characters (letters, digits, space, . _ - +)")
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large (max 25 MB)")
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                preds = {k: z[k] for k in TEST_SPLITS if k in z.files}
        except Exception as exc:  # not a valid npz
            raise HTTPException(422, f"cannot read the npz file: {exc}") from exc
        try:
            rows = evaluate_submission(dataset(), preds)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        for r in rows:
            r["model"] = model
        leaderboard.add_entries(db_path, rows, author=author, source="submission")
        window.append(now)
        return {"model": model, "results": {r["split"]: r["metrics"] for r in rows}}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
