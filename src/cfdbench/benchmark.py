"""Benchmark runner: fit models on the train split, evaluate on the test splits."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from . import models
from .dataset import CFDDataset
from .metrics import evaluate

TEST_SPLITS = ("test_interp", "test_extrap")


def hidden_test_inputs(ds: CFDDataset) -> dict[str, np.ndarray]:
    """What a participant is given for the test splits: Reynolds numbers only."""
    return {s: ds.subset(s)[0] for s in TEST_SPLITS if s in ds.split_names()}


def run_model(ds: CFDDataset, name: str, **kwargs) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fit one registered model and evaluate it on every test split.

    Returns the leaderboard rows and the predictions (to inspect or plot them).
    """
    re_train, f_train = ds.subset("train")
    model = models.create(name, **kwargs)

    t0 = time.perf_counter()
    model.fit(re_train, f_train, ds.mask)
    fit_time = time.perf_counter() - t0

    rows, preds = [], {}
    for split in TEST_SPLITS:
        if split not in ds.split_names():
            continue
        re_test, truth = ds.subset(split)
        t0 = time.perf_counter()
        pred = model.predict(re_test)
        predict_time = time.perf_counter() - t0
        preds[split] = pred
        rows.append(
            {
                "model": name,
                "split": split,
                "metrics": evaluate(pred, truth, ds.mask, wake_start=int(ds.config["cx"] + ds.config["diameter"] / 2)),
                "fit_time_s": round(fit_time, 4),
                "predict_time_s": round(predict_time, 5),
                "n_test": int(len(re_test)),
                "dataset_version": ds.version,
                "dataset_checksum": ds.checksum,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return rows, preds


def run_benchmark(ds: CFDDataset, names: list[str] | None = None) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
    names = names or list(models.REGISTRY)
    all_rows, all_preds = [], {}
    for name in names:
        rows, preds = run_model(ds, name)
        all_rows += rows
        all_preds[name] = preds
    return all_rows, all_preds


def evaluate_submission(ds: CFDDataset, predictions: dict[str, np.ndarray]) -> list[dict]:
    """Score external predictions. ``predictions[split]`` has shape (N, 2, nx, ny)
    and follows the ascending Reynolds order of ``hidden_test_inputs``."""
    rows = []
    for split in TEST_SPLITS:
        if split not in ds.split_names():
            continue
        if split not in predictions:
            raise ValueError(f"missing predictions for split '{split}'")
        _, truth = ds.subset(split)
        pred = np.asarray(predictions[split], dtype=np.float32)
        rows.append(
            {
                "split": split,
                "metrics": evaluate(pred, truth, ds.mask, wake_start=int(ds.config["cx"] + ds.config["diameter"] / 2)),
                "n_test": int(len(truth)),
                "dataset_version": ds.version,
                "dataset_checksum": ds.checksum,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return rows
