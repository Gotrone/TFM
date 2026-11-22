"""Post-processing: evaluation metrics for predicted velocity fields.

All metrics are computed on fluid cells only (the obstacle is excluded) and
averaged over the samples of a split. ``pred`` and ``truth`` have shape
(N, 2, nx, ny) with channels (u, v), in units of the inlet velocity.
"""

from __future__ import annotations

import numpy as np

METRICS = ("mse", "mae", "rel_l2", "max_abs", "wake_rel_l2", "div_rms")

#: Metric used to rank models on the leaderboard (lower is better).
RANKING_METRIC = "rel_l2"


def _check(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> None:
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape} vs truth {truth.shape}")
    if truth.ndim != 4 or truth.shape[1] != 2:
        raise ValueError("fields must have shape (N, 2, nx, ny)")
    if mask.shape != truth.shape[2:]:
        raise ValueError("mask shape does not match the fields")
    if not np.isfinite(pred).all():
        raise ValueError("prediction contains NaN or infinite values")


def divergence(fields: np.ndarray) -> np.ndarray:
    """Central-difference divergence du/dx + dv/dy, shape (N, nx-2, ny-2)."""
    u, v = fields[:, 0], fields[:, 1]
    dudx = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / 2
    dvdy = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / 2
    return dudx + dvdy


def evaluate(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, wake_start: int | None = None) -> dict:
    """Compute every metric. Returns plain floats, averaged over the samples.

    * ``mse`` / ``mae``: mean squared / absolute error on both velocity components.
    * ``rel_l2``: ||pred - truth|| / ||truth|| (per sample), the ranking metric.
    * ``max_abs``: largest absolute error of a cell.
    * ``wake_rel_l2``: same as ``rel_l2`` but only downstream of ``wake_start`` (the wake).
    * ``div_rms``: RMS divergence of the prediction on fluid cells away from the obstacle:
      a physical-consistency check (the mean flow of an incompressible fluid is divergence-free).
    """
    _check(pred, truth, mask)
    fluid = ~mask
    diff = (pred - truth)[:, :, fluid]  # (N, 2, n_fluid)
    ref = truth[:, :, fluid]

    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rel = np.linalg.norm(diff.reshape(len(diff), -1), axis=1) / np.linalg.norm(ref.reshape(len(ref), -1), axis=1)
    max_abs = float(np.abs(diff).max())

    if wake_start is None:
        wake_start = mask.shape[0] // 2
    wake = fluid.copy()
    wake[:wake_start] = False
    wd = (pred - truth)[:, :, wake]
    wr = truth[:, :, wake]
    wake_rel = np.linalg.norm(wd.reshape(len(wd), -1), axis=1) / np.linalg.norm(wr.reshape(len(wr), -1), axis=1)

    # divergence on cells whose 3x3 neighbourhood is fully fluid
    inner = fluid.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            inner &= np.roll(np.roll(fluid, dx, axis=0), dy, axis=1)
    inner_c = inner[1:-1, 1:-1]
    div_rms = float(np.sqrt(np.mean(divergence(pred)[:, inner_c] ** 2)))

    return {
        "mse": mse,
        "mae": mae,
        "rel_l2": float(rel.mean()),
        "max_abs": max_abs,
        "wake_rel_l2": float(wake_rel.mean()),
        "div_rms": div_rms,
    }


def per_sample_rel_l2(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Relative L2 error of every sample (used for error-vs-Re plots)."""
    _check(pred, truth, mask)
    fluid = ~mask
    diff = (pred - truth)[:, :, fluid].reshape(len(pred), -1)
    ref = truth[:, :, fluid].reshape(len(truth), -1)
    return np.linalg.norm(diff, axis=1) / np.linalg.norm(ref, axis=1)
