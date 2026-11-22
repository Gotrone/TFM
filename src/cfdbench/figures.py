"""Figures for the report and the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .dataset import CFDDataset  # noqa: E402
from .metrics import per_sample_rel_l2  # noqa: E402
from .preprocessing import velocity_magnitude  # noqa: E402


def _show(ax, field, mask, **kw):
    img = np.ma.masked_where(mask, field).T
    im = ax.imshow(img, origin="lower", **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def dataset_overview(ds: CFDDataset, out: Path) -> Path:
    """Mean velocity magnitude for every training Reynolds number."""
    re, fields = ds.subset("train")
    mag = velocity_magnitude(fields)
    fig, axes = plt.subplots(len(re), 1, figsize=(8, 1.5 * len(re)), constrained_layout=True)
    for ax, r, m in zip(np.atleast_1d(axes), re, mag, strict=True):
        im = _show(ax, m, ds.mask, cmap="viridis", vmin=0, vmax=1.6)
        ax.set_ylabel(f"Re={int(r)}", rotation=0, labelpad=30, va="center")
    fig.colorbar(im, ax=axes, shrink=0.6, label="|u| / u_in (time-averaged)")
    fig.suptitle("Training set: mean velocity magnitude behind the cylinder")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def model_comparison(ds: CFDDataset, preds: dict, model: str, split: str, index: int, out: Path) -> Path:
    """Truth / prediction / absolute error of one test case."""
    re, truth = ds.subset(split)
    pred = preds[model][split]
    t, p = velocity_magnitude(truth[index]), velocity_magnitude(pred[index])
    err = np.abs(t - p)
    fig, axes = plt.subplots(3, 1, figsize=(8, 5), constrained_layout=True)
    for ax, data, title, kw in zip(
        axes,
        (t, p, err),
        ("Ground truth (LBM)", f"Prediction ({model})", "Absolute error"),
        (dict(cmap="viridis", vmin=0, vmax=1.6), dict(cmap="viridis", vmin=0, vmax=1.6), dict(cmap="inferno")),
        strict=True,
    ):
        im = _show(ax, data, ds.mask, **kw)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.suptitle(f"{split}, Re = {int(re[index])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def error_vs_reynolds(ds: CFDDataset, preds: dict, out: Path) -> Path:
    """Relative L2 error of each model against Re, on all test points."""
    re_train, _ = ds.subset("train")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for model, by_split in preds.items():
        xs, ys = [], []
        for split, pred in by_split.items():
            re, truth = ds.subset(split)
            xs += list(re)
            ys += list(per_sample_rel_l2(pred, truth, ds.mask))
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=model)
    ax.axvspan(re_train.min(), re_train.max(), color="grey", alpha=0.12, label="training range")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel("relative L2 error")
    ax.set_title("Prediction error on the test Reynolds numbers")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
