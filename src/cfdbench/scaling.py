"""Performance studies of the data generation (the compute-heavy part of the suite).

* :func:`process_scaling`: strong scaling of the dataset generation with the number of
  worker processes (CPU, one simulation per task).
* :func:`grid_scaling`: cost of one solver step on the CPU and on the GPU as the grid grows,
  in million lattice updates per second (MLUPS).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from .dataset import _run_case
from .lbm import CylinderConfig, simulate


def process_scaling(workers=(1, 2, 4, 8, 16), n_cases: int = 16, steps: int = 1000) -> list[dict]:
    """Time the same batch of ``n_cases`` short simulations with 1, 2, 4... worker processes."""
    cfg = CylinderConfig(warmup_steps=int(steps * 0.8), average_steps=int(steps * 0.2))
    jobs = [(int(re), cfg.__dict__.copy()) for re in np.linspace(40, 180, n_cases)]
    rows = []
    for w in workers:
        t0 = time.perf_counter()
        if w == 1:
            list(map(_run_case, jobs))
        else:
            with ProcessPoolExecutor(max_workers=w) as pool:
                list(pool.map(_run_case, jobs))
        rows.append({"workers": w, "seconds": time.perf_counter() - t0})
    base = rows[0]["seconds"]
    for r in rows:
        r["speedup"] = base / r["seconds"]
        r["efficiency"] = r["speedup"] / r["workers"]
        r["seconds"] = round(r["seconds"], 2)
        r["speedup"] = round(r["speedup"], 2)
        r["efficiency"] = round(r["efficiency"], 3)
    return rows


def _time_steps(cfg: CylinderConfig, backend: str, dtype: str, steps: int) -> float:
    """Seconds per step, measured after a short warm-up (kernel compilation, caches)."""
    simulate(100, replace(cfg, warmup_steps=3, average_steps=2), backend=backend, dtype=dtype)
    c = replace(cfg, warmup_steps=steps, average_steps=1)
    t0 = time.perf_counter()
    simulate(100, c, backend=backend, dtype=dtype)  # simulate() ends with a host copy -> GPU is synchronized
    return (time.perf_counter() - t0) / (steps + 1)


def grid_scaling(factors=(1, 2, 4, 8, 16), steps: int = 30, backends=(("cpu", "float64"), ("gpu", "float64"), ("gpu", "float32"))) -> list[dict]:
    """Seconds per step and MLUPS for grids ``factor`` times larger than the dataset grid (in x and y)."""
    base = CylinderConfig()
    rows = []
    for k in factors:
        cfg = replace(base, nx=base.nx * k, ny=base.ny * k, diameter=base.diameter * k, cx=base.cx * k)
        for backend, dtype in backends:
            try:
                sec = _time_steps(cfg, backend, dtype, steps)
            except Exception as exc:  # e.g. no GPU or out of memory
                rows.append({"factor": k, "cells": cfg.nx * cfg.ny, "backend": backend, "dtype": dtype, "error": str(exc)[:80]})
                continue
            rows.append(
                {
                    "factor": k,
                    "grid": f"{cfg.nx}x{cfg.ny}",
                    "cells": cfg.nx * cfg.ny,
                    "backend": backend,
                    "dtype": dtype,
                    "ms_per_step": round(sec * 1000, 3),
                    "mlups": round(cfg.nx * cfg.ny / sec / 1e6, 2),
                }
            )
    return rows


def plot_process_scaling(rows: list[dict], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = [r["workers"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax[0].plot(w, [r["speedup"] for r in rows], "o-", label="measured")
    ax[0].plot(w, w, "k--", alpha=0.5, label="ideal")
    ax[0].axvline(8, color="grey", alpha=0.3)
    ax[0].set(xlabel="worker processes", ylabel="speedup", xscale="log", yscale="log", title="Strong scaling (CPU)")
    ax[0].set_xticks(w, [str(x) for x in w])
    ax[0].legend()
    ax[1].plot(w, [r["efficiency"] for r in rows], "o-")
    ax[1].set(xlabel="worker processes", ylabel="parallel efficiency", ylim=(0, 1.1), xscale="log", title="Efficiency = speedup / workers")
    ax[1].set_xticks(w, [str(x) for x in w])
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Dataset generation, 16 short simulations (8 physical cores, 16 threads)", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_grid_scaling(rows: list[dict], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for backend, dtype in sorted({(r["backend"], r["dtype"]) for r in rows}):
        pts = [r for r in rows if r["backend"] == backend and r["dtype"] == dtype and "mlups" in r]
        ax.plot([r["cells"] for r in pts], [r["mlups"] for r in pts], "o-", label=f"{backend.upper()} {dtype}")
    ax.set(xscale="log", yscale="log", xlabel="grid cells", ylabel="MLUPS (higher is better)", title="Solver throughput vs grid size")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
