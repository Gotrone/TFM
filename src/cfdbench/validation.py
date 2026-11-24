"""Solver validation: vortex-shedding frequency (Strouhal number) against the literature.

The reference is the Williamson and Brown (1998) series fit for an unconfined cylinder,
``St = 0.2665 - 1.018 / sqrt(Re)`` (valid for roughly 49 < Re < 180). Our channel has
periodic top and bottom boundaries and therefore a blockage ``B = D / ny``, which raises
St, so a wider domain is also run to show the effect of the blockage.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from .lbm import CylinderConfig, simulate


def williamson_strouhal(reynolds: float) -> float:
    return 0.2665 - 1.018 / np.sqrt(reynolds)


def dominant_frequency(signal: np.ndarray) -> float:
    """Dominant frequency (cycles per step) of a signal: zero-padded FFT + parabolic peak refinement."""
    s = signal - signal.mean()
    n = 1 << int(np.ceil(np.log2(len(s) * 8)))
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
    k = int(np.argmax(spec[1:])) + 1
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return (k + shift) / n


def _measure(args: tuple[float, dict]) -> dict:
    re, cfg_dict = args
    cfg = CylinderConfig(**cfg_dict)
    probe = (cfg.cx + 3 * cfg.diameter, cfg.cy + cfg.diameter // 2)
    out = simulate(re, cfg, seed=int(re), probe=probe)
    st = dominant_frequency(out["probe_v"]) * cfg.diameter / cfg.u_in
    return {"reynolds": re, "ny": cfg.ny, "blockage": round(cfg.diameter / cfg.ny, 3), "strouhal": float(st), "reference": float(williamson_strouhal(re))}


def validate(reynolds=(100, 140), workers: int | None = None) -> list[dict]:
    """Measure St for the dataset geometry and for a domain twice as wide (half the blockage)."""
    base = CylinderConfig(warmup_steps=10000, average_steps=12000)
    wide = replace(base, ny=2 * base.ny)
    jobs = [(re, cfg.__dict__.copy()) for cfg in (base, wide) for re in reynolds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_measure, jobs))
    for r in rows:
        r["rel_error"] = round(abs(r["strouhal"] - r["reference"]) / r["reference"], 3)
    return rows
