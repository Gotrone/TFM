"""Dataset generation, storage and loading.

Samples are time-averaged velocity fields ``(u, v)`` of a 2D flow past a circular
cylinder, one per Reynolds number. The task is: *given Re, predict the mean flow
field*. Splits are defined on Re so that interpolation and extrapolation can be
evaluated separately.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lbm import CylinderConfig, cylinder_mask, simulate

DATASET_NAME = "cylinder-flow-lbm"
DATASET_VERSION = "2.0"

#: Reynolds numbers of each split.
SPLITS: dict[str, list[int]] = {
    "train": [40, 60, 80, 100, 120, 140],
    "test_interp": [45, 50, 55, 65, 70, 75, 85, 90, 95, 105, 110, 115, 125, 130, 135],  # between training points
    "test_extrap": [30, 35, 150, 160, 170, 180],  # outside the training range
}

DEFAULT_PATH = Path("data") / "cylinder_flow.npz"


@dataclass
class CFDDataset:
    """In-memory dataset. ``fields`` has shape (N, 2, nx, ny): channels are (u, v)."""

    reynolds: np.ndarray
    fields: np.ndarray
    inst_fields: np.ndarray
    mask: np.ndarray  # (nx, ny), True inside the cylinder
    split: np.ndarray  # (N,) split name of every sample
    config: dict
    version: str = DATASET_VERSION
    checksum: str = ""

    def subset(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(reynolds, fields)`` of a split, sorted by Reynolds number."""
        idx = np.where(self.split == name)[0]
        idx = idx[np.argsort(self.reynolds[idx])]
        return self.reynolds[idx], self.fields[idx]

    def split_names(self) -> list[str]:
        return [s for s in SPLITS if s in set(self.split.tolist())]

    def info(self) -> dict:
        return {
            "name": DATASET_NAME,
            "version": self.version,
            "checksum": self.checksum,
            "grid": [int(self.fields.shape[2]), int(self.fields.shape[3])],
            "channels": ["u", "v"],
            "n_samples": int(len(self.reynolds)),
            "splits": {s: sorted(int(r) for r in self.reynolds[self.split == s]) for s in self.split_names()},
            "simulation": self.config,
        }


def case_list(splits: dict[str, list[int]] | None = None) -> list[tuple[int, str]]:
    """All (Reynolds number, split) pairs in a fixed order (used to index SLURM array jobs)."""
    return [(re, name) for name, res in (splits or SPLITS).items() for re in res]


def _run_case(args: tuple) -> tuple[int, dict]:
    reynolds, cfg_dict, *opts = args
    backend, dtype = opts if opts else ("cpu", "float64")
    return reynolds, simulate(reynolds, CylinderConfig(**cfg_dict), seed=reynolds, backend=backend, dtype=dtype)


def _write_dataset(path: str | Path, cfg: CylinderConfig, cases: list[tuple[int, str]], results: dict) -> Path:
    reynolds = np.array([re for re, _ in cases], dtype=np.float32)
    split = np.array([name for _, name in cases])
    fields = np.stack([[results[re]["u_mean"], results[re]["v_mean"]] for re, _ in cases]).astype(np.float32)
    inst = np.stack([[results[re]["u_inst"], results[re]["v_inst"]] for re, _ in cases]).astype(np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        reynolds=reynolds,
        fields=fields,
        inst_fields=inst,
        mask=cylinder_mask(cfg),
        split=split,
        config=json.dumps(cfg.__dict__.copy()),
        version=DATASET_VERSION,
    )
    return path


def generate(
    path: str | Path = DEFAULT_PATH,
    config: CylinderConfig | None = None,
    workers: int | None = None,
    splits: dict[str, list[int]] | None = None,
    backend: str = "cpu",
    dtype: str = "float64",
) -> Path:
    """Run every simulation and write the dataset file (compressed ``.npz``).

    ``workers`` processes run on the CPU. With ``backend="gpu"`` the cases run one after the
    other on the GPU (a single process owns the device).
    """
    cfg = config or CylinderConfig()
    cases = case_list(splits)
    cfg_dict = cfg.__dict__.copy()
    jobs = [(re, cfg_dict, backend, dtype) for re, _ in cases]
    if workers == 1 or backend == "gpu":
        results = dict(map(_run_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = dict(pool.map(_run_case, jobs))
    return _write_dataset(path, cfg, cases, results)


def generate_case(index: int, out_dir: str | Path, config: CylinderConfig | None = None, backend: str = "cpu", dtype: str = "float64") -> Path:
    """Simulate one case of :func:`case_list` and save it (for SLURM job arrays)."""
    cfg = config or CylinderConfig()
    cases = case_list()
    if not 0 <= index < len(cases):
        raise IndexError(f"case index {index} out of range (0..{len(cases) - 1})")
    re, split = cases[index]
    _, out = _run_case((re, cfg.__dict__.copy(), backend, dtype))
    path = Path(out_dir) / f"case_re{re:03d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, split=split, reynolds=re, config=json.dumps(cfg.__dict__.copy()), **out)
    return path


def assemble(case_dir: str | Path, path: str | Path = DEFAULT_PATH) -> Path:
    """Combine the files written by :func:`generate_case` into the final dataset."""
    cases = case_list()
    results, cfg_json = {}, None
    for re, _ in cases:
        f = Path(case_dir) / f"case_re{re:03d}.npz"
        if not f.exists():
            raise FileNotFoundError(f"missing case file {f} (job failed or still running?)")
        with np.load(f, allow_pickle=False) as z:
            results[re] = {k: z[k] for k in ("u_mean", "v_mean", "u_inst", "v_inst")}
            cfg_json = json.loads(str(z["config"]))
    return _write_dataset(path, CylinderConfig(**cfg_json), cases, results)


def checksum(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_dataset(path: str | Path = DEFAULT_PATH) -> CFDDataset:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        return CFDDataset(
            reynolds=z["reynolds"],
            fields=z["fields"],
            inst_fields=z["inst_fields"],
            mask=z["mask"],
            split=z["split"],
            config=json.loads(str(z["config"])),
            version=str(z["version"]),
            checksum=checksum(path),
        )
