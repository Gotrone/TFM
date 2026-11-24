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


def _run_case(args: tuple[int, dict]) -> tuple[int, dict]:
    reynolds, cfg_dict = args
    return reynolds, simulate(reynolds, CylinderConfig(**cfg_dict), seed=reynolds)


def generate(
    path: str | Path = DEFAULT_PATH,
    config: CylinderConfig | None = None,
    workers: int | None = None,
    splits: dict[str, list[int]] | None = None,
) -> Path:
    """Run every simulation and write the dataset file (compressed ``.npz``)."""
    cfg = config or CylinderConfig()
    splits = splits or SPLITS
    cases = [(re, name) for name, res in splits.items() for re in res]
    cfg_dict = cfg.__dict__.copy()

    jobs = [(re, cfg_dict) for re, _ in cases]
    if workers == 1:
        results = dict(map(_run_case, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = dict(pool.map(_run_case, jobs))

    reynolds = np.array([re for re, _ in cases], dtype=np.float32)
    split = np.array([name for _, name in cases])
    fields = np.stack([[results[re]["u_mean"], results[re]["v_mean"]] for re, _ in cases]).astype(np.float32)
    inst = np.stack([[results[re]["u_inst"], results[re]["v_inst"]] for re, _ in cases]).astype(np.float32)
    mask = cylinder_mask(cfg)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        reynolds=reynolds,
        fields=fields,
        inst_fields=inst,
        mask=mask,
        split=split,
        config=json.dumps(cfg_dict),
        version=DATASET_VERSION,
    )
    return path


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
