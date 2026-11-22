import numpy as np
import pytest

from cfdbench.dataset import CFDDataset, checksum  # noqa: F401


def _field(re: float, nx: int = 24, ny: int = 12) -> np.ndarray:
    """Smooth synthetic velocity field depending on Re (divergence is not zero, that's fine)."""
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
    t = re / 100.0
    u = 1 - 0.5 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) * (5 + 3 * t))
    v = 0.2 * t * np.sin(2 * np.pi * x) * np.cos(np.pi * y)
    return np.stack([u, v]).astype(np.float32)


@pytest.fixture(scope="session")
def tiny_dataset() -> CFDDataset:
    splits = {"train": [40, 60, 80, 100, 120], "test_interp": [50, 70, 90], "test_extrap": [30, 140]}
    re, names = [], []
    for s, vals in splits.items():
        re += vals
        names += [s] * len(vals)
    fields = np.stack([_field(r) for r in re])
    mask = np.zeros((24, 12), dtype=bool)
    mask[10:13, 5:8] = True
    fields[:, :, mask] = 0.0
    return CFDDataset(
        reynolds=np.array(re, dtype=np.float32),
        fields=fields,
        inst_fields=fields.copy(),
        mask=mask,
        split=np.array(names),
        config={"cx": 10, "diameter": 3, "nx": 24, "ny": 12},
        checksum="test",
    )
