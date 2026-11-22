"""Standard preprocessing tools: normalization, masking, resampling, image export."""

from __future__ import annotations

import numpy as np


class Standardizer:
    """Per-channel z-score normalization fitted on *fluid* cells of the training set.

    Solid cells (inside the obstacle) are excluded from the statistics and set to
    zero after the transform, so they never influence a model.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, fields: np.ndarray, mask: np.ndarray) -> Standardizer:
        fluid = ~mask
        values = fields[:, :, fluid]  # (N, C, n_fluid)
        self.mean = values.mean(axis=(0, 2))
        self.std = values.std(axis=(0, 2)) + 1e-8
        return self

    def transform(self, fields: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._check()
        out = (fields - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        out[:, :, mask] = 0.0
        return out.astype(np.float32)

    def inverse(self, fields: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._check()
        out = fields * self.std[None, :, None, None] + self.mean[None, :, None, None]
        out[:, :, mask] = 0.0
        return out.astype(np.float32)

    def _check(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer is not fitted")


def velocity_magnitude(fields: np.ndarray) -> np.ndarray:
    """Magnitude of a (..., 2, nx, ny) velocity array -> (..., nx, ny)."""
    return np.sqrt(fields[..., 0, :, :] ** 2 + fields[..., 1, :, :] ** 2)


def downsample(fields: np.ndarray, factor: int) -> np.ndarray:
    """Block-average the last two axes by an integer ``factor`` (crops the remainder)."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return fields
    nx, ny = fields.shape[-2:]
    nx2, ny2 = nx - nx % factor, ny - ny % factor
    cropped = fields[..., :nx2, :ny2]
    shape = cropped.shape[:-2] + (nx2 // factor, factor, ny2 // factor, factor)
    return cropped.reshape(shape).mean(axis=(-3, -1))


def crop(fields: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    """Crop a region of interest on the last two axes."""
    return fields[..., x0:x1, y0:y1]


def field_to_image(field2d: np.ndarray, mask: np.ndarray | None = None, vmin=None, vmax=None) -> np.ndarray:
    """Convert a scalar field (nx, ny) to an RGB uint8 image (ny, nx, 3), y pointing up.

    Image-based flow representation used by vision models; solid cells are painted grey.
    """
    from matplotlib import colormaps

    vmin = float(np.nanmin(field2d)) if vmin is None else vmin
    vmax = float(np.nanmax(field2d)) if vmax is None else vmax
    scaled = np.clip((field2d - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    rgb = (colormaps["viridis"](scaled)[..., :3] * 255).astype(np.uint8)
    if mask is not None:
        rgb[mask] = (128, 128, 128)
    return np.transpose(rgb, (1, 0, 2))[::-1]
