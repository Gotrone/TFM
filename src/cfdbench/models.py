"""Baseline prediction models and the model registry.

A model maps a Reynolds number to a velocity field. To add your own, subclass
:class:`Model`, implement ``fit`` and ``predict`` and decorate it with
``@register("my-model")``.
"""

from __future__ import annotations

import numpy as np

from .preprocessing import Standardizer

REGISTRY: dict[str, type[Model]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


def create(name: str, **kwargs) -> Model:
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}', available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


class Model:
    """Interface: ``fit(reynolds, fields, mask)`` then ``predict(reynolds) -> fields``."""

    name = "model"

    def fit(self, reynolds: np.ndarray, fields: np.ndarray, mask: np.ndarray) -> Model:
        raise NotImplementedError

    def predict(self, reynolds: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@register("freestream")
class FreeStream(Model):
    """Trivial baseline: uniform flow u = 1, v = 0 (no obstacle effect at all)."""

    def fit(self, reynolds, fields, mask):
        self.shape = fields.shape[1:]
        self.mask = mask
        return self

    def predict(self, reynolds):
        out = np.zeros((len(reynolds),) + self.shape, dtype=np.float32)
        out[:, 0] = 1.0
        out[:, :, self.mask] = 0.0
        return out


@register("mean")
class MeanField(Model):
    """Average training field, whatever the Reynolds number."""

    def fit(self, reynolds, fields, mask):
        self.mean = fields.mean(axis=0)
        return self

    def predict(self, reynolds):
        return np.repeat(self.mean[None], len(reynolds), axis=0)


@register("nearest")
class NearestRe(Model):
    """Field of the closest training Reynolds number."""

    def fit(self, reynolds, fields, mask):
        self.re = np.asarray(reynolds, dtype=float)
        self.fields = fields
        return self

    def predict(self, reynolds):
        idx = [int(np.argmin(np.abs(self.re - r))) for r in np.asarray(reynolds)]
        return self.fields[idx].copy()


@register("linear")
class LinearInterp(Model):
    """Pixel-wise linear interpolation between training Reynolds numbers (clamped outside)."""

    def fit(self, reynolds, fields, mask):
        order = np.argsort(reynolds)
        self.re = np.asarray(reynolds, dtype=float)[order]
        self.fields = fields[order]
        return self

    def predict(self, reynolds):
        out = np.empty((len(reynolds),) + self.fields.shape[1:], dtype=np.float32)
        for k, r in enumerate(np.asarray(reynolds, dtype=float)):
            r = float(np.clip(r, self.re[0], self.re[-1]))
            j = int(np.searchsorted(self.re, r, side="right"))
            j = min(max(j, 1), len(self.re) - 1)
            t = (r - self.re[j - 1]) / (self.re[j] - self.re[j - 1])
            out[k] = (1 - t) * self.fields[j - 1] + t * self.fields[j]
        return out


class _PODModel(Model):
    """Reduced-order model: POD basis of the training fields + a regressor on Re.

    Steps: standardize, SVD of the snapshot matrix, keep ``n_modes`` modes, regress
    the modal coefficients on Re, reconstruct and de-standardize.
    """

    def __init__(self, n_modes: int = 5) -> None:
        self.n_modes = n_modes

    def _make_regressor(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def fit(self, reynolds, fields, mask):
        self.mask = mask
        self.scaler = Standardizer().fit(fields, mask)
        z = self.scaler.transform(fields, mask)
        flat = z.reshape(len(z), -1)
        self.mean_ = flat.mean(axis=0)
        _, _, vt = np.linalg.svd(flat - self.mean_, full_matrices=False)
        k = min(self.n_modes, len(flat) - 1)
        self.basis = vt[:k]  # (k, features)
        coeffs = (flat - self.mean_) @ self.basis.T
        self.re_mean = float(np.mean(reynolds))
        self.re_std = float(np.std(reynolds)) + 1e-8
        x = self._scale(reynolds)
        self.regressors = [self._make_regressor(x, coeffs[:, j]) for j in range(k)]
        self.shape = fields.shape[1:]
        return self

    def _scale(self, reynolds):
        return ((np.asarray(reynolds, dtype=float) - self.re_mean) / self.re_std)[:, None]

    def predict(self, reynolds):
        x = self._scale(reynolds)
        coeffs = np.stack([reg.predict(x) for reg in self.regressors], axis=1)
        flat = coeffs @ self.basis + self.mean_
        z = flat.reshape((len(x),) + self.shape)
        return self.scaler.inverse(z, self.mask)


@register("pod-poly")
class PODPoly(_PODModel):
    """POD + polynomial regression of the modal coefficients on Re."""

    def __init__(self, n_modes: int = 5, degree: int = 3) -> None:
        super().__init__(n_modes)
        self.degree = degree

    def _make_regressor(self, x, y):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures

        deg = min(self.degree, len(x) - 1)
        return make_pipeline(PolynomialFeatures(deg), Ridge(alpha=1e-6)).fit(x, y)


@register("pod-gp")
class PODGaussianProcess(_PODModel):
    """POD + Gaussian-process regression of the modal coefficients on Re."""

    def _make_regressor(self, x, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-6, (1e-10, 1e-2))
        return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(x, y)
