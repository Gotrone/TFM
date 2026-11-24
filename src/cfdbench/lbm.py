"""Minimal 2D lattice Boltzmann (D2Q9) solver for flow past a circular cylinder.

It is used to generate the reference ("ground truth") flow fields of the
benchmark. The scheme is the regularized BGK model with:

* uniform inlet velocity (equilibrium boundary) on the left,
* zero-gradient outlet on the right,
* periodic top / bottom boundaries,
* half-way bounce-back on the cylinder.

Everything is lattice units (dx = dt = 1). The Reynolds number is
``Re = u_in * D / nu`` with ``D`` the cylinder diameter.

The solver runs on the CPU (NumPy) or on an NVIDIA GPU (CuPy, ``backend="gpu"``); both
use the same code through the array module ``xp``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# D2Q9 lattice
C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CS2 = 1.0 / 3.0


def get_backend(backend: str = "cpu"):
    """Return the array module: ``numpy`` for "cpu", ``cupy`` for "gpu"."""
    if backend == "cpu":
        return np
    if backend == "gpu":
        try:
            import cupy
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError("GPU backend needs CuPy: pip install 'cupy-cuda12x[ctk]'") from exc
        return cupy
    raise ValueError("backend must be 'cpu' or 'gpu'")


def _const(xp, dtype):
    """Lattice constants as arrays of the chosen module and precision."""
    return (
        xp.asarray(C[:, 0], dtype=dtype)[:, None, None],
        xp.asarray(C[:, 1], dtype=dtype)[:, None, None],
        xp.asarray(W, dtype=dtype)[:, None, None],
    )


@dataclass(frozen=True)
class CylinderConfig:
    """Geometry and numerical parameters of a cylinder-flow case."""

    nx: int = 200
    ny: int = 72
    diameter: int = 14
    cx: int = 44  # cylinder centre (x, in lattice nodes)
    u_in: float = 0.08  # inlet velocity (lattice units, Mach ~ 0.14)
    warmup_steps: int = 10000
    average_steps: int = 3200

    @property
    def cy(self) -> int:
        return self.ny // 2


def cylinder_mask(cfg: CylinderConfig) -> np.ndarray:
    """Boolean (nx, ny) array, True inside the cylinder."""
    x, y = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.ny), indexing="ij")
    return (x - cfg.cx) ** 2 + (y - cfg.cy) ** 2 <= (cfg.diameter / 2) ** 2


def equilibrium(rho, ux, uy, xp=np):
    """Second-order equilibrium distribution, shape (9, nx, ny)."""
    cx, cy, w = _const(xp, rho.dtype)
    cu = cx * ux + cy * uy
    usq = ux**2 + uy**2
    return w * rho * (1 + cu / CS2 + cu**2 / (2 * CS2**2) - usq / (2 * CS2))


def _moments(f, xp=np):
    cx, cy, _ = _const(xp, f.dtype)
    rho = f.sum(axis=0)
    ux = (cx * f).sum(axis=0) / rho
    uy = (cy * f).sum(axis=0) / rho
    return rho, ux, uy


def _regularize(f_neq, xp=np):
    """Project the non-equilibrium part on the second-order Hermite basis."""
    cx, cy, w = _const(xp, f_neq.dtype)
    pi_xx = (cx**2 * f_neq).sum(axis=0)
    pi_yy = (cy**2 * f_neq).sum(axis=0)
    pi_xy = (cx * cy * f_neq).sum(axis=0)
    out = (cx**2 - CS2) * pi_xx + (cy**2 - CS2) * pi_yy + 2 * cx * cy * pi_xy
    return w * out / (2 * CS2**2)


def viscosity_for_reynolds(reynolds: float, cfg: CylinderConfig) -> float:
    return cfg.u_in * cfg.diameter / reynolds


def simulate(
    reynolds: float,
    cfg: CylinderConfig | None = None,
    seed: int = 0,
    probe: tuple[int, int] | None = None,
    backend: str = "cpu",
    dtype: str = "float64",
) -> dict:
    """Run one case and return the time-averaged and last instantaneous fields.

    Returns a dict with ``u_mean``, ``v_mean`` (time averages over
    ``cfg.average_steps`` steps after the warm-up), ``u_inst``, ``v_inst`` (last
    frame) and ``mask``. Velocities are in units of the inlet velocity.
    If ``probe=(ix, iy)`` is given, the transverse velocity at that node is recorded at every
    averaging step and returned as ``probe_v`` (used to measure the shedding frequency).
    ``backend`` is "cpu" or "gpu"; ``dtype`` is "float64" or "float32".
    """
    cfg = cfg or CylinderConfig()
    xp = get_backend(backend)
    dt = np.dtype(dtype)
    nu = viscosity_for_reynolds(reynolds, cfg)
    omega = 1.0 / (nu / CS2 + 0.5)
    mask_np = cylinder_mask(cfg)
    mask = xp.asarray(mask_np)
    opp = list(OPP)

    rng = np.random.default_rng(seed)
    ux0 = np.full((cfg.nx, cfg.ny), cfg.u_in)
    uy0 = 1e-4 * cfg.u_in * rng.standard_normal((cfg.nx, cfg.ny))  # tiny noise to trigger shedding
    ux0[mask_np] = 0.0
    ux, uy = xp.asarray(ux0, dtype=dt), xp.asarray(uy0, dtype=dt)
    f = equilibrium(xp.ones((cfg.nx, cfg.ny), dtype=dt), ux, uy, xp)
    f_inlet = equilibrium(
        xp.ones((1, cfg.ny), dtype=dt), xp.full((1, cfg.ny), cfg.u_in, dtype=dt), xp.zeros((1, cfg.ny), dtype=dt), xp
    )[:, 0, :]

    probe_v: list = []
    u_sum = xp.zeros((cfg.nx, cfg.ny), dtype=dt)
    v_sum = xp.zeros((cfg.nx, cfg.ny), dtype=dt)

    for step in range(cfg.warmup_steps + cfg.average_steps):
        rho, ux, uy = _moments(f, xp)
        feq = equilibrium(rho, ux, uy, xp)
        f_neq = _regularize(f - feq, xp)
        f_post = feq + (1 - omega) * f_neq

        # bounce-back on the cylinder
        f_post[:, mask] = f[opp][:, mask]

        # streaming
        for i in range(9):
            f[i] = xp.roll(xp.roll(f_post[i], int(C[i, 0]), axis=0), int(C[i, 1]), axis=1)

        # open boundaries
        f[:, 0, :] = f_inlet
        f[:, -1, :] = f[:, -2, :]

        if step >= cfg.warmup_steps:
            u_sum += ux
            v_sum += uy
            if probe is not None:
                probe_v.append(uy[probe])

    if not bool(xp.isfinite(f).all()):
        raise RuntimeError(f"Simulation diverged for Re={reynolds}")

    def host(a):
        return a.get() if xp is not np else a

    n = cfg.average_steps
    u_mean = host(u_sum).astype(np.float64) / n / cfg.u_in
    v_mean = host(v_sum).astype(np.float64) / n / cfg.u_in
    u_inst = host(ux).astype(np.float64) / cfg.u_in
    v_inst = host(uy).astype(np.float64) / cfg.u_in
    for arr in (u_mean, v_mean, u_inst, v_inst):
        arr[mask_np] = 0.0
    return {
        "u_mean": u_mean.astype(np.float32),
        "v_mean": v_mean.astype(np.float32),
        "u_inst": u_inst.astype(np.float32),
        "v_inst": v_inst.astype(np.float32),
        "mask": mask_np,
        "probe_v": host(xp.stack(probe_v)).astype(np.float64) / cfg.u_in if probe_v else np.array([]),
    }
