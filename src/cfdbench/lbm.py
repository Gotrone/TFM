"""Minimal 2D lattice Boltzmann (D2Q9) solver for flow past a circular cylinder.

It is used to generate the reference ("ground truth") flow fields of the
benchmark. The scheme is the regularized BGK model with:

* uniform inlet velocity (equilibrium boundary) on the left,
* zero-gradient outlet on the right,
* periodic top / bottom boundaries,
* half-way bounce-back on the cylinder.

Everything is lattice units (dx = dt = 1). The Reynolds number is
``Re = u_in * D / nu`` with ``D`` the cylinder diameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# D2Q9 lattice
C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CS2 = 1.0 / 3.0


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


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order equilibrium distribution, shape (9, nx, ny)."""
    cu = C[:, 0, None, None] * ux + C[:, 1, None, None] * uy
    usq = ux**2 + uy**2
    return W[:, None, None] * rho * (1 + cu / CS2 + cu**2 / (2 * CS2**2) - usq / (2 * CS2))


def _moments(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    ux = (C[:, 0, None, None] * f).sum(axis=0) / rho
    uy = (C[:, 1, None, None] * f).sum(axis=0) / rho
    return rho, ux, uy


def _regularize(f_neq: np.ndarray) -> np.ndarray:
    """Project the non-equilibrium part on the second-order Hermite basis."""
    pi_xx = (C[:, 0, None, None] ** 2 * f_neq).sum(axis=0)
    pi_yy = (C[:, 1, None, None] ** 2 * f_neq).sum(axis=0)
    pi_xy = (C[:, 0, None, None] * C[:, 1, None, None] * f_neq).sum(axis=0)
    q_xx = C[:, 0] ** 2 - CS2
    q_yy = C[:, 1] ** 2 - CS2
    q_xy = C[:, 0] * C[:, 1]
    out = (
        q_xx[:, None, None] * pi_xx + q_yy[:, None, None] * pi_yy + 2 * q_xy[:, None, None] * pi_xy
    )
    return W[:, None, None] * out / (2 * CS2**2)


def viscosity_for_reynolds(reynolds: float, cfg: CylinderConfig) -> float:
    return cfg.u_in * cfg.diameter / reynolds


def simulate(reynolds: float, cfg: CylinderConfig | None = None, seed: int = 0) -> dict:
    """Run one case and return the time-averaged and last instantaneous fields.

    Returns a dict with ``u_mean``, ``v_mean`` (time averages over
    ``cfg.average_steps`` steps after the warm-up), ``u_inst``, ``v_inst`` (last
    frame) and ``mask``. Velocities are in units of the inlet velocity.
    """
    cfg = cfg or CylinderConfig()
    nu = viscosity_for_reynolds(reynolds, cfg)
    tau = nu / CS2 + 0.5
    omega = 1.0 / tau
    mask = cylinder_mask(cfg)

    rng = np.random.default_rng(seed)
    ux = np.full((cfg.nx, cfg.ny), cfg.u_in)
    uy = 1e-4 * cfg.u_in * rng.standard_normal((cfg.nx, cfg.ny))  # tiny noise to trigger shedding
    ux[mask] = 0.0
    f = equilibrium(np.ones((cfg.nx, cfg.ny)), ux, uy)
    f_inlet = equilibrium(np.ones((1, cfg.ny)), np.full((1, cfg.ny), cfg.u_in), np.zeros((1, cfg.ny)))[
        :, 0, :
    ]

    u_sum = np.zeros((cfg.nx, cfg.ny))
    v_sum = np.zeros((cfg.nx, cfg.ny))

    for step in range(cfg.warmup_steps + cfg.average_steps):
        rho, ux, uy = _moments(f)
        feq = equilibrium(rho, ux, uy)
        f_neq = _regularize(f - feq)
        f_post = feq + (1 - omega) * f_neq

        # bounce-back on the cylinder
        f_post[:, mask] = f[OPP][:, mask]

        # streaming
        for i in range(9):
            f[i] = np.roll(np.roll(f_post[i], C[i, 0], axis=0), C[i, 1], axis=1)

        # open boundaries
        f[:, 0, :] = f_inlet
        f[:, -1, :] = f[:, -2, :]

        if step >= cfg.warmup_steps:
            u_sum += ux
            v_sum += uy

    if not np.isfinite(f).all():
        raise RuntimeError(f"Simulation diverged for Re={reynolds}")

    n = cfg.average_steps
    u_mean = u_sum / n / cfg.u_in
    v_mean = v_sum / n / cfg.u_in
    u_inst = ux / cfg.u_in
    v_inst = uy / cfg.u_in
    for arr in (u_mean, v_mean, u_inst, v_inst):
        arr[mask] = 0.0
    return {
        "u_mean": u_mean.astype(np.float32),
        "v_mean": v_mean.astype(np.float32),
        "u_inst": u_inst.astype(np.float32),
        "v_inst": v_inst.astype(np.float32),
        "mask": mask,
    }
