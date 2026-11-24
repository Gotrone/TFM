import numpy as np
import pytest

from cfdbench import models
from cfdbench.benchmark import evaluate_submission, run_benchmark
from cfdbench.lbm import CylinderConfig, cylinder_mask, equilibrium, simulate
from cfdbench.metrics import divergence, evaluate
from cfdbench.preprocessing import Standardizer, downsample, field_to_image, velocity_magnitude


# --- lbm ---------------------------------------------------------------------------------------
def test_equilibrium_conserves_mass_and_momentum():
    rho = np.full((4, 4), 1.2)
    ux, uy = np.full((4, 4), 0.05), np.full((4, 4), -0.02)
    feq = equilibrium(rho, ux, uy)
    assert np.allclose(feq.sum(axis=0), rho)
    from cfdbench.lbm import C

    assert np.allclose((C[:, 0, None, None] * feq).sum(axis=0), rho * ux)
    assert np.allclose((C[:, 1, None, None] * feq).sum(axis=0), rho * uy)


def test_cylinder_mask_geometry():
    cfg = CylinderConfig(nx=60, ny=30, diameter=10, cx=20)
    m = cylinder_mask(cfg)
    assert m[20, 15] and not m[0, 0]
    assert abs(m.sum() - np.pi * 25) < 12


def test_short_simulation_is_finite_and_blocked_inside_cylinder():
    cfg = CylinderConfig(nx=60, ny=30, diameter=8, cx=15, warmup_steps=150, average_steps=50)
    out = simulate(80, cfg)
    assert np.isfinite(out["u_mean"]).all()
    assert (out["u_mean"][out["mask"]] == 0).all()
    assert 0.5 < out["u_mean"][-5].mean() < 1.5  # roughly the free-stream velocity at the outlet


# --- preprocessing -----------------------------------------------------------------------------
def test_standardizer_roundtrip(tiny_dataset):
    ds = tiny_dataset
    s = Standardizer().fit(ds.fields, ds.mask)
    z = s.transform(ds.fields, ds.mask)
    assert abs(z[:, :, ~ds.mask].mean()) < 1e-3
    assert (z[:, :, ds.mask] == 0).all()
    assert np.allclose(s.inverse(z, ds.mask), ds.fields, atol=1e-5)


def test_downsample_block_mean():
    a = np.arange(16, dtype=float).reshape(4, 4)
    assert np.allclose(downsample(a, 2), [[2.5, 4.5], [10.5, 12.5]])
    assert downsample(np.ones((2, 5, 7)), 2).shape == (2, 2, 3)


def test_field_to_image_shape_and_mask(tiny_dataset):
    img = field_to_image(velocity_magnitude(tiny_dataset.fields[0]), tiny_dataset.mask)
    assert img.shape == (12, 24, 3) and img.dtype == np.uint8


# --- metrics -----------------------------------------------------------------------------------
def test_perfect_prediction_has_zero_error(tiny_dataset):
    ds = tiny_dataset
    _, truth = ds.subset("test_interp")
    m = evaluate(truth.copy(), truth, ds.mask)
    assert m["mse"] == 0 and m["mae"] == 0 and m["rel_l2"] == 0 and m["max_abs"] == 0


def test_metrics_scale_with_error(tiny_dataset):
    ds = tiny_dataset
    _, truth = ds.subset("test_interp")
    small = evaluate(truth + 0.01, truth, ds.mask)
    big = evaluate(truth + 0.1, truth, ds.mask)
    assert big["mse"] > small["mse"] and big["rel_l2"] > small["rel_l2"]
    assert small["mae"] == pytest.approx(0.01, rel=1e-3)


def test_metrics_reject_bad_input(tiny_dataset):
    ds = tiny_dataset
    _, truth = ds.subset("test_interp")
    with pytest.raises(ValueError):
        evaluate(truth[:1], truth, ds.mask)
    bad = truth.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate(bad, truth, ds.mask)


def test_divergence_of_constant_field_is_zero():
    f = np.ones((1, 2, 10, 10))
    assert np.allclose(divergence(f), 0)


# --- models and benchmark ----------------------------------------------------------------------
def test_registry_contains_baselines():
    assert {"freestream", "mean", "nearest", "linear", "pod-poly", "pod-gp", "pod-mlp"} <= set(models.REGISTRY)
    with pytest.raises(KeyError):
        models.create("nope")


def test_linear_interpolation_is_exact_on_training_points(tiny_dataset):
    ds = tiny_dataset
    re, f = ds.subset("train")
    pred = models.create("linear").fit(re, f, ds.mask).predict(re)
    assert np.allclose(pred, f, atol=1e-6)


def test_nearest_picks_closest(tiny_dataset):
    ds = tiny_dataset
    re, f = ds.subset("train")
    pred = models.create("nearest").fit(re, f, ds.mask).predict(np.array([62.0]))
    assert np.allclose(pred[0], f[1])  # Re=60


def test_pod_models_reproduce_smooth_training_data(tiny_dataset):
    ds = tiny_dataset
    re, f = ds.subset("train")
    for name in ("pod-poly", "pod-gp", "pod-mlp"):
        pred = models.create(name, n_modes=4).fit(re, f, ds.mask).predict(re)
        err = np.linalg.norm(pred - f) / np.linalg.norm(f)
        assert err < 0.05, (name, err)


def test_benchmark_ranks_smart_models_above_trivial_ones(tiny_dataset):
    rows, preds = run_benchmark(tiny_dataset, ["freestream", "mean", "linear", "pod-poly"])
    interp = {r["model"]: r["metrics"]["rel_l2"] for r in rows if r["split"] == "test_interp"}
    assert interp["linear"] < interp["mean"] < interp["freestream"] + 1e-9 or interp["linear"] < interp["freestream"]
    assert interp["pod-poly"] < interp["freestream"]
    assert set(preds["linear"]) == {"test_interp", "test_extrap"}


def test_evaluate_submission_requires_all_splits(tiny_dataset):
    _, truth = tiny_dataset.subset("test_interp")
    with pytest.raises(ValueError):
        evaluate_submission(tiny_dataset, {"test_interp": truth})
    _, truth_x = tiny_dataset.subset("test_extrap")
    rows = evaluate_submission(tiny_dataset, {"test_interp": truth, "test_extrap": truth_x})
    assert all(r["metrics"]["rel_l2"] == 0 for r in rows)


# --- per-case generation and assembly (SLURM job arrays) ---------------------------------------
def test_generate_case_and_assemble_match_generate(tmp_path):
    from dataclasses import replace

    from cfdbench import dataset
    from cfdbench.lbm import CylinderConfig

    cfg = CylinderConfig(nx=50, ny=24, diameter=6, cx=12, warmup_steps=30, average_steps=10)
    splits = {"train": [40, 60, 80], "test_interp": [50], "test_extrap": [30]}
    old = dataset.SPLITS
    dataset.SPLITS = splits  # shrink the case list for the test
    try:
        n = len(dataset.case_list())
        for i in range(n):
            dataset.generate_case(i, tmp_path / "cases", cfg)
        a = dataset.assemble(tmp_path / "cases", tmp_path / "a.npz")
        b = dataset.generate(tmp_path / "b.npz", cfg, workers=1, splits=splits)
        da, db = dataset.load_dataset(a), dataset.load_dataset(b)
        assert np.allclose(da.fields, db.fields) and (da.split == db.split).all()
        with pytest.raises(FileNotFoundError):
            (tmp_path / "cases" / "case_re050.npz").unlink()
            dataset.assemble(tmp_path / "cases", tmp_path / "c.npz")
    finally:
        dataset.SPLITS = old
    assert replace(cfg).nx == 50
