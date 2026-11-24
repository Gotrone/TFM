"""Command-line interface: ``cfdbench <command>``."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import leaderboard, models
from .benchmark import TEST_SPLITS, evaluate_submission, run_benchmark
from .dataset import DEFAULT_PATH, assemble, case_list, generate, generate_case, load_dataset
from .lbm import CylinderConfig

DEFAULT_DB = Path("results") / "leaderboard.db"


def _config(a: argparse.Namespace) -> CylinderConfig:
    cfg = CylinderConfig()
    if getattr(a, "warmup", None) is not None or getattr(a, "average", None) is not None:
        cfg = replace(cfg, warmup_steps=a.warmup or cfg.warmup_steps, average_steps=a.average or cfg.average_steps)
    return cfg


def _cmd_generate(a: argparse.Namespace) -> None:
    path = generate(a.output, config=_config(a), workers=a.workers, backend=a.backend, dtype=a.dtype)
    print(f"dataset written to {path}")


def _cmd_generate_case(a: argparse.Namespace) -> None:
    n = len(case_list())
    index = a.index if a.index is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
    if index < 0:
        raise SystemExit(f"give --index (0..{n - 1}) or run inside a SLURM job array")
    print(f"case {index}/{n - 1} written to {generate_case(index, a.out, _config(a), a.backend, a.dtype)}")


def _cmd_assemble(a: argparse.Namespace) -> None:
    print(f"dataset written to {assemble(a.cases, a.output)}")


def _cmd_info(a: argparse.Namespace) -> None:
    print(json.dumps(load_dataset(a.data).info(), indent=2))


def _cmd_benchmark(a: argparse.Namespace) -> None:
    from . import figures

    ds = load_dataset(a.data)
    rows, preds = run_benchmark(ds, a.models or None)
    if a.reset:
        leaderboard.reset(a.db)
    leaderboard.add_entries(a.db, rows, source="baseline")
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(rows, indent=2))
    figures.dataset_overview(ds, out / "figures" / "dataset_overview.png")
    figures.error_vs_reynolds(ds, preds, out / "figures" / "error_vs_reynolds.png")
    best = min((r for r in rows if r["split"] == "test_interp"), key=lambda r: r["metrics"]["rel_l2"])["model"]
    figures.model_comparison(ds, preds, best, "test_interp", 2, out / "figures" / f"comparison_{best}_interp.png")
    figures.model_comparison(ds, preds, best, "test_extrap", len(preds[best]["test_extrap"]) - 1, out / "figures" / f"comparison_{best}_extrap.png")
    _print_board(a.db)


def _print_board(db: Path) -> None:
    for split in TEST_SPLITS:
        print(f"\n== {split} (ranked by rel_l2) ==")
        print(f"{'#':>2} {'model':<12} {'rel_l2':>9} {'mse':>10} {'mae':>9} {'wake_rel_l2':>12} {'div_rms':>9}")
        for e in leaderboard.get_leaderboard(db, split):
            m = e["metrics"]
            print(f"{e['rank']:>2} {e['model']:<12} {m['rel_l2']:>9.4f} {m['mse']:>10.2e} {m['mae']:>9.4f} {m['wake_rel_l2']:>12.4f} {m['div_rms']:>9.4f}")


def _cmd_leaderboard(a: argparse.Namespace) -> None:
    _print_board(a.db)


def _cmd_evaluate(a: argparse.Namespace) -> None:
    ds = load_dataset(a.data)
    with np.load(a.submission, allow_pickle=False) as z:
        preds = {k: z[k] for k in TEST_SPLITS if k in z.files}
    rows = evaluate_submission(ds, preds)
    for r in rows:
        print(r["split"], json.dumps(r["metrics"], indent=2))


def _cmd_scaling(a: argparse.Namespace) -> None:
    from . import scaling

    out = Path(a.results)
    if a.study in ("process", "all"):
        rows = scaling.process_scaling(tuple(a.workers))
        (out / "scaling_processes.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "scaling_processes.json").write_text(json.dumps(rows, indent=2))
        scaling.plot_process_scaling(rows, out / "figures" / "scaling_processes.png")
        for r in rows:
            print(f"{r['workers']:>2} workers  {r['seconds']:>7.2f} s  speedup {r['speedup']:>5.2f}  efficiency {r['efficiency']:.2f}")
    if a.study in ("grid", "all"):
        rows = scaling.grid_scaling()
        (out / "scaling_grid.json").write_text(json.dumps(rows, indent=2))
        scaling.plot_grid_scaling(rows, out / "figures" / "scaling_grid.png")
        for r in rows:
            print(r)


def _cmd_serve(a: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ["CFDBENCH_DATA"] = str(a.data)
    os.environ["CFDBENCH_DB"] = str(a.db)
    uvicorn.run("cfdbench.web.app:app", host=a.host, port=a.port)


def _solver_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--warmup", type=int, default=None, help="override warm-up steps (smoke tests)")
    p.add_argument("--average", type=int, default=None, help="override averaging steps (smoke tests)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cfdbench", description="CFD flow prediction benchmarking suite")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run the LBM simulations and write the dataset")
    g.add_argument("--output", default=DEFAULT_PATH)
    g.add_argument("--workers", type=int, default=None, help="parallel processes (default: all cores)")
    _solver_options(g)
    g.set_defaults(func=_cmd_generate)

    gc = sub.add_parser("generate-case", help="simulate one case (index of the case list, or SLURM_ARRAY_TASK_ID)")
    gc.add_argument("--index", type=int, default=None)
    gc.add_argument("--out", default="data/cases")
    _solver_options(gc)
    gc.set_defaults(func=_cmd_generate_case)

    asm = sub.add_parser("assemble", help="combine the per-case files into the dataset")
    asm.add_argument("--cases", default="data/cases")
    asm.add_argument("--output", default=DEFAULT_PATH)
    asm.set_defaults(func=_cmd_assemble)

    i = sub.add_parser("info", help="print the dataset description")
    i.add_argument("--data", default=DEFAULT_PATH)
    i.set_defaults(func=_cmd_info)

    b = sub.add_parser("benchmark", help="run baseline models, store results, make figures")
    b.add_argument("--data", default=DEFAULT_PATH)
    b.add_argument("--db", default=DEFAULT_DB)
    b.add_argument("--results", default="results")
    b.add_argument("--models", nargs="*", choices=sorted(models.REGISTRY))
    b.add_argument("--reset", action="store_true", help="empty the leaderboard first")
    b.set_defaults(func=_cmd_benchmark)

    lb = sub.add_parser("leaderboard", help="print the leaderboard")
    lb.add_argument("--db", default=DEFAULT_DB)
    lb.set_defaults(func=_cmd_leaderboard)

    e = sub.add_parser("evaluate", help="score a submission .npz locally")
    e.add_argument("submission")
    e.add_argument("--data", default=DEFAULT_PATH)
    e.set_defaults(func=_cmd_evaluate)

    sc = sub.add_parser("scaling", help="performance studies: worker processes and CPU vs GPU")
    sc.add_argument("--study", choices=["process", "grid", "all"], default="all")
    sc.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    sc.add_argument("--results", default="results")
    sc.set_defaults(func=_cmd_scaling)

    s = sub.add_parser("serve", help="start the web platform")
    s.add_argument("--data", default=DEFAULT_PATH)
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
