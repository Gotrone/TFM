# CFD Benchmarking – Evaluation Prototype

Prototype of the evaluation pipeline for a benchmarking suite for CFD (computational fluid dynamics) prediction models, started as part of my Master's thesis topic in High Performance Computing at Universidade da Coruña (2024-2025).

> **Status: early prototype (work in progress).** The notebook validates the workflow on synthetic data; real CFD datasets and prediction models are the next step.

## What the notebook does

1. **Data generation** – synthetic 100 × 100 "ground truth" velocity field and a simulated prediction (ground truth + Gaussian noise).
2. **Preprocessing** – min-max normalization of both fields to [0, 1].
3. **Evaluation** – Mean Squared Error (MSE) and Mean Absolute Error (MAE).
4. **Visualization** – heatmaps of ground truth, prediction and absolute error.

Results on synthetic data: MSE ≈ 0.0056, MAE ≈ 0.062.

## Repository structure

```
notebooks/
└── TFM.ipynb   # full prototype: generation, evaluation, visualization
```

## Getting started

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook notebooks/TFM.ipynb
```

## Roadmap

- Replace synthetic data with real CFD datasets (cylinder flow, channel turbulence).
- Plug in actual prediction models.
- Automate batch evaluations and result tracking (leaderboard).
- Add interactive dashboards.

## Tech stack

Python · NumPy · pandas · Matplotlib · Jupyter
