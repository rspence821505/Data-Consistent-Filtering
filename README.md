# A Data-Consistent Approach to Ensemble Filtering

Code, numerical results and reproduction scripts for the paper

> **A Data-Consistent Approach to Ensemble Filtering**
> Rylan Spence¹, Troy Butler², Clint Dawson¹
> ¹ Oden Institute for Computational Science and Engineering, University of Texas, Austin, TX
> ² Department of Mathematical and Statistical Sciences, University of Colorado Denver, Denver, CO
>
> arXiv preprint [arXiv:2605.11282](https://arxiv.org/abs/2605.11282) [math.ST], 2026.

Every figure of the paper can be regenerated from the numerical results stored
here in about 15 seconds (**Level 1**). Every Lorenz-96 experiment can also be
rerun from scratch on a workstation (**Level 2**). The high-dimensional
shallow-water experiment is reproduced from its saved run outputs (Level 1
only).

* [Overview](#overview)
* [Quick start](#quick-start)
* [Repository structure](#repository-structure)
* [Installation](#installation)
* [Reproducibility](#reproducibility)
* [Computational requirements](#computational-requirements)
* [Data availability](#data-availability)
* [Citation](#citation) · [License](#license)

## Overview

Ensemble data assimilation becomes difficult when the ensemble is much smaller
than the state and the stacked observation vector. The empirical covariances
are then rank deficient and dominated by sampling error. The paper introduces
the **Quantity-of-Interest Principal Component Analysis Ensemble Data-Consistent
Filter (QPCA-EnDCF)**, a deterministic filter motivated by data-consistent
inversion. It regularizes the analysis in the *whitened forecast–observation
mismatch* space. Over an assimilation window, the forecast residuals are whitened
by the observation-error covariance, and the correction is restricted to the
leading rank-κ principal subspace of their empirical covariance. The projected
mismatch is then mapped back to the state through an empirical EnDCF gain.

The main contributions are:

* **Method.** QPCA-EnDCF regularizes the update in the space where agreement with
  the data is measured, not through localization or a transform of the state or
  ensemble space. It needs no perturbed observations.
* **Theory.** Concentration and spectral-projector perturbation bounds for the
  empirical whitened-mismatch covariance and its retained subspace, and an exact
  bias–variance decomposition of the analysis mean. The truncation bias is governed
  by the discarded component of the population mean residual; finite-ensemble
  variability depends on the ensemble size N, the rank κ and the spectral gap.
* **Lorenz-96 experiments** (N = 10 in a 100-dimensional stacked observation
  space). Compared with sequential and four-dimensional stochastic EnKF and
  deterministic ETKF, QPCA-EnDCF is markedly better calibrated (settled
  spread–skill ratio 0.82 against 0.07–0.11). It has lower mean-squared error,
  trading a larger squared bias for an 82–84% reduction in across-realization
  variance. Sweeps over N and κ support reading κ as a regularization parameter.
* **High-dimensional shallow-water experiment** (idealized inlet, ~2×10⁵ state
  degrees of freedom, ~1500 stacked observations, N = 10). QPCA-EnDCF can be
  implemented without assembling the state covariance and remains numerically
  viable through the prescribed shelf-crossing endpoints in all four
  storm–localization configurations.

The paper is available on arXiv:
[arXiv:2605.11282](https://arxiv.org/abs/2605.11282).

## Quick start

```bash
git clone https://github.com/rspence821505/Data-Consistent-Filtering.git
cd Data-Consistent-Filtering
conda env create -f environment.yml
conda activate data-consistent-filtering

python scripts/reproduce_figures.py --compare     # all 8 figures, ~15 s
```

The figures are written to `figures/generated/published/`, with the numbers each
one reports in `figures/generated/published/summaries/`. `--compare` checks every
PNG pixel by pixel against the one used in the manuscript (`figures/paper/`).

## Repository structure

```
Data-Consistent-Filtering/
├── README.md, LICENSE, environment.yml
├── scripts/                          entry points
│   ├── reproduce_figures.py          Level 1: figures from saved numerical results
│   ├── run_all_experiments.py        Level 2: rerun the Lorenz-96 experiments
│   ├── extract_results.py            raw experiment output -> processed results (shared interface)
│   └── verify_results.py             regenerated vs published numbers; spot checks
├── src/                              Python package (imported as `src`)
│   ├── filters/                      Seq-EnKF, 4D-EnKF, Seq-/4D-(L)ETKF, QPCA-EnDCF
│   ├── models/, utils/               Lorenz-96 model, observations, localization, metrics
│   ├── experiments/                  Lorenz-96 experiment drivers (resumable, parallel)
│   │   ├── baseline.py               N = 10 comparison and ensemble-size sweep (Figs 1-3)
│   │   ├── coverage.py               common-endpoint rank diagnostic (Fig 2)
│   │   ├── bias_variance_mc.py       hierarchical bias-variance experiment (Fig 4)
│   │   ├── n_kappa_bias_variance.py  (N, kappa) response surface (Fig 5)
│   │   └── restart_vs_cycled.py      restart-law calibration used by the (N, kappa) experiment
│   ├── processing/                   reduction to processed results; inlet tables; checksums
│   ├── figures/                      one module per paper figure (fig1_... to fig8_...)
│   └── config.py, paths.py           paths only (no experimental parameters)
├── experiments/lorenz96/             YAML configurations of the Lorenz-96 experiments
├── data/shallow_water/               idealized-inlet mesh (drawn in Figure 6)
├── results/
│   ├── published/                    numerical results behind the paper's figures (+ MANIFEST.json)
│   │   ├── lorenz96/                 processed Lorenz-96 results (.npz)
│   │   └── shallow_water/            47 per-run JSON outputs and the derived tables
│   └── regenerated/                  Level 2 output (git-ignored)
├── figures/
│   ├── paper/                        the 8 PNGs used in the manuscript (comparison reference)
│   └── generated/                    Level 1 output (git-ignored)
├── notebooks/reproduce_figures.ipynb interactive version of Level 1
└── tests/                            unit tests of drivers and kernels; Level 1 regression test
```

## Installation

Requires conda (e.g. [Miniforge](https://conda-forge.org/download/)).

```bash
git clone https://github.com/rspence821505/Data-Consistent-Filtering.git
cd Data-Consistent-Filtering
conda env create -f environment.yml          # ~1-5 min
conda activate data-consistent-filtering
python -m pytest tests/test_reproduction.py  # optional, ~15 s: checks Level 1
```

All commands below are run from the repository root with this environment
active. The environment pins Python 3.12, NumPy 2.5.1, SciPy 1.18.0,
Matplotlib 3.11.0 with FreeType 2.14.3, OpenBLAS 0.3.33, PyYAML 6.0.3,
joblib 1.5.3, threadpoolctl 3.6.0, h5py 3.16.0, Pillow 12.3.0, JupyterLab 4.6.1
and pytest 9.1.1, the versions the published results and figures were produced
with (see [Numerical verification](#numerical-verification)).

## Reproducibility

Two levels of reproduction share one data interface:

```
 Level 2 (complete)                         shared interface                   Level 1 (fast)
 scripts/run_all_experiments.py  --raw-->   scripts/extract_results.py  --->   scripts/reproduce_figures.py
                                            results/<set>/lorenz96/*.npz        figures/generated/<set>/
                                            results/published/shallow_water/*
         <set> = published (committed, never overwritten) | regenerated (git-ignored)
```

The published processed results were produced by the same reduction code
(`src/processing/`) from the original raw checkpoints (2.4 GB, not included).
Rerunning an experiment writes only under `results/regenerated/`.

### Level 1: figures from saved numerical results

```bash
python scripts/reproduce_figures.py                         # all figures from results/published/
python scripts/reproduce_figures.py --figures 4 5           # a subset
python scripts/reproduce_figures.py --compare               # pixel comparison with figures/paper/
python scripts/reproduce_figures.py --results regenerated   # from Level 2 output
```

or open `notebooks/reproduce_figures.ipynb`. No experiment is rerun. Each
figure module also reports the numbers the paper quotes for it: settled
spread–skill ratios, coverage and tail rates, settled RMSE by N, bias² /
variance / MSE and their ratios, log–log slopes and the κ minimizing the total
error, R_E and the component fractions. The plotting code is the code that made
the published figures, taken unchanged from the original analysis notebooks and
scripts. Only the input and output paths changed.

### Figures, code and data

| Paper figure | Description | Level 1 code (`src/figures/`) | Processed data (`results/published/`) | Level 2 experiment |
|---|---|---|---|---|
| 1 `fig04t_calibration_collapse_ribbons.png` | Median spread–skill ratio and IQR of five filters, Lorenz-96, N = 10 | `fig1_calibration_ribbons.py` | `lorenz96/calibration_N10.npz` | `baseline` |
| 2 `fig04s_postburnin_coverage_reliability.png` | Post-burn-in central coverage, tail balance, outside-ensemble rate | `fig2_coverage_reliability.py` | `lorenz96/coverage_ranks_N10.npz`, `lorenz96/calibration_N10.npz` | `coverage` + `baseline` |
| 3 `fig04p_rmse_by_ensemble_size.png` | Settled RMSE for N = 5…30, RMSE per added member, QPCA-EnDCF lead over Seq-ETKF | `fig3_rmse_by_ensemble_size.py` | `lorenz96/ensemble_size_sweep.npz` | `ensemble-sweep` |
| 4 `fig07j2_hierarchical_bias_variance_summary.png` | Bias², variance, MSE over 100 truths × 100 realizations × 1000 windows, and their stability | `fig4_hierarchical_bias_variance.py` | `lorenz96/hierarchical_bias_variance.npz` | `hierarchical` |
| 5 `fig07p_n_kappa_bias_variance_surfaces.png` | Debiased bias², variance and T over (N, κ), independent restarts | `fig5_n_kappa_surfaces.py` | `lorenz96/n_kappa_bootstrap.npz` | `n-kappa` |
| 6 `fig01_scale_matching_setup.png` | Inlet geometry, observations, vortex track; storm–localization design | `fig6_inlet_setup.py` | `shallow_water/scale_matching_regime_map_v3.csv` and the mesh `data/shallow_water/Ideal_Inlet.h5` | — (saved run outputs) |
| 7 `fig02_rmse_calibration_trajectories.png` | Per-window RMSE_{E,w} and γ_{E,w}, QPCA-EnDCF vs 4D-LETKF | `fig7_inlet_trajectories.py` | `shallow_water/runs/*/result_qpca_endcf.json` | — (saved run outputs) |
| 8 `fig05_metric_sensitivity.png` | Endpoint RMSE_E per realization, R_E, component fractions of RMSE_E² | `fig8_inlet_endpoint_error.py` | `shallow_water/runs/*/result_qpca_endcf.json` | — (saved run outputs) |

**Table 2** (per-window RMSE_{E,w} and γ_{E,w} of all four filters in the inlet
experiment) is `results/published/shallow_water/all_filter_per_window_v3.json`.
`python scripts/extract_results.py --shallow-water` rebuilds it, and the
regime-map table behind Figure 6, from the published run outputs into
`results/regenerated/shallow_water/`, printing Table 2 in the paper's layout;
`python scripts/verify_results.py` then compares the rebuilt tables with the
published ones.

### Level 2: rerunning the experiments

**Lorenz-96 (Figures 1–5).** One command reruns the five experiments with the
configurations, seeds and settings used for the paper, and reduces their output:

```bash
python scripts/run_all_experiments.py --list                            # experiments, figures, cost
python scripts/run_all_experiments.py --n-jobs 10                       # all five
python scripts/run_all_experiments.py --experiments baseline coverage   # Figures 1-2 only
python scripts/reproduce_figures.py --results regenerated
python scripts/verify_results.py                                        # compare with the published numbers
```

| Experiment | Figures | What it runs | Wall clock, 10 workers |
|---|---|---|---|
| `baseline` | 1–2 | 5 filters × 100 seeds × 1000 windows, N = 10 | ~4 min (7.2 min measured with 6 workers on a busy machine) |
| `coverage` | 2 | the same runs, recording the truth's rank at every window | ~4 min (5.0 min measured with 6 workers) |
| `ensemble-sweep` | 3 | `baseline` at N ∈ {5, 10, 15, 20, 25, 30} | ~1–1.5 h (projected) |
| `hierarchical` | 4 | 117 truth seeds × 100 realizations × 1000 windows × 5 filters | ~1 h (4.7 core-hours, projected) |
| `n-kappa` | 5 | 15 (N, κ) points × 2 regimes × 50 truth seeds × 100 realizations | ~1–2 h (9.2 core-hours, projected) |

Measured or projected on a 15-core Apple M5 Pro. The projections come from the
drivers' own `--dry-run` estimators. Each experiment runs on its own and is
resumable: repeating a command continues from the last checkpoint. Raw
checkpoints go to `results/regenerated/raw/lorenz96/` (move them with
`DCF_RAW_RESULTS_DIR`). The drivers can also be called directly, e.g.
`python -m src.experiments.bias_variance_mc --help`.

The (N, κ) experiment holds a restart forecast law fixed across its grid. By
default the law is recalibrated from scratch (seconds), which reproduces the
published law's identifier (`2fbc05cb06f87eaa`) and its matrices to ~5×10⁻¹⁴.
`--reuse-published-restart-law` starts from the published law
(`results/published/lorenz96/restart_law_2fbc05cb06f87eaa.npz`) instead.

**Shallow water (Figures 6–8, Table 2): saved run outputs only.** This
experiment is not rerun. Its 47 runs (4 storm–localization configurations ×
4 filters × 3 realizations; one of the 48 produced no output) were computed on
TACC Vista with a FEniCSx (DOLFINx 0.10) shallow-water solver on 8 MPI ranks per
run. That solver is not part of this repository. The per-window diagnostics of
every run are distributed in `results/published/shallow_water/runs/`, and each
file records its run configuration under `config`. Figures 6–8 and Table 2 are
reproduced from them; `--results regenerated` draws these three figures from the
published outputs as well.

### Numerical verification

`scripts/verify_results.py` first checks that every published file still has the
SHA-256 recorded in `results/published/MANIFEST.json`. It then compares each
regenerated file with its published counterpart:

* **IDENTICAL**: bit-for-bit equal.
* **ROUNDOFF**: equal to a relative tolerance of 10⁻⁹.
* **DIFFERENT**: anything larger. The Lorenz-96 experiments are chaotic: a
  last-bit change in one window grows until a trajectory becomes a *new sample of
  the same experiment*. The script then compares each figure-level statistic
  through a paired z-score across seeds or cells. |z| < 4 is reported as
  consistent with sampling variability; |z| ≥ 4 is flagged as a meaningful
  discrepancy.

`python scripts/verify_results.py --spot-check` (~3 min) reruns one seed or one
cell of every Lorenz-96 experiment from scratch and compares it with the
published per-seed or per-cell values.

What to expect:

* **Lorenz-96 on osx-arm64** (Apple Silicon, where the published results were
  computed) with `environment.yml`: bit-for-bit reproduction. OpenBLAS must be
  0.3.33. With 0.3.34, the dense eigendecomposition in QPCA-EnDCF (Figures 1–3)
  differs from the published run by 10⁻¹⁵ (relative) at window 2. The difference
  grows to 10⁻¹⁰ by window 12 and decorrelates the trajectory by window ~50.
  The other four filters, and the batched kernels behind Figures 4–5, were
  unaffected.
* **Lorenz-96 on other platforms** (e.g. Linux x86-64): OpenBLAS uses different
  CPU kernels, so roundoff differences grow as above. Expect the figure-level
  statistics to agree within sampling variability, not bit for bit. The truth
  seeds on which the unlocalized Seq-ETKF diverges in Figure 4 (12 of 117, a
  selection the paper discusses) can then change; `verify_results.py` falls back
  to unpaired comparisons.
* **Shallow-water tables:** rebuilding them from the published run files under
  Python ≥ 3.12 changes some averaged columns in the last bits (relative
  differences ≤ 3×10⁻¹⁶, reported as ROUNDOFF). Python 3.12 made `sum()` of
  floats compensated; the published tables were built under Python 3.11.

### Verification status

Checked while this repository was assembled, in a fresh environment created from
`environment.yml` on macOS 26 / Apple M5 Pro:

| Item | Status |
|---|---|
| Figures 1–8 from `results/published/` | **Verified**: all 8 pixel-identical to the manuscript PNGs |
| Numbers reported with the figures | **Verified** against the values quoted in the manuscript text |
| Table 2 | **Verified**: rebuilt from the run outputs; every entry matches the manuscript after rounding |
| Published processed files | **Verified**: reduced from the original raw checkpoints by `src/processing/`; the (N, κ) bootstrap draws equal the driver's own summary exactly |
| Level 2, Lorenz-96, Figures 1–2 | **Verified**: `baseline` and `coverage` rerun in full from scratch; all regenerated values bit-identical to the published ones, both figures pixel-identical |
| Level 2, Lorenz-96, Figures 3–5 | **Spot-checked**: one seed of the ensemble sweep, one truth seed of the hierarchical experiment and two (N, κ) cells rerun from scratch, all bit-identical (`verify_results.py --spot-check`); the full `ensemble-sweep`, `hierarchical` and `n-kappa` runs were not repeated |
| Shallow-water experiment | Not rerun (Level 1 only, by design); Figures 6–8 and Table 2 reproduced from the saved run outputs |
| Unit tests (`python -m pytest tests`) | **Passing**: 92 tests, ~2 min |

## Computational requirements

| Task | Hardware | Time | Memory / disk |
|---|---|---|---|
| Level 1, all figures | any laptop | ~15 s | < 2 GB RAM |
| Level 2, Lorenz-96, all five experiments | multicore workstation (10 workers) | ~3–5 h | ~2 GB RAM (~4 GB while reducing the (N, κ) run); 2.2 GB of raw checkpoints |
| Level 2, `--spot-check` | any laptop | ~3 min | < 1 GB RAM |

Runtimes scale roughly inversely with `--n-jobs`. The Lorenz-96 drivers pin
BLAS to one thread per worker.

## Data availability

* **Included:** all processed numerical results behind the figures
  (`results/published/`, 23 MB, checksummed); the 47 per-run outputs of the
  shallow-water experiment; the inlet mesh (`data/shallow_water/`); the manuscript
  PNGs used as comparison references (`figures/paper/`).
* **Not included:** the raw Lorenz-96 checkpoints (2.4 GB), which no figure
  needs and Level 2 regenerates; and the shallow-water solver and its
  full-field output, since that experiment is reproduced from its per-run
  diagnostics.
* **No external data:** every experiment is a twin experiment whose truth and
  observations are generated from fixed seeds by the code in this repository.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{spence_butler_dawson_data_consistent_filtering,
  author        = {Spence, Rylan and Butler, Troy and Dawson, Clint},
  title         = {A Data-Consistent Approach to Ensemble Filtering},
  year          = {2026},
  eprint        = {2605.11282},
  archivePrefix = {arXiv},
  primaryClass  = {math.ST},
  doi           = {10.48550/arXiv.2605.11282},
  url           = {https://arxiv.org/abs/2605.11282}
}
```

The entry will be updated with the journal reference once the paper is
published.

## License

MIT License (`LICENSE`), © 2026 Rylan Spence.
