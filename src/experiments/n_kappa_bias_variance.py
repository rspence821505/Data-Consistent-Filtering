"""Ensemble-size by retained-rank ``(N, kappa)`` bias--variance experiment (notebook 10, section 9).

Appendix~D of the paper makes the QPCA-EnDCF error bounds depend on *two*
design numbers rather than one.  Writing $d=mL$ for the stacked observation
dimension, the bias bound carries a truncation term

    ||(I - Pi_{kappa,s}) mu_{E,s}||^2,

which can only shrink as ``kappa`` grows, and both the bias and the variance
bounds carry a projector-estimation term with the stability factor

    kappa / delta_kappa^2,      delta_kappa = lambda_kappa - lambda_{kappa+1},

which grows with ``kappa`` and with the closing of the cutoff gap.  The
remaining terms are O(1/N).  So the theory predicts a tradeoff *in kappa* at
fixed ``N``, a decay *in N* at fixed ``kappa``, and an interaction between them
-- none of which the published experiments touch, because every one of them
pins ``kappa = 1`` and ``N = 10``.

This module measures that surface.  It runs QPCA-EnDCF over the grid

    N     in {5, 10, 20, 40},
    kappa in {1, 2, 4, 8},        kappa <= N - 1,

in two regimes, and stores, per ``(regime, N, kappa, truth seed)``:

* the bias--variance decomposition of the analysis-mean error (the same
  window-averaged, state-dimension-normalized estimators the rest of notebook 10
  uses, via ``restart_vs_cycled.summarize_cell``);
* per-realization spectral diagnostics at the *retained/discarded boundary*
  rather than at the ``kappa = 1`` boundary the published diagnostics use; and
* per-window projector diagnostics that need more than one realization --
  chiefly the across-realization variability of the empirical rank-``kappa``
  projector, which is the empirical stand-in for the theory's
  ``E||Pi_hat - Pi||_F^2``.

Regimes
-------
``cycled``
    Ordinary cycling.  The analysis ensemble of window ``w`` is the forecast
    ensemble of window ``w+1``, which is what the filter actually does and what
    every other section of notebook 10 measures.

``restart``
    The theory-compatible companion.  Every evaluated window starts from a
    freshly drawn forecast ensemble, is analysed once, and is discarded.  The
    shared forecast-mean error ``b`` is drawn ONCE PER WINDOW and shared by all
    ``M`` realizations (``mean_error_mode="per_window"``), so conditional on the
    window every realization faces the *same* forecast distribution and the M
    ensembles are i.i.d. draws from it.  That is exactly the setting of
    Definition "Bias--Variance Assumptions" in Appendix~D: the expectation
    defining bias and variance is over independent ensemble draws with the truth
    and the observations held fixed.

    Section 7's primary design uses ``per_realization`` instead, which keeps the
    forecast-mean uncertainty a real forecast ensemble carries but puts an
    N-independent floor of order ``tr(P_mean)/n`` into the across-realization
    variance.  That floor is harmless for the A_cycle ratio section 7 reports and
    fatal for a question about how variance responds to ``N`` and ``kappa``, so
    this experiment uses the companion design as its primary restart regime and
    says so wherever a number is read off it.

Restart law
-----------
The restart forecast law ``(P_mean, P_within)`` is held **fixed across the whole
grid** and is the one section 7 already calibrated: the common, equal-weight pool
over the three filters, estimated at ``N = 10`` and ``kappa = 1`` on truth seeds
disjoint from the evaluation set.  Holding it fixed is the same argument section 7
makes for holding it fixed across ``N``, applied to ``kappa`` as well: if the law
moved with the design point, the response surface would mix the sampling effect
the theory predicts with a moving target distribution and would answer no clean
question.  ``resolve_calibration_law`` finds that law on disk, verifies it was
built under a matching configuration, and records its identifier in the manifest
and in every checkpoint; only if no matching run exists does it calibrate one
itself, under exactly the same specification, and cache it.

Pairing
-------
The ensemble streams are keyed by ``(truth seed, window)`` and
``(truth seed, window, N)`` and are **not** keyed by ``kappa`` (QPCA-EnDCF draws
no observation perturbations at all).  Consequently, at a fixed ``N``:

* in the restart regime every ``kappa`` sees bit-identical forecast ensembles at
  every evaluated window, so the ``kappa`` comparison is paired at the level of
  the ensemble draw rather than merely at the level of the truth seed;
* in the cycled regime every ``kappa`` starts from a bit-identical window-0
  ensemble and diverges only through the analyses themselves.

Run it as::

    python -m src.experiments.n_kappa_bias_variance --n-kappa --smoke-test
    python -m src.experiments.n_kappa_bias_variance --n-kappa --n-jobs 10

Everything is resumable: one checkpoint per ``(regime, N, kappa, truth seed)``
cell, keyed by a deterministic configuration hash, and a re-run skips whatever
is already complete.

What this experiment does NOT establish is listed in ``INTERPRETATION_CAVEATS``
below and repeated in the notebook.  The single most important entry: the
population objects of Appendix~D -- ``Sigma_{E,s}``, its eigenvalues, the
population projector ``Pi_{kappa,s}`` and the population residual mean
``mu_{E,s}`` -- are not observable here.  Every diagnostic below is an
*empirical* quantity, and the empirical cutoff gap is a plug-in proxy for
``delta_kappa`` that is itself biased by finite ``N``.
"""

from __future__ import annotations

# BLAS threading must be pinned *before* numpy is imported, otherwise every
# worker process spawns its own thread pool and they fight over the cores.
import os as _os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DIAGNOSTICS_EXPERIMENT_DIR, DIAGNOSTICS_RESULTS_DIR
from src.experiments.bias_variance_mc import (
    _advance,
    _localized_gain,
    _lowrank_gain,
    _thin_svd,
    generate_truth_batch,
)
from src.experiments.restart_vs_cycled import (
    CALIBRATION_SEED_OFFSET,
    DEFAULT_RESULTS_ROOT as RESTART_RESULTS_ROOT,
    RNG_BASE_ENTROPY,
    TRUTH_SEED_BASE,
    ExperimentSettings as RestartSettings,
    RestartDistribution,
    _config_for_windows,
    _diverged_realizations,
    _iter_tasks,
    _sanitize,
    _slug,
    build_window_context,
    calibrate_restart_distribution,
    config_hash,
    discover_runs,
    load_calibration,
    make_seed_sets,
    percentile_interval,
    restart_anomaly_generator,
    restart_mean_generator,
    save_calibration,
    summarize_cell,
    validate_kappa,
)
from src.utils.config import load_experiment_config

# --------------------------------------------------------------------------- #
# Constants and design decisions
# --------------------------------------------------------------------------- #

#: The two regimes compared.  ``cycled`` first, because it is what the filter
#: actually does; ``restart`` is the theory-compatible companion.
REGIMES = ("cycled", "restart")

#: The only filter in this experiment.  The question is about QPCA-EnDCF's own
#: two design numbers; a stochastic EnKF has no ``kappa`` and would contribute a
#: flat line to every panel.
FILTER_NAME = "QPCA-EnDCF"

DEFAULT_RESULTS_ROOT = DIAGNOSTICS_RESULTS_DIR / "n_kappa_bias_variance"
DEFAULT_CONFIG_PATH = DIAGNOSTICS_EXPERIMENT_DIR / "n_kappa_bias_variance.yaml"

#: Where a section 7 run's already-calibrated restart law is looked for.
DEFAULT_CALIBRATION_SEARCH_ROOT = RESTART_RESULTS_ROOT

#: The calibration specification.  Identical to section 7's, deliberately: if a
#: section 7 run on disk was built under exactly this spec, its law is *the same
#: object* this module would compute, and reusing it costs nothing and keeps the
#: two experiments on one restart distribution.
CALIBRATION_SPEC = {
    "filter_names": ("Seq-EnKF", "4D-EnKF", "QPCA-EnDCF"),
    "kappa": 1,
    "calibration_ensemble_size": 10,
    "calibration_n_mc": 20,
    "calibration_n_windows": 300,
    "calibration_spinup_windows": 50,
    "eigenvalue_floor": 1.0e-6,
    "shrinkage": 0.0,
    "n_calibration_seeds": 10,
    "restart_mode": "common",
}

#: Per-realization diagnostics, in stored column order.  Appending is safe;
#: reordering is not, so new entries go on the end.
#:
#: ``normalized_cutoff_gap`` is ``(lambda_kappa - lambda_{kappa+1}) / lambda_1``,
#: i.e. the gap at the RETAINED/DISCARDED boundary, normalized by the leading
#: eigenvalue so it is dimensionless and comparable across ``N``.  Note this is
#: *not* the quantity ``bias_variance_mc`` stores under the same name: that one
#: is hard-wired to the ``kappa = 1`` boundary, ``(lambda_1 - lambda_2)/lambda_1``,
#: which coincides with this only at ``kappa = 1``.
#:
#: ``cutoff_gap`` is the same difference unnormalized, so the theory's stability
#: factor ``kappa / delta_kappa^2`` can be formed directly from what is stored.
REALIZATION_DIAGNOSTICS = (
    "leading_eigenvalue",
    "leading_eigenvalue_fraction",
    "retained_variance_fraction",
    "cutoff_gap",
    "normalized_cutoff_gap",
    "stability_factor",
    "retained_mean_mismatch_fraction",
    "discarded_mean_mismatch_fraction",
    "mean_residual_energy",
)

#: Per-window diagnostics: the ones that need more than one realization.
#:
#: ``projector_variability`` is ``kappa - ||Pbar||_F^2`` with
#: ``Pbar = M^-1 sum_t Pi_hat_t``.  Two equivalent readings, both exact:
#:
#:     kappa - ||Pbar||_F^2
#:       = M^-1 sum_t ||Pi_hat_t - Pbar||_F^2                (a Frobenius variance)
#:       = (2 M^2)^-1 sum_{t,t'} ||Pi_hat_t - Pi_hat_{t'}||_F^2
#:       = mean over realization pairs of sum_i sin^2(theta_i),
#:
#: where ``theta_i`` are the principal angles between the two retained subspaces.
#: Dividing by ``kappa`` therefore gives a number in [0, 1]: the mean squared
#: sine of the principal angles between two independently estimated retained
#: subspaces.  0 means every realization found the same subspace.
#:
#: ``projector_error_vs_mean_subspace`` is ``M^-1 sum_t ||Pi_hat_t - Pibar||_F^2``
#: with ``Pibar`` the rank-``kappa`` projector onto the leading eigenvectors of
#: ``Pbar``.  It is the closer analogue of the theory's
#: ``E||Pi_hat_{kappa,s} - Pi_{kappa,s}||_F^2``, with the across-realization mean
#: subspace standing in for the unobservable population one -- which makes it a
#: lower bound on the true quantity, not an estimate of it.
PROJECTOR_DIAGNOSTICS = (
    "projector_variability",
    "projector_variability_normalized",
    "projector_error_vs_mean_subspace",
    "pooled_mean_residual_energy",
    "pooled_truncation_energy",
    "pooled_discarded_mean_mismatch_fraction",
    "n_valid_realizations",
)

#: Metric columns produced by the bootstrap, in stored column order.
#: ``total`` is the decomposition-consistent total error
#: ``T = Bias^2_db + Variance``, which is what every comparison in this
#: experiment ranks on.  It sits below the stored plug-in MSE by exactly
#: ``Var/(M-1)``; both are carried so that relationship stays checkable.
METRIC_NAMES = ("mse", "bias_squared", "bias_squared_debiased", "variance", "total")

INTERPRETATION_CAVEATS = (
    "Every spectral quantity here is EMPIRICAL. Appendix D's delta_kappa, "
    "Pi_{kappa,s} and mu_{E,s} are population objects of the forecast law and are "
    "not observable from a finite ensemble. The stored cutoff gap is a plug-in "
    "proxy whose own bias grows as N shrinks, which is the regime the experiment "
    "is most interested in.",
    "projector_error_vs_mean_subspace measures deviation from the "
    "across-realization MEAN subspace, not from the population subspace. Because "
    "the mean subspace is itself fitted to the same realizations, this "
    "systematically understates E||Pi_hat - Pi||_F^2.",
    "The restart regime removes inherited filter dependence; it does NOT make the "
    "evaluated truth windows temporally independent, because they are drawn from "
    "one chaotic trajectory. The bootstrap therefore resamples whole truth seeds "
    "and never individual windows.",
    "The covariance-matched restart is a controlled statistical construction, not "
    "an operational forecasting procedure: it is centred on a perturbed truth, "
    "which no real forecast system has access to.",
    "kappa <= N - 1 is a rank constraint, not a tuning range: the centered "
    "whitened residual matrix has rank at most N - 1, so the grid is triangular "
    "and (N=5, kappa=8) does not exist. Comparisons 'across kappa at fixed N' are "
    "therefore over different kappa sets at different N.",
    "An empirically optimal kappa is the argmin of a noisy surface over a "
    "four-point grid. Where two design points are within each other's intervals, "
    "the ranking between them is not resolved by this experiment.",
    "All results are for one unlocalized, uninflated Lorenz-96 configuration at "
    "one observation density and noise level. Nothing here separates the "
    "truncation and projector-estimation terms of the Appendix D bound: it "
    "measures the total each design point produces, alongside diagnostics that "
    "are consistent or inconsistent with each mechanism.",
)


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #


def valid_design_points(
    ensemble_sizes: Sequence[int],
    kappa_values: Sequence[int],
    stacked_obs_dimension: int,
) -> List[Tuple[int, int]]:
    """The ``(N, kappa)`` pairs that exist, in row-major ``(N, kappa)`` order.

    ``kappa <= min(N - 1, d)`` is enforced rather than clipped: the centered
    whitened residual matrix has rank at most ``min(N - 1, d)``, so a larger
    ``kappa`` retains numerical-noise directions and the corresponding design
    point is not the same object as its neighbours.  ``validate_kappa`` raises on
    exactly this condition; here the pair is simply absent from the grid.
    """
    pairs = []
    for N in sorted({int(N) for N in ensemble_sizes}):
        for kappa in sorted({int(k) for k in kappa_values}):
            if 1 <= kappa <= min(N - 1, int(stacked_obs_dimension)):
                pairs.append((N, kappa))
    if not pairs:
        raise ValueError(
            f"No valid (N, kappa) pair in N={list(ensemble_sizes)}, "
            f"kappa={list(kappa_values)} under kappa <= min(N-1, d)."
        )
    return pairs


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass
class NKappaSettings:
    """Everything the driver needs beyond the base YAML model/observation block."""

    ensemble_sizes: Tuple[int, ...] = (5, 10, 20, 40)
    kappa_values: Tuple[int, ...] = (1, 2, 4, 8)

    n_mc: int = 60  # M, inner ensemble/filter realizations
    n_windows: int = 60  # number of EVALUATED windows
    window_stride: int = 5  # s
    spinup_windows: int = 50  # windows discarded before the first evaluated one

    n_evaluation_seeds: int = 20
    truth_seed_base: int = TRUTH_SEED_BASE
    calibration_seed_offset: int = CALIBRATION_SEED_OFFSET
    evaluation_seeds: Tuple[int, ...] = ()

    regimes: Tuple[str, ...] = REGIMES
    #: How the shared forecast-mean error ``b`` is drawn in the restart regime.
    #: ``per_window`` is the theory-compatible choice and the default here; see
    #: the module docstring for why it, rather than section 7's primary
    #: ``per_realization`` design, is what an N- and kappa-response study needs.
    mean_error_mode: str = "per_window"

    label: str = "grid"

    # Filled in by ``settings_from_config``; a plain default keeps the dataclass
    # constructible in tests.
    stacked_obs_dimension: int = 100

    def __post_init__(self) -> None:
        self.ensemble_sizes = tuple(int(N) for N in self.ensemble_sizes)
        self.kappa_values = tuple(int(k) for k in self.kappa_values)
        self.regimes = tuple(str(g) for g in self.regimes)
        unknown = [g for g in self.regimes if g not in REGIMES]
        if unknown:
            raise ValueError(f"Unknown regime(s) {unknown}; expected {list(REGIMES)}.")
        if self.mean_error_mode not in ("per_window", "per_realization"):
            raise ValueError(f"Unknown mean_error_mode: {self.mean_error_mode!r}")
        if self.window_stride < 1:
            raise ValueError("window_stride must be >= 1.")
        if self.n_windows < 1:
            raise ValueError("n_windows must be >= 1.")
        if self.n_mc < 2:
            raise ValueError("n_mc must be >= 2; a variance needs two realizations.")
        if not self.evaluation_seeds:
            # The same seed family, and the same evaluation/calibration split by
            # index, that section 7 and the published 100x100 run use -- so a
            # design point at N=10 sits on the same truth trajectories as those.
            _calibration, evaluation = make_seed_sets(
                CALIBRATION_SPEC["n_calibration_seeds"],
                self.n_evaluation_seeds,
                self.truth_seed_base,
                self.calibration_seed_offset,
            )
            self.evaluation_seeds = tuple(evaluation)
        self.evaluation_seeds = tuple(int(s) for s in self.evaluation_seeds)
        # Raises if a requested (N, kappa) grid is empty.
        valid_design_points(
            self.ensemble_sizes, self.kappa_values, self.stacked_obs_dimension
        )

    @property
    def calibration_seeds(self) -> Tuple[int, ...]:
        """The disjoint truth seeds the restart law is calibrated on."""
        calibration, _evaluation = make_seed_sets(
            CALIBRATION_SPEC["n_calibration_seeds"],
            max(self.n_evaluation_seeds, len(self.evaluation_seeds)),
            self.truth_seed_base,
            self.calibration_seed_offset,
        )
        return tuple(int(s) for s in calibration)

    @property
    def design_points(self) -> List[Tuple[int, int]]:
        return valid_design_points(
            self.ensemble_sizes, self.kappa_values, self.stacked_obs_dimension
        )

    @property
    def evaluation_windows(self) -> np.ndarray:
        """0-based indices of the evaluated windows.

        Window ``w`` starts at truth index ``w * L`` and ends at ``(w + 1) * L``;
        the recorded analysis mean is the one at the window endpoint.
        """
        return self.spinup_windows + self.window_stride * np.arange(self.n_windows)

    @property
    def total_cycled_windows(self) -> int:
        """Windows the cycled regime must integrate to reach the last evaluated one."""
        return int(self.evaluation_windows[-1]) + 1


def settings_from_config(config: Dict, **overrides) -> NKappaSettings:
    """Build settings from the YAML ``n_kappa`` block plus command-line overrides."""
    block = dict(config.get("n_kappa", {}) or {})
    block.update({k: v for k, v in overrides.items() if v is not None})
    known = set(NKappaSettings.__dataclass_fields__)
    unknown = set(block) - known
    if unknown:
        raise ValueError(f"Unknown n_kappa settings: {sorted(unknown)}")
    block["stacked_obs_dimension"] = int(config["observations"]["m"]) * int(
        config["assimilation"]["window_length"]
    )
    return NKappaSettings(**block)


def restart_settings_for_calibration(
    config: Dict, settings: NKappaSettings
) -> RestartSettings:
    """A ``restart_vs_cycled`` settings object describing only the calibration.

    Used for two things and nothing else: to compute the fallback restart law
    through the section 7 code path, and to state precisely what a section 7 run
    on disk has to match before its law is reused.  The evaluation-side fields
    are section 7 defaults and are never exercised.
    """
    return RestartSettings(
        ensemble_sizes=(CALIBRATION_SPEC["calibration_ensemble_size"],),
        filter_names=CALIBRATION_SPEC["filter_names"],
        kappa=CALIBRATION_SPEC["kappa"],
        n_calibration_seeds=CALIBRATION_SPEC["n_calibration_seeds"],
        n_evaluation_seeds=len(settings.evaluation_seeds),
        truth_seed_base=settings.truth_seed_base,
        calibration_seed_offset=settings.calibration_seed_offset,
        calibration_ensemble_size=CALIBRATION_SPEC["calibration_ensemble_size"],
        calibration_n_mc=CALIBRATION_SPEC["calibration_n_mc"],
        calibration_n_windows=CALIBRATION_SPEC["calibration_n_windows"],
        calibration_spinup_windows=CALIBRATION_SPEC["calibration_spinup_windows"],
        restart_mode=CALIBRATION_SPEC["restart_mode"],
        eigenvalue_floor=CALIBRATION_SPEC["eigenvalue_floor"],
        shrinkage=CALIBRATION_SPEC["shrinkage"],
        label="n_kappa_calibration",
        stacked_obs_dimension=settings.stacked_obs_dimension,
    )


# --------------------------------------------------------------------------- #
# Restart law: reuse section 7's, or compute one to the same specification
# --------------------------------------------------------------------------- #


def _model_block(config: Dict) -> Dict:
    return {
        "model": {
            "n": int(config["model"]["n"]),
            "F": float(config["model"]["F"]),
            "dt": float(config["model"]["dt"]),
            "spinup_steps": int(config["model"].get("spinup_steps", 0)),
        },
        "observations": {
            "m": int(config["observations"]["m"]),
            "observe_every": int(config["observations"]["observe_every"]),
            "noise_std": float(config["observations"]["noise_std"]),
            "steps_per_obs": int(config["observations"]["steps_per_obs"]),
        },
        "assimilation": {
            "window_length": int(config["assimilation"]["window_length"]),
        },
    }


def _calibration_matches(manifest: Dict, config: Dict, settings: NKappaSettings) -> bool:
    """True if a ``restart_vs_cycled`` run's law was built to ``CALIBRATION_SPEC``.

    Every field that changes what the calibration computed is compared: the
    model and observation blocks, the calibration design, the filter set it was
    pooled over, the ``kappa`` those filters ran at, the stabilization, and the
    calibration truth seeds themselves.  Anything that only changes the
    *evaluation* side of a section 7 run -- its ensemble sizes, its window
    design, its evaluation seeds -- is ignored, because the law does not depend
    on it.
    """
    try:
        scientific = manifest["scientific_config"]
        restart = scientific["restart_distribution"]
        design = scientific["design"]
    except KeyError:
        return False

    ours = _model_block(config)
    for block in ours:
        if scientific.get(block) != ours[block]:
            return False
    if tuple(scientific["filters"]["names"]) != CALIBRATION_SPEC["filter_names"]:
        return False
    if int(scientific["filters"]["kappa"]) != CALIBRATION_SPEC["kappa"]:
        return False
    if scientific["filters"].get("qpca_localization_radius") != (
        config.get("filters", {}).get("qpca_endcf", {}) or {}
    ).get("localization_radius", None):
        return False
    if restart.get("mode") != CALIBRATION_SPEC["restart_mode"]:
        return False
    for key in (
        "calibration_ensemble_size",
        "calibration_n_mc",
        "calibration_n_windows",
        "calibration_spinup_windows",
    ):
        if int(restart.get(key, -1)) != int(CALIBRATION_SPEC[key]):
            return False
    if float(restart.get("eigenvalue_floor", -1.0)) != float(
        CALIBRATION_SPEC["eigenvalue_floor"]
    ):
        return False
    if float(restart.get("shrinkage", -1.0)) != float(CALIBRATION_SPEC["shrinkage"]):
        return False
    if int(design.get("rng_base_entropy", -1)) != int(RNG_BASE_ENTROPY):
        return False
    return tuple(int(s) for s in design.get("calibration_seeds", ())) == tuple(
        settings.calibration_seeds
    )


def _calibration_cache_path(config: Dict, settings: NKappaSettings, root: Path) -> Path:
    """Where a locally computed law is cached, keyed by its own specification."""
    payload = dict(_model_block(config))
    payload["calibration"] = {
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in CALIBRATION_SPEC.items()},
        "calibration_seeds": [int(s) for s in settings.calibration_seeds],
        "rng_base_entropy": int(RNG_BASE_ENTROPY),
        "qpca_localization_radius": (
            config.get("filters", {}).get("qpca_endcf", {}) or {}
        ).get("localization_radius", None),
    }
    digest = config_hash(payload)
    return Path(root) / "_calibration" / f"restart_distribution_{digest[:12]}.npz"


def resolve_calibration_law(
    config: Dict,
    settings: NKappaSettings,
    results_root: Path | None = None,
    search_root: Path | None = None,
    n_jobs: int = 1,
    verbose: int = 0,
    allow_calibration: bool = True,
) -> Tuple[RestartDistribution, Dict[str, object]]:
    """The fixed restart law for the whole grid, plus its provenance.

    Resolution order, most-preferred first:

    1. a ``restart_vs_cycled`` run on disk whose manifest matches
       ``CALIBRATION_SPEC`` exactly -- its law is the same object this module
       would compute, and reusing it keeps both experiments on one distribution;
    2. a law this module has already computed and cached under
       ``<results_root>/_calibration/``;
    3. a fresh calibration through ``restart_vs_cycled.calibrate_restart_distribution``,
       written to that cache.

    Raises rather than silently substituting a different law if
    ``allow_calibration`` is False and neither (1) nor (2) is available.
    """
    results_root = Path(results_root) if results_root is not None else DEFAULT_RESULTS_ROOT
    search_root = (
        Path(search_root) if search_root is not None else DEFAULT_CALIBRATION_SEARCH_ROOT
    )

    for run in discover_runs(search_root):
        path = Path(run["results_dir"]) / "calibration" / "restart_distribution.npz"
        if not path.exists() or not _calibration_matches(run["manifest"], config, settings):
            continue
        calibration = load_calibration(path)
        law = calibration["common"]
        return law, {
            "source": "restart_vs_cycled run",
            "path": str(path),
            "run_label": run["label"],
            "run_config_hash": run["config_hash"],
            "identifier": law.identifier,
            "name": law.name,
        }

    cache_path = _calibration_cache_path(config, settings, results_root)
    if cache_path.exists():
        law = load_calibration(cache_path)["common"]
        return law, {
            "source": "n_kappa calibration cache",
            "path": str(cache_path),
            "identifier": law.identifier,
            "name": law.name,
        }

    if not allow_calibration:
        raise FileNotFoundError(
            "No restart law matching CALIBRATION_SPEC was found under "
            f"{search_root} or {cache_path.parent}. Re-run without "
            "--no-calibrate to compute one here, or pass "
            "--reuse-published-restart-law to scripts/run_all_experiments.py."
        )

    calibration_settings = restart_settings_for_calibration(config, settings)
    t0 = time.perf_counter()
    print(
        "No matching restart law on disk; calibrating one to CALIBRATION_SPEC on "
        f"{len(calibration_settings.calibration_seeds)} held-out truth seeds "
        f"(N={calibration_settings.calibration_ensemble_size}, "
        f"M={calibration_settings.calibration_n_mc}, "
        f"W={calibration_settings.calibration_n_windows})...",
        flush=True,
    )
    calibration = calibrate_restart_distribution(
        config, calibration_settings, n_jobs=n_jobs, verbose=verbose
    )
    save_calibration(cache_path, calibration, calibration_settings)
    print(f"  done in {time.perf_counter() - t0:.1f}s -> {cache_path}", flush=True)
    law = calibration["common"]
    return law, {
        "source": "calibrated here",
        "path": str(cache_path),
        "identifier": law.identifier,
        "name": law.name,
    }


# --------------------------------------------------------------------------- #
# Configuration hashing
# --------------------------------------------------------------------------- #


def scientific_config(
    config: Dict, settings: NKappaSettings, restart_distribution_id: str
) -> Dict:
    """The configuration fields a checkpoint's validity depends on.

    Excludes anything that changes only how the run is scheduled (worker count,
    verbosity, output paths, resume behaviour) and includes everything that
    changes what is computed -- including the identifier of the restart law,
    because a different law is a different experiment even at identical settings.
    """
    scientific = _model_block(config)
    scientific["filters"] = {
        "names": [FILTER_NAME],
        "kappa_values": [int(k) for k in settings.kappa_values],
        "inflation": {FILTER_NAME: 1.0},
        "qpca_localization_radius": (
            config.get("filters", {}).get("qpca_endcf", {}) or {}
        ).get("localization_radius", None),
    }
    scientific["design"] = {
        "n_mc": int(settings.n_mc),
        "n_windows": int(settings.n_windows),
        "window_stride": int(settings.window_stride),
        "spinup_windows": int(settings.spinup_windows),
        "ensemble_sizes": [int(N) for N in settings.ensemble_sizes],
        "design_points": [[int(N), int(k)] for N, k in settings.design_points],
        "regimes": list(settings.regimes),
        "truth_seed_base": int(settings.truth_seed_base),
        "calibration_seed_offset": int(settings.calibration_seed_offset),
        "evaluation_seeds": [int(s) for s in settings.evaluation_seeds],
        "rng_base_entropy": int(RNG_BASE_ENTROPY),
    }
    scientific["restart_distribution"] = {
        "identifier": str(restart_distribution_id),
        "mean_error_mode": settings.mean_error_mode,
        "spec": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in CALIBRATION_SPEC.items()
        },
    }
    return scientific


# --------------------------------------------------------------------------- #
# One-window QPCA-EnDCF kernel with spectral and projector diagnostics
# --------------------------------------------------------------------------- #
#
# The analysis algebra below is the *imported* helpers -- ``_advance``,
# ``_thin_svd``, ``_lowrank_gain``, ``_localized_gain`` -- in the same order and
# with the same arguments as ``restart_vs_cycled.run_one_window``'s QPCA branch,
# so this kernel and that one are the same filter.  It is not a re-derivation:
# ``tests/test_n_kappa_bias_variance.py`` asserts the two produce bit-identical
# analysis means on the same input, which is what stops the two drifting apart.
#
# What is added is diagnostics, and they are computed from quantities the update
# already formed (``singular_values``, ``V_kappa``, ``e_bar``), so collecting
# them changes no arithmetic that feeds the analysis.


def _projector_summary(
    projector_sum: np.ndarray,
    residual_mean_sum: np.ndarray,
    kappa: int,
    n_valid: int,
) -> Dict[str, float]:
    """Reduce the within-window accumulators to the per-window diagnostics.

    ``projector_sum`` is ``sum_t Pi_hat_t`` and ``residual_mean_sum`` is
    ``sum_t e_bar_t``, both over the realizations that stayed finite at this
    window.  Everything below is exact algebra on those two objects; nothing is
    estimated twice.
    """
    empty = {name: np.nan for name in PROJECTOR_DIAGNOSTICS}
    empty["n_valid_realizations"] = float(n_valid)
    if n_valid < 2:
        # A single realization has no across-realization variability to measure.
        # Recorded as NaN rather than as 0, which would read as "measured, and
        # the subspace was perfectly stable".
        return empty

    mean_projector = projector_sum / n_valid
    # M^-1 sum_t ||Pi_t - Pbar||_F^2 = M^-1 sum_t ||Pi_t||_F^2 - ||Pbar||_F^2,
    # and ||Pi_t||_F^2 = tr(Pi_t) = kappa exactly for any rank-kappa projector.
    variability = float(kappa) - float(np.sum(mean_projector * mean_projector))
    # Clipped only against roundoff: the identity above makes it non-negative.
    variability = max(variability, 0.0)

    values, vectors = np.linalg.eigh(mean_projector)
    order = np.argsort(values)[::-1]
    leading = vectors[:, order[:kappa]]  # the mean retained subspace
    # M^-1 sum_t ||Pi_t - Pibar||_F^2 = 2 kappa - 2 tr(Pbar Pibar)
    #                                 = 2 kappa - 2 sum_{i<=kappa} mu_i(Pbar).
    error_vs_mean = 2.0 * float(kappa) - 2.0 * float(
        np.sum(values[order[:kappa]])
    )
    error_vs_mean = max(error_vs_mean, 0.0)

    pooled_mean = residual_mean_sum / n_valid
    pooled_energy = float(pooled_mean @ pooled_mean)
    retained = leading.T @ pooled_mean
    truncation_energy = max(pooled_energy - float(retained @ retained), 0.0)

    return {
        "projector_variability": variability,
        "projector_variability_normalized": variability / float(kappa),
        "projector_error_vs_mean_subspace": error_vs_mean,
        "pooled_mean_residual_energy": pooled_energy,
        "pooled_truncation_energy": truncation_energy,
        "pooled_discarded_mean_mismatch_fraction": (
            truncation_energy / pooled_energy if pooled_energy > 0.0 else np.nan
        ),
        "n_valid_realizations": float(n_valid),
    }


def run_qpca_window(
    block: np.ndarray,
    obs_window: np.ndarray,
    H: np.ndarray,
    context,
    M: int,
    N: int,
    kappa: int,
    collect_diagnostics: bool = False,
    exclude: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, Dict[str, float] | None]:
    """Advance ``(n, M*N)`` through one window, analyse it, and optionally diagnose it.

    Parameters
    ----------
    block : ndarray, shape (n, M*N)
        Window-initial forecast ensembles; realization ``t`` owns columns
        ``t*N : (t+1)*N``.
    obs_window : ndarray, shape (L, m)
        The window's observations, in time order.
    kappa : int
        Retained rank.  Must satisfy ``1 <= kappa <= min(N - 1, d)``.
    collect_diagnostics : bool
        Compute the spectral and projector diagnostics.  Off for the cycled
        regime's non-evaluated windows, which exist only to carry the ensemble
        forward.
    exclude : ndarray of bool, shape (M,), optional
        Realizations already known to be dead.  ``_sanitize`` replaces a blown-up
        realization's columns with zeros so LAPACK is not handed NaNs on the
        healthy realizations' behalf, and a zero Lorenz-96 state is perfectly
        finite -- it relaxes toward F.  A window that re-tested from scratch would
        therefore find such a realization "recovered" and fold a fabricated
        trajectory into the across-realization projector statistics.  The cycled
        runner passes its accumulated divergence mask here so that cannot happen.

    Returns
    -------
    block, analysis_means (M, n), diverged (M,), realization diagnostics
    (M, len(REALIZATION_DIAGNOSTICS)) or None, per-window diagnostics or None.
    """
    n, m, L = context.n, context.m, context.L
    mL = m * L

    Y_window = np.empty((L, m, M, N))
    for ell in range(L):
        block = _advance(block, context.steps_per_obs, context.dt, context.F)
        Y_window[ell] = (H @ block).reshape(m, M, N)

    # As in the reference kernel: the 4D filters analyse only at the window end,
    # so a non-finite value propagates and is caught once.
    diverged = _diverged_realizations(block, M, N) | ~np.isfinite(Y_window).all(
        axis=(0, 1, 3)
    )
    if exclude is not None:
        diverged = diverged | np.asarray(exclude, dtype=bool)
    _sanitize(block, diverged, M, N)
    Y_window[:, :, diverged, :] = 0.0

    z_stack = obs_window.ravel()
    Xv = block.reshape(n, M, N)

    realization_diagnostics = (
        np.full((M, len(REALIZATION_DIAGNOSTICS)), np.nan) if collect_diagnostics else None
    )
    projector_sum = np.zeros((mL, mL)) if collect_diagnostics else None
    residual_mean_sum = np.zeros(mL) if collect_diagnostics else None
    n_valid = 0

    for i in range(M):
        if diverged[i]:
            continue
        Y_stack = Y_window[:, :, i, :].reshape(mL, N)
        E = context.whiten @ (Y_stack - z_stack[:, None])
        e_bar = E.mean(axis=1)
        E_c = E - e_bar[:, None]
        eigenvecs, singular_values, _vt = _thin_svd(E_c)
        V_kappa = eigenvecs[:, :kappa]
        Delta_obs = context.R_L_chol @ (-V_kappa @ (V_kappa.T @ E))

        X_i = Xv[:, i, :]
        A_x = X_i - X_i.mean(axis=1, keepdims=True)
        A_y = Y_stack - Y_stack.mean(axis=1, keepdims=True)
        if context.taper is None:
            K = _lowrank_gain(A_x, A_y, context.eps_qpca, N)
        else:
            K = _localized_gain(A_x, A_y, context.taper, N)
        Xv[:, i, :] = X_i + K @ Delta_obs

        if not collect_diagnostics:
            continue

        # ---- diagnostics, from quantities the update already formed --------- #
        # The nonzero eigenvalues of the empirical residual covariance
        # Sigma_hat_E = E_c E_c^T / (N - 1) are singular_values^2 / (N - 1), in
        # descending order.  Beyond index N-2 they are numerically zero: the
        # centered matrix has rank at most N-1, which is also why kappa is capped
        # there.  The paper's convention lambda_{d+1} := 0 extends the spectrum
        # with zeros, which is what the padding below reproduces.
        eigenvalues = singular_values**2 / (N - 1)
        leading = float(eigenvalues[0])
        spectrum_sum = float(eigenvalues.sum())
        lambda_kappa = float(eigenvalues[kappa - 1])
        lambda_next = float(eigenvalues[kappa]) if eigenvalues.size > kappa else 0.0
        gap = lambda_kappa - lambda_next
        normalized_gap = gap / leading if leading > 0.0 else np.nan
        stability = kappa / gap**2 if gap > 0.0 else np.inf

        mean_energy = float(e_bar @ e_bar)
        retained = V_kappa.T @ e_bar
        retained_fraction = (
            float(retained @ retained) / mean_energy if mean_energy > 0.0 else np.nan
        )

        realization_diagnostics[i] = (
            leading,
            leading / spectrum_sum if spectrum_sum > 0.0 else np.nan,
            float(eigenvalues[:kappa].sum()) / spectrum_sum
            if spectrum_sum > 0.0
            else np.nan,
            gap,
            normalized_gap,
            stability,
            retained_fraction,
            1.0 - retained_fraction if np.isfinite(retained_fraction) else np.nan,
            mean_energy,
        )
        projector_sum += V_kappa @ V_kappa.T
        residual_mean_sum += e_bar
        n_valid += 1

    analysis_means = block.reshape(n, M, N).mean(axis=2).T.copy()  # (M, n)
    analysis_means[diverged] = np.nan

    window_diagnostics = (
        _projector_summary(projector_sum, residual_mean_sum, kappa, n_valid)
        if collect_diagnostics
        else None
    )
    return block, analysis_means, diverged, realization_diagnostics, window_diagnostics


# --------------------------------------------------------------------------- #
# Cell runners
# --------------------------------------------------------------------------- #


def _empty_diagnostics(M: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.full((M, W, len(REALIZATION_DIAGNOSTICS)), np.nan),
        np.full((W, len(PROJECTOR_DIAGNOSTICS)), np.nan),
    )


def _store_window_diagnostics(
    target: np.ndarray, index: int, summary: Dict[str, float]
) -> None:
    target[index] = [summary[name] for name in PROJECTOR_DIAGNOSTICS]


def run_restart_cell(
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    context,
    distribution: RestartDistribution,
    truth_seed: int,
    ensemble_size: int,
    kappa: int,
    n_realizations: int,
    evaluation_windows: Sequence[int],
    mean_error_mode: str = "per_window",
) -> Dict[str, np.ndarray]:
    """Independent-restart regime for one ``(N, kappa, truth seed)`` cell.

    Each evaluated window draws a fresh ensemble, runs exactly one window, and
    discards it.  The loop body holds the only reference to ``block``, so there
    is no path by which an analysis ensemble could reach the next window.

    With ``mean_error_mode="per_window"`` every realization at a window faces the
    same forecast distribution, so the M ensembles are i.i.d. draws from it and
    the across-realization bias and variance are the objects Appendix D defines.
    """
    L, n = context.L, context.n
    M, N = int(n_realizations), int(ensemble_size)
    shared_mean = mean_error_mode == "per_window"
    W = len(evaluation_windows)

    errors = np.full((M, W, n), np.nan)
    diverged = np.zeros((M, W), dtype=bool)
    realization_diagnostics, window_diagnostics = _empty_diagnostics(M, W)

    for index, w in enumerate(evaluation_windows):
        w = int(w)
        block = distribution.draw(
            truth[w * L],
            N,
            M,
            restart_mean_generator(truth_seed, w),
            restart_anomaly_generator(truth_seed, w, N),
            shared_mean_across_realizations=shared_mean,
        )
        block, analysis_means, failed, realization, window = run_qpca_window(
            block,
            observations[w * L : (w + 1) * L],
            H,
            context,
            M,
            N,
            kappa,
            collect_diagnostics=True,
        )
        errors[:, index, :] = analysis_means - truth[(w + 1) * L][None, :]
        diverged[:, index] = failed
        realization_diagnostics[:, index, :] = realization
        _store_window_diagnostics(window_diagnostics, index, window)
        del block  # explicit: nothing survives to the next evaluated window

    return {
        "errors": errors,
        "diverged": diverged,
        "realization_diagnostics": realization_diagnostics,
        "window_diagnostics": window_diagnostics,
    }


def run_cycled_cell(
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    context,
    distribution: RestartDistribution,
    truth_seed: int,
    ensemble_size: int,
    kappa: int,
    n_realizations: int,
    evaluation_windows: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Ordinary cycling for one ``(N, kappa, truth seed)`` cell.

    The window-0 ensemble is drawn from the *same* law with the *same*
    ``(r, w=0)`` streams the restart regime would use, so the two regimes start
    from identical ensembles and diverge only through cycling.  The window-0 draw
    always uses the per-realization mean error, whatever the restart regime is
    configured to do: cycled realizations must start from genuinely distinct
    ensembles, and the initialization is forgotten long before the first
    evaluated window in any case.
    """
    L, n = context.L, context.n
    M, N = int(n_realizations), int(ensemble_size)
    evaluation_windows = [int(w) for w in evaluation_windows]
    window_index = {w: i for i, w in enumerate(evaluation_windows)}
    total_windows = max(evaluation_windows) + 1
    W = len(evaluation_windows)

    block = distribution.draw(
        truth[0],
        N,
        M,
        restart_mean_generator(truth_seed, 0),
        restart_anomaly_generator(truth_seed, 0, N),
        shared_mean_across_realizations=False,
    )

    errors = np.full((M, W, n), np.nan)
    diverged = np.zeros((M, W), dtype=bool)
    ever_diverged = np.zeros(M, dtype=bool)
    realization_diagnostics, window_diagnostics = _empty_diagnostics(M, W)

    for w in range(total_windows):
        evaluated = w in window_index
        # Once a cycled realization has gone non-finite it never recovers, so the
        # accumulated mask is carried into the kernel rather than re-derived
        # there: a sanitized realization is finite from the next window on, and
        # would otherwise be counted as healthy in the projector statistics.
        block, analysis_means, failed, realization, window = run_qpca_window(
            block,
            observations[w * L : (w + 1) * L],
            H,
            context,
            M,
            N,
            kappa,
            collect_diagnostics=evaluated,
            exclude=ever_diverged,
        )
        ever_diverged |= failed
        if not evaluated:
            continue
        index = window_index[w]
        errors[:, index, :] = analysis_means - truth[(w + 1) * L][None, :]
        diverged[:, index] = ever_diverged
        errors[ever_diverged, index, :] = np.nan
        realization[ever_diverged] = np.nan
        realization_diagnostics[:, index, :] = realization
        _store_window_diagnostics(window_diagnostics, index, window)

    return {
        "errors": errors,
        "diverged": diverged,
        "realization_diagnostics": realization_diagnostics,
        "window_diagnostics": window_diagnostics,
    }


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


def cell_checkpoint_path(
    results_dir: Path, regime: str, ensemble_size: int, kappa: int, truth_seed: int
) -> Path:
    return (
        Path(results_dir)
        / "cells"
        / f"{regime}_{_slug(FILTER_NAME)}_N{int(ensemble_size)}"
        f"_k{int(kappa)}_truth_seed_{int(truth_seed)}.npz"
    )


def cell_checkpoint_matches(
    path: Path, expected_hash: str, n_mc: int, n_windows: int
) -> bool:
    """True only if the checkpoint was produced by the current configuration."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["config_hash"]) == expected_hash
                and int(data["n_realizations"]) == int(n_mc)
                and int(data["n_windows"]) == int(n_windows)
            )
    except (OSError, ValueError, KeyError):
        return False


def _cell_task(
    config: Dict,
    settings: NKappaSettings,
    regime: str,
    ensemble_size: int,
    kappa: int,
    truth_seed: int,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    distribution: RestartDistribution,
    results_dir: Path,
    expected_hash: str,
):
    """Run one ``(regime, N, kappa, truth seed)`` cell and checkpoint it."""
    started = time.perf_counter()
    try:
        validate_kappa(kappa, ensemble_size, settings.stacked_obs_dimension)
        context = build_window_context(config, kappa)
        windows = [int(w) for w in settings.evaluation_windows]

        common = (
            truth,
            observations,
            H,
            context,
            distribution,
            int(truth_seed),
            int(ensemble_size),
            int(kappa),
            int(settings.n_mc),
            windows,
        )
        result = (
            run_cycled_cell(*common)
            if regime == "cycled"
            else run_restart_cell(*common, mean_error_mode=settings.mean_error_mode)
        )

        summary = summarize_cell(result["errors"], result["diverged"])
        summary.update(
            {
                "realization_diagnostics": result["realization_diagnostics"].astype(
                    np.float32
                ),
                "window_diagnostics": result["window_diagnostics"],
                "realization_diagnostic_names": np.asarray(REALIZATION_DIAGNOSTICS),
                "window_diagnostic_names": np.asarray(PROJECTOR_DIAGNOSTICS),
                "config_hash": np.asarray(expected_hash),
                "experiment": np.asarray("n_kappa_bias_variance"),
                "regime": np.asarray(regime),
                "filter": np.asarray(FILTER_NAME),
                "ensemble_size": np.asarray(int(ensemble_size)),
                "kappa": np.asarray(int(kappa)),
                "truth_seed": np.asarray(int(truth_seed)),
                # generate_observations seeds the noise stream with the truth
                # seed, so the two are the same integer by construction; both are
                # recorded so that stays visible if it ever changes.
                "observation_noise_seed": np.asarray(int(truth_seed)),
                "evaluation_windows": np.asarray(windows),
                "window_stride": np.asarray(int(settings.window_stride)),
                "spinup_windows": np.asarray(int(settings.spinup_windows)),
                "restart_distribution_id": np.asarray(distribution.identifier),
                "restart_distribution_name": np.asarray(distribution.name),
                "mean_error_mode": np.asarray(
                    settings.mean_error_mode if regime == "restart" else "per_realization"
                ),
                "inflation": np.asarray(1.0),
                "localization_radius": np.asarray(
                    -1.0
                    if (config.get("filters", {}).get("qpca_endcf", {}) or {}).get(
                        "localization_radius"
                    )
                    is None
                    else float(config["filters"]["qpca_endcf"]["localization_radius"])
                ),
                "rng_base_entropy": np.asarray(int(RNG_BASE_ENTROPY)),
                "ensemble_stream_spec": np.asarray(
                    json.dumps(
                        {
                            "generator": "numpy.random.Generator "
                            "(SeedSequence over an integer entropy list)",
                            "shared_mean_error_b": [
                                RNG_BASE_ENTROPY,
                                0,
                                "truth_seed",
                                "window",
                            ],
                            "within_ensemble_anomaly": [
                                RNG_BASE_ENTROPY,
                                1,
                                "truth_seed",
                                "window",
                                "ensemble_size",
                            ],
                            "observation_perturbation": "none (QPCA-EnDCF is "
                            "deterministic after initialization)",
                            "kappa_keys_no_stream": True,
                        },
                        sort_keys=True,
                    )
                ),
            }
        )

        path = cell_checkpoint_path(
            results_dir, regime, ensemble_size, kappa, truth_seed
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **summary)
    except Exception as exc:  # noqa: BLE001 -- returned, so the pool survives
        return (
            regime,
            int(ensemble_size),
            int(kappa),
            int(truth_seed),
            time.perf_counter() - started,
            0,
            f"{type(exc).__name__}: {exc}",
        )

    return (
        regime,
        int(ensemble_size),
        int(kappa),
        int(truth_seed),
        time.perf_counter() - started,
        int(summary["n_diverged_realization_windows"]),
        None,
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def resolve_results_dir(
    config: Dict,
    settings: NKappaSettings,
    restart_distribution_id: str,
    root: Path | None = None,
) -> Tuple[Path, str, Dict]:
    """``(directory, hash, scientific config)`` for the current settings.

    The directory name carries the label *and* the leading 8 hex digits of the
    configuration hash, so a smoke run and a production run never share a
    directory even if someone reuses the label.
    """
    scientific = scientific_config(config, settings, restart_distribution_id)
    digest = config_hash(scientific)
    root = Path(root) if root is not None else DEFAULT_RESULTS_ROOT
    return root / f"{settings.label}_{digest[:8]}", digest, scientific


def write_manifest(
    results_dir: Path,
    scientific: Dict,
    digest: str,
    settings: NKappaSettings,
    calibration_provenance: Dict[str, object],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "n_kappa_bias_variance",
        "config_hash": digest,
        "scientific_config": scientific,
        "label": settings.label,
        "design_points": [[int(N), int(k)] for N, k in settings.design_points],
        "evaluation_windows": [int(w) for w in settings.evaluation_windows],
        "total_cycled_windows": int(settings.total_cycled_windows),
        "regimes": list(settings.regimes),
        "calibration": dict(calibration_provenance),
        "caveats": list(INTERPRETATION_CAVEATS),
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


def run_n_kappa_experiment(
    config: Dict,
    settings: NKappaSettings,
    n_jobs: int = 1,
    results_root: Path | None = None,
    resume: bool = True,
    verbose: int = 0,
    allow_calibration: bool = True,
) -> Dict:
    """Run every requested ``(regime, N, kappa, truth seed)`` cell, resumably."""
    law, provenance = resolve_calibration_law(
        config,
        settings,
        results_root=results_root,
        n_jobs=n_jobs,
        verbose=verbose,
        allow_calibration=allow_calibration,
    )
    results_dir, digest, scientific = resolve_results_dir(
        config, settings, law.identifier, results_root
    )
    write_manifest(results_dir, scientific, digest, settings, provenance)

    print(f"Results directory: {results_dir}", flush=True)
    print(f"Configuration hash: {digest}", flush=True)
    print(
        f"Restart law '{law.name}' ({law.identifier}) from {provenance['source']}: "
        f"{provenance['path']}",
        flush=True,
    )

    points = settings.design_points
    cells = [
        (regime, N, kappa, seed)
        for regime in settings.regimes
        for N, kappa in points
        for seed in settings.evaluation_seeds
    ]
    pending = [
        cell
        for cell in cells
        if not (
            resume
            and cell_checkpoint_path(results_dir, cell[0], cell[1], cell[2], cell[3]).exists()
            and cell_checkpoint_matches(
                cell_checkpoint_path(results_dir, cell[0], cell[1], cell[2], cell[3]),
                digest,
                settings.n_mc,
                settings.n_windows,
            )
        )
    ]

    excluded = [
        (int(N), int(k))
        for N in settings.ensemble_sizes
        for k in settings.kappa_values
        if (int(N), int(k)) not in set(points)
    ]
    print(
        f"Grid: {len(settings.regimes)} regimes x {len(points)} valid (N, kappa) "
        f"points x {len(settings.evaluation_seeds)} truth seeds = {len(cells)} cells "
        f"({len(pending)} to run); M={settings.n_mc}, {settings.n_windows} evaluated "
        f"windows at stride {settings.window_stride} after {settings.spinup_windows} "
        f"spin-up windows ({settings.total_cycled_windows} windows integrated per "
        "cycled cell)",
        flush=True,
    )
    if excluded:
        print(
            "  excluded by the rank constraint kappa <= N - 1: "
            + ", ".join(f"(N={N}, kappa={k})" for N, k in excluded),
            flush=True,
        )

    if pending:
        needed = sorted({seed for _g, _N, _k, seed in pending})
        evaluation_config = _config_for_windows(config, settings.total_cycled_windows)
        t0 = time.perf_counter()
        truth_data = generate_truth_batch(evaluation_config, needed)
        print(
            f"Generated {len(needed)} truth trajectories in "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )

        tasks = [
            (
                evaluation_config,
                settings,
                regime,
                N,
                kappa,
                seed,
                truth_data[seed][0],
                truth_data[seed][1],
                truth_data[seed][2],
                law,
                results_dir,
                digest,
            )
            for regime, N, kappa, seed in pending
        ]

        t_start = time.perf_counter()
        completed = 0
        failures: List[Tuple] = []
        divergences: List[Tuple] = []
        for regime, N, kappa, seed, seconds, n_diverged, error in _iter_tasks(
            _cell_task, tasks, n_jobs, verbose
        ):
            completed += 1
            if error is None:
                note = "" if n_diverged == 0 else f", {n_diverged} diverged"
                print(
                    f"  [{completed}/{len(tasks)}] wrote {regime:<7} N={N:<2} "
                    f"kappa={kappa:<2} truth seed {seed} ({seconds:.1f}s{note}, "
                    f"{time.perf_counter() - t_start:.0f}s elapsed)",
                    flush=True,
                )
                if n_diverged:
                    divergences.append((regime, N, kappa, seed, n_diverged))
            else:
                failures.append((regime, N, kappa, seed, error))
                print(
                    f"  [{completed}/{len(tasks)}] FAILED {regime} N={N} "
                    f"kappa={kappa} truth seed {seed}: {error}",
                    flush=True,
                )

        elapsed = time.perf_counter() - t_start
        print(
            f"\nCompleted {len(tasks) - len(failures)}/{len(tasks)} cells in "
            f"{elapsed:.0f}s ({elapsed / 60:.1f} min).",
            flush=True,
        )
        if divergences:
            print(f"\nDivergence recorded in {len(divergences)} cell(s):", flush=True)
            for regime, N, kappa, seed, count in divergences:
                print(
                    f"  {regime:<7} N={N:<2} kappa={kappa:<2} truth seed {seed}: "
                    f"{count} (realization, window) pairs non-finite",
                    flush=True,
                )
        if failures:
            print(f"\n{len(failures)} cell(s) FAILED and have no checkpoint:", flush=True)
            for regime, N, kappa, seed, error in failures:
                print(f"  {regime} N={N} kappa={kappa} seed {seed}: {error}", flush=True)
            print("Rerun the same command to retry; complete cells are skipped.", flush=True)

    cells_loaded = load_cells(results_dir, settings, digest)
    return {
        "results_dir": results_dir,
        "config_hash": digest,
        "restart_law": law,
        "calibration": provenance,
        "cells": cells_loaded,
        "settings": settings,
    }


# --------------------------------------------------------------------------- #
# Loading and analysis
# --------------------------------------------------------------------------- #

CellKey = Tuple[str, int, int, int]  # (regime, N, kappa, truth seed)


def discover_n_kappa_runs(root: Path | None = None) -> List[Dict[str, object]]:
    """Every completed or partial run under ``root``, most cells first.

    Used by the notebook so that what it reports always comes from a run that
    actually exists on disk, rather than from settings retyped alongside it.
    """
    root = Path(root) if root is not None else DEFAULT_RESULTS_ROOT
    if not root.exists():
        return []
    runs = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("experiment") != "n_kappa_bias_variance":
            continue
        cell_dir = manifest_path.parent / "cells"
        runs.append(
            {
                "results_dir": manifest_path.parent,
                "label": manifest.get("label", manifest_path.parent.name),
                "config_hash": manifest["config_hash"],
                "n_cells": len(list(cell_dir.glob("*.npz"))) if cell_dir.exists() else 0,
                "manifest": manifest,
            }
        )
    return sorted(runs, key=lambda run: run["n_cells"], reverse=True)


def settings_from_manifest(manifest: Dict) -> NKappaSettings:
    """Rebuild the exact settings a run was executed with.

    The manifest carries the full scientific configuration, so the notebook never
    has to restate the design -- and therefore cannot silently disagree with the
    checkpoints it is plotting.
    """
    scientific = manifest["scientific_config"]
    design = scientific["design"]
    return NKappaSettings(
        ensemble_sizes=tuple(design["ensemble_sizes"]),
        kappa_values=tuple(scientific["filters"]["kappa_values"]),
        n_mc=int(design["n_mc"]),
        n_windows=int(design["n_windows"]),
        window_stride=int(design["window_stride"]),
        spinup_windows=int(design["spinup_windows"]),
        n_evaluation_seeds=len(design["evaluation_seeds"]),
        truth_seed_base=int(design["truth_seed_base"]),
        calibration_seed_offset=int(design["calibration_seed_offset"]),
        evaluation_seeds=tuple(design["evaluation_seeds"]),
        regimes=tuple(design["regimes"]),
        mean_error_mode=scientific["restart_distribution"]["mean_error_mode"],
        label=manifest.get("label", "grid"),
        stacked_obs_dimension=int(scientific["observations"]["m"])
        * int(scientific["assimilation"]["window_length"]),
    )


def load_cells(
    results_dir: Path,
    settings: NKappaSettings,
    digest: str | None = None,
) -> Dict[CellKey, Dict[str, np.ndarray]]:
    """Load every checkpoint that matches the current configuration.

    Checkpoints written under a different configuration hash are skipped, not
    silently mixed in.
    """
    results_dir = Path(results_dir)
    if digest is None:
        manifest = results_dir / "manifest.json"
        if not manifest.exists():
            return {}
        digest = json.loads(manifest.read_text())["config_hash"]

    cells: Dict[CellKey, Dict[str, np.ndarray]] = {}
    for regime in settings.regimes:
        for N, kappa in settings.design_points:
            for seed in settings.evaluation_seeds:
                path = cell_checkpoint_path(results_dir, regime, N, kappa, seed)
                if not path.exists():
                    continue
                with np.load(path, allow_pickle=False) as data:
                    if str(data["config_hash"]) != digest:
                        continue
                    cells[(regime, int(N), int(kappa), int(seed))] = {
                        key: data[key] for key in data.files
                    }
    return cells


def complete_truth_seeds(
    cells: Dict[CellKey, Dict[str, np.ndarray]], settings: NKappaSettings
) -> Dict[str, object]:
    """Truth seeds usable for the paired analysis, plus what was excluded and why.

    A seed counts only when every cell of the grid -- every regime and every
    valid ``(N, kappa)`` -- is present and divergence-free on it, because the
    whole surface is compared as a paired object.  A seed is never dropped at one
    design point and kept at another.

    ``per_point`` reports, per ``(N, kappa)``, the seeds complete at that design
    point alone, so a design point that is unusable is visible rather than
    silently shrinking the common set.
    """
    required = [
        (regime, N, kappa)
        for regime in settings.regimes
        for N, kappa in settings.design_points
    ]

    def _status(regime, N, kappa, seed) -> str:
        """``"ok"``, ``"missing"`` or ``"diverged"`` for one cell.

        The two failure modes are kept apart on purpose: a missing cell means the
        run is unfinished, a diverged one means the filter blew up.  Reporting
        both as "incomplete" would let an unfinished run read as a scientific
        finding about instability.
        """
        cell = cells.get((regime, int(N), int(kappa), int(seed)))
        if cell is None:
            return "missing"
        if int(cell["n_valid_realizations"]) != int(cell["n_realizations"]):
            return "diverged"
        return "ok"

    per_point: Dict[Tuple[int, int], List[int]] = {}
    for N, kappa in settings.design_points:
        per_point[(int(N), int(kappa))] = [
            int(seed)
            for seed in settings.evaluation_seeds
            if all(_status(regime, N, kappa, seed) == "ok" for regime in settings.regimes)
        ]

    common = [
        int(seed)
        for seed in settings.evaluation_seeds
        if all(_status(regime, N, kappa, seed) == "ok" for regime, N, kappa in required)
    ]

    excluded: Dict[int, List[str]] = {}
    n_missing = n_diverged = 0
    for seed in settings.evaluation_seeds:
        reasons = []
        for regime, N, kappa in required:
            status = _status(regime, N, kappa, seed)
            if status == "ok":
                continue
            reasons.append(f"{regime}/N={N}/kappa={kappa} ({status})")
            n_missing += status == "missing"
            n_diverged += status == "diverged"
        if reasons:
            excluded[int(seed)] = reasons

    return {
        "common": common,
        "per_point": per_point,
        "excluded": excluded,
        "n_missing_cells": n_missing,
        "n_diverged_cells": n_diverged,
        "n_required_cells": len(required) * len(settings.evaluation_seeds),
    }


def diagnostic_table(
    cells: Dict[CellKey, Dict[str, np.ndarray]],
    seeds: Sequence[int],
    settings: NKappaSettings,
) -> Dict[Tuple[str, int, int], Dict[str, np.ndarray]]:
    """Per-seed window-averaged diagnostics for every cell.

    Returns ``{(regime, N, kappa): {diagnostic: array over seeds}}``.  Averaging
    across seeds is left to the caller so that the across-seed spread stays
    available for intervals.

    Per-realization diagnostics are reduced with a mean over realizations and
    then over evaluated windows; per-window diagnostics only over windows.
    ``nanmean`` is used because a diverged ``(realization, window)`` pair carries
    NaN by construction -- but the seeds this is called with are the complete
    ones, where no such pair exists.

    Non-finite entries are folded into the NaNs first.  Only one diagnostic can
    produce them: ``stability_factor`` is ``kappa/delta_kappa^2``, which is
    ``+inf`` at a numerically degenerate gap, and a single infinity would swallow
    the whole average.  It is a heavy-tailed quantity in any case -- the mean is
    dominated by the smallest gaps -- so the normalized gap, not the stability
    factor, is the robust summary and the one the figures and the printed table
    use.
    """
    out: Dict[Tuple[str, int, int], Dict[str, np.ndarray]] = {}
    for regime in settings.regimes:
        for N, kappa in settings.design_points:
            rows: Dict[str, List[float]] = {
                name: [] for name in (*REALIZATION_DIAGNOSTICS, *PROJECTOR_DIAGNOSTICS)
            }
            for seed in seeds:
                cell = cells[(regime, int(N), int(kappa), int(seed))]
                realization = np.asarray(cell["realization_diagnostics"], dtype=float)
                window = np.asarray(cell["window_diagnostics"], dtype=float)
                realization = np.where(np.isfinite(realization), realization, np.nan)
                with np.errstate(invalid="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
                    per_realization = np.nanmean(realization, axis=(0, 1))
                    per_window = np.nanmean(window, axis=0)
                for j, name in enumerate(REALIZATION_DIAGNOSTICS):
                    rows[name].append(float(per_realization[j]))
                for j, name in enumerate(PROJECTOR_DIAGNOSTICS):
                    rows[name].append(float(per_window[j]))
            out[(regime, int(N), int(kappa))] = {
                name: np.asarray(values) for name, values in rows.items()
            }
    return out


def _gram_key(stride_multiplier: int) -> str:
    if stride_multiplier == 1:
        return "error_gram"
    if stride_multiplier == 2:
        return "error_gram_stride2"
    raise ValueError("Only stride multipliers 1 and 2 are stored.")


def bootstrap_grid(
    cells: Dict[CellKey, Dict[str, np.ndarray]],
    seeds: Sequence[int],
    settings: NKappaSettings,
    n_bootstrap: int = 2000,
    n_inner: int = 250,
    random_seed: int = 20260810,
    stride_multiplier: int = 1,
) -> Dict[str, object]:
    """Paired hierarchical bootstrap over the whole ``(regime, N, kappa)`` grid.

    Outer level
        Whole truth seeds, resampled with replacement.  Every evaluated window
        belonging to a resampled seed is kept: windows drawn from one chaotic
        trajectory are serially correlated, so they are not independent bootstrap
        units and are never resampled individually.

    Inner level
        Complete Monte Carlo realizations, resampled by multinomial counts.

    The *same* outer and inner indices are used at every design point, so every
    comparison the summary reports -- across ``kappa``, across ``N``, and between
    regimes -- is paired.

    Also returns, per ``(regime, N)``, the bootstrap distribution of the
    ``kappa`` that minimizes the decomposition-consistent total error, as
    ``argmin_share[(regime, N)][kappa]``.  That is a resampling frequency, not a
    posterior probability, and it is only defined over the ``kappa`` values that
    exist at that ``N``.
    """
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("No complete truth seeds are available.")
    key = _gram_key(stride_multiplier)
    rng = np.random.default_rng(random_seed)
    n_seeds = len(seeds)
    M = int(settings.n_mc)

    grid = [
        (regime, int(N), int(kappa))
        for regime in settings.regimes
        for N, kappa in settings.design_points
    ]

    # Inner draws are shared across the grid: one multinomial count vector per
    # (seed, inner draw), reused by every cell.
    inner = {cell: np.empty((n_seeds, n_inner, len(METRIC_NAMES))) for cell in grid}
    for seed_index, seed in enumerate(seeds):
        counts = rng.multinomial(M, np.full(M, 1.0 / M), size=n_inner)
        for cell in grid:
            gram = cells[(cell[0], cell[1], cell[2], seed)][key]
            if gram.shape != (M, M):
                # Only complete cells are poolable: an incomplete one would be
                # silently reweighted against a different denominator.
                raise ValueError(
                    f"{cell} at truth seed {seed} has a {gram.shape} error Gram, "
                    f"expected ({M}, {M}). Restrict the seed list to the output of "
                    "complete_truth_seeds()."
                )
            mse = counts @ np.diag(gram) / M
            bias2 = np.einsum("bi,ij,bj->b", counts, gram, counts, optimize=True) / M**2
            variance = mse - bias2
            debiased = bias2 - variance / (M - 1)
            inner[cell][seed_index] = np.column_stack(
                (mse, bias2, debiased, variance, debiased + variance)
            )

    draws = {cell: np.empty((n_bootstrap, len(METRIC_NAMES))) for cell in grid}
    for b in range(n_bootstrap):
        outer = rng.integers(0, n_seeds, size=n_seeds)
        inner_index = rng.integers(0, n_inner, size=n_seeds)
        for cell in grid:
            draws[cell][b] = inner[cell][outer, inner_index].mean(axis=0)

    total_column = METRIC_NAMES.index("total")
    argmin_share: Dict[Tuple[str, int], Dict[int, float]] = {}
    for regime in settings.regimes:
        for N in sorted({N for N, _k in settings.design_points}):
            kappas = [k for Np, k in settings.design_points if Np == N]
            if len(kappas) < 2:
                continue
            stack = np.column_stack(
                [draws[(regime, int(N), int(k))][:, total_column] for k in kappas]
            )
            winners = np.argmin(stack, axis=1)
            argmin_share[(regime, int(N))] = {
                int(k): float(np.mean(winners == j)) for j, k in enumerate(kappas)
            }

    return {
        "metric_names": METRIC_NAMES,
        "truth_seeds": np.asarray(seeds),
        "design_points": [(int(N), int(k)) for N, k in settings.design_points],
        "draws": draws,
        "argmin_share": argmin_share,
        "n_bootstrap": int(n_bootstrap),
        "n_inner": int(n_inner),
        "random_seed": int(random_seed),
        "stride_multiplier": int(stride_multiplier),
    }


def surface(
    bootstrap: Dict[str, object],
    settings: NKappaSettings,
    regime: str,
    metric: str,
    statistic: str = "median",
) -> np.ndarray:
    """The ``(len(ensemble_sizes), len(kappa_values))`` response surface.

    Cells excluded by ``kappa <= N - 1`` are ``np.nan``, so a masked colormap
    leaves them blank rather than interpolating across a design point that does
    not exist.
    """
    column = METRIC_NAMES.index(metric)
    sizes = sorted({int(N) for N in settings.ensemble_sizes})
    kappas = sorted({int(k) for k in settings.kappa_values})
    grid = np.full((len(sizes), len(kappas)), np.nan)
    for i, N in enumerate(sizes):
        for j, kappa in enumerate(kappas):
            cell = (regime, N, kappa)
            if cell not in bootstrap["draws"]:
                continue
            values = bootstrap["draws"][cell][:, column]
            if statistic == "median":
                grid[i, j] = float(np.median(values))
            elif statistic == "lower":
                grid[i, j] = percentile_interval(values)[1]
            elif statistic == "upper":
                grid[i, j] = percentile_interval(values)[2]
            else:
                raise ValueError(f"Unknown statistic: {statistic!r}")
    return grid


def diagnostic_surface(
    table: Dict[Tuple[str, int, int], Dict[str, np.ndarray]],
    settings: NKappaSettings,
    regime: str,
    diagnostic: str,
    statistic: str = "mean",
) -> np.ndarray:
    """The ``(N, kappa)`` surface of one diagnostic, averaged across truth seeds."""
    sizes = sorted({int(N) for N in settings.ensemble_sizes})
    kappas = sorted({int(k) for k in settings.kappa_values})
    grid = np.full((len(sizes), len(kappas)), np.nan)
    reduce = {"mean": np.nanmean, "median": np.nanmedian}[statistic]
    for i, N in enumerate(sizes):
        for j, kappa in enumerate(kappas):
            values = table.get((regime, N, kappa))
            if values is None:
                continue
            grid[i, j] = float(reduce(values[diagnostic]))
    return grid


def save_summary(
    results_dir: Path,
    bootstrap: Dict[str, object],
    table: Dict[Tuple[str, int, int], Dict[str, np.ndarray]],
    settings: NKappaSettings,
    seed_report: Dict[str, object],
) -> Tuple[Path, Path]:
    """Write the machine-readable summary: a CSV table and an NPZ of the draws.

    The NPZ carries the full bootstrap draw matrices, not just the intervals, so
    any other interval or comparison can be recomputed without rerunning the
    experiment.
    """
    import csv

    summary_dir = Path(results_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for regime in settings.regimes:
        for N, kappa in settings.design_points:
            row: Dict[str, object] = {
                "regime": regime,
                "ensemble_size": int(N),
                "kappa": int(kappa),
            }
            for metric in METRIC_NAMES:
                mid, lo, hi = percentile_interval(
                    bootstrap["draws"][(regime, int(N), int(kappa))][
                        :, METRIC_NAMES.index(metric)
                    ]
                )
                row[metric] = mid
                row[f"{metric}_lo"] = lo
                row[f"{metric}_hi"] = hi
            for name, values in table[(regime, int(N), int(kappa))].items():
                row[name] = float(np.nanmean(values))
            rows.append(row)

    csv_path = summary_dir / "n_kappa_summary.csv"
    fieldnames = ["regime", "ensemble_size", "kappa"] + sorted(
        {key for row in rows for key in row} - {"regime", "ensemble_size", "kappa"}
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload: Dict[str, np.ndarray] = {
        "metric_names": np.asarray(METRIC_NAMES),
        "realization_diagnostic_names": np.asarray(REALIZATION_DIAGNOSTICS),
        "window_diagnostic_names": np.asarray(PROJECTOR_DIAGNOSTICS),
        "design_points": np.asarray(bootstrap["design_points"]),
        "truth_seeds": np.asarray(bootstrap["truth_seeds"]),
        "n_bootstrap": np.asarray(bootstrap["n_bootstrap"]),
        "n_inner": np.asarray(bootstrap["n_inner"]),
        "random_seed": np.asarray(bootstrap["random_seed"]),
        "stride_multiplier": np.asarray(bootstrap["stride_multiplier"]),
        "n_mc": np.asarray(settings.n_mc),
        "window_stride": np.asarray(settings.window_stride),
        "spinup_windows": np.asarray(settings.spinup_windows),
        "mean_error_mode": np.asarray(settings.mean_error_mode),
        "excluded_truth_seeds": np.asarray(
            json.dumps({str(k): v for k, v in seed_report["excluded"].items()}, sort_keys=True)
        ),
        "argmin_share": np.asarray(
            json.dumps(
                {f"{g}|N{N}": share for (g, N), share in bootstrap["argmin_share"].items()},
                sort_keys=True,
            )
        ),
        "summary_rows": np.asarray(json.dumps(rows, sort_keys=True)),
    }
    for (regime, N, kappa), values in bootstrap["draws"].items():
        payload[f"draws__{regime}__N{N}__k{kappa}"] = values
    for (regime, N, kappa), values in table.items():
        for name, series in values.items():
            payload[f"diag__{regime}__N{N}__k{kappa}__{name}"] = series

    npz_path = summary_dir / "n_kappa_bootstrap.npz"
    np.savez_compressed(npz_path, **payload)
    return csv_path, npz_path


def load_run(results_dir: Path) -> Dict[str, object]:
    """Load one run's manifest, settings and cells together.

    Returns ``{}`` if the directory holds no manifest, so a notebook can guard on
    it the same way the other sections guard on missing checkpoints.
    """
    results_dir = Path(results_dir)
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    settings = settings_from_manifest(manifest)
    cells = load_cells(results_dir, settings, manifest["config_hash"])
    return {
        "results_dir": results_dir,
        "manifest": manifest,
        "settings": settings,
        "config_hash": manifest["config_hash"],
        "calibration": manifest.get("calibration", {}),
        "cells": cells,
        "seed_report": complete_truth_seeds(cells, settings) if cells else None,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_summary_table(
    bootstrap: Dict[str, object],
    table: Dict[Tuple[str, int, int], Dict[str, np.ndarray]],
    settings: NKappaSettings,
) -> None:
    header = (
        f"{'Regime':<9}{'N':>4}{'kappa':>7}{'Bias^2 (db)':>24}{'Variance':>24}"
        f"{'Total':>24}{'Proj. var/kappa':>17}{'Norm. gap':>11}{'Discarded':>11}"
    )
    print("\n" + header)
    print("-" * len(header))
    for regime in settings.regimes:
        for N, kappa in settings.design_points:
            draws = bootstrap["draws"][(regime, int(N), int(kappa))]
            values = table[(regime, int(N), int(kappa))]

            def interval(metric):
                mid, lo, hi = percentile_interval(draws[:, METRIC_NAMES.index(metric)])
                return f"{mid:.3f} [{lo:.3f}, {hi:.3f}]"

            print(
                f"{regime:<9}{N:>4}{kappa:>7}"
                f"{interval('bias_squared_debiased'):>24}"
                f"{interval('variance'):>24}"
                f"{interval('total'):>24}"
                f"{np.nanmean(values['projector_variability_normalized']):>17.4f}"
                f"{np.nanmean(values['normalized_cutoff_gap']):>11.4f}"
                f"{np.nanmean(values['discarded_mean_mismatch_fraction']):>11.4f}"
            )

    print("\nkappa minimizing the decomposition-consistent total T = Bias^2_db + Var")
    print("(share of paired bootstrap draws in which each kappa wins at that N)")
    for (regime, N), share in sorted(bootstrap["argmin_share"].items()):
        ordered = ", ".join(
            f"kappa={k}: {100 * v:.0f}%" for k, v in sorted(share.items()) if v > 0.0
        )
        print(f"  {regime:<9} N={N:<3} {ordered}")
    print(
        "\nIntervals are 95% paired hierarchical-bootstrap percentile intervals "
        f"({bootstrap['n_bootstrap']} outer draws over "
        f"{len(bootstrap['truth_seeds'])} truth seeds). They quantify the sampling "
        "uncertainty OF THE ESTIMATE, not the physical spread across truth "
        "trajectories."
    )


def dry_run(config: Dict, settings: NKappaSettings, n_jobs: int) -> int:
    """Time one probe cell per (regime, N, kappa) and project the total cost."""
    from src.experiments.restart_vs_cycled import bootstrap_restart_distribution

    probe_windows = min(8, settings.n_windows)
    probe_settings = NKappaSettings(
        **{
            **{
                field: getattr(settings, field)
                for field in NKappaSettings.__dataclass_fields__
            },
            "n_windows": probe_windows,
            "evaluation_seeds": settings.evaluation_seeds[:1],
        }
    )
    seed = int(settings.evaluation_seeds[0])
    evaluation_config = _config_for_windows(config, probe_settings.total_cycled_windows)
    truth, observations, H = generate_truth_batch(evaluation_config, [seed])[seed]
    distribution = bootstrap_restart_distribution(int(config["model"]["n"]))

    print(
        f"Probe: {probe_windows} evaluated windows, M={settings.n_mc}, "
        f"{probe_settings.total_cycled_windows} windows per cycled cell\n"
    )
    total = 0.0
    for regime in settings.regimes:
        for N, kappa in settings.design_points:
            context = build_window_context(evaluation_config, kappa)
            args = (
                truth,
                observations,
                H,
                context,
                distribution,
                seed,
                N,
                kappa,
                settings.n_mc,
                [int(w) for w in probe_settings.evaluation_windows],
            )
            t0 = time.perf_counter()
            if regime == "cycled":
                run_cycled_cell(*args)
            else:
                run_restart_cell(*args, mean_error_mode=settings.mean_error_mode)
            measured = time.perf_counter() - t0
            # Cost is linear in the number of windows integrated, which for the
            # cycled regime includes the strided-out ones.
            scale = (
                settings.n_windows / probe_windows
                if regime == "restart"
                else settings.total_cycled_windows / probe_settings.total_cycled_windows
            )
            projected = measured * scale * len(settings.evaluation_seeds)
            total += projected
            print(
                f"  {regime:<7} N={N:<3} kappa={kappa:<2} {measured:6.2f}s probe -> "
                f"{projected / 60:7.2f} min for all "
                f"{len(settings.evaluation_seeds)} truth seeds"
            )

    print(
        f"\nTotal: {total / 3600:.2f} core-hours; at ~50% parallel efficiency on "
        f"{n_jobs} workers, ~{total / (0.5 * max(n_jobs, 1)) / 60:.0f} min wall clock."
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

SMOKE_TEST_SETTINGS = dict(
    ensemble_sizes=(5, 10),
    kappa_values=(1, 2),
    n_mc=4,
    n_windows=4,
    window_stride=2,
    spinup_windows=3,
    n_evaluation_seeds=2,
    label="smoke",
)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="(N, kappa) bias-variance response surface for QPCA-EnDCF."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--n-kappa",
        action="store_true",
        help="Run the experiment (explicit opt-in, matching the notebook's guard).",
    )
    parser.add_argument("--ensemble-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--kappa-values", type=int, nargs="+", default=None)
    parser.add_argument(
        "--regime", choices=(*REGIMES, "both"), default=None, help="Default: both."
    )
    parser.add_argument("--n-mc", type=int, default=None, help="Inner realizations M.")
    parser.add_argument(
        "--n-windows", type=int, default=None, help="Number of EVALUATED windows."
    )
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--spinup-windows", type=int, default=None)
    parser.add_argument("--n-truth-seeds", type=int, default=None)
    parser.add_argument(
        "--mean-error-mode",
        choices=("per_window", "per_realization"),
        default=None,
        help="How the shared forecast-mean error b is drawn in the restart "
        "regime. 'per_window' (default) is the theory-compatible companion.",
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--results-dir", type=Path, default=None, help="Results root.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip cells with a matching checkpoint (default).",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--no-calibrate",
        dest="allow_calibration",
        action="store_false",
        default=True,
        help="Fail rather than calibrating a restart law if none is on disk.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-inner", type=int, default=250)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)

    overrides = dict(
        ensemble_sizes=tuple(args.ensemble_sizes) if args.ensemble_sizes else None,
        kappa_values=tuple(args.kappa_values) if args.kappa_values else None,
        n_mc=args.n_mc,
        n_windows=args.n_windows,
        window_stride=args.window_stride,
        spinup_windows=args.spinup_windows,
        n_evaluation_seeds=args.n_truth_seeds,
        mean_error_mode=args.mean_error_mode,
        label=args.label,
        regimes=(
            None if args.regime in (None, "both") else (args.regime,)
        ),
    )
    if args.smoke_test:
        for key, value in SMOKE_TEST_SETTINGS.items():
            if overrides.get(key) is None:
                overrides[key] = value

    settings = settings_from_config(config, **overrides)

    if args.dry_run:
        return dry_run(config, settings, args.n_jobs)

    if not args.n_kappa:
        print(
            "Nothing to do. Pass --n-kappa to run the experiment, or --dry-run to "
            "project its cost.\n\nFor example:\n"
            "  python -m src.experiments.n_kappa_bias_variance --n-kappa --smoke-test"
        )
        return 0

    if args.smoke_test:
        print("*** SMOKE TEST -- results are NOT a scientific result ***", flush=True)

    outcome = run_n_kappa_experiment(
        config,
        settings,
        n_jobs=args.n_jobs,
        results_root=args.results_dir,
        resume=args.resume,
        verbose=args.verbose,
        allow_calibration=args.allow_calibration,
    )

    seed_report = complete_truth_seeds(outcome["cells"], settings)
    print(
        f"\nComplete truth seeds usable for the paired analysis: "
        f"{len(seed_report['common'])}/{len(settings.evaluation_seeds)}"
    )
    if seed_report["excluded"]:
        print("  excluded (incomplete or diverged in at least one cell):")
        for seed, reasons in seed_report["excluded"].items():
            shown = ", ".join(reasons[:4])
            more = "" if len(reasons) <= 4 else f", +{len(reasons) - 4} more"
            print(f"    {seed}: {shown}{more}")

    if not seed_report["common"]:
        print("\nNo complete truth seed; stopping before the paired summary.")
        return 0

    bootstrap = bootstrap_grid(
        outcome["cells"],
        seed_report["common"],
        settings,
        n_bootstrap=args.n_bootstrap,
        n_inner=args.n_inner,
    )
    table = diagnostic_table(outcome["cells"], seed_report["common"], settings)
    print_summary_table(bootstrap, table, settings)
    csv_path, npz_path = save_summary(
        outcome["results_dir"], bootstrap, table, settings, seed_report
    )
    print(f"\nWrote {csv_path}\nWrote {npz_path}")

    if args.smoke_test:
        print(
            "\n*** SMOKE TEST -- the numbers above come from a deliberately tiny "
            "grid and must not be quoted as a result. ***"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
