"""
Independent-restart versus cycled-ensemble experiment (notebook 10, section 7).

The bias--variance theory assumes that, at a fixed assimilation window, the
window-initial forecast ensemble is an i.i.d. sample from a forecast
distribution.  Ordinary filtering cycles violate that assumption: the analysis
ensemble of window ``w`` *is* the forecast ensemble of window ``w+1``, so
members are neither independent of one another nor of the previous window.

This module runs a matched pair of regimes so the assumption can be probed
directly.

``cycled``
    Ordinary cycling, ``X^a_{r,t,w} -> X^f_{r,t,w+1}``.

``restart``
    Every evaluated window begins from a freshly drawn forecast ensemble,
    advances through exactly one window, is analysed, and is then discarded.
    No analysis ensemble is ever carried between evaluated restart windows.

The restart forecast law is

    x^{(j),f}_{r,t,w,0} = x^true_{r,k_{w-1}} + b_{r,t,w} + eps~^{(j)}_{r,t,w},

with a shared mean error ``b ~ N(0, P_mean)``, within-ensemble anomalies
``eps ~ N(0, P_within)`` drawn i.i.d. and then *exactly* centered, so that the
shared mean error and the within-ensemble spread are controlled separately.
``P_mean`` and ``P_within`` are calibrated on a set of truth seeds that is
disjoint from the evaluation seeds (see ``calibrate_restart_distribution``).

Terminology, used consistently in names and docstrings:

    r  outer truth / observation-sequence index    N  ensemble size
    t  inner ensemble/filter realization index     M  inner realizations
    w  assimilation-window index                   W  assimilation windows
    j  ensemble-member index                       L  observations per window
    n  state dimension                             g  regime, restart | cycled

Run it as::

    python -m src.experiments.restart_vs_cycled --restart-vs-cycled --smoke-test
    python -m src.experiments.restart_vs_cycled --restart-vs-cycled \\
        --ensemble-sizes 5 10 20 40 --n-truth-seeds 50 --n-mc 100 \\
        --n-windows 150 --window-stride 5 --spinup-windows 50 --n-jobs 10

Everything is resumable: one checkpoint per ``(regime, filter, N, truth seed)``
cell, keyed by a deterministic configuration hash, and a re-run skips whatever
is already complete.

What this experiment does *not* establish is stated in
``INTERPRETATION_CAVEATS`` and repeated in the notebook: restarting ensembles
does not make the sampled truth windows temporally independent, and a fitted
scaling exponent does not prove a theorem.
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
import hashlib
import json
import sys
import time
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import linalg

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DIAGNOSTICS_EXPERIMENT_DIR, DIAGNOSTICS_RESULTS_DIR
from src.experiments.bias_variance_mc import (
    FILTER_NAMES,
    _advance,
    _check_isotropic_obs_cov,
    _localized_gain,
    _lowrank_gain,
    _qpca_taper,
    _thin_svd,
    generate_truth_batch,
    make_truth_seeds,
)
from src.utils.config import load_experiment_config

# --------------------------------------------------------------------------- #
# Constants and design decisions
# --------------------------------------------------------------------------- #

#: The two regimes compared.  ``g`` in the write-up.
REGIMES = ("restart", "cycled")

#: Default results root.  Deliberately *not* ``bias_variance_analysis_100x100``:
#: this experiment must never overwrite the published hierarchical checkpoints.
DEFAULT_RESULTS_ROOT = DIAGNOSTICS_RESULTS_DIR / "restart_vs_cycled"

DEFAULT_CONFIG_PATH = DIAGNOSTICS_EXPERIMENT_DIR / "restart_vs_cycled.yaml"

#: Truth-seed family.  Identical to the published hierarchical run, so the new
#: cycled cells at N=10 sit on the *same* truth trajectories as
#: ``bias_variance_analysis_100x100`` and can be cross-checked against it.
TRUTH_SEED_BASE = 20260728

#: Evaluation seeds are the first ``n_evaluation`` of that family.  Calibration
#: seeds start at this index -- past the 101 seeds the published run drew -- so
#: no calibration trajectory can ever be an evaluation trajectory, and neither
#: set touches the one seed (index 67) on which Seq-EnKF is known to diverge.
CALIBRATION_SEED_OFFSET = 101

#: Ensemble size at which the restart distribution is calibrated.  It is held
#: FIXED across the ensemble-size sweep on purpose: if the restart forecast law
#: changed with N, the N-dependence of the analysis-mean variance would mix the
#: sampling effect the theory predicts with a moving target distribution, and
#: the fitted exponent would answer no clean question.
DEFAULT_CALIBRATION_ENSEMBLE_SIZE = 10

#: Entropy for every ``numpy.random.Generator`` in this experiment.  Python's
#: ``hash()`` is process-randomized and is never used for a scientific seed.
RNG_BASE_ENTROPY = 20260807

#: Stream kinds.  Fixed integers, not names, so the stream a draw comes from
#: cannot change if a dict happens to be reordered.
_STREAM_RESTART_MEAN = 0  # keyed (r, w)        -- N- and regime-independent
_STREAM_RESTART_ANOMALY = 1  # keyed (r, w, N)  -- regime-independent
_STREAM_OBS_PERTURBATION = 2  # keyed (filter, r, w, N) -- regime-independent

#: Filter stream indices.  Appending a filter must not repaint an existing one,
#: so these are assigned once and never reordered.
_FILTER_STREAM_INDEX = {
    "Seq-EnKF": 0,
    "4D-EnKF": 1,
    "Seq-LETKF": 2,
    "4D-LETKF": 3,
    "QPCA-EnDCF": 4,
}

#: The filters this experiment supports.  Deliberately NOT
#: ``bias_variance_mc.FILTER_NAMES``: that tuple is the cycled decomposition's
#: filter set and has grown to five, whereas ``run_one_window`` implements only
#: these three.  Inheriting it would redefine this experiment silently, change
#: its configuration hash, and invalidate every checkpoint under
#: ``results/diagnostics/restart_vs_cycled/``.  Adding a filter here means
#: implementing its one-window kernel and paying for the recomputation.
RESTART_SUPPORTED_FILTERS = ("Seq-EnKF", "4D-EnKF", "QPCA-EnDCF")

#: Restart-distribution constructions.  ``common`` is the primary design.
RESTART_DISTRIBUTION_MODES = ("common", "filter_marginal", "isotropic")

#: How the shared forecast-mean error ``b`` is drawn.  These answer different
#: questions and both are run; neither is a substitute for the other.
#:
#: ``per_realization`` -- ``b_{r,t,w}``, redrawn for every inner realization.
#:     The primary design.  It keeps the forecast-mean uncertainty that a real
#:     forecast ensemble carries, rather than centring every restart on the
#:     truth, which would give an oracle forecast mean.
#:
#:     It has a consequence that must be stated whenever a scaling exponent is
#:     read off it: because ``b`` is realization-dependent and does not depend
#:     on ``N``, the analysis-mean variance across realizations carries an
#:     N-independent floor of order ``tr(P_mean)/n`` propagated through the
#:     analysis.  A fitted exponent on the *total* variance is therefore bounded
#:     away from 1 by construction, and a small exponent here is not evidence
#:     against the theory.
#:
#: ``per_window`` -- ``b_{r,w}``, one draw shared by all ``M`` realizations at a
#:     window.  Every realization then faces the SAME forecast distribution,
#:     which is the situation the theory assumes, so the realization-to-
#:     realization variance comes only from finite-ensemble sampling and the
#:     observation perturbations.  This is the configuration in which "does the
#:     analysis-mean variance scale as N^-1?" is a well-posed question, and it
#:     is run as a labelled companion to the primary design.
MEAN_ERROR_MODES = ("per_realization", "per_window")

INTERPRETATION_CAVEATS = (
    "Restarting the ensemble removes inherited filter dependence; it does NOT "
    "make the evaluated truth windows temporally independent, because they are "
    "drawn from one chaotic trajectory. The bootstrap therefore resamples whole "
    "truth seeds and never individual windows.",
    "A_cycle is a ratio of variance magnitudes. It is not a test of statistical "
    "independence and must not be read as one.",
    "A fitted slope whose interval contains 1 is consistent with N^-1 scaling; "
    "it does not confirm the theorem.",
    "The covariance-matched restart is a controlled statistical construction, "
    "not an operational forecasting procedure: it is centred on a perturbed "
    "truth, which no real forecast system has access to.",
    "This experiment does not replace a deterministic square-root baseline, and "
    "it does not attribute the variance difference to any single mechanism.",
)


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


def make_seed_sets(
    n_calibration: int,
    n_evaluation: int,
    base_seed: int = TRUTH_SEED_BASE,
    calibration_offset: int = CALIBRATION_SEED_OFFSET,
) -> Tuple[List[int], List[int]]:
    """Disjoint calibration and evaluation truth seeds from one seed family.

    Both come from ``SeedSequence(base_seed)``, partitioned *by index*:
    evaluation takes the leading block (the same trajectories the published
    100x100 run used), calibration takes a block starting at
    ``calibration_offset``.  Disjointness is asserted rather than assumed --
    ``generate_state`` returns uint32 values, so a collision is improbable but
    not impossible, and a silent overlap would leak evaluation data into the
    restart covariances.
    """
    if n_calibration < 1 or n_evaluation < 1:
        raise ValueError("Both seed counts must be positive.")
    if calibration_offset < n_evaluation:
        raise ValueError(
            f"calibration_offset ({calibration_offset}) must be at least "
            f"n_evaluation ({n_evaluation}) or the two blocks overlap by index."
        )

    total = calibration_offset + n_calibration
    family = make_truth_seeds(total, base_seed)
    evaluation = family[:n_evaluation]
    calibration = family[calibration_offset : calibration_offset + n_calibration]

    overlap = set(evaluation) & set(calibration)
    if overlap:
        raise ValueError(
            "Calibration and evaluation truth seeds collide on "
            f"{sorted(overlap)}. Change --truth-seed-base; the restart "
            "covariances must never see an evaluation trajectory."
        )
    return calibration, evaluation


def _generator(*keys: int) -> np.random.Generator:
    """A ``Generator`` addressed by a tuple of non-negative integer keys.

    ``SeedSequence`` hashes the entropy sequence, so distinct key tuples give
    independent streams and the same tuple always gives the same stream --
    across processes, machines and runs.
    """
    entropy = [int(RNG_BASE_ENTROPY)] + [int(k) for k in keys]
    if any(k < 0 for k in entropy):
        raise ValueError(f"Stream keys must be non-negative; got {entropy}.")
    return np.random.default_rng(np.random.SeedSequence(entropy))


def restart_mean_generator(truth_seed: int, window: int) -> np.random.Generator:
    """Stream for the shared forecast-mean error ``b`` at ``(r, w)``.

    Deliberately independent of ``N`` and of the regime: the same mean error is
    then reused at every ensemble size, so the ensemble-size sweep isolates the
    within-ensemble sampling term instead of also resampling ``b``.
    """
    return _generator(_STREAM_RESTART_MEAN, truth_seed, window)


def restart_anomaly_generator(
    truth_seed: int, window: int, ensemble_size: int
) -> np.random.Generator:
    """Stream for the within-ensemble anomalies at ``(r, w, N)``."""
    return _generator(_STREAM_RESTART_ANOMALY, truth_seed, window, ensemble_size)


def observation_perturbation_generator(
    filter_name: str, truth_seed: int, window: int, ensemble_size: int
) -> np.random.Generator:
    """Stream for a filter's observation perturbations at ``(filter, r, w, N)``.

    Keyed by filter, so a filter draws the same numbers regardless of which
    other filters are in the run.  *Not* keyed by regime, so the restart and
    cycled runs of one filter consume identical perturbations at matched
    ``(r, w, t)`` -- common random numbers across the comparison of interest.
    """
    if filter_name not in _FILTER_STREAM_INDEX:
        raise ValueError(f"Unknown filter name: {filter_name!r}")
    return _generator(
        _STREAM_OBS_PERTURBATION,
        _FILTER_STREAM_INDEX[filter_name],
        truth_seed,
        window,
        ensemble_size,
    )


# --------------------------------------------------------------------------- #
# Configuration hashing
# --------------------------------------------------------------------------- #


def scientific_config(config: Dict, settings: "ExperimentSettings") -> Dict:
    """The configuration fields a checkpoint's validity depends on.

    Excludes anything that changes only how the run is scheduled (worker count,
    verbosity, output paths, resume behaviour) -- those must not invalidate a
    checkpoint -- and includes everything that changes what is computed.
    """
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
        "filters": {
            "names": list(settings.filter_names),
            "kappa": int(settings.kappa),
            "inflation": {name: 1.0 for name in settings.filter_names},
            "qpca_localization_radius": (
                config.get("filters", {}).get("qpca_endcf", {}) or {}
            ).get("localization_radius", None),
        },
        "design": {
            "n_mc": int(settings.n_mc),
            "n_windows": int(settings.n_windows),
            "window_stride": int(settings.window_stride),
            "spinup_windows": int(settings.spinup_windows),
            "ensemble_sizes": [int(N) for N in settings.ensemble_sizes],
            "truth_seed_base": int(settings.truth_seed_base),
            "calibration_seed_offset": int(settings.calibration_seed_offset),
            "calibration_seeds": [int(s) for s in settings.calibration_seeds],
            "evaluation_seeds": [int(s) for s in settings.evaluation_seeds],
            "rng_base_entropy": int(RNG_BASE_ENTROPY),
        },
        "restart_distribution": {
            "mode": settings.restart_mode,
            "mean_error_mode": settings.mean_error_mode,
            "calibration_ensemble_size": int(settings.calibration_ensemble_size),
            "calibration_n_mc": int(settings.calibration_n_mc),
            "calibration_n_windows": int(settings.calibration_n_windows),
            "calibration_spinup_windows": int(settings.calibration_spinup_windows),
            "eigenvalue_floor": float(settings.eigenvalue_floor),
            "shrinkage": float(settings.shrinkage),
        },
    }


def config_hash(scientific: Dict) -> str:
    """Deterministic sha256 of the canonicalized scientific configuration.

    ``json.dumps(..., sort_keys=True)`` is stable across processes and Python
    versions, unlike ``hash()``, which is randomized per process and must never
    key a scientific artefact.
    """
    payload = json.dumps(scientific, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass
class ExperimentSettings:
    """Everything the driver needs beyond the base YAML model/observation block."""

    ensemble_sizes: Tuple[int, ...] = (5, 10, 20, 40)
    filter_names: Tuple[str, ...] = RESTART_SUPPORTED_FILTERS
    kappa: int = 1

    n_mc: int = 100  # M
    n_windows: int = 150  # number of EVALUATED windows
    window_stride: int = 5  # s
    spinup_windows: int = 50  # windows discarded before the first evaluated one

    n_calibration_seeds: int = 10
    n_evaluation_seeds: int = 50
    truth_seed_base: int = TRUTH_SEED_BASE
    calibration_seed_offset: int = CALIBRATION_SEED_OFFSET
    calibration_seeds: Tuple[int, ...] = ()
    evaluation_seeds: Tuple[int, ...] = ()

    calibration_ensemble_size: int = DEFAULT_CALIBRATION_ENSEMBLE_SIZE
    calibration_n_mc: int = 20
    calibration_n_windows: int = 300
    calibration_spinup_windows: int = 50

    restart_mode: str = "common"
    mean_error_mode: str = "per_realization"
    eigenvalue_floor: float = 1e-6  # relative to the largest eigenvalue
    shrinkage: float = 0.0  # linear shrinkage toward (tr/n) I

    label: str = "run"

    def __post_init__(self) -> None:
        self.ensemble_sizes = tuple(int(N) for N in self.ensemble_sizes)
        self.filter_names = tuple(self.filter_names)
        unsupported = [f for f in self.filter_names if f not in RESTART_SUPPORTED_FILTERS]
        if unsupported:
            # Caught here rather than inside a worker: without this the run gets
            # through calibration and dies per-cell with an opaque dispatch
            # error, after minutes of wasted compute.
            raise ValueError(
                f"restart_vs_cycled supports {list(RESTART_SUPPORTED_FILTERS)}; "
                f"got unsupported {unsupported}. Implement a one-window kernel in "
                "run_one_window before adding a filter to this experiment."
            )
        if self.restart_mode not in RESTART_DISTRIBUTION_MODES:
            raise ValueError(
                f"restart_mode must be one of {RESTART_DISTRIBUTION_MODES}; "
                f"got {self.restart_mode!r}"
            )
        if self.mean_error_mode not in MEAN_ERROR_MODES:
            raise ValueError(
                f"mean_error_mode must be one of {MEAN_ERROR_MODES}; "
                f"got {self.mean_error_mode!r}"
            )
        if self.window_stride < 1:
            raise ValueError("window_stride must be >= 1.")
        if self.n_windows < 1:
            raise ValueError("n_windows must be >= 1.")
        if not self.calibration_seeds or not self.evaluation_seeds:
            calibration, evaluation = make_seed_sets(
                self.n_calibration_seeds,
                self.n_evaluation_seeds,
                self.truth_seed_base,
                self.calibration_seed_offset,
            )
            self.calibration_seeds = tuple(calibration)
            self.evaluation_seeds = tuple(evaluation)
        self.calibration_seeds = tuple(int(s) for s in self.calibration_seeds)
        self.evaluation_seeds = tuple(int(s) for s in self.evaluation_seeds)
        overlap = set(self.calibration_seeds) & set(self.evaluation_seeds)
        if overlap:
            raise ValueError(
                f"Calibration and evaluation truth seeds overlap on {sorted(overlap)}."
            )
        for N in self.ensemble_sizes:
            validate_kappa(self.kappa, N, self.stacked_obs_dimension)
        validate_kappa(
            self.kappa, self.calibration_ensemble_size, self.stacked_obs_dimension
        )

    # The stacked observation dimension d = m * L is filled in by ``from_config``;
    # a plain default keeps the dataclass constructible in tests.
    stacked_obs_dimension: int = 100

    @property
    def evaluation_windows(self) -> np.ndarray:
        """0-based indices of the evaluated windows, ``w`` in the write-up.

        Window ``w`` starts at truth index ``w * L`` and ends at ``(w + 1) * L``;
        the recorded analysis mean is the one at the window endpoint.
        """
        return self.spinup_windows + self.window_stride * np.arange(self.n_windows)

    @property
    def total_cycled_windows(self) -> int:
        """Windows the cycled regime must integrate to reach the last evaluated one."""
        return int(self.evaluation_windows[-1]) + 1


def settings_from_config(config: Dict, **overrides) -> ExperimentSettings:
    """Build settings from the YAML ``restart_vs_cycled`` block plus overrides."""
    block = dict(config.get("restart_vs_cycled", {}) or {})
    block.update({k: v for k, v in overrides.items() if v is not None})
    known = {f for f in ExperimentSettings.__dataclass_fields__}
    unknown = set(block) - known
    if unknown:
        raise ValueError(f"Unknown restart_vs_cycled settings: {sorted(unknown)}")
    block.setdefault(
        "kappa", int((config.get("filters", {}).get("qpca_endcf", {}) or {}).get("k", 1))
    )
    block["stacked_obs_dimension"] = int(config["observations"]["m"]) * int(
        config["assimilation"]["window_length"]
    )
    return ExperimentSettings(**block)


def validate_kappa(kappa: int, ensemble_size: int, stacked_obs_dimension: int) -> int:
    """Enforce ``1 <= kappa <= min(N - 1, d)``.

    The retained subspace is spanned by the centered whitened residuals, whose
    rank is at most ``N - 1``; asking for more modes silently retains numerical
    noise directions, so it is rejected rather than clipped.
    """
    kappa = int(kappa)
    upper = min(int(ensemble_size) - 1, int(stacked_obs_dimension))
    if kappa < 1:
        raise ValueError(f"kappa must be >= 1; got {kappa}.")
    if kappa > upper:
        raise ValueError(
            f"kappa={kappa} is invalid at N={ensemble_size}, d={stacked_obs_dimension}: "
            f"the centered residual matrix has rank at most min(N-1, d) = {upper}."
        )
    return kappa


# --------------------------------------------------------------------------- #
# Restart distribution
# --------------------------------------------------------------------------- #


@dataclass
class RestartDistribution:
    """A stabilized ``(P_mean, P_within)`` pair plus its provenance.

    ``factor_mean`` and ``factor_within`` are the square-root factors used for
    sampling; storing them keeps the draw a single matrix product and makes the
    stabilization auditable (the eigenvalues they were built from are kept).
    """

    name: str
    P_mean: np.ndarray
    P_within: np.ndarray
    factor_mean: np.ndarray
    factor_within: np.ndarray
    eigenvalues_mean_raw: np.ndarray
    eigenvalues_within_raw: np.ndarray
    eigenvalues_mean: np.ndarray
    eigenvalues_within: np.ndarray
    effective_rank_mean: float
    effective_rank_within: float
    n_samples: int
    provenance: Dict = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        """Short, stable identifier recorded in every checkpoint."""
        payload = json.dumps(
            {
                "name": self.name,
                "P_mean": np.round(self.P_mean, 10).tolist(),
                "P_within": np.round(self.P_within, 10).tolist(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def draw(
        self,
        truth_state: np.ndarray,
        ensemble_size: int,
        n_realizations: int,
        rng_mean: np.random.Generator,
        rng_anomaly: np.random.Generator,
        shared_mean_across_realizations: bool = False,
    ) -> np.ndarray:
        """Draw ``M`` independent restart ensembles about one truth state.

        Returns an ``(n, M * N)`` block laid out so that realization ``t`` owns
        columns ``t*N : (t+1)*N`` -- the same layout the batched kernels of
        ``bias_variance_mc`` use, so a whole window advances in one ``rk4_step``
        sweep.

        The anomalies are centered *exactly* after being drawn, so the ensemble
        mean is ``x_true + b`` by construction and the shared mean error is not
        contaminated by within-ensemble sampling noise.

        ``shared_mean_across_realizations`` selects between the two mean-error
        designs; see ``MEAN_ERROR_MODES`` for what each one is for.
        """
        n = self.P_mean.shape[0]
        M, N = int(n_realizations), int(ensemble_size)
        if truth_state.shape != (n,):
            raise ValueError(
                f"truth_state has shape {truth_state.shape}, expected ({n},)."
            )

        if shared_mean_across_realizations:
            # b_{r,w}: one mean error for the whole window, so all M
            # realizations face the SAME forecast distribution.
            b = np.broadcast_to(
                rng_mean.standard_normal(n) @ self.factor_mean.T, (M, n)
            )
        else:
            # b_{r,t,w}: one shared mean error per realization.
            b = rng_mean.standard_normal((M, n)) @ self.factor_mean.T  # (M, n)

        # eps^{(j)}_{r,t,w}: i.i.d. within-ensemble anomalies, then exactly centered.
        eps = rng_anomaly.standard_normal((M, N, n)) @ self.factor_within.T
        eps -= eps.mean(axis=1, keepdims=True)

        members = truth_state[None, None, :] + b[:, None, :] + eps  # (M, N, n)
        return np.ascontiguousarray(members.reshape(M * N, n).T)  # (n, M*N)


def _stabilize_covariance(
    raw: np.ndarray, eigenvalue_floor: float, shrinkage: float, what: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Symmetrize, shrink, floor the spectrum, and return a sampling factor.

    Returns ``(P, factor, eigenvalues_raw, eigenvalues, effective_rank)``.

    A sample covariance estimated from finitely many correlated draws is
    symmetric only up to roundoff and can carry small negative eigenvalues.
    Both are fixed explicitly rather than left to a Cholesky to discover:
    the spectrum is clipped at zero, optionally shrunk toward ``(tr/n) I``, and
    then floored at ``eigenvalue_floor`` times the largest eigenvalue so the
    factor is well defined.  The raw spectrum is returned alongside so the
    notebook can report exactly what the stabilization changed.
    """
    raw = np.asarray(raw, dtype=float)
    n = raw.shape[0]
    symmetric = 0.5 * (raw + raw.T)

    # Before the eigendecomposition, not after: handed a NaN, LAPACK reports
    # "Eigenvalues did not converge", which reads as a numerical quirk and hides
    # the fact that the calibration produced a non-finite covariance.
    if not np.isfinite(symmetric).all():
        raise FloatingPointError(
            f"{what}: non-finite entries in the raw covariance. The calibration "
            "run produced non-finite forecast statistics; investigate those "
            "rather than regularizing them away."
        )

    eigenvalues_raw = np.linalg.eigvalsh(symmetric)
    if eigenvalues_raw.max() <= 0.0:
        raise ValueError(
            f"{what}: the estimated covariance has no positive eigenvalue "
            f"(max {eigenvalues_raw.max():.3e}); it cannot be used to sample."
        )

    if shrinkage > 0.0:
        target = (np.trace(symmetric) / n) * np.eye(n)
        symmetric = (1.0 - shrinkage) * symmetric + shrinkage * target

    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, 0.0)
    floor = eigenvalue_floor * values.max()
    values = np.maximum(values, floor)

    P = (vectors * values) @ vectors.T
    P = 0.5 * (P + P.T)
    factor = vectors * np.sqrt(values)  # factor @ factor.T = P

    # Participation-ratio effective rank: (sum l)^2 / sum l^2, which is n for a
    # flat spectrum and ~1 for a rank-one one.  Reported, not enforced.
    effective_rank = float(values.sum() ** 2 / np.sum(values**2))
    return P, factor, eigenvalues_raw, values, effective_rank


class _RestartCalibrationAccumulator:
    """Running sums for the restart covariances, one instance per filter.

    Accumulates over calibration truth seeds ``r``, realizations ``t`` and
    post-spin-up windows ``w`` the two objects the restart law needs:

        e^mean_{r,t,w} = xbar^f_{r,t,w} - x^true_{r,k_{w-1}}
        a^{(j)}_{r,t,w} = x^{(j),f}_{r,t,w} - xbar^f_{r,t,w}

    plus lag-1 quantities for the dependence diagnostics.  Sums, not stacked
    samples: the calibration touches ~10^5 ensembles and storing them would cost
    more memory than the whole experiment.
    """

    def __init__(self, n: int):
        self.n = n
        self.count = 0
        self.sum_mean_error = np.zeros(n)
        self.sum_mean_outer = np.zeros((n, n))
        self.sum_within = np.zeros((n, n))
        # Pooled member-error second moments (marginal forecast-error covariance).
        self.member_count = 0
        self.sum_member_error = np.zeros(n)
        self.sum_member_outer = np.zeros((n, n))
        # Lag-1 persistence, per state component, for member and mean errors.
        self._lag = {
            key: dict(count=0, sx=np.zeros(n), sy=np.zeros(n), sxx=np.zeros(n),
                      syy=np.zeros(n), sxy=np.zeros(n))
            for key in ("member", "mean")
        }
        self._previous: Dict[str, np.ndarray | None] = {"member": None, "mean": None}

    def update(self, ensembles: np.ndarray, truth_state: np.ndarray) -> None:
        """Absorb one window-start ensemble block ``(n, M*N)``."""
        n = self.n
        # (M * N, n) view, matching the (n, M*N) layout used everywhere here.
        members = ensembles.T.reshape(-1, n)
        errors = members - truth_state[None, :]
        self.member_count += errors.shape[0]
        self.sum_member_error += errors.sum(axis=0)
        self.sum_member_outer += errors.T @ errors

    def update_structured(
        self, ensembles: np.ndarray, truth_state: np.ndarray, M: int, N: int
    ) -> None:
        """Absorb one window-start block with its ``(M, N)`` structure known."""
        n = self.n
        block = ensembles.T.reshape(M, N, n)
        mean = block.mean(axis=1)  # (M, n)
        anomalies = block - mean[:, None, :]  # (M, N, n)
        mean_error = mean - truth_state[None, :]  # (M, n)

        self.count += M
        self.sum_mean_error += mean_error.sum(axis=0)
        self.sum_mean_outer += mean_error.T @ mean_error
        # E[ (1/(N-1)) sum_j a a^T ], summed over realizations.
        flat = anomalies.reshape(M * N, n)
        self.sum_within += (flat.T @ flat) / (N - 1)

        self.update(ensembles, truth_state)

        member_error = (block - truth_state[None, None, :]).reshape(M * N, n)
        self._update_lag("member", member_error)
        self._update_lag("mean", mean_error)

    def _update_lag(self, key: str, current: np.ndarray) -> None:
        previous = self._previous[key]
        if previous is not None and previous.shape == current.shape:
            acc = self._lag[key]
            acc["count"] += current.shape[0]
            acc["sx"] += previous.sum(axis=0)
            acc["sy"] += current.sum(axis=0)
            acc["sxx"] += (previous**2).sum(axis=0)
            acc["syy"] += (current**2).sum(axis=0)
            acc["sxy"] += (previous * current).sum(axis=0)
        self._previous[key] = current.copy()

    def break_sequence(self) -> None:
        """Called between truth seeds: window w+1 of seed A does not follow seed B."""
        self._previous = {"member": None, "mean": None}

    def merge(self, other: "_RestartCalibrationAccumulator") -> None:
        self.count += other.count
        self.sum_mean_error += other.sum_mean_error
        self.sum_mean_outer += other.sum_mean_outer
        self.sum_within += other.sum_within
        self.member_count += other.member_count
        self.sum_member_error += other.sum_member_error
        self.sum_member_outer += other.sum_member_outer
        for key, acc in self._lag.items():
            for field_name, value in other._lag[key].items():
                acc[field_name] = acc[field_name] + value

    def raw_covariances(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(P_mean, P_within)`` before stabilization.

        ``P_mean`` is the *centered* covariance of the forecast-mean error, as
        specified.  The sample mean is reported separately in
        ``diagnostics()``: a systematically non-zero mean error is a property of
        the filter, and injecting it into the restart law would hard-code a
        deterministic bias into every drawn ensemble.
        """
        if self.count < 2:
            raise ValueError("Fewer than two calibration samples were accumulated.")
        mean = self.sum_mean_error / self.count
        P_mean = self.sum_mean_outer / self.count - np.outer(mean, mean)
        P_mean *= self.count / (self.count - 1)  # unbiased
        P_within = self.sum_within / self.count
        return P_mean, P_within

    def diagnostics(self) -> Dict[str, np.ndarray]:
        mean = self.sum_mean_error / max(self.count, 1)
        member_mean = self.sum_member_error / max(self.member_count, 1)
        member_cov = self.sum_member_outer / max(self.member_count, 1) - np.outer(
            member_mean, member_mean
        )
        out = {
            "n_samples": np.asarray(self.count),
            "n_member_samples": np.asarray(self.member_count),
            "mean_error_mean": mean,
            "mean_error_mean_energy": np.asarray(float(mean @ mean)),
            "member_error_covariance": 0.5 * (member_cov + member_cov.T),
        }
        for key, acc in self._lag.items():
            count = max(acc["count"], 1)
            cov = acc["sxy"] / count - (acc["sx"] / count) * (acc["sy"] / count)
            var_x = acc["sxx"] / count - (acc["sx"] / count) ** 2
            var_y = acc["syy"] / count - (acc["sy"] / count) ** 2
            denominator = np.sqrt(np.maximum(var_x * var_y, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                correlation = np.where(denominator > 0.0, cov / denominator, np.nan)
            out[f"lag1_{key}_correlation_per_component"] = correlation
            out[f"lag1_{key}_correlation"] = np.asarray(float(np.nanmean(correlation)))
            out[f"lag1_{key}_pairs"] = np.asarray(acc["count"])
        return out


def build_restart_distribution(
    raw_mean: np.ndarray,
    raw_within: np.ndarray,
    name: str,
    n_samples: int,
    eigenvalue_floor: float,
    shrinkage: float,
    provenance: Dict | None = None,
) -> RestartDistribution:
    """Stabilize a raw ``(P_mean, P_within)`` pair into a sampleable law."""
    P_mean, factor_mean, raw_eig_mean, eig_mean, rank_mean = _stabilize_covariance(
        raw_mean, eigenvalue_floor, shrinkage, f"{name}/P_mean"
    )
    (
        P_within,
        factor_within,
        raw_eig_within,
        eig_within,
        rank_within,
    ) = _stabilize_covariance(raw_within, eigenvalue_floor, shrinkage, f"{name}/P_within")
    return RestartDistribution(
        name=name,
        P_mean=P_mean,
        P_within=P_within,
        factor_mean=factor_mean,
        factor_within=factor_within,
        eigenvalues_mean_raw=raw_eig_mean,
        eigenvalues_within_raw=raw_eig_within,
        eigenvalues_mean=eig_mean,
        eigenvalues_within=eig_within,
        effective_rank_mean=rank_mean,
        effective_rank_within=rank_within,
        n_samples=int(n_samples),
        provenance=dict(provenance or {}),
    )


def isotropic_restart_distribution(
    reference: RestartDistribution, eigenvalue_floor: float
) -> RestartDistribution:
    """Trace-matched isotropic control.

    Provided ONLY as an explicitly labelled sensitivity experiment.  It keeps
    the total mean-error and within-ensemble energy of ``reference`` but throws
    away every spatial correlation, which for a Lorenz-96 forecast ensemble is a
    substantial distortion -- so it must never stand in for the calibrated law
    in a headline result.
    """
    n = reference.P_mean.shape[0]
    scale_mean = np.trace(reference.P_mean) / n
    scale_within = np.trace(reference.P_within) / n
    return build_restart_distribution(
        scale_mean * np.eye(n),
        scale_within * np.eye(n),
        name=f"{reference.name}+isotropic",
        n_samples=reference.n_samples,
        eigenvalue_floor=eigenvalue_floor,
        shrinkage=0.0,
        provenance={
            "derived_from": reference.name,
            "kind": "trace-matched isotropic control (labelled sensitivity only)",
        },
    )


def validate_restart_draws(
    distribution: RestartDistribution,
    ensemble_size: int,
    n_draws: int = 4000,
    seed: int = 991,
) -> Dict[str, float]:
    """Check that drawn ensembles reproduce the intended covariance traces.

    Returns both the trace ratios and relative Frobenius errors, so a failure
    says whether the *scale* or the *structure* is off.
    """
    n = distribution.P_mean.shape[0]
    truth_state = np.zeros(n)
    rng_mean = np.random.default_rng(seed)
    rng_anomaly = np.random.default_rng(seed + 1)
    N = int(ensemble_size)

    block = distribution.draw(truth_state, N, n_draws, rng_mean, rng_anomaly)
    members = block.T.reshape(n_draws, N, n)
    mean_error = members.mean(axis=1)  # exactly b, by construction
    anomalies = members - mean_error[:, None, :]

    empirical_mean = np.cov(mean_error, rowvar=False, ddof=1)
    flat = anomalies.reshape(n_draws * N, n)
    empirical_within = (flat.T @ flat) / (n_draws * (N - 1))

    def _report(empirical, target):
        return (
            float(np.trace(empirical) / np.trace(target)),
            float(
                np.linalg.norm(empirical - target, "fro")
                / np.linalg.norm(target, "fro")
            ),
        )

    trace_mean, frobenius_mean = _report(empirical_mean, distribution.P_mean)
    trace_within, frobenius_within = _report(empirical_within, distribution.P_within)
    return {
        "n_draws": float(n_draws),
        "ensemble_size": float(N),
        "trace_ratio_mean": trace_mean,
        "trace_ratio_within": trace_within,
        "relative_frobenius_mean": frobenius_mean,
        "relative_frobenius_within": frobenius_within,
        "target_trace_mean": float(np.trace(distribution.P_mean)),
        "target_trace_within": float(np.trace(distribution.P_within)),
        "max_abs_anomaly_sum": float(np.abs(anomalies.sum(axis=1)).max()),
    }


# --------------------------------------------------------------------------- #
# One-window kernels
# --------------------------------------------------------------------------- #
#
# These reproduce the analysis algebra of ``bias_variance_mc``'s batched kernels
# exactly -- the gain, the whitening, the QPCA projection and the member update
# are the *imported* helpers, not re-derived here -- but they are scoped to a
# single assimilation window so the same code drives both regimes.  Restart and
# cycled then differ in exactly one respect: what the window starts from.
#
# Note on Seq-EnKF: within one window it performs its native L sequential
# updates, one per observation time, and the recorded quantity is the analysis
# mean at the window endpoint.  Collapsing it to a single window-end update
# would change the filter, which this experiment must not do.


@dataclass
class _WindowContext:
    """Matrices reused at every window; built once per cell."""

    n: int
    m: int
    noise_std: float
    L: int
    steps_per_obs: int
    F: float
    dt: float
    eps_enkf: float
    eps_qpca: float
    R_L_chol: np.ndarray
    whiten: np.ndarray
    taper: np.ndarray | None
    kappa: int


def build_window_context(config: Dict, kappa: int) -> _WindowContext:
    n = int(config["model"]["n"])
    m = int(config["observations"]["m"])
    noise_std = float(config["observations"]["noise_std"])
    L = int(config["assimilation"]["window_length"])
    variance = _check_isotropic_obs_cov(config)

    R_L = np.kron(np.eye(L), noise_std**2 * np.eye(m))
    R_L_chol = linalg.cholesky(R_L, lower=True)
    eye_mL = np.eye(m * L)
    whiten = linalg.solve_triangular(R_L_chol, eye_mL, lower=True).T

    return _WindowContext(
        n=n,
        m=m,
        noise_std=noise_std,
        L=L,
        steps_per_obs=int(config["observations"]["steps_per_obs"]),
        F=float(config["model"]["F"]),
        dt=float(config["model"]["dt"]),
        # Matches the reference kernels: S = P_yy + sigma^2 I + 1e-6 I for the
        # stochastic filters, S = P_yy + 1e-6 I for QPCA-EnDCF.
        eps_enkf=variance + 1e-6,
        eps_qpca=1e-6,
        R_L_chol=R_L_chol,
        whiten=whiten,
        taper=_qpca_taper(config),
        kappa=int(kappa),
    )


def _diverged_realizations(block: np.ndarray, M: int, N: int) -> np.ndarray:
    """Boolean mask over realizations whose ensemble holds a non-finite value."""
    finite = np.isfinite(block).reshape(block.shape[0], M, N).all(axis=(0, 2))
    return ~finite


def _sanitize(block: np.ndarray, diverged: np.ndarray, M: int, N: int) -> None:
    """Zero out the columns of diverged realizations, in place.

    A diverged realization is recorded and reported, never quietly dropped; the
    zeroing exists only so that LAPACK is not handed NaNs on behalf of the
    realizations that are still healthy.  A zero anomaly matrix yields a zero
    gain in both ``_lowrank_gain`` and ``_localized_gain``, so the arithmetic
    stays well defined.
    """
    if not diverged.any():
        return
    view = block.reshape(block.shape[0], M, N)
    view[:, diverged, :] = 0.0


def run_one_window(
    filter_name: str,
    block: np.ndarray,
    obs_window: np.ndarray,
    H: np.ndarray,
    context: _WindowContext,
    M: int,
    N: int,
    rng_obs: np.random.Generator | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance ``(n, M*N)`` through one window and analyse it.

    Parameters
    ----------
    block : ndarray, shape (n, M*N)
        Window-initial forecast ensembles; realization ``t`` owns columns
        ``t*N : (t+1)*N``.
    obs_window : ndarray, shape (L, m)
        The window's observations, in time order.
    rng_obs : Generator or None
        Source of the observation perturbations.  ``None`` for QPCA-EnDCF,
        which draws none.

    Returns
    -------
    block, analysis_means (M, n), diverged (M,)
    """
    n, m, L = context.n, context.m, context.L
    mL = m * L

    # Divergence ACCUMULATES over the window rather than being re-tested at each
    # observation time. _sanitize replaces a blown-up realization's columns with
    # zeros so LAPACK is not handed NaNs on the healthy realizations' behalf, and
    # a zero Lorenz-96 state is perfectly finite -- it relaxes toward F. Testing
    # afresh at the next observation time would therefore find it "recovered"
    # and record a fabricated trajectory as a valid result.
    diverged = np.zeros(M, dtype=bool)

    if filter_name == "Seq-EnKF":
        # One draw for the whole window: (L, M, N, m). Drawing per observation
        # time from the same stream would be equivalent, but this keeps the
        # number of Generator calls independent of L.
        perturbations = context.noise_std * rng_obs.standard_normal((L, M, N, m))
        for ell in range(L):
            block = _advance(block, context.steps_per_obs, context.dt, context.F)
            diverged |= _diverged_realizations(block, M, N)
            _sanitize(block, diverged, M, N)

            Xv = block.reshape(n, M, N)
            Yv = (H @ block).reshape(m, M, N)
            z = obs_window[ell]
            for i in range(M):
                if diverged[i]:
                    continue
                X_i, Y_i = Xv[:, i, :], Yv[:, i, :]
                A_x = X_i - X_i.mean(axis=1, keepdims=True)
                A_y = Y_i - Y_i.mean(axis=1, keepdims=True)
                K = _lowrank_gain(A_x, A_y, context.eps_enkf, N)
                Xv[:, i, :] = X_i + K @ (
                    z[:, None] + perturbations[ell, i].T - Y_i
                )
    else:
        Y_window = np.empty((L, m, M, N))
        for ell in range(L):
            block = _advance(block, context.steps_per_obs, context.dt, context.F)
            Y_window[ell] = (H @ block).reshape(m, M, N)
        # No mid-window sanitizing here: the 4D filters analyse only at the
        # window end, so a non-finite value simply propagates and is caught once.
        diverged |= _diverged_realizations(block, M, N) | ~np.isfinite(
            Y_window
        ).all(axis=(0, 1, 3))
        _sanitize(block, diverged, M, N)
        Y_window[:, :, diverged, :] = 0.0

        z_stack = obs_window.ravel()
        Xv = block.reshape(n, M, N)

        if filter_name == "4D-EnKF":
            perturbations = context.R_L_chol @ rng_obs.standard_normal(
                (M, N, mL)
            ).reshape(M * N, mL).T
            perturbations = perturbations.reshape(mL, M, N)
            for i in range(M):
                if diverged[i]:
                    continue
                Y_stack = Y_window[:, :, i, :].reshape(mL, N)
                X_i = Xv[:, i, :]
                A_x = X_i - X_i.mean(axis=1, keepdims=True)
                A_y = Y_stack - Y_stack.mean(axis=1, keepdims=True)
                K = _lowrank_gain(A_x, A_y, context.eps_enkf, N)
                Xv[:, i, :] = X_i + K @ (
                    z_stack[:, None] + perturbations[:, i, :] - Y_stack
                )
        elif filter_name == "QPCA-EnDCF":
            for i in range(M):
                if diverged[i]:
                    continue
                Y_stack = Y_window[:, :, i, :].reshape(mL, N)
                E = context.whiten @ (Y_stack - z_stack[:, None])
                E_c = E - E.mean(axis=1, keepdims=True)
                eigenvecs, _s, _vt = _thin_svd(E_c)
                V_kappa = eigenvecs[:, : context.kappa]
                Delta_obs = context.R_L_chol @ (-V_kappa @ (V_kappa.T @ E))

                X_i = Xv[:, i, :]
                A_x = X_i - X_i.mean(axis=1, keepdims=True)
                A_y = Y_stack - Y_stack.mean(axis=1, keepdims=True)
                if context.taper is None:
                    K = _lowrank_gain(A_x, A_y, context.eps_qpca, N)
                else:
                    K = _localized_gain(A_x, A_y, context.taper, N)
                Xv[:, i, :] = X_i + K @ Delta_obs
        else:
            raise ValueError(f"Unsupported filter for this experiment: {filter_name!r}")

    diverged = diverged | _diverged_realizations(block, M, N)
    analysis_means = block.reshape(n, M, N).mean(axis=2).T.copy()  # (M, n)
    analysis_means[diverged] = np.nan
    return block, analysis_means, diverged


def requires_observation_perturbations(filter_name: str) -> bool:
    """QPCA-EnDCF is deterministic after initialization; the EnKFs are not."""
    return filter_name in ("Seq-EnKF", "4D-EnKF")


# --------------------------------------------------------------------------- #
# Cell runners
# --------------------------------------------------------------------------- #


def run_restart_cell(
    filter_name: str,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    context: _WindowContext,
    distribution: RestartDistribution,
    truth_seed: int,
    ensemble_size: int,
    n_realizations: int,
    evaluation_windows: Sequence[int],
    mean_error_mode: str = "per_realization",
) -> Dict[str, np.ndarray]:
    """Independent-restart regime for one ``(filter, N, truth seed)`` cell.

    Each evaluated window draws a fresh ensemble, runs exactly one window, and
    discards it.  Nothing is carried between windows -- the loop body holds the
    only reference to ``block``, so there is no path by which an analysis
    ensemble could reach the next evaluated window.

    ``mean_error_mode`` selects how the shared forecast-mean error is drawn; see
    ``MEAN_ERROR_MODES``.
    """
    if mean_error_mode not in MEAN_ERROR_MODES:
        raise ValueError(f"Unknown mean_error_mode: {mean_error_mode!r}")
    shared_mean = mean_error_mode == "per_window"
    L, n = context.L, context.n
    M, N = int(n_realizations), int(ensemble_size)

    errors = np.full((M, len(evaluation_windows), n), np.nan)
    diverged = np.zeros((M, len(evaluation_windows)), dtype=bool)

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
        rng_obs = (
            observation_perturbation_generator(filter_name, truth_seed, w, N)
            if requires_observation_perturbations(filter_name)
            else None
        )
        block, analysis_means, failed = run_one_window(
            filter_name,
            block,
            observations[w * L : (w + 1) * L],
            H,
            context,
            M,
            N,
            rng_obs,
        )
        errors[:, index, :] = analysis_means - truth[(w + 1) * L][None, :]
        diverged[:, index] = failed
        del block  # explicit: nothing survives to the next evaluated window

    return {"errors": errors, "diverged": diverged}


def run_cycled_cell(
    filter_name: str,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    context: _WindowContext,
    distribution: RestartDistribution,
    truth_seed: int,
    ensemble_size: int,
    n_realizations: int,
    evaluation_windows: Sequence[int],
    window_start_hook: Callable[[int, np.ndarray], None] | None = None,
) -> Dict[str, np.ndarray]:
    """Ordinary cycling for one ``(filter, N, truth seed)`` cell.

    The window-0 ensemble is drawn from the *same* restart law with the *same*
    ``(r, w=0, t)`` streams the restart regime would use, so the two regimes
    start from identical ensembles and diverge only through cycling.

    ``window_start_hook(w, block)`` is called with the window-initial forecast
    ensemble before it is advanced; the calibration pass uses it to accumulate
    the forecast statistics the restart law is built from.

    The window-0 draw always uses the per-realization mean error, whatever the
    restart regime is configured to do: cycled realizations must start from
    genuinely distinct ensembles, and the initialization is in any case
    forgotten long before the first evaluated window.
    """
    L, n = context.L, context.n
    M, N = int(n_realizations), int(ensemble_size)
    evaluation_windows = [int(w) for w in evaluation_windows]
    window_index = {w: i for i, w in enumerate(evaluation_windows)}
    total_windows = max(evaluation_windows) + 1

    block = distribution.draw(
        truth[0],
        N,
        M,
        restart_mean_generator(truth_seed, 0),
        restart_anomaly_generator(truth_seed, 0, N),
        shared_mean_across_realizations=False,
    )

    errors = np.full((M, len(evaluation_windows), n), np.nan)
    diverged = np.zeros((M, len(evaluation_windows)), dtype=bool)
    ever_diverged = np.zeros(M, dtype=bool)

    for w in range(total_windows):
        if window_start_hook is not None:
            window_start_hook(w, block)
        rng_obs = (
            observation_perturbation_generator(filter_name, truth_seed, w, N)
            if requires_observation_perturbations(filter_name)
            else None
        )
        block, analysis_means, failed = run_one_window(
            filter_name,
            block,
            observations[w * L : (w + 1) * L],
            H,
            context,
            M,
            N,
            rng_obs,
        )
        ever_diverged |= failed
        if w in window_index:
            index = window_index[w]
            errors[:, index, :] = analysis_means - truth[(w + 1) * L][None, :]
            # Once a cycled realization has gone non-finite it never recovers,
            # so the whole tail of that realization is marked, not just the
            # window where the blow-up was first detected.
            diverged[:, index] = ever_diverged
            errors[ever_diverged, index, :] = np.nan

    return {"errors": errors, "diverged": diverged}


# --------------------------------------------------------------------------- #
# Restart-distribution calibration
# --------------------------------------------------------------------------- #


def bootstrap_restart_distribution(n: int) -> RestartDistribution:
    """The isotropic law used only to seed the *calibration* runs' window 0.

    It reproduces the existing driver's ``_initial_ensemble``
    (``x_true + 0.5 xi_0 + 0.5 xi_j``), i.e. ``P_mean = P_within = 0.25 I``.
    Its role is purely to start the calibration cycles somewhere; the first
    ``calibration_spinup_windows`` windows are discarded before any statistic is
    accumulated, by which point a Lorenz-96 filter has long forgotten its
    initial ensemble.  It is never a restart law in a reported result.
    """
    quarter = 0.25 * np.eye(n)
    return build_restart_distribution(
        quarter,
        quarter,
        name="bootstrap-isotropic",
        n_samples=0,
        eigenvalue_floor=0.0,
        shrinkage=0.0,
        provenance={"kind": "calibration bootstrap only; never a reported law"},
    )


def _calibration_task(
    config: Dict,
    settings: ExperimentSettings,
    truth_seed: int,
    filter_name: str,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
):
    """Cycle one ``(calibration seed, filter)`` pair and return its accumulator."""
    n = int(config["model"]["n"])
    N = int(settings.calibration_ensemble_size)
    M = int(settings.calibration_n_mc)
    context = build_window_context(config, settings.kappa)

    accumulator = _RestartCalibrationAccumulator(n)
    spinup = int(settings.calibration_spinup_windows)
    L = context.L

    def hook(w: int, block: np.ndarray) -> None:
        if w < spinup:
            return
        if not np.isfinite(block).all():
            return
        accumulator.update_structured(block, truth[w * L], M, N)

    windows = list(range(int(settings.calibration_n_windows)))
    try:
        result = run_cycled_cell(
            filter_name,
            truth,
            observations,
            H,
            context,
            bootstrap_restart_distribution(n),
            truth_seed,
            N,
            M,
            windows,
            window_start_hook=hook,
        )
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        return truth_seed, filter_name, None, f"{type(exc).__name__}: {exc}"

    accumulator.break_sequence()
    n_diverged = int(result["diverged"].any(axis=1).sum())
    return truth_seed, filter_name, (accumulator, n_diverged), None


def calibrate_restart_distribution(
    config: Dict,
    settings: ExperimentSettings,
    n_jobs: int = 1,
    verbose: int = 0,
) -> Dict[str, object]:
    """Estimate ``(P_mean, P_within)`` from calibration truth seeds only.

    Returns a dict with the primary distribution under ``"primary"``, the
    per-filter marginal-matched laws under ``"filter_marginal"``, the labelled
    isotropic control under ``"isotropic"``, the dependence diagnostics, and the
    full provenance.

    Design
    ------
    The primary law is *common*: every filter starts its restart windows from a
    statistically identical forecast ensemble.  It is built by pooling the three
    filters' calibration statistics with equal weight -- a neutral, matched
    construction that does not privilege QPCA-EnDCF, which the filter-marginal
    alternative would.  The filter-marginal laws are computed at the same time
    and reported as a sensitivity analysis; they answer a different question
    (each filter faced with *its own* forecast distribution) and the two
    interpretations must not be conflated.
    """
    seeds = list(settings.calibration_seeds)
    if set(seeds) & set(settings.evaluation_seeds):
        raise ValueError("Calibration seeds must be disjoint from evaluation seeds.")

    calibration_config = _config_for_windows(config, settings.calibration_n_windows)
    truth_data = generate_truth_batch(calibration_config, seeds)

    tasks = [
        (
            calibration_config,
            settings,
            seed,
            name,
            truth_data[seed][0],
            truth_data[seed][1],
            truth_data[seed][2],
        )
        for seed in seeds
        for name in settings.filter_names
    ]

    n = int(config["model"]["n"])
    per_filter = {name: _RestartCalibrationAccumulator(n) for name in settings.filter_names}
    divergence: Dict[str, int] = {name: 0 for name in settings.filter_names}
    failures: List[Tuple[int, str, str]] = []

    for seed, name, payload, error in _iter_tasks(_calibration_task, tasks, n_jobs, verbose):
        if error is not None:
            failures.append((seed, name, error))
            continue
        accumulator, n_diverged = payload
        per_filter[name].merge(accumulator)
        divergence[name] += n_diverged

    if failures:
        for seed, name, error in failures:
            print(f"  calibration FAILED {name} seed {seed}: {error}", flush=True)
        raise RuntimeError(
            f"{len(failures)} calibration task(s) failed; the restart law cannot be "
            "estimated from a partial calibration set."
        )

    # --- filter-marginal laws (sensitivity) --------------------------------- #
    filter_marginal: Dict[str, RestartDistribution] = {}
    for name, accumulator in per_filter.items():
        raw_mean, raw_within = accumulator.raw_covariances()
        filter_marginal[name] = build_restart_distribution(
            raw_mean,
            raw_within,
            name=f"filter-marginal[{name}]",
            n_samples=accumulator.count,
            eigenvalue_floor=settings.eigenvalue_floor,
            shrinkage=settings.shrinkage,
            provenance={
                "kind": "filter-marginal matched (sensitivity analysis)",
                "filter": name,
                "calibration_seeds": [int(s) for s in seeds],
            },
        )

    # --- common law (primary): equal-weight pool over filters ---------------- #
    raw = [per_filter[name].raw_covariances() for name in settings.filter_names]
    pooled_mean = np.mean([r[0] for r in raw], axis=0)
    pooled_within = np.mean([r[1] for r in raw], axis=0)
    total_samples = sum(per_filter[name].count for name in settings.filter_names)
    common = build_restart_distribution(
        pooled_mean,
        pooled_within,
        name="common-pooled",
        n_samples=total_samples,
        eigenvalue_floor=settings.eigenvalue_floor,
        shrinkage=settings.shrinkage,
        provenance={
            "kind": "common restart law, equal-weight pool across filters",
            "filters": list(settings.filter_names),
            "calibration_seeds": [int(s) for s in seeds],
            "calibration_ensemble_size": int(settings.calibration_ensemble_size),
            "calibration_n_mc": int(settings.calibration_n_mc),
            "calibration_n_windows": int(settings.calibration_n_windows),
            "calibration_spinup_windows": int(settings.calibration_spinup_windows),
            "eigenvalue_floor": float(settings.eigenvalue_floor),
            "shrinkage": float(settings.shrinkage),
        },
    )

    isotropic = isotropic_restart_distribution(common, settings.eigenvalue_floor)

    if settings.restart_mode == "common":
        primary = common
    elif settings.restart_mode == "isotropic":
        primary = isotropic
    else:  # filter_marginal
        primary = filter_marginal  # resolved per filter by the driver

    diagnostics = {
        name: accumulator.diagnostics() for name, accumulator in per_filter.items()
    }

    return {
        "primary": primary,
        "common": common,
        "filter_marginal": filter_marginal,
        "isotropic": isotropic,
        "diagnostics": diagnostics,
        "divergence": divergence,
        "calibration_seeds": seeds,
    }


def resolve_distribution(
    calibration: Dict[str, object], filter_name: str, mode: str
) -> RestartDistribution:
    """The restart law a given filter should draw from, under ``mode``."""
    if mode == "filter_marginal":
        return calibration["filter_marginal"][filter_name]
    if mode == "isotropic":
        return calibration["isotropic"]
    return calibration["common"]


def _config_for_windows(config: Dict, n_windows: int) -> Dict:
    """Copy of ``config`` whose truth/observation horizon covers ``n_windows``."""
    from copy import deepcopy

    out = deepcopy(config)
    out["assimilation"]["n_windows"] = int(n_windows)
    out["assimilation"]["total_observations"] = int(n_windows) * int(
        config["assimilation"]["window_length"]
    )
    return out


# --------------------------------------------------------------------------- #
# Parallel plumbing
# --------------------------------------------------------------------------- #


def _iter_tasks(function, tasks: List[Tuple], n_jobs: int, verbose: int):
    """Yield task results as they complete, from one process pool."""
    if n_jobs == 1 or len(tasks) <= 1:
        for task in tasks:
            yield function(*task)
        return

    import joblib
    from joblib import Parallel, delayed

    jobs = [delayed(function)(*task) for task in tasks]
    ctx = (
        joblib.parallel_config(backend="loky", inner_max_num_threads=1)
        if hasattr(joblib, "parallel_config")
        else _nullcontext()
    )
    with ctx:
        try:
            yield from Parallel(
                n_jobs=n_jobs,
                backend="loky",
                batch_size=1,
                verbose=verbose,
                return_as="generator_unordered",
            )(jobs)
        except TypeError:  # joblib < 1.3 has no return_as
            yield from Parallel(
                n_jobs=n_jobs, backend="loky", batch_size=1, verbose=verbose
            )(jobs)


# --------------------------------------------------------------------------- #
# Cell summaries and checkpoints
# --------------------------------------------------------------------------- #


def summarize_cell(errors: np.ndarray, diverged: np.ndarray) -> Dict[str, np.ndarray]:
    """Reduce ``(M, W, n)`` analysis-mean errors to the stored sufficient statistics.

    What is stored, and why:

    ``error_gram`` (M, M)
        ``G[t,t'] = (1/(W n)) sum_{w,i} e_{t,w,i} e_{t',w,i}``, the same
        normalization ``bias_variance_mc`` uses.  For multinomial realization
        counts ``c`` it gives, exactly,

            MSE*    = c . diag(G) / M
            Bias^2* = c^T G c / M^2

        window-averaged and state-dimension-normalized, so the whole inner
        bootstrap is an ``O(M^2)`` quadratic form rather than a re-reduction of
        the raw errors.

    ``error_gram_stride2``
        The same object over every other evaluated window, i.e. effective
        stride ``2s``.  It makes the stride-sensitivity check free instead of a
        second production run.

    ``sum_error`` (W, n) and ``sum_sq_error`` (W,)
        Exact per-window sums, so point estimates can be recomputed on any
        subset of windows afterwards without rerunning anything.
    """
    M, W, n = errors.shape
    realization_ok = ~diverged.any(axis=1)
    valid = errors[realization_ok]
    n_valid = int(realization_ok.sum())

    out: Dict[str, np.ndarray] = {
        "n_realizations": np.asarray(M),
        "n_valid_realizations": np.asarray(n_valid),
        "realization_ok": realization_ok,
        "diverged": diverged,
        "n_diverged_realization_windows": np.asarray(int(diverged.sum())),
        "n_windows": np.asarray(W),
        "state_dimension": np.asarray(n),
    }

    if n_valid < 2:
        # Nothing estimable. Recorded explicitly rather than written as zeros
        # that would silently read as "measured, and small".
        empty2 = np.full((n_valid, n_valid), np.nan)
        out.update(
            {
                "error_gram": empty2,
                "error_gram_stride2": empty2,
                "sum_error": np.full((W, n), np.nan),
                "sum_sq_error": np.full(W, np.nan),
                "bias_squared_trace": np.full(W, np.nan),
                "variance_trace": np.full(W, np.nan),
                "mse_trace": np.full(W, np.nan),
                "bias_squared_debiased_trace": np.full(W, np.nan),
            }
        )
        return out

    flat = valid.reshape(n_valid, W * n)
    out["error_gram"] = (flat @ flat.T) / (W * n)
    sub = valid[:, ::2]
    flat2 = sub.reshape(n_valid, sub.shape[1] * n)
    out["error_gram_stride2"] = (flat2 @ flat2.T) / (sub.shape[1] * n)

    out["sum_error"] = valid.sum(axis=0)
    out["sum_sq_error"] = (valid**2).sum(axis=(0, 2))

    expectation = valid.mean(axis=0)  # (W, n)
    bias_squared = (expectation**2).mean(axis=1)  # (W,)
    mse = (valid**2).mean(axis=(0, 2))  # (W,)
    # Exact for the population variance across realizations, so the plug-in
    # identity MSE = Bias^2 + Var holds to roundoff by construction.
    variance = mse - bias_squared
    out["bias_squared_trace"] = bias_squared
    out["variance_trace"] = variance
    out["mse_trace"] = mse
    out["bias_squared_debiased_trace"] = (
        bias_squared - variance / (n_valid - 1) if n_valid > 1 else np.full(W, np.nan)
    )
    return out


def cell_checkpoint_path(
    results_dir: Path, regime: str, filter_name: str, ensemble_size: int, truth_seed: int
) -> Path:
    slug = filter_name.lower().replace("-", "_")
    return (
        Path(results_dir)
        / "cells"
        / f"{regime}_{slug}_N{int(ensemble_size)}_truth_seed_{int(truth_seed)}.npz"
    )


def cell_checkpoint_matches(
    path: Path,
    expected_hash: str,
    n_mc: int,
    n_windows: int,
    restart_distribution_id: str,
) -> bool:
    """True only if the checkpoint was produced by the current configuration.

    A checkpoint from a different configuration hash, ensemble count, window
    count or restart law is *not* loaded.  Mixing them inside one figure is the
    failure mode this guards against.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["config_hash"]) == expected_hash
                and int(data["n_realizations"]) == int(n_mc)
                and int(data["n_windows"]) == int(n_windows)
                and str(data["restart_distribution_id"]) == restart_distribution_id
            )
    except (OSError, ValueError, KeyError):
        return False


def _cell_task(
    config: Dict,
    settings: ExperimentSettings,
    regime: str,
    filter_name: str,
    ensemble_size: int,
    truth_seed: int,
    truth: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    distribution: RestartDistribution,
    results_dir: Path,
    expected_hash: str,
):
    """Run one ``(regime, filter, N, truth seed)`` cell and checkpoint it."""
    started = time.perf_counter()
    try:
        validate_kappa(settings.kappa, ensemble_size, settings.stacked_obs_dimension)
        context = build_window_context(config, settings.kappa)
        windows = [int(w) for w in settings.evaluation_windows]

        common = (
            filter_name,
            truth,
            observations,
            H,
            context,
            distribution,
            int(truth_seed),
            int(ensemble_size),
            int(settings.n_mc),
            windows,
        )
        result = (
            run_restart_cell(*common, mean_error_mode=settings.mean_error_mode)
            if regime == "restart"
            else run_cycled_cell(*common)
        )

        summary = summarize_cell(result["errors"], result["diverged"])
        summary.update(
            {
                "config_hash": np.asarray(expected_hash),
                "regime": np.asarray(regime),
                "filter": np.asarray(filter_name),
                "ensemble_size": np.asarray(int(ensemble_size)),
                "kappa": np.asarray(int(settings.kappa)),
                "truth_seed": np.asarray(int(truth_seed)),
                # generate_observations seeds the noise stream with the truth
                # seed, so the two are the same integer by construction; both
                # are recorded so that stays visible if it ever changes.
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
                                _STREAM_RESTART_MEAN,
                                "truth_seed",
                                "window",
                            ],
                            "within_ensemble_anomaly": [
                                RNG_BASE_ENTROPY,
                                _STREAM_RESTART_ANOMALY,
                                "truth_seed",
                                "window",
                                "ensemble_size",
                            ],
                            "observation_perturbation": [
                                RNG_BASE_ENTROPY,
                                _STREAM_OBS_PERTURBATION,
                                "filter_index",
                                "truth_seed",
                                "window",
                                "ensemble_size",
                            ],
                            "filter_index": _FILTER_STREAM_INDEX,
                        },
                        sort_keys=True,
                    )
                ),
            }
        )

        path = cell_checkpoint_path(
            results_dir, regime, filter_name, ensemble_size, truth_seed
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **summary)
    except Exception as exc:  # noqa: BLE001 -- returned, so the pool survives
        return (
            regime,
            filter_name,
            int(ensemble_size),
            int(truth_seed),
            time.perf_counter() - started,
            int(0),
            f"{type(exc).__name__}: {exc}",
        )

    return (
        regime,
        filter_name,
        int(ensemble_size),
        int(truth_seed),
        time.perf_counter() - started,
        int(summary["n_diverged_realization_windows"]),
        None,
    )


# --------------------------------------------------------------------------- #
# Calibration persistence
# --------------------------------------------------------------------------- #

_DISTRIBUTION_ARRAY_FIELDS = (
    "P_mean",
    "P_within",
    "factor_mean",
    "factor_within",
    "eigenvalues_mean_raw",
    "eigenvalues_within_raw",
    "eigenvalues_mean",
    "eigenvalues_within",
)


def _distribution_to_arrays(prefix: str, distribution: RestartDistribution) -> Dict:
    out = {
        f"{prefix}__{name}": getattr(distribution, name)
        for name in _DISTRIBUTION_ARRAY_FIELDS
    }
    out[f"{prefix}__scalars"] = np.asarray(
        json.dumps(
            {
                "name": distribution.name,
                "effective_rank_mean": distribution.effective_rank_mean,
                "effective_rank_within": distribution.effective_rank_within,
                "n_samples": distribution.n_samples,
                "identifier": distribution.identifier,
                "provenance": distribution.provenance,
            },
            sort_keys=True,
        )
    )
    return out


def _distribution_from_arrays(prefix: str, data) -> RestartDistribution:
    scalars = json.loads(str(data[f"{prefix}__scalars"]))
    return RestartDistribution(
        name=scalars["name"],
        effective_rank_mean=scalars["effective_rank_mean"],
        effective_rank_within=scalars["effective_rank_within"],
        n_samples=scalars["n_samples"],
        provenance=scalars["provenance"],
        **{name: data[f"{prefix}__{name}"] for name in _DISTRIBUTION_ARRAY_FIELDS},
    )


def save_calibration(path: Path, calibration: Dict[str, object], settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    payload.update(_distribution_to_arrays("common", calibration["common"]))
    payload.update(_distribution_to_arrays("isotropic", calibration["isotropic"]))
    for name, distribution in calibration["filter_marginal"].items():
        payload.update(
            _distribution_to_arrays(f"filter_marginal__{_slug(name)}", distribution)
        )
    for name, diagnostics in calibration["diagnostics"].items():
        for key, value in diagnostics.items():
            payload[f"diagnostics__{_slug(name)}__{key}"] = value
    payload["filter_names"] = np.asarray(list(calibration["filter_marginal"]))
    payload["calibration_seeds"] = np.asarray(calibration["calibration_seeds"])
    payload["divergence"] = np.asarray(json.dumps(calibration["divergence"], sort_keys=True))
    payload["restart_mode"] = np.asarray(settings.restart_mode)
    np.savez_compressed(path, **payload)


def load_calibration(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        filter_names = [str(x) for x in data["filter_names"]]
        common = _distribution_from_arrays("common", data)
        isotropic = _distribution_from_arrays("isotropic", data)
        filter_marginal = {
            name: _distribution_from_arrays(f"filter_marginal__{_slug(name)}", data)
            for name in filter_names
        }
        diagnostics: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in filter_names}
        for key in data.files:
            if not key.startswith("diagnostics__"):
                continue
            _, slug, field_name = key.split("__", 2)
            for name in filter_names:
                if _slug(name) == slug:
                    diagnostics[name][field_name] = data[key]
        mode = str(data["restart_mode"])
        calibration = {
            "common": common,
            "isotropic": isotropic,
            "filter_marginal": filter_marginal,
            "diagnostics": diagnostics,
            "divergence": json.loads(str(data["divergence"])),
            "calibration_seeds": [int(s) for s in data["calibration_seeds"]],
        }
    calibration["primary"] = (
        filter_marginal if mode == "filter_marginal" else calibration[mode]
    )
    return calibration


def _slug(name: str) -> str:
    return name.lower().replace("-", "_")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def resolve_results_dir(
    config: Dict, settings: ExperimentSettings, root: Path | None = None
) -> Tuple[Path, str, Dict]:
    """``(directory, hash, scientific config)`` for the current settings.

    The directory name carries the label *and* the leading 8 hex digits of the
    configuration hash, so a smoke run and a production run never share a
    directory even if someone reuses the label.
    """
    scientific = scientific_config(config, settings)
    digest = config_hash(scientific)
    root = Path(root) if root is not None else DEFAULT_RESULTS_ROOT
    return root / f"{settings.label}_{digest[:8]}", digest, scientific


def write_manifest(
    results_dir: Path,
    scientific: Dict,
    digest: str,
    settings: ExperimentSettings,
    regimes: Sequence[str] = REGIMES,
) -> None:
    """Write the manifest, recording which regimes this run actually covers.

    ``regimes`` matters on reload: a ``--regime restart`` companion run holds no
    cycled cells, and a loader that assumed both would decide every truth seed
    was incomplete.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "restart_vs_cycled",
        "config_hash": digest,
        "scientific_config": scientific,
        "label": settings.label,
        "evaluation_windows": [int(w) for w in settings.evaluation_windows],
        "total_cycled_windows": int(settings.total_cycled_windows),
        "regimes": list(regimes),
        "caveats": list(INTERPRETATION_CAVEATS),
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def run_restart_vs_cycled(
    config: Dict,
    settings: ExperimentSettings,
    n_jobs: int = 1,
    results_root: Path | None = None,
    resume: bool = True,
    verbose: int = 5,
    regimes: Sequence[str] = REGIMES,
) -> Dict:
    """Run every requested ``(regime, filter, N, truth seed)`` cell, resumably."""
    results_dir, digest, scientific = resolve_results_dir(config, settings, results_root)
    write_manifest(results_dir, scientific, digest, settings, regimes)
    print(f"Results directory: {results_dir}", flush=True)
    print(f"Configuration hash: {digest}", flush=True)

    # --- restart-distribution calibration ----------------------------------- #
    calibration_path = results_dir / "calibration" / "restart_distribution.npz"
    if resume and calibration_path.exists():
        calibration = load_calibration(calibration_path)
        print(
            f"Loaded restart distribution '{calibration['common'].name}' "
            f"({calibration['common'].identifier}) from {calibration_path}",
            flush=True,
        )
    else:
        t0 = time.perf_counter()
        print(
            f"Calibrating the restart distribution on {len(settings.calibration_seeds)} "
            f"held-out truth seeds "
            f"(N={settings.calibration_ensemble_size}, M={settings.calibration_n_mc}, "
            f"W={settings.calibration_n_windows}, spin-up "
            f"{settings.calibration_spinup_windows})...",
            flush=True,
        )
        calibration = calibrate_restart_distribution(config, settings, n_jobs, verbose)
        save_calibration(calibration_path, calibration, settings)
        print(
            f"  done in {time.perf_counter() - t0:.1f}s -> {calibration_path}",
            flush=True,
        )

    validation = {
        int(N): validate_restart_draws(calibration["common"], N)
        for N in settings.ensemble_sizes
    }
    np.savez_compressed(
        results_dir / "calibration" / "draw_validation.npz",
        report=np.asarray(json.dumps(validation, sort_keys=True)),
    )
    for N, report in validation.items():
        print(
            f"  restart draws at N={N:>2}: trace ratio "
            f"mean {report['trace_ratio_mean']:.4f}, within "
            f"{report['trace_ratio_within']:.4f}; max |sum of anomalies| "
            f"{report['max_abs_anomaly_sum']:.2e}",
            flush=True,
        )

    # --- cells --------------------------------------------------------------- #
    cells = [
        (regime, name, N, seed)
        for regime in regimes
        for name in settings.filter_names
        for N in settings.ensemble_sizes
        for seed in settings.evaluation_seeds
    ]
    distributions = {
        name: resolve_distribution(calibration, name, settings.restart_mode)
        for name in settings.filter_names
    }

    pending = []
    for regime, name, N, seed in cells:
        path = cell_checkpoint_path(results_dir, regime, name, N, seed)
        if (
            resume
            and path.exists()
            and cell_checkpoint_matches(
                path,
                digest,
                settings.n_mc,
                settings.n_windows,
                distributions[name].identifier,
            )
        ):
            continue
        pending.append((regime, name, N, seed))

    print(
        f"Grid: {len(regimes)} regimes x {len(settings.filter_names)} filters x "
        f"{len(settings.ensemble_sizes)} ensemble sizes x "
        f"{len(settings.evaluation_seeds)} truth seeds = {len(cells)} cells "
        f"({len(pending)} to run); M={settings.n_mc}, "
        f"{settings.n_windows} evaluated windows at stride {settings.window_stride} "
        f"after {settings.spinup_windows} spin-up windows "
        f"({settings.total_cycled_windows} windows integrated per cycled cell)",
        flush=True,
    )

    if pending:
        needed = sorted({seed for _g, _f, _N, seed in pending})
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
                name,
                N,
                seed,
                truth_data[seed][0],
                truth_data[seed][1],
                truth_data[seed][2],
                distributions[name],
                results_dir,
                digest,
            )
            for regime, name, N, seed in pending
        ]

        t_start = time.perf_counter()
        completed = 0
        failures: List[Tuple] = []
        divergences: List[Tuple] = []
        for regime, name, N, seed, seconds, n_diverged, error in _iter_tasks(
            _cell_task, tasks, n_jobs, verbose
        ):
            completed += 1
            if error is None:
                note = "" if n_diverged == 0 else f", {n_diverged} diverged (realization, window)"
                print(
                    f"  [{completed}/{len(tasks)}] wrote {regime:<7} {name:<11} "
                    f"N={N:<2} truth seed {seed} ({seconds:.1f}s{note}, "
                    f"{time.perf_counter() - t_start:.0f}s elapsed)",
                    flush=True,
                )
                if n_diverged:
                    divergences.append((regime, name, N, seed, n_diverged))
            else:
                failures.append((regime, name, N, seed, error))
                print(
                    f"  [{completed}/{len(tasks)}] FAILED {regime} {name} N={N} "
                    f"truth seed {seed}: {error}",
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
            for regime, name, N, seed, count in divergences:
                print(
                    f"  {regime:<7} {name:<11} N={N:<2} truth seed {seed}: "
                    f"{count} (realization, window) pairs non-finite",
                    flush=True,
                )
        if failures:
            print(f"\n{len(failures)} cell(s) FAILED and have no checkpoint:", flush=True)
            for regime, name, N, seed, error in failures:
                print(f"  {regime} {name} N={N} truth seed {seed}: {error}", flush=True)
            print("Rerun the same command to retry; complete cells are skipped.", flush=True)

    return {
        "results_dir": results_dir,
        "config_hash": digest,
        "calibration": calibration,
        "draw_validation": validation,
        "cells": load_cells(results_dir, settings, digest, distributions, regimes),
        "regimes": tuple(regimes),
    }


# --------------------------------------------------------------------------- #
# Loading and analysis
# --------------------------------------------------------------------------- #

CellKey = Tuple[str, str, int, int]  # (regime, filter, N, truth seed)
METRIC_NAMES = ("mse", "bias_squared", "bias_squared_debiased", "variance")


def discover_runs(root: Path | None = None) -> List[Dict[str, object]]:
    """Every completed or partial run under ``root``, newest-looking first.

    Used by the notebook so that what it reports always comes from a run that
    actually exists on disk, rather than from settings retyped alongside it.
    """
    root = Path(root) if root is not None else DEFAULT_RESULTS_ROOT
    if not root.exists():
        return []
    runs = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
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


def settings_from_manifest(manifest: Dict) -> ExperimentSettings:
    """Rebuild the exact settings a run was executed with.

    The manifest carries the full scientific configuration, so the notebook
    never has to restate the design -- and therefore cannot silently disagree
    with the checkpoints it is plotting.
    """
    scientific = manifest["scientific_config"]
    design = scientific["design"]
    restart = scientific["restart_distribution"]
    return ExperimentSettings(
        ensemble_sizes=tuple(design["ensemble_sizes"]),
        filter_names=tuple(scientific["filters"]["names"]),
        kappa=int(scientific["filters"]["kappa"]),
        n_mc=int(design["n_mc"]),
        n_windows=int(design["n_windows"]),
        window_stride=int(design["window_stride"]),
        spinup_windows=int(design["spinup_windows"]),
        n_calibration_seeds=len(design["calibration_seeds"]),
        n_evaluation_seeds=len(design["evaluation_seeds"]),
        truth_seed_base=int(design["truth_seed_base"]),
        calibration_seed_offset=int(design["calibration_seed_offset"]),
        calibration_seeds=tuple(design["calibration_seeds"]),
        evaluation_seeds=tuple(design["evaluation_seeds"]),
        calibration_ensemble_size=int(restart["calibration_ensemble_size"]),
        calibration_n_mc=int(restart["calibration_n_mc"]),
        calibration_n_windows=int(restart["calibration_n_windows"]),
        calibration_spinup_windows=int(restart["calibration_spinup_windows"]),
        restart_mode=restart["mode"],
        mean_error_mode=restart.get("mean_error_mode", "per_realization"),
        eigenvalue_floor=float(restart["eigenvalue_floor"]),
        shrinkage=float(restart["shrinkage"]),
        label=manifest.get("label", "run"),
        stacked_obs_dimension=int(scientific["observations"]["m"])
        * int(scientific["assimilation"]["window_length"]),
    )


def load_run(results_dir: Path) -> Dict[str, object]:
    """Load one run's manifest, settings, calibration and cells together.

    Returns ``{}`` if the directory holds no manifest, so a notebook can guard
    on it the same way section 1 guards on missing hierarchical checkpoints.
    """
    results_dir = Path(results_dir)
    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    settings = settings_from_manifest(manifest)

    calibration_path = results_dir / "calibration" / "restart_distribution.npz"
    calibration = load_calibration(calibration_path) if calibration_path.exists() else None

    validation_path = results_dir / "calibration" / "draw_validation.npz"
    validation = None
    if validation_path.exists():
        with np.load(validation_path, allow_pickle=False) as data:
            validation = {
                int(k): v for k, v in json.loads(str(data["report"])).items()
            }

    regimes = tuple(manifest.get("regimes", REGIMES))
    cells = load_cells(results_dir, settings, manifest["config_hash"], regimes=regimes)
    return {
        "results_dir": results_dir,
        "manifest": manifest,
        "settings": settings,
        "config_hash": manifest["config_hash"],
        "regimes": regimes,
        "calibration": calibration,
        "draw_validation": validation,
        "cells": cells,
        "seed_report": (
            complete_truth_seeds(cells, settings, regimes) if cells else None
        ),
    }


def load_cells(
    results_dir: Path,
    settings: ExperimentSettings,
    digest: str | None = None,
    distributions: Dict[str, RestartDistribution] | None = None,
    regimes: Sequence[str] = REGIMES,
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
    for regime in regimes:
        for name in settings.filter_names:
            expected_id = (
                distributions[name].identifier if distributions is not None else None
            )
            for N in settings.ensemble_sizes:
                for seed in settings.evaluation_seeds:
                    path = cell_checkpoint_path(results_dir, regime, name, N, seed)
                    if not path.exists():
                        continue
                    with np.load(path, allow_pickle=False) as data:
                        if str(data["config_hash"]) != digest:
                            continue
                        if (
                            expected_id is not None
                            and str(data["restart_distribution_id"]) != expected_id
                        ):
                            continue
                        cells[(regime, name, int(N), int(seed))] = {
                            key: data[key] for key in data.files
                        }
    return cells


def complete_truth_seeds(
    cells: Dict[CellKey, Dict[str, np.ndarray]],
    settings: ExperimentSettings,
    regimes: Sequence[str] = REGIMES,
) -> Dict[str, object]:
    """Truth seeds usable for paired analysis, plus what was excluded and why.

    Two seed sets are returned.

    ``common``
        Seeds complete and divergence-free in *every* cell -- every regime,
        filter and ensemble size.  This is the set the paired analysis uses,
        including the variance-slope fit, because a slope fitted across
        ensemble sizes with different seed sets is not a paired quantity.

    ``per_size``
        Per ``N``, the seeds complete across regimes and filters at that ``N``
        alone.  Reported so that an ensemble size which is unusable for some
        filter is visible rather than silently shrinking the common set.

    A seed is never dropped for one filter and kept for another.
    """
    required = [
        (regime, name, N)
        for regime in regimes
        for name in settings.filter_names
        for N in settings.ensemble_sizes
    ]

    def _status(regime, name, N, seed) -> str:
        """``"ok"``, ``"missing"`` or ``"diverged"`` for one cell.

        The two failure modes are kept apart on purpose: a missing cell means
        the run is unfinished, a diverged one means the filter blew up. Reporting
        both as "incomplete" would let an unfinished run read as a scientific
        finding about instability.
        """
        cell = cells.get((regime, name, int(N), int(seed)))
        if cell is None:
            return "missing"
        if int(cell["n_valid_realizations"]) != int(cell["n_realizations"]):
            return "diverged"
        return "ok"

    def _ok(regime, name, N, seed) -> bool:
        return _status(regime, name, N, seed) == "ok"

    per_size: Dict[int, List[int]] = {}
    for N in settings.ensemble_sizes:
        per_size[int(N)] = [
            int(seed)
            for seed in settings.evaluation_seeds
            if all(
                _ok(regime, name, N, seed)
                for regime in regimes
                for name in settings.filter_names
            )
        ]

    common = [
        int(seed)
        for seed in settings.evaluation_seeds
        if all(_ok(regime, name, N, seed) for regime, name, N in required)
    ]

    excluded: Dict[int, List[str]] = {}
    n_missing = 0
    n_diverged = 0
    for seed in settings.evaluation_seeds:
        reasons = []
        for regime, name, N in required:
            status = _status(regime, name, N, seed)
            if status == "ok":
                continue
            reasons.append(f"{regime}/{name}/N={N} ({status})")
            n_missing += status == "missing"
            n_diverged += status == "diverged"
        if reasons:
            excluded[int(seed)] = reasons

    return {
        "common": common,
        "per_size": per_size,
        "excluded": excluded,
        "n_missing_cells": n_missing,
        "n_diverged_cells": n_diverged,
        "n_required_cells": len(required) * len(settings.evaluation_seeds),
    }


def _gram_key(stride_multiplier: int) -> str:
    if stride_multiplier == 1:
        return "error_gram"
    if stride_multiplier == 2:
        return "error_gram_stride2"
    raise ValueError("Only stride multipliers 1 and 2 are stored.")


def point_estimates(
    cells: Dict[CellKey, Dict[str, np.ndarray]],
    seeds: Sequence[int],
    settings: ExperimentSettings,
    regimes: Sequence[str] = REGIMES,
) -> Dict[Tuple[str, str, int], Dict[str, np.ndarray]]:
    """Per-seed window-averaged bias^2 / variance / MSE for every cell.

    Returns ``{(regime, filter, N): {metric: array over seeds}}``.  Averaging
    across seeds is left to the caller so that the across-seed spread stays
    available for the figure whiskers.
    """
    out: Dict[Tuple[str, str, int], Dict[str, np.ndarray]] = {}
    for regime in regimes:
        for name in settings.filter_names:
            for N in settings.ensemble_sizes:
                rows = {metric: [] for metric in METRIC_NAMES}
                for seed in seeds:
                    cell = cells[(regime, name, int(N), int(seed))]
                    rows["mse"].append(float(cell["mse_trace"].mean()))
                    rows["bias_squared"].append(float(cell["bias_squared_trace"].mean()))
                    rows["bias_squared_debiased"].append(
                        float(cell["bias_squared_debiased_trace"].mean())
                    )
                    rows["variance"].append(float(cell["variance_trace"].mean()))
                out[(regime, name, int(N))] = {
                    metric: np.asarray(values) for metric, values in rows.items()
                }
    return out


def fit_variance_slope(
    ensemble_sizes: Sequence[int], variances: Sequence[float]
) -> Tuple[float, float]:
    """OLS fit of ``log Var = alpha - beta log N``; returns ``(alpha, beta)``.

    Requires at least three ensemble sizes: a two-point "fit" has no residual
    degrees of freedom and its slope is an interpolation, not an estimate.
    """
    sizes = np.asarray(ensemble_sizes, dtype=float)
    values = np.asarray(variances, dtype=float)
    if sizes.size < 3:
        raise ValueError(
            f"A scaling exponent needs at least three ensemble sizes; got {sizes.size}."
        )
    finite = np.isfinite(values) & (values > 0.0)
    if finite.sum() < 3:
        return np.nan, np.nan
    x = np.log(sizes[finite])
    y = np.log(values[finite])
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(-slope)


def hierarchical_bootstrap_regimes(
    cells: Dict[CellKey, Dict[str, np.ndarray]],
    seeds: Sequence[int],
    settings: ExperimentSettings,
    n_bootstrap: int = 2000,
    n_inner: int = 250,
    random_seed: int = 20260807,
    stride_multiplier: int = 1,
    regimes: Sequence[str] = REGIMES,
) -> Dict[str, object]:
    """Paired hierarchical bootstrap over the whole ``(regime, filter, N)`` grid.

    Outer level
        Whole truth seeds, resampled with replacement.  Every evaluated window
        belonging to a resampled seed is kept: windows drawn from one chaotic
        trajectory are serially correlated, so they are not independent
        bootstrap units and are never resampled individually.

    Inner level
        Complete Monte Carlo realizations, resampled by multinomial counts.

    The *same* outer and inner indices are used for every regime, filter and
    ensemble size, so every comparison the summary reports -- between filters,
    between regimes, and across ensemble sizes -- is paired.
    """
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("No complete truth seeds are available.")
    key = _gram_key(stride_multiplier)
    rng = np.random.default_rng(random_seed)
    n_seeds = len(seeds)
    M = int(settings.n_mc)

    grid = [
        (regime, name, int(N))
        for regime in regimes
        for name in settings.filter_names
        for N in settings.ensemble_sizes
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
            inner[cell][seed_index] = np.column_stack((mse, bias2, debiased, variance))

    draws = {cell: np.empty((n_bootstrap, len(METRIC_NAMES))) for cell in grid}
    for b in range(n_bootstrap):
        outer = rng.integers(0, n_seeds, size=n_seeds)
        inner_index = rng.integers(0, n_inner, size=n_seeds)
        for cell in grid:
            draws[cell][b] = inner[cell][outer, inner_index].mean(axis=0)

    variance_column = METRIC_NAMES.index("variance")
    sizes = [int(N) for N in settings.ensemble_sizes]

    slopes: Dict[Tuple[str, str], np.ndarray] = {}
    intercepts: Dict[Tuple[str, str], np.ndarray] = {}
    if len(sizes) < 3:
        # No slope is reported from two ensemble sizes. A two-point "fit" has no
        # residual degrees of freedom, so its exponent is an interpolation
        # between two noisy numbers dressed up as an estimate. The grid is
        # reported as unfittable instead.
        print(
            f"Variance-slope fit skipped: {len(sizes)} ensemble size(s) requested, "
            "at least 3 are required.",
            flush=True,
        )
    else:
        for regime in regimes:
            for name in settings.filter_names:
                stack = np.column_stack(
                    [draws[(regime, name, N)][:, variance_column] for N in sizes]
                )
                fitted = np.array(
                    [fit_variance_slope(sizes, row) for row in stack]
                )  # (n_bootstrap, 2)
                intercepts[(regime, name)] = fitted[:, 0]
                slopes[(regime, name)] = fitted[:, 1]

    amplification: Dict[Tuple[str, int], np.ndarray] = {}
    if "restart" in regimes and "cycled" in regimes:
        for name in settings.filter_names:
            for N in sizes:
                restart = draws[("restart", name, N)][:, variance_column]
                cycled = draws[("cycled", name, N)][:, variance_column]
                amplification[(name, N)] = np.divide(
                    cycled,
                    restart,
                    out=np.full(n_bootstrap, np.nan),
                    where=restart > 0.0,
                )

    return {
        "metric_names": METRIC_NAMES,
        "truth_seeds": np.asarray(seeds),
        "ensemble_sizes": sizes,
        "draws": draws,
        "slopes": slopes,
        "intercepts": intercepts,
        "amplification": amplification,
        "n_bootstrap": int(n_bootstrap),
        "n_inner": int(n_inner),
        "random_seed": int(random_seed),
        "stride_multiplier": int(stride_multiplier),
    }


def percentile_interval(values, level: float = 95.0) -> Tuple[float, float, float]:
    """``(median, lower, upper)`` of a bootstrap draw vector."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    half = (100.0 - level) / 2.0
    lo, mid, hi = np.percentile(values, [half, 50.0, 100.0 - half])
    return float(mid), float(lo), float(hi)


def build_summary_table(
    bootstrap: Dict[str, object],
    settings: ExperimentSettings,
    primary_size: int = 10,
) -> List[Dict[str, object]]:
    """One row per filter: restart/cycled beta, A_cycle, and MSE at ``primary_size``."""
    rows: List[Dict[str, object]] = []
    draws = bootstrap["draws"]
    mse_column = METRIC_NAMES.index("mse")
    for name in settings.filter_names:
        row: Dict[str, object] = {"filter": name}
        for regime in ("restart", "cycled"):
            key = (regime, name)
            if key in bootstrap["slopes"]:
                mid, lo, hi = percentile_interval(bootstrap["slopes"][key])
                row[f"{regime}_beta"] = mid
                row[f"{regime}_beta_lo"] = lo
                row[f"{regime}_beta_hi"] = hi
            cell = (regime, name, int(primary_size))
            if cell in draws:
                mid, lo, hi = percentile_interval(draws[cell][:, mse_column])
                row[f"{regime}_mse_N{primary_size}"] = mid
                row[f"{regime}_mse_N{primary_size}_lo"] = lo
                row[f"{regime}_mse_N{primary_size}_hi"] = hi
        amp = bootstrap["amplification"].get((name, int(primary_size)))
        if amp is not None:
            mid, lo, hi = percentile_interval(amp)
            row[f"A_cycle_N{primary_size}"] = mid
            row[f"A_cycle_N{primary_size}_lo"] = lo
            row[f"A_cycle_N{primary_size}_hi"] = hi
        rows.append(row)
    return rows


def save_summary(
    results_dir: Path,
    bootstrap: Dict[str, object],
    settings: ExperimentSettings,
    seed_report: Dict[str, object],
    primary_size: int = 10,
) -> Tuple[Path, Path]:
    """Write the machine-readable summary: a CSV table and an NPZ of the draws.

    The NPZ carries the full bootstrap draw matrices, not just the intervals,
    so any other interval or comparison can be recomputed without rerunning the
    experiment.
    """
    import csv

    summary_dir = Path(results_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = build_summary_table(bootstrap, settings, primary_size)
    csv_path = summary_dir / "restart_vs_cycled_summary.csv"
    fieldnames = sorted({key for row in rows for key in row} - {"filter"})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filter", *fieldnames])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload: Dict[str, np.ndarray] = {
        "metric_names": np.asarray(METRIC_NAMES),
        "ensemble_sizes": np.asarray(bootstrap["ensemble_sizes"]),
        "truth_seeds": np.asarray(bootstrap["truth_seeds"]),
        "n_bootstrap": np.asarray(bootstrap["n_bootstrap"]),
        "n_inner": np.asarray(bootstrap["n_inner"]),
        "random_seed": np.asarray(bootstrap["random_seed"]),
        "stride_multiplier": np.asarray(bootstrap["stride_multiplier"]),
        "n_mc": np.asarray(settings.n_mc),
        "window_stride": np.asarray(settings.window_stride),
        "spinup_windows": np.asarray(settings.spinup_windows),
        "excluded_truth_seeds": np.asarray(
            json.dumps(
                {str(k): v for k, v in seed_report["excluded"].items()}, sort_keys=True
            )
        ),
        "summary_rows": np.asarray(json.dumps(rows, sort_keys=True)),
    }
    for (regime, name, N), values in bootstrap["draws"].items():
        payload[f"draws__{regime}__{_slug(name)}__N{N}"] = values
    for (regime, name), values in bootstrap["slopes"].items():
        payload[f"beta__{regime}__{_slug(name)}"] = values
    for (name, N), values in bootstrap["amplification"].items():
        payload[f"amplification__{_slug(name)}__N{N}"] = values

    npz_path = summary_dir / "restart_vs_cycled_bootstrap.npz"
    np.savez_compressed(npz_path, **payload)
    return csv_path, npz_path


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_summary_table(
    bootstrap: Dict[str, object],
    settings: ExperimentSettings,
    primary_size: int = 10,
) -> None:
    rows = build_summary_table(bootstrap, settings, primary_size)
    header = (
        f"{'Filter':<12}{'Restart beta':>24}{'Cycled beta':>24}"
        f"{f'A_cycle({primary_size})':>22}"
        f"{f'Restart MSE (N={primary_size})':>28}{f'Cycled MSE (N={primary_size})':>28}"
    )
    print("\n" + header)
    print("-" * len(header))

    def fmt(row, prefix, digits=2):
        mid = row.get(prefix)
        if mid is None or not np.isfinite(mid):
            return "n/a"
        lo, hi = row.get(f"{prefix}_lo"), row.get(f"{prefix}_hi")
        return f"{mid:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

    for row in rows:
        print(
            f"{row['filter']:<12}"
            f"{fmt(row, 'restart_beta', 3):>24}"
            f"{fmt(row, 'cycled_beta', 3):>24}"
            f"{fmt(row, f'A_cycle_N{primary_size}', 3):>22}"
            f"{fmt(row, f'restart_mse_N{primary_size}', 3):>28}"
            f"{fmt(row, f'cycled_mse_N{primary_size}', 3):>28}"
        )
    print(
        "\nIntervals are 95% paired hierarchical-bootstrap percentile intervals "
        f"({bootstrap['n_bootstrap']} outer draws over "
        f"{len(bootstrap['truth_seeds'])} truth seeds).\n"
        "They are narrow because each truth seed already averages "
        "150 evaluated windows x 100\nrealizations, so the across-seed spread of "
        "these means is small. They quantify the\nsampling uncertainty OF THE "
        "ESTIMATE, not the physical spread across truth\ntrajectories, and a tight "
        "interval is not a claim that the underlying quantity is\nuniform over "
        "trajectories."
    )


def dry_run(config: Dict, settings: ExperimentSettings, n_jobs: int) -> int:
    """Time one probe cell per (regime, filter) and project the total cost."""
    probe_windows = min(20, settings.n_windows)
    probe = ExperimentSettings(
        **{
            **{
                field_name: getattr(settings, field_name)
                for field_name in ExperimentSettings.__dataclass_fields__
            },
            "n_windows": probe_windows,
            "n_evaluation_seeds": 1,
            "evaluation_seeds": settings.evaluation_seeds[:1],
            "calibration_seeds": settings.calibration_seeds,
        }
    )
    seed = int(settings.evaluation_seeds[0])
    evaluation_config = _config_for_windows(config, probe.total_cycled_windows)
    truth, observations, H = generate_truth_batch(evaluation_config, [seed])[seed]
    context = build_window_context(evaluation_config, settings.kappa)
    distribution = bootstrap_restart_distribution(int(config["model"]["n"]))

    print(
        f"Probe: {probe_windows} evaluated windows, M={settings.n_mc}, "
        f"{probe.total_cycled_windows} windows per cycled cell\n"
    )
    total = 0.0
    for regime in REGIMES:
        for name in settings.filter_names:
            for N in settings.ensemble_sizes:
                probe_args = (
                    name,
                    truth,
                    observations,
                    H,
                    context,
                    distribution,
                    seed,
                    N,
                    settings.n_mc,
                    [int(w) for w in probe.evaluation_windows],
                )
                t0 = time.perf_counter()
                if regime == "restart":
                    run_restart_cell(
                        *probe_args, mean_error_mode=settings.mean_error_mode
                    )
                else:
                    run_cycled_cell(*probe_args)
                measured = time.perf_counter() - t0
                # Cost is linear in the number of windows integrated, which for
                # the cycled regime includes the strided-out ones.
                scale = (
                    settings.n_windows / probe_windows
                    if regime == "restart"
                    else settings.total_cycled_windows / probe.total_cycled_windows
                )
                projected = measured * scale * len(settings.evaluation_seeds)
                total += projected
                print(
                    f"  {regime:<7} {name:<11} N={N:<3} {measured:6.2f}s probe -> "
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
    n_mc=4,
    n_windows=6,
    window_stride=2,
    spinup_windows=3,
    n_calibration_seeds=2,
    n_evaluation_seeds=2,
    calibration_n_mc=4,
    calibration_n_windows=12,
    calibration_spinup_windows=3,
    label="smoke",
)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independent-restart versus cycled-ensemble bias-variance experiment."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--restart-vs-cycled",
        action="store_true",
        help="Run the experiment (explicit opt-in, matching the notebook's guard).",
    )
    parser.add_argument(
        "--regime",
        choices=("restart", "cycled", "both"),
        default="both",
        help="Which regime(s) to run. Default: both.",
    )
    parser.add_argument("--ensemble-sizes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--n-truth-seeds", type=int, default=None, help="Number of EVALUATION truth seeds."
    )
    parser.add_argument("--n-calibration-seeds", type=int, default=None)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit evaluation truth seeds (overrides --n-truth-seeds).",
    )
    parser.add_argument(
        "--calibration-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit calibration truth seeds (must be disjoint from evaluation).",
    )
    parser.add_argument("--n-mc", type=int, default=None, help="Inner realizations M.")
    parser.add_argument(
        "--n-windows", type=int, default=None, help="Number of EVALUATED windows."
    )
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--spinup-windows", type=int, default=None)
    parser.add_argument("--kappa", type=int, default=None, help="QPCA-EnDCF retained rank.")
    parser.add_argument(
        "--restart-mode", choices=RESTART_DISTRIBUTION_MODES, default=None
    )
    parser.add_argument(
        "--mean-error-mode",
        choices=MEAN_ERROR_MODES,
        default=None,
        help="How the shared forecast-mean error b is drawn. 'per_realization' "
        "(default) is the primary design and keeps forecast-mean uncertainty, at "
        "the cost of an N-independent variance floor. 'per_window' shares one b "
        "across all realizations at a window, which is the configuration in "
        "which the N^-1 scaling question is well posed.",
    )
    parser.add_argument("--eigenvalue-floor", type=float, default=None)
    parser.add_argument("--shrinkage", type=float, default=None)
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
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-inner", type=int, default=250)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)

    overrides = dict(
        ensemble_sizes=tuple(args.ensemble_sizes) if args.ensemble_sizes else None,
        n_mc=args.n_mc,
        n_windows=args.n_windows,
        window_stride=args.window_stride,
        spinup_windows=args.spinup_windows,
        n_evaluation_seeds=args.n_truth_seeds,
        n_calibration_seeds=args.n_calibration_seeds,
        evaluation_seeds=tuple(args.evaluation_seeds) if args.evaluation_seeds else None,
        calibration_seeds=tuple(args.calibration_seeds) if args.calibration_seeds else None,
        kappa=args.kappa,
        restart_mode=args.restart_mode,
        mean_error_mode=args.mean_error_mode,
        eigenvalue_floor=args.eigenvalue_floor,
        shrinkage=args.shrinkage,
        label=args.label,
    )
    if args.smoke_test:
        for key, value in SMOKE_TEST_SETTINGS.items():
            overrides.setdefault(key, None)
            if overrides.get(key) is None:
                overrides[key] = value

    settings = settings_from_config(config, **overrides)

    if args.dry_run:
        return dry_run(config, settings, args.n_jobs)

    if not args.restart_vs_cycled:
        print(
            "Nothing to do. Pass --restart-vs-cycled to run the experiment, or "
            "--dry-run to project its cost.\n\nFor example:\n"
            "  python -m src.experiments.restart_vs_cycled --restart-vs-cycled "
            "--smoke-test"
        )
        return 0

    regimes = REGIMES if args.regime == "both" else (args.regime,)
    if args.smoke_test:
        print("*** SMOKE TEST -- results are NOT a scientific result ***", flush=True)

    outcome = run_restart_vs_cycled(
        config,
        settings,
        n_jobs=args.n_jobs,
        results_root=args.results_dir,
        resume=args.resume,
        verbose=args.verbose,
        regimes=regimes,
    )

    seed_report = complete_truth_seeds(outcome["cells"], settings, regimes)
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

    if not seed_report["common"] or len(regimes) < 2:
        print("\nNot enough complete cells for the paired summary; stopping here.")
        return 0

    bootstrap = hierarchical_bootstrap_regimes(
        outcome["cells"],
        seed_report["common"],
        settings,
        n_bootstrap=args.n_bootstrap,
        n_inner=args.n_inner,
        regimes=regimes,
    )
    primary_size = 10 if 10 in settings.ensemble_sizes else settings.ensemble_sizes[0]
    print_summary_table(bootstrap, settings, primary_size)
    csv_path, npz_path = save_summary(
        outcome["results_dir"], bootstrap, settings, seed_report, primary_size
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
