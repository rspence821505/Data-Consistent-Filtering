"""
Metrics for data assimilation performance evaluation.
"""

import numpy as np
from src.utils.linalg import sym_posdef_inverse


def compute_rmse(x_mean, x_true):
    """
    Root mean square error between estimate and truth.

    Parameters
    ----------
    x_mean : ndarray, shape (n,)
        Estimated state (e.g., ensemble mean).
    x_true : ndarray, shape (n,)
        True state.

    Returns
    -------
    rmse : float
        Root mean square error.
    """
    if x_mean.ndim == 1 or x_true.ndim == 1:
        return np.sqrt(np.mean((x_mean - x_true) ** 2))
    return np.sqrt(np.mean((x_mean - x_true) ** 2, axis=1))


def rmse(x_mean, x_true):
    """
    Backwards-compatible wrapper for legacy imports.
    """
    return compute_rmse(x_mean, x_true)


def rel_data_misfit(z, Hx_mean, R_inv_sqrt):
    """
    Relative data misfit in whitened metric:
        || R^{-1/2} (z - H x_mean) ||_2 / || R^{-1/2} z ||_2

    Parameters
    ----------
    z : ndarray, shape (m,)
        Observation vector.
    Hx_mean : ndarray, shape (m,)
        Predicted observation H @ x_mean.
    R_inv_sqrt : ndarray, shape (m, m)
        Square root of inverse observation error covariance.

    Returns
    -------
    misfit : float
        Relative data misfit (0 = perfect fit).
    """
    num = np.linalg.norm(R_inv_sqrt @ (z - Hx_mean))
    den = np.linalg.norm(R_inv_sqrt @ z)
    return num / (den + 1e-12)


def compute_relative_misfit(estimates, observations, H, R):
    R_inv = sym_posdef_inverse(R)
    R_inv_sqrt = np.linalg.cholesky(R_inv)
    misfits = []
    for x_mean, z in zip(estimates, observations):
        residual = z - H @ x_mean
        num = np.linalg.norm(R_inv_sqrt @ residual)
        den = np.linalg.norm(R_inv_sqrt @ z) + 1e-12
        misfits.append(num / den)
    return np.array(misfits)


def compute_ensemble_spread(ensemble: np.ndarray) -> np.ndarray:
    """
    Compute ensemble spread (square root of trace of ensemble covariance).

    Parameters
    ----------
    ensemble : ndarray, shape (n_windows, n_states, n_ensemble)
        Full ensemble trajectory

    Returns
    -------
    spread : ndarray, shape (n_windows,)
        Total ensemble spread at each window
    """
    # check dimensions
    if ensemble.ndim != 3:
        n_windows = 1
        n_states, n_ensemble = ensemble.shape
        ensemble = ensemble.reshape((n_windows, n_states, n_ensemble))
    else:
        n_windows, n_states, n_ensemble = ensemble.shape

    spread = np.zeros(n_windows)

    for w in range(n_windows):
        # Compute ensemble variance for each state
        variances = np.var(ensemble[w, :, :], axis=1, ddof=1)
        # Total spread is sqrt of sum of variances (sqrt of trace of covariance)
        spread[w] = np.sqrt(np.sum(variances)) / np.sqrt(n_states)

    return spread


def compute_rank_histogram(
    truth: np.ndarray, ensemble: np.ndarray, n_bins: int = 11
) -> np.ndarray:
    """
    Compute rank histogram for ensemble calibration assessment.

    For each state and time, determine the rank of the truth value
    among the ensemble members. A flat histogram indicates good calibration.

    Parameters
    ----------
    truth : ndarray, shape (n_windows, n_states)
        True state trajectory
    ensemble : ndarray, shape (n_windows, n_states, n_ensemble)
        Full ensemble trajectory
    n_bins : int
        Number of bins (should be n_ensemble + 1)

    Returns
    -------
    histogram : ndarray, shape (n_bins,)
        Rank histogram counts
    """
    n_windows, n_states, n_ensemble = ensemble.shape

    # Should have n_ensemble + 1 bins
    if n_bins != n_ensemble + 1:
        print(
            f"Warning: n_bins={n_bins} but should be {n_ensemble + 1} for {n_ensemble} ensemble members"
        )

    ranks = []

    for w in range(n_windows):
        for s in range(n_states):
            # Get truth and ensemble members for this state/time
            truth_val = truth[w, s]
            ens_vals = ensemble[w, s, :]

            # Count how many ensemble members are less than truth
            rank = np.sum(ens_vals < truth_val)
            ranks.append(rank)

    # Create histogram
    histogram, _ = np.histogram(ranks, bins=np.arange(n_bins + 1))

    return histogram


def compute_spread_skill_ratio(spread: np.ndarray, rmse: np.ndarray) -> float:
    """
    Compute mean spread-skill ratio.

    A ratio close to 1 indicates good calibration.
    Ratio < 1: underdispersed (spread too small)
    Ratio > 1: overdispersed (spread too large)

    Parameters
    ----------
    spread : ndarray
        Ensemble spread time series
    rmse : ndarray
        RMSE time series

    Returns
    -------
    ratio : float
        Mean spread / RMSE ratio
    """
    return np.mean(spread / rmse)


def compute_correlation(spread: np.ndarray, rmse: np.ndarray) -> float:
    """
    Compute Pearson correlation between spread and RMSE.

    High correlation indicates that ensemble spread tracks actual errors well.

    Parameters
    ----------
    spread : ndarray
        Ensemble spread time series
    rmse : ndarray
        RMSE time series

    Returns
    -------
    correlation : float
        Pearson correlation coefficient
    """
    return np.corrcoef(spread, rmse)[0, 1]
