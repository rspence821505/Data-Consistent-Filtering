"""
Sequential stochastic ensemble Kalman filter.
"""

import numpy as np
from numpy.linalg import cholesky
from .base_filter import BaseFilter
from ..utils.linalg import cov_and_anoms, sym_posdef_inverse


class StochasticEnKF(BaseFilter):
    """
    Regular stochastic EnKF with perturbed observations.
    One update per observation time (sequential EnKF).
    Particles are columns: X in R^{n x N}.
    """

    def __init__(self, H, R, stabilize=True):
        """
        Initialize sequential stochastic EnKF.

        Parameters
        ----------
        H : ndarray, shape (m, n)
            Observation operator.
        R : ndarray, shape (m, m)
            Observation error covariance.
        stabilize : bool, optional
            Whether to stabilize innovation covariance (default: True).
        """
        super().__init__(H, R)
        self.stabilize = stabilize
        self.R_chol = cholesky(R)

    def update(self, X, z, rng):
        """
        Perform analysis update with perturbed observations.

        Parameters
        ----------
        X : ndarray, shape (n, N)
            Current ensemble (columns = particles).
        z : ndarray, shape (m,)
            Observation at this assimilation time.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        X_a : ndarray, shape (n, N)
            Updated (analysis) ensemble.
        """
        H, R = self.H, self.R
        n, N = X.shape

        # Ensemble stats
        P, _ = cov_and_anoms(X)  # (n x n)
        HX = H @ X  # (m x N)
        Pyy = (
            (HX - HX.mean(axis=1, keepdims=True))
            @ (HX - HX.mean(axis=1, keepdims=True)).T
            / (N - 1)
        )
        S = Pyy + R  # EnKF innovation covariance

        S_inv = sym_posdef_inverse(S)
        K = P @ H.T @ S_inv  # (n x m)

        # Perturb observations
        Eps = self.R_chol @ rng.standard_normal(size=(R.shape[0], N))
        Y = z[:, None] + Eps  # (m x N)

        # Update
        X_a = X + K @ (Y - HX)
        return X_a
