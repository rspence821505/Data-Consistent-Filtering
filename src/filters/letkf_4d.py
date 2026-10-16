"""
Four-dimensional localized ensemble transform Kalman filter (4D-LETKF).

Implements Algorithm 4 of the paper.  The window's L observation times are
stacked into a single observation vector, as in the 4D stochastic EnKF, but the
analysis is a deterministic localized square-root transform in ensemble space
targeting the window-endpoint state.
"""

import numpy as np

from .base_filter import BaseFilter
from .seq_letkf import _ensemble_transform


class LETKF4D(BaseFilter):
    """Four-dimensional localized ensemble transform Kalman filter (Algorithm 4).

    Parameters
    ----------
    H : ndarray, shape (m, n)
        Observation operator.
    R : ndarray, shape (m, m)
        Per-time observation error covariance.  Must be diagonal; the stacked
        covariance is ``R_L = I_L kron R``.
    window_len : int
        Number of observation times stacked per assimilation window.
    localization : ndarray, shape (n, m * window_len), optional
        Taper weights over the *stacked* observation vector, as returned by
        ``build_localization_weights(..., window_len=L)``.  ``None`` gives the
        unlocalized 4D ETKF.
    inflation : float, optional
        Multiplicative inflation applied about the analysis mean (default 1.0).
    """

    def __init__(self, H, R, window_len, localization=None, inflation=1.0):
        super().__init__(H, R)

        R = np.asarray(R, dtype=float)
        off_diagonal = R - np.diag(np.diag(R))
        if np.abs(off_diagonal).max() > 1e-12:
            raise ValueError(
                "LETKF4D requires a diagonal observation covariance; "
                "localization tapers R_L^{-1} entrywise."
            )
        self.window_len = int(window_len)
        self.m = R.shape[0]
        # R_L = I_L kron R is diagonal, so its inverse diagonal is R^{-1} tiled.
        self.r_inv_diag = np.tile(1.0 / np.diag(R), self.window_len)
        self.localization = localization
        self.inflation = float(inflation)

    def update(self, X_path, z_stack):
        """Deterministic localized analysis at a window endpoint.

        Parameters
        ----------
        X_path : sequence of ndarray
            Length-``L`` sequence of ensembles ``(n, N)`` at the window's
            observation times, ordered in time.
        z_stack : ndarray, shape (m * L,)
            Stacked observations for the window.

        Returns
        -------
        X_a_end : ndarray, shape (n, N)
            Analysis ensemble at the window endpoint.
        """
        H = self.H
        L = self.window_len
        if len(X_path) != L:
            raise ValueError(
                f"X_path holds {len(X_path)} times but window_len is {L}."
            )

        # Stacked predicted observations across the window.
        Z_stack = np.vstack([H @ X_t for X_t in X_path])  # (mL, N)
        z_mean = Z_stack.mean(axis=1)
        A_z = Z_stack - z_mean[:, None]
        innovation = z_stack - z_mean

        # The analysis targets the window-endpoint state.
        X_end = X_path[-1]
        n, N = X_end.shape
        x_mean = X_end.mean(axis=1)
        A_x = X_end - x_mean[:, None]

        weights = (
            self.localization
            if self.localization is not None
            else np.ones((n, A_z.shape[0]))
        )
        W = _ensemble_transform(A_z, innovation, weights, self.r_inv_diag, N)

        X_a = x_mean[:, None] + np.einsum("qn,qnm->qm", A_x, W, optimize=True)

        if self.inflation != 1.0:
            a_mean = X_a.mean(axis=1, keepdims=True)
            X_a = a_mean + self.inflation * (X_a - a_mean)

        return X_a
