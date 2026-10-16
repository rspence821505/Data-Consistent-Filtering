"""
Sequential localized ensemble transform Kalman filter (Seq-LETKF).

Implements Algorithm 3 of the paper: a deterministic square-root update carried
out region by region in the N-dimensional ensemble space, with localization
restricting which observations enter each region's transform.  Unlike the
stochastic EnKF baselines there are no perturbed observations, so the update
contributes no observation-perturbation variance.
"""

import numpy as np

from .base_filter import BaseFilter


def _ensemble_transform(A_z, innovation, weights, r_inv_diag, N):
    """Local ensemble-space transforms for every analysis region at once.

    Region ``q`` of Algorithm 3 is a single state index, so the regions differ
    only in their taper row.  Every region therefore solves an ``N x N`` system
    of the same shape, and the whole set is evaluated as one batched
    eigendecomposition rather than a Python loop over the state dimension.

    Parameters
    ----------
    A_z : ndarray, shape (d, N)
        Predicted-observation anomalies.
    innovation : ndarray, shape (d,)
        ``z - mean(H X)``.
    weights : ndarray, shape (n_regions, d)
        Localization taper for each region.
    r_inv_diag : ndarray, shape (d,)
        Diagonal of ``R^{-1}``.
    N : int
        Ensemble size.

    Returns
    -------
    W : ndarray, shape (n_regions, N, N)
        ``W_q = wbar_q 1^T + [(N - 1) Ptilde_q]^{1/2}``.
    """
    # Localized observation precision, one row per region: diag(rho_q) R^{-1}.
    precision = weights * r_inv_diag[None, :]

    # C_q = A_z^T diag(rho_q) R^{-1} A_z  and  b_q = A_z^T diag(rho_q) R^{-1} d
    C = np.einsum("dn,qd,dm->qnm", A_z, precision, A_z, optimize=True)
    b = np.einsum("dn,qd,d->qn", A_z, precision, innovation, optimize=True)

    # One symmetric eigendecomposition supplies both the analysis covariance and
    # its square root: with (N-1)I + C = V diag(mu) V^T,
    #   Ptilde          = V diag(1 / mu) V^T
    #   [(N-1)Ptilde]^½ = V diag(sqrt((N-1) / mu)) V^T.
    target = C + (N - 1) * np.eye(N)[None, :, :]
    target = 0.5 * (target + np.transpose(target, (0, 2, 1)))
    mu, V = np.linalg.eigh(target)
    mu = np.maximum(mu, 1e-12)

    Vt = np.transpose(V, (0, 2, 1))
    P_tilde = np.einsum("qnk,qk,qkm->qnm", V, 1.0 / mu, Vt, optimize=True)
    W_tilde = np.einsum(
        "qnk,qk,qkm->qnm", V, np.sqrt((N - 1) / mu), Vt, optimize=True
    )

    w_bar = np.einsum("qnm,qm->qn", P_tilde, b, optimize=True)
    return w_bar[:, :, None] + W_tilde


class SequentialLETKF(BaseFilter):
    """Sequential localized ensemble transform Kalman filter (Algorithm 3).

    Parameters
    ----------
    H : ndarray, shape (m, n)
        Observation operator.
    R : ndarray, shape (m, m)
        Observation error covariance.  Must be diagonal: localization tapers the
        observation precision entrywise, which is only a reweighting when the
        observation errors are uncorrelated.
    localization : ndarray, shape (n, m), optional
        Taper weights from ``build_localization_weights``.  ``None`` gives the
        unlocalized ETKF.
    inflation : float, optional
        Multiplicative inflation applied about the analysis mean (default 1.0,
        i.e. none).
    """

    def __init__(self, H, R, localization=None, inflation=1.0):
        super().__init__(H, R)

        R = np.asarray(R, dtype=float)
        off_diagonal = R - np.diag(np.diag(R))
        if np.abs(off_diagonal).max() > 1e-12:
            raise ValueError(
                "SequentialLETKF requires a diagonal observation covariance; "
                "localization tapers R^{-1} entrywise."
            )
        self.r_inv_diag = 1.0 / np.diag(R)
        self.localization = localization
        self.inflation = float(inflation)

    def update(self, X, z):
        """Deterministic localized analysis at one observation time.

        Parameters
        ----------
        X : ndarray, shape (n, N)
            Forecast ensemble (columns are members).
        z : ndarray, shape (m,)
            Observation at this time.

        Returns
        -------
        X_a : ndarray, shape (n, N)
            Analysis ensemble.
        """
        H = self.H
        n, N = X.shape

        x_mean = X.mean(axis=1)
        A_x = X - x_mean[:, None]

        Z = H @ X
        z_mean = Z.mean(axis=1)
        A_z = Z - z_mean[:, None]
        innovation = z - z_mean

        weights = (
            self.localization
            if self.localization is not None
            else np.ones((n, A_z.shape[0]))
        )
        W = _ensemble_transform(A_z, innovation, weights, self.r_inv_diag, N)

        # Region q is state index q, so only row q of the state anomalies enters:
        # X_q^a = xbar_q 1^T + A_{x,q} W_q.
        X_a = x_mean[:, None] + np.einsum("qn,qnm->qm", A_x, W, optimize=True)

        if self.inflation != 1.0:
            a_mean = X_a.mean(axis=1, keepdims=True)
            X_a = a_mean + self.inflation * (X_a - a_mean)

        return X_a
