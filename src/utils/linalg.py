"""
Linear algebra utilities for data assimilation.
"""

import numpy as np
from numpy.linalg import eigh, cholesky, solve


def sym_posdef_inverse(A, jitter=1e-6):
    """
    Compute inverse of symmetric positive definite matrix via Cholesky.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive definite matrix.
    jitter : float, optional
        Regularization parameter to ensure positive definiteness (default: 1e-6).

    Returns
    -------
    A_inv : ndarray, shape (n, n)
        Inverse of A.
    """
    # Symmetrize and add regularization to ensure positive definiteness
    A_sym = 0.5 * (A + A.T) + jitter * np.eye(A.shape[0])
    try:
        L = cholesky(A_sym)
        return solve(L.T, solve(L, np.eye(A.shape[0])))
    except np.linalg.LinAlgError:
        # If Cholesky still fails, use eigenvalue decomposition with stronger regularization
        w, V = eigh(A_sym)
        # Adaptively set jitter based on matrix scale
        scale = np.abs(w).max()
        adaptive_jitter = max(jitter, 1e-4 * scale, 1e-8)
        # Threshold eigenvalues to ensure positive definiteness
        w_reg = np.maximum(w, adaptive_jitter)
        # Compute inverse directly from eigendecomposition: A^{-1} = V @ diag(1/w) @ V.T
        return V @ np.diag(1.0 / w_reg) @ V.T


def stabilize_spd_like(S, R, eps=1e-6):
    """
    Raise S's spectrum so that lambda_min(S) >= 1% of lambda_max(R).

    The floor is one percent of the largest eigenvalue of R, not lambda_max(R)
    itself -- a hundred times weaker than the name suggests, and load-bearing at
    that value: it is the floor QPCA-EnDCF's published Lorenz-96 results were
    produced with, and the one its localized branch matches when it decides
    which directions of P_yy to retain. Changing it changes those results.

    Useful in the 4D stacked setting as well.

    Parameters
    ----------
    S : ndarray, shape (n, n)
        Covariance matrix to stabilize.
    R : ndarray, shape (m, m)
        Reference covariance (observation error).
    eps : float, optional
        Small positive constant for numerical stability (default: 1e-6).

    Returns
    -------
    S_stabilized : ndarray, shape (n, n)
        Stabilized covariance matrix.
    """
    wS, _ = eigh(S)
    wR, _ = eigh(R)
    # Ensure minimum eigenvalue is positive and larger than max of R
    min_eig = max(eps, wR.max() * 1e-2)  # At least 1% of max(R) eigenvalue
    if wS.min() < min_eig:
        delta = (min_eig - wS.min()) + eps
        S = S + delta * np.eye(S.shape[0])
    return S


def block_diag_repeat(A, k):
    """
    Return block diagonal diag(A, A, ..., A) with k blocks.

    Parameters
    ----------
    A : ndarray, shape (m, m)
        Matrix to repeat on diagonal.
    k : int
        Number of blocks.

    Returns
    -------
    B : ndarray, shape (k*m, k*m)
        Block diagonal matrix.
    """
    return np.kron(np.eye(k), A)


def cov_and_anoms(X):
    """
    Compute ensemble covariance and centered anomalies.

    Parameters
    ----------
    X : ndarray, shape (d, N)
        Ensemble matrix with columns = particles.

    Returns
    -------
    C : ndarray, shape (d, d)
        Ensemble covariance matrix.
    Xc : ndarray, shape (d, N)
        Column-centered anomalies.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    C = (Xc @ Xc.T) / (X.shape[1] - 1)
    return C, Xc


def spd_inverse(A, jitter=1e-10):
    """
    Convenience wrapper around ``sym_posdef_inverse`` with optional jitter.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    return sym_posdef_inverse(0.5 * (A + A.T) + jitter * np.eye(n))


def cholesky_whitening(R, jitter=1e-8):
    """
    Compute Cholesky square root and inverse square root of SPD matrix R.

    Returns
    -------
    R_sqrt : ndarray
        Lower-triangular factor L such that L L^T = R.
    R_inv_sqrt : ndarray
        Matrix W such that W^T W = R^{-1}.
    """
    R = np.asarray(R, dtype=float)
    n = R.shape[0]
    L = cholesky(0.5 * (R + R.T) + jitter * np.eye(n))
    L_inv = solve(L, np.eye(n))
    R_inv_sqrt = L_inv.T
    return L, R_inv_sqrt
