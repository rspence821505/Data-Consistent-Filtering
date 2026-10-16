"""
QPCA Ensemble Data Consistency Filter (4D variant).
"""

import numpy as np
from numpy.linalg import eigh, cholesky
from .base_filter import BaseFilter
from ..utils.linalg import (
    cov_and_anoms,
    sym_posdef_inverse,
    stabilize_spd_like,
    block_diag_repeat,
)


class QPCAEnDCF(BaseFilter):
    """
      QPCA–EnDCF consistent with seq_mud:
    - Build stacked, per-particle residuals E_stack = R_block^{-1/2} (HX_stack - z_stack 1^T)
    - PCA in data (stacked) coordinates on centered whitened residuals
    - Project residuals onto top-k eigenvectors
    - De-whiten projected residuals and apply 4D gain K = P_xy S^{-1}
      with S = P_yy (optionally stabilized vs R_block)
    """

    def __init__(self, H, R, window_len, k=1, stabilize=True, localization=None):
        """
        Initialize QPCA-EnDCF 4D filter.

        Parameters
        ----------
        H : ndarray, shape (m, n)
            Observation operator.
        R : ndarray, shape (m, m)
            Observation error covariance.
        window_len : int
            Number of observation times per assimilation window.
        k : int, optional
            Number of PCA modes to retain (default: 1).
        stabilize : bool, optional
            Whether to stabilize innovation covariance (default: True).
        localization : ndarray, shape (n, m * window_len), optional
            Gaspari-Cohn taper applied to the ensemble state--observation
            cross-covariance ``P_xy``, as in the paper's inlet study.  ``None``
            (the default) is the filter as specified for the Lorenz-96
            experiments: regularizing the whitened observation mismatch rather
            than imposing a spatial structure is the point of the method, so
            supplying a taper is a deliberate departure used to separate
            localization from residual-space truncation.

            Note that the taper also changes how ``S`` is inverted; see
            ``update``.
        """
        super().__init__(H, R)
        self.window_len = window_len
        self.k = k
        self.stabilize = stabilize
        self.localization = localization
        self.m = R.shape[0]

        # Block structures
        self.R_block = block_diag_repeat(R, window_len)  # (mL x mL)
        # Add small jitter to ensure positive definiteness
        jitter = 1e-10
        R_block_reg = self.R_block + jitter * np.eye(self.R_block.shape[0])
        self.R_block_chol = cholesky(R_block_reg)  # R_block^{1/2} = L
        # Compute R_block^{-1/2} = (L^{-1})^T directly from Cholesky factor
        # This is more numerically stable than computing inverse then Cholesky
        L_inv = np.linalg.solve(self.R_block_chol, np.eye(self.R_block_chol.shape[0]))
        self.R_block_inv_sqrt = L_inv.T  # (R_block^{-1/2})^T

    def update(self, X_path, z_stack, return_diagnostics=False):
        """
        Perform 4D analysis update with whitened PCA filtering.

        Parameters
        ----------
        X_path : list of ndarray
            List of length L with state ensembles at each obs time in the window.
            Each element is X_t with shape (n, N).
        z_stack : ndarray, shape (mL,)
            Stacked observation vector at those times.
        return_diagnostics : bool, optional
            Also return the intermediate quantities this update already builds.
            Off by default, and off is the path every experiment takes: the flag
            adds no arithmetic, only a dict of references to arrays that were
            computed either way, so the analysis is bit-identical with it on or
            off (pinned by a test in the original development repository). It exists so that a
            figure or diagnostic can show *what the filter actually did* rather
            than a re-derivation that could drift from this code.

        Returns
        -------
        X_a_end : ndarray, shape (n, N)
            Updated ensemble at window end.
        diagnostics : dict, optional
            Returned only when ``return_diagnostics`` is True, as
            ``(X_a_end, diagnostics)``. Keys, in the notation of the paper's
            Section 3.4:

            ``D``           :math:`\\mathbf{D}_w=\\mathbf{Z}^f_w-\\mathbf{z}_w\\mathbf{1}^\\top`, (d, N)
            ``E``           :math:`\\mathbf{E}_w=\\mathbf{R}_L^{-1/2}\\mathbf{D}_w`, (d, N)
            ``e_bar``       :math:`\\bar{\\mathbf{e}}_w`, the *uncentered* residual mean, (d,)
            ``E_c``         :math:`\\mathbf{E}_{c,w}=\\mathbf{E}_w-\\bar{\\mathbf{e}}_w\\mathbf{1}^\\top`, (d, N)
            ``Sigma_E``     :math:`\\widehat{\\boldsymbol\\Sigma}_{E,w}`, (d, d)
            ``eigvals``     all eigenvalues of ``Sigma_E``, descending, (d,)
            ``eigvecs``     matching eigenvectors as columns, descending, (d, d)
            ``V_kappa``     :math:`\\widehat{\\mathbf{V}}_\\kappa`, (d, kappa)
            ``Q_pca``       :math:`\\mathbf{Q}_{\\mathrm{PCA}}=\\widehat{\\mathbf{V}}_\\kappa^\\top\\mathbf{E}_w`, (kappa, N)
            ``Delta_white`` :math:`\\boldsymbol\\Delta_{\\mathrm{white}}=-\\widehat{\\boldsymbol\\Pi}_\\kappa\\mathbf{E}_w`, (d, N)
            ``Delta_obs``   :math:`\\boldsymbol\\Delta_{\\mathrm{obs}}=\\mathbf{R}_L^{1/2}\\boldsymbol\\Delta_{\\mathrm{white}}`, (d, N)
            ``Delta_x``     :math:`\\boldsymbol\\Delta_x=\\mathbf{K}^{\\mathrm{DC}}_w\\boldsymbol\\Delta_{\\mathrm{obs}}`, (n, N)
            ``K``           the empirical gain actually used, (n, d)
            ``X_f``         the forecast ensemble at the window end, (n, N)

            Note the eigenpairs are re-ordered to descending here, while the
            update itself slices ``eigh``'s ascending output; the retained
            subspace is the same either way and is returned as ``V_kappa``.
        """
        H = self.H
        L = self.window_len
        n, N = X_path[-1].shape
        m = self.m

        # Build stacked H X for each time in the window
        HX_blocks = [H @ X_path[t] for t in range(L)]  # each (m x N)
        HX_stack = np.vstack(HX_blocks)  # (mL x N)

        # Whitened residuals in stacked data space
        E = self.R_block_inv_sqrt.T @ (HX_stack - z_stack[:, None])  # (mL x N)
        Ec = E - E.mean(axis=1, keepdims=True)
        C = (Ec @ Ec.T) / (N - 1)  # (mL x mL)

        # Top-k eigenvectors (largest eigenvalues)
        w, V = eigh(C)  # ascending
        Vk = V[:, -self.k :]  # (mL x k)

        # QPCA map (whitened space), WME sign
        Q_qpca = -Vk @ (Vk.T @ E)  # (mL x N)

        # Ensemble anomalies at window end and in stacked obs space
        _, A_x_end = cov_and_anoms(X_path[-1])  # (n x N)
        _, A_y = cov_and_anoms(HX_stack)  # (mL x N)

        # Cross- and auto-covariances
        P_xy = (A_x_end @ A_y.T) / (N - 1)  # (n x mL)
        P_yy = (A_y @ A_y.T) / (N - 1)  # (mL x mL)

        if self.localization is None:
            # Innovation covariance for EnDCF-style gain
            S = P_yy
            if self.stabilize:
                S = stabilize_spd_like(S, self.R_block)
            S_inv = sym_posdef_inverse(S)

            # Gain and de-whitened correction
            K = P_xy @ S_inv  # (n x mL)
        else:
            # Localized gain: taper P_xy, then invert P_yy on its range only.
            #
            # Untapered, P_xy = A_x A_y^T/(N-1) has its row space inside the
            # ensemble span, so whatever the stabilized inverse does in the
            # (mL - N + 1)-dimensional null space of P_yy is annihilated.
            # Tapering entrywise destroys that: measured over 200 draws, a
            # median 58% of the tapered cross-covariance's norm lands outside
            # that span (as 1 - ||rho*P_xy P||/||rho*P_xy||, P the projector
            # onto range(A_y)), where the stabilization floor would amplify it.
            #
            # In practice that amplification is NOT what decides the answer
            # here, and it would be wrong to claim otherwise: QPCA's correction
            # corr_obs is itself built from the residual spectrum and lies
            # almost entirely inside range(P_yy), so the stabilized inverse and
            # the pseudo-inverse give increments differing by only ~1.5e-4
            # relative (median over 200 draws; max 1.9e-4). The pseudo-inverse
            # is used because it does not *depend* on that coincidence -- it is
            # well posed for any taper and any correction -- not because the
            # stabilized inverse visibly fails.
            #
            # With P_yy = U diag(s^2/(N-1)) U^T from the thin SVD of A_y,
            # P_yy^+ = U diag((N-1)/s^2) U^T over the retained directions.
            # Directions are retained only above the same floor
            # stabilize_spd_like applies (1% of the largest R_block eigenvalue),
            # so a collapsed ensemble yields a zero gain rather than dividing a
            # vanishing cross-covariance by a vanishing eigenvalue.
            U, s, _ = np.linalg.svd(A_y, full_matrices=False)
            eigenvalues = s**2 / (N - 1)
            floor = 1e-2 * float(np.max(np.diag(self.R_block)))
            keep = eigenvalues > floor
            d = np.zeros_like(s)
            d[keep] = (N - 1) / s[keep] ** 2
            K = ((self.localization * P_xy) @ U * d) @ U.T  # (n x mL)
        corr_obs = self.R_block_chol @ Q_qpca  # (mL x N)
        X_a_end = X_path[-1] + K @ corr_obs  # (n x N)
        if not return_diagnostics:
            return X_a_end

        # Nothing below is used by the update; it repackages what was already
        # computed. `Q_qpca` above is the *lifted* correction -Vk Vk^T E, which
        # is Delta_white; the kappa-dimensional coordinate map of the paper is
        # Vk^T E, recovered here without recomputing anything upstream.
        order = np.argsort(w)[::-1]
        diagnostics = {
            "D": HX_stack - z_stack[:, None],
            "E": E,
            "e_bar": E.mean(axis=1),
            "E_c": Ec,
            "Sigma_E": C,
            "eigvals": w[order],
            "eigvecs": V[:, order],
            "V_kappa": Vk,
            "Q_pca": Vk.T @ E,
            "Delta_white": Q_qpca,
            "Delta_obs": corr_obs,
            "Delta_x": K @ corr_obs,
            "K": K,
            "X_f": X_path[-1],
        }
        return X_a_end, diagnostics
