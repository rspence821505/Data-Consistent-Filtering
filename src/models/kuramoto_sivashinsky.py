"""
Kuramoto-Sivashinsky model implementation with ETDRK4 time stepping.
"""

import numpy as np


class KS:
    """
    Lightweight Kuramoto–Sivashinsky integrator with stateful API
    using spectral ETDRK4 (Kassam–Trefethen) and 2/3-rule dealiasing.

    PDE (periodic):
        u_t = -u u_x - u_xx - u_xxxx
    so in Fourier space:
        u_hat_t = (k^2 - k^4) u_hat + N_hat,
    where N(u) = -u u_x.
    """

    def __init__(self, n: int, L: float, dt: float, dealias: bool = True, M: int = 16):
        """
        Parameters
        ----------
        n : int
            Number of spatial grid points.
        L : float
            Domain length.
        dt : float
            Time step.
        dealias : bool, optional
            If True, apply 2/3-rule dealiasing for the quadratic nonlinearity.
        M : int, optional
            Number of points for contour integration in ETDRK4 coefficient
            computation (Kassam–Trefethen). Typical values: 16 or 32.
        """
        self.n = n
        self.L = L
        self.dt = dt
        self.state = np.zeros(n, dtype=float)
        self.dealias = dealias

        # Spectral grid: wavenumbers
        dx = L / n
        self.k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)

        # Linear operator for u_t = -u u_x - u_xx - u_xxxx
        # In Fourier space: L(k) = k^2 - k^4
        self.linear_op = self.k**2 - self.k**4

        # Dealiasing mask (2/3 rule) in Fourier space
        if self.dealias:
            k_abs = np.abs(self.k)
            k_max = k_abs.max()
            cutoff = (2.0 / 3.0) * k_max
            self.dealias_mask = (k_abs <= cutoff).astype(float)
        else:
            self.dealias_mask = np.ones_like(self.k, dtype=float)

        # Precompute ETDRK4 coefficients
        self._precompute_etdrk4(M)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_state(self, x0):
        """Set the current physical state."""
        self.state = np.array(x0, dtype=float)

    def get_state(self):
        """Return a copy of the current physical state."""
        return self.state.copy()

    def step(self, steps: int = 1):
        """Advance the KS solution by the given number of time steps."""
        for _ in range(steps):
            self.state = self.etdrk4_step(self.state)
        return self.state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _precompute_etdrk4(self, M: int):
        """
        Precompute ETDRK4 coefficients (Kassam–Trefethen) for u_t = L u + N(u).

        Uses contour integration on the complex plane for each mode of L.
        """
        h = self.dt
        L = self.linear_op.astype(complex)

        self.E = np.exp(h * L)
        self.E2 = np.exp(h * L / 2.0)

        # Contour integration on unit circle (Kassam–Trefethen 2005)
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # shape (M,)
        LR = h * L[:, None] + r[None, :]  # shape (n, M)
        exp_LR = np.exp(LR)

        self.Q = h * np.mean((exp_LR - 1.0) / LR, axis=1)
        self.f1 = h * np.mean(
            (-4.0 - LR + exp_LR * (4.0 - 3.0 * LR + LR**2)) / (LR**3),
            axis=1,
        )
        self.f2 = h * np.mean(
            (2.0 + LR + exp_LR * (-2.0 + LR)) / (LR**3),
            axis=1,
        )
        self.f3 = h * np.mean(
            (-4.0 - 3.0 * LR - LR**2 + exp_LR * (4.0 - LR)) / (LR**3),
            axis=1,
        )

        # Ensure everything is complex arrays of shape (n,)
        self.Q = self.Q.astype(complex)
        self.f1 = self.f1.astype(complex)
        self.f2 = self.f2.astype(complex)
        self.f3 = self.f3.astype(complex)
        self.E = self.E.astype(complex)
        self.E2 = self.E2.astype(complex)

    def _nonlinear(self, u):
        """
        Nonlinear term N(u) = -u * u_x evaluated in physical space,
        with spectral derivative and 2/3-rule dealiasing.

        Parameters
        ----------
        u : ndarray, shape (n,)
            Physical-space state.

        Returns
        -------
        N_u : ndarray, shape (n,)
            Nonlinear term in physical space.
        """
        # Transform u to Fourier space and dealias
        u_hat = np.fft.fft(u)
        u_hat *= self.dealias_mask

        # Spectral derivative u_x
        ux_hat = 1j * self.k * u_hat
        ux = np.fft.ifft(ux_hat).real

        # Quadratic nonlinearity
        nonlinear = -u * ux

        # Optionally dealias nonlinear term as well
        nonlinear_hat = np.fft.fft(nonlinear)
        nonlinear_hat *= self.dealias_mask
        nonlinear = np.fft.ifft(nonlinear_hat).real

        return nonlinear

    def etdrk4_step(self, u):
        """
        Single ETDRK4 step for u_t = L u + N(u), where L is diagonal in
        Fourier space and N is computed in physical space.
        """
        # Initial transform
        u_hat = np.fft.fft(u).astype(complex)
        u_hat *= self.dealias_mask

        # N(u)
        Nv = self._nonlinear(u)
        Nv_hat = np.fft.fft(Nv).astype(complex)
        Nv_hat *= self.dealias_mask

        # ETDRK4 stages (Kassam–Trefethen)
        a_hat = self.E2 * u_hat + self.Q * Nv_hat
        a = np.fft.ifft(a_hat).real
        Na = self._nonlinear(a)
        Na_hat = np.fft.fft(Na).astype(complex)
        Na_hat *= self.dealias_mask

        b_hat = self.E2 * u_hat + self.Q * Na_hat
        b = np.fft.ifft(b_hat).real
        Nb = self._nonlinear(b)
        Nb_hat = np.fft.fft(Nb).astype(complex)
        Nb_hat *= self.dealias_mask

        c_hat = self.E2 * a_hat + self.Q * (2.0 * Nb_hat - Nv_hat)
        c = np.fft.ifft(c_hat).real
        Nc = self._nonlinear(c)
        Nc_hat = np.fft.fft(Nc).astype(complex)
        Nc_hat *= self.dealias_mask

        # Final combination
        u_hat_next = (
            self.E * u_hat
            + self.f1 * Nv_hat
            + 2.0 * self.f2 * (Na_hat + Nb_hat)
            + self.f3 * Nc_hat
        )

        u_next = np.fft.ifft(u_hat_next).real
        return u_next


# ----------------------------------------------------------------------
# Standalone helpers
# ----------------------------------------------------------------------
def ks_rhs(u, k, dealias_mask=None):
    """
    Nonlinear Kuramoto–Sivashinsky term in physical space.

    PDE:
        u_t = -u u_x - u_xx - u_xxxx

    This function returns only the nonlinear part:
        N(u) = -u u_x.

    Parameters
    ----------
    u : ndarray, shape (n,)
        State vector in physical space.
    k : ndarray, shape (n,)
        Wavenumbers.
    dealias_mask : ndarray, shape (n,), optional
        Optional 2/3-rule mask in Fourier space.

    Returns
    -------
    nonlinear : ndarray, shape (n,)
        Nonlinear term N(u) in physical space.
    """
    u_hat = np.fft.fft(u)
    if dealias_mask is not None:
        u_hat = u_hat * dealias_mask

    ux_hat = 1j * k * u_hat
    ux = np.fft.ifft(ux_hat).real

    nonlinear = -u * ux

    if dealias_mask is not None:
        nonlinear_hat = np.fft.fft(nonlinear) * dealias_mask
        nonlinear = np.fft.ifft(nonlinear_hat).real

    return nonlinear


def integrate_model(x0, dt, steps, L, dealias=True, M=16):
    """
    Integrate the Kuramoto–Sivashinsky model forward in time
    using the KS class and ETDRK4.

    Parameters
    ----------
    x0 : ndarray, shape (n,)
        Initial state in physical space.
    dt : float
        Time step.
    steps : int
        Number of time steps.
    L : float
        Domain length.
    dealias : bool, optional
        Whether to use 2/3-rule dealiasing.
    M : int, optional
        Number of points for contour integration in ETDRK4 coefficient
        computation (passed to KS).

    Returns
    -------
    traj : ndarray, shape (steps+1, n)
        Trajectory with initial condition at traj[0].
    """
    n = x0.size
    model = KS(n=n, L=L, dt=dt, dealias=dealias, M=M)
    model.set_state(x0)

    traj = np.zeros((steps + 1, n), dtype=float)
    traj[0] = x0
    for t in range(steps):
        traj[t + 1] = model.step(1)

    return traj
