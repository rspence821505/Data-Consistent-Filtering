"""
Lorenz-96 model implementation.
"""

import numpy as np


class Lorenz96:
    """Lightweight Lorenz-96 integrator with stateful API for notebooks."""

    def __init__(self, n, F, dt):
        self.n = n
        self.F = F
        self.dt = dt
        self.state = np.zeros(n, dtype=float)

    def set_state(self, x0):
        self.state = np.array(x0, dtype=float)

    def step(self, steps=1):
        for _ in range(steps):
            self.state = rk4_step(l96_rhs, self.state, self.dt, F=self.F)
        return self.state

    def get_state(self):
        return self.state.copy()


def l96_rhs(x, F=8.0):
    """
    Right-hand side of Lorenz-96 ODE system.

    dx_k/dt = (x_{k+1} - x_{k-2}) * x_{k-1} - x_k + F

    The cyclic index arithmetic is expressed with `np.roll` along the state
    axis, so a whole ensemble can be advanced in one call instead of member by
    member. Results are bit-identical to the per-element formulation.

    Parameters
    ----------
    x : ndarray, shape (n,) or (n, N)
        State vector, or an ensemble with members as columns.
    F : float, optional
        Forcing parameter (default: 8.0).

    Returns
    -------
    dxdt : ndarray, same shape as `x`
        Time derivative.
    """
    return (
        (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(x, 1, axis=0) - x + F
    )


def rk4_step(f, x, dt, **kwargs):
    """
    Fourth-order Runge-Kutta time step.

    Parameters
    ----------
    f : callable
        Right-hand side function f(x, **kwargs).
    x : ndarray, shape (n,)
        Current state.
    dt : float
        Time step.
    **kwargs : dict
        Additional arguments to f.

    Returns
    -------
    x_next : ndarray, shape (n,)
        State at next time step.
    """
    k1 = f(x, **kwargs)
    k2 = f(x + 0.5 * dt * k1, **kwargs)
    k3 = f(x + 0.5 * dt * k2, **kwargs)
    k4 = f(x + dt * k3, **kwargs)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_model(x0, dt, steps, F=8.0):
    """
    Integrate Lorenz-96 model forward in time.

    Parameters
    ----------
    x0 : ndarray, shape (n,)
        Initial state.
    dt : float
        Time step.
    steps : int
        Number of integration steps.
    F : float, optional
        Forcing parameter (default: 8.0).

    Returns
    -------
    traj : ndarray, shape (steps+1, n)
        Trajectory with initial condition at traj[0].
    """
    traj = np.zeros((steps + 1, x0.size))
    traj[0] = x0
    x = x0.copy()
    for t in range(steps):
        x = rk4_step(l96_rhs, x, dt, F=F)
        traj[t + 1] = x
    return traj
