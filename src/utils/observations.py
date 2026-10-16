"""
Observation operator utilities.
"""

import numpy as np

from ..models.lorenz96 import l96_rhs, rk4_step
from ..models.kuramoto_sivashinsky import integrate_model as ks_integrate_model


def initialize_ensemble(
    truth_0: np.ndarray,
    N: int,
    common_weight: float = 0.5,
    individual_weight: float = 0.5,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Initialize ensemble with perturbations around first truth state.

    Args:
        truth_0: Initial true state (n,)
        N: Ensemble size
        common_weight: Shared perturbation weight.
        individual_weight: Per-member perturbation weight.
        seed: Optional seed for reproducibility (ignored if rng provided).
        rng: Optional numpy Generator to draw perturbations from.

    Returns:
        X0: (n, N) initial ensemble matrix
    """
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, but not both.")

    if rng is None:
        rng = np.random.default_rng(seed)

    n = len(truth_0)

    xi_0 = rng.standard_normal(n)
    X0 = np.zeros((n, N))

    for j in range(N):
        xi_j = rng.standard_normal(n)
        X0[:, j] = truth_0 + common_weight * xi_0 + individual_weight * xi_j

    return X0


def generate_truth(x0, K, steps_per_obs, dt, F=8.0, spinup_steps=2000, model_type='lorenz96', L=None):
    """
    Generate truth trajectory using specified model from src.

    Parameters
    ----------
    x0 : ndarray
        Initial condition
    K : int
        Number of observation times
    steps_per_obs : int
        Integration steps between observations
    dt : float
        Time step
    F : float
        Forcing parameter (for Lorenz-96)
    spinup_steps : int
        Spinup steps before saving trajectory
    model_type : str
        Model type: 'lorenz96' or 'ks'
    L : float, optional
        Domain length (required for KS model)

    Returns
    -------
    truth : ndarray, shape (K+1, n)
        Truth trajectory (includes initial condition)
    """
    if model_type == 'ks':
        if L is None:
            raise ValueError("Domain length L is required for KS model")

        # Spinup
        n = x0.size
        x = x0.copy()
        if spinup_steps > 0:
            x = ks_integrate_model(x, dt, spinup_steps, L)[-1]

        # Generate trajectory
        truth = np.zeros((K + 1, n))
        truth[0] = x

        for k in range(K):
            traj = ks_integrate_model(x, dt, steps_per_obs, L)
            x = traj[-1]
            truth[k + 1] = x

        return truth

    else:  # lorenz96
        # Spinup
        x = x0.copy()
        for _ in range(spinup_steps):
            x = rk4_step(l96_rhs, x, dt, F=F)

        # Generate trajectory
        n = x.size
        truth = np.zeros((K + 1, n))
        truth[0] = x

        for k in range(K):
            for _ in range(steps_per_obs):
                x = rk4_step(l96_rhs, x, dt, F=F)
            truth[k + 1] = x

        return truth


print("Lorenz-96 model utilities implemented using src/models/lorenz96.py")


def generate_observations(truth, H, sigma_obs, seed=None, rng=None):
    """
    Generate synthetic observations with Gaussian noise.

    Parameters
    ----------
    truth : ndarray, shape (K+1, n)
        Truth trajectory
    H : ndarray, shape (m, n)
        Observation operator
    sigma_obs : float
        Observation noise standard deviation
    seed : int, optional
        Seed for the *global* legacy generator. Kept for the callers that were
        written against it; see the warning below.
    rng : numpy.random.Generator, optional
        Draw the noise from this generator instead. Preferred: it makes the
        observation errors independent of anything else the caller has drawn,
        which ``seed`` does not guarantee.

    Warning
    -------
    The ``seed`` path re-seeds the *global* legacy stream, so it silently
    correlates the observation errors with anything the caller drew from that
    same stream under the same seed. In ``src/experiments/baseline.py`` the
    truth's initial condition used to be drawn that way, which made the first
    ``ceil(n/m)`` observation-error vectors literally equal to ``x0``: with
    n = 40, m = 20 the errors at k = 0 and k = 1 were ``x0[:20]`` and
    ``x0[20:]`` exactly. Pass ``rng`` to avoid it.

    ``seed`` is retained, and its draw order is unchanged, because
    ``bias_variance_mc.py`` and ``restart_vs_cycled.py`` key long-lived
    checkpoints to it -- switching them would invalidate those caches for a
    defect that only touches the first observation times.

    Returns
    -------
    observations : ndarray, shape (K, m)
        Noisy observations (excluding initial time)
    """
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, but not both.")
    if seed is not None:
        np.random.seed(seed)
    draw = np.random if rng is None else rng

    K = truth.shape[0] - 1
    m = H.shape[0]

    observations = np.zeros((K, m))
    for k in range(K):
        observations[k] = H @ truth[k + 1] + sigma_obs * draw.standard_normal(m)

    return observations


print("Observation utilities implemented using src/utils/observations.py")


def build_obs_operator(n, observe_every=2):
    """
    Build observation operator that observes every k-th state variable.

    Parameters
    ----------
    n : int
        State dimension.
    observe_every : int, optional
        Observe every k-th state (default: 2).

    Returns
    -------
    H : ndarray, shape (m, n)
        Observation operator (sparse identity).
    obs_idx : ndarray, shape (m,)
        Indices of observed state variables.
    """
    idx = np.arange(0, n, observe_every)
    m = len(idx)
    H = np.zeros((m, n))
    H[np.arange(m), idx] = 1.0
    return H, idx


def multiplicative_inflation(X, inflation_factor):
    """
    Apply multiplicative covariance inflation to ensemble.

    Parameters
    ----------
    X : ndarray, shape (n, N)
        Ensemble matrix (columns = particles)
    inflation_factor : float
        Multiplicative inflation factor (>= 1.0)

    Returns
    -------
    X_inflated : ndarray, shape (n, N)
        Inflated ensemble
    """
    if inflation_factor == 1.0:
        return X

    x_mean = X.mean(axis=1, keepdims=True)
    X_anom = X - x_mean
    return x_mean + inflation_factor * X_anom


def _propagate_ensemble(ensemble, steps, dt, F=None, model_type='lorenz96', L=None):
    """
    Propagate each ensemble member forward in time.

    Parameters
    ----------
    ensemble : ndarray, shape (n, N)
        Ensemble matrix
    steps : int
        Number of time steps
    dt : float
        Time step size
    F : float, optional
        Forcing parameter (for Lorenz-96)
    model_type : str
        Model type: 'lorenz96' or 'ks'
    L : float, optional
        Domain length (required for KS model)

    Returns
    -------
    ensemble : ndarray, shape (n, N)
        Propagated ensemble
    """
    if model_type == 'ks':
        if L is None:
            raise ValueError("Domain length L is required for KS model")
        for j in range(ensemble.shape[1]):
            traj = ks_integrate_model(ensemble[:, j], dt, steps, L)
            ensemble[:, j] = traj[-1]
    else:  # lorenz96
        if F is None:
            F = 8.0
        # l96_rhs is shape-agnostic, so the whole ensemble advances in one call.
        for _ in range(steps):
            ensemble = rk4_step(l96_rhs, ensemble, dt, F=F)
    return ensemble
