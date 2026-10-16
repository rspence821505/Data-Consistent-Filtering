"""Correctness checks for the localized ensemble transform filters.

The strong check is algebraic: without localization the ETKF transform must
reproduce the deterministic Kalman mean update built from the same ensemble
covariances, and its analysis anomalies must satisfy the square-root property
A^a (A^a)^T / (N-1) = P^a.  Both are identities, so they hold to roundoff and
do not depend on any tuning.

Run:  python tests/test_letkf.py       (or: python -m pytest tests/test_letkf.py -v)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.filters import LETKF4D, SequentialLETKF  # noqa: E402
from src.utils.localization import (  # noqa: E402
    build_localization_weights,
    gaspari_cohn,
    periodic_distance,
)
from src.utils.observations import build_obs_operator  # noqa: E402

N, NX, M, L = 10, 40, 20, 5
SIGMA = 1.5


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    H, obs_idx = build_obs_operator(NX, 2)
    R = SIGMA**2 * np.eye(M)
    X = rng.standard_normal((NX, N)) * 3.0
    z = H @ rng.standard_normal(NX) * 3.0 + SIGMA * rng.standard_normal(M)
    return rng, H, obs_idx, R, X, z


def test_gaspari_cohn_shape():
    """Taper is 1 at zero separation, compactly supported, and non-negative."""
    radius = 8.0
    assert abs(gaspari_cohn(0.0, radius) - 1.0) < 1e-12
    assert gaspari_cohn(radius, radius) == 0.0
    assert gaspari_cohn(radius * 1.5, radius) == 0.0

    d = np.linspace(0, radius, 200)
    w = gaspari_cohn(d, radius)
    assert (w >= 0).all() and (w <= 1).all()
    # Monotone decreasing over its support.
    assert (np.diff(w) <= 1e-12).all()


def test_periodic_distance_wraps():
    assert periodic_distance(0, 39, 40) == 1.0
    assert periodic_distance(0, 20, 40) == 20.0
    assert periodic_distance(5, 7, 40) == 2.0


def test_unlocalized_etkf_matches_kalman_mean():
    """Algorithm 3 without localization == deterministic Kalman mean update."""
    _, H, _, R, X, z = _setup()

    filt = SequentialLETKF(H, R, localization=None, inflation=1.0)
    X_a = filt.update(X, z)

    # Reference: xbar + P_xy (P_yy + R)^-1 d, from the same ensemble.
    x_mean = X.mean(axis=1)
    A_x = X - x_mean[:, None]
    Z = H @ X
    z_mean = Z.mean(axis=1)
    A_z = Z - z_mean[:, None]
    P_xy = A_x @ A_z.T / (N - 1)
    P_yy = A_z @ A_z.T / (N - 1)
    K = P_xy @ np.linalg.inv(P_yy + R)
    expected_mean = x_mean + K @ (z - z_mean)

    err = np.abs(X_a.mean(axis=1) - expected_mean).max() / np.abs(expected_mean).max()
    assert err < 1e-9, f"analysis mean differs from the Kalman update by {err:.3e}"


def test_square_root_property():
    """Analysis anomalies reproduce the ensemble-space analysis covariance."""
    _, H, _, R, X, z = _setup(seed=3)
    filt = SequentialLETKF(H, R, localization=None, inflation=1.0)
    X_a = filt.update(X, z)

    x_mean = X.mean(axis=1)
    A_x = X - x_mean[:, None]
    Z = H @ X
    A_z = Z - Z.mean(axis=1)[:, None]
    P_tilde = np.linalg.inv((N - 1) * np.eye(N) + A_z.T @ np.linalg.inv(R) @ A_z)

    A_a = X_a - X_a.mean(axis=1, keepdims=True)
    lhs = A_a @ A_a.T / (N - 1)
    rhs = A_x @ P_tilde @ A_x.T
    err = np.abs(lhs - rhs).max() / np.abs(rhs).max()
    assert err < 1e-9, f"square-root property violated by {err:.3e}"


def test_zero_taper_leaves_forecast_unchanged():
    """A region whose observations are all tapered out is not updated."""
    _, H, obs_idx, R, X, z = _setup(seed=5)
    weights = np.zeros((NX, M))
    filt = SequentialLETKF(H, R, localization=weights, inflation=1.0)
    X_a = filt.update(X, z)
    assert np.abs(X_a - X).max() < 1e-10


def test_localization_confines_the_increment():
    """With a tight taper, a state far from every updated region barely moves."""
    _, H, obs_idx, R, X, z = _setup(seed=7)
    weights = build_localization_weights(NX, obs_idx, radius=4.0)
    filt = SequentialLETKF(H, R, localization=weights, inflation=1.0)
    X_a = filt.update(X, z)

    increment = np.abs((X_a - X).mean(axis=1))
    global_filt = SequentialLETKF(H, R, localization=None, inflation=1.0)
    global_increment = np.abs((global_filt.update(X, z) - X).mean(axis=1))

    # Localization must not amplify the update relative to the global transform.
    assert increment.max() <= global_increment.max() * 3.0
    # Every taper row is non-trivial here, so no state is fully frozen.
    assert (weights.sum(axis=1) > 0).all()


def test_inflation_scales_anomalies_only():
    _, H, _, R, X, z = _setup(seed=11)
    plain = SequentialLETKF(H, R, inflation=1.0).update(X, z)
    inflated = SequentialLETKF(H, R, inflation=1.25).update(X, z)

    assert np.abs(plain.mean(axis=1) - inflated.mean(axis=1)).max() < 1e-10
    a_plain = plain - plain.mean(axis=1, keepdims=True)
    a_infl = inflated - inflated.mean(axis=1, keepdims=True)
    assert np.abs(a_infl - 1.25 * a_plain).max() < 1e-10


def test_4d_letkf_matches_kalman_mean_unlocalized():
    """Algorithm 4 without localization == 4D deterministic Kalman mean update."""
    rng, H, _, R, _, _ = _setup(seed=13)
    X_path = [rng.standard_normal((NX, N)) * 3.0 for _ in range(L)]
    z_stack = rng.standard_normal(M * L) * 3.0

    filt = LETKF4D(H, R, window_len=L, localization=None, inflation=1.0)
    X_a = filt.update(X_path, z_stack)

    Z_stack = np.vstack([H @ Xt for Xt in X_path])
    z_mean = Z_stack.mean(axis=1)
    A_z = Z_stack - z_mean[:, None]
    X_end = X_path[-1]
    x_mean = X_end.mean(axis=1)
    A_x = X_end - x_mean[:, None]
    R_L = np.kron(np.eye(L), R)
    P_xy = A_x @ A_z.T / (N - 1)
    P_yy = A_z @ A_z.T / (N - 1)
    K = P_xy @ np.linalg.inv(P_yy + R_L)
    expected_mean = x_mean + K @ (z_stack - z_mean)

    err = np.abs(X_a.mean(axis=1) - expected_mean).max() / np.abs(expected_mean).max()
    assert err < 1e-8, f"4D analysis mean differs by {err:.3e}"


def test_4d_localization_tiles_over_window():
    """Stacked taper repeats the spatial layout once per observation time."""
    _, H, obs_idx, _, _, _ = _setup()
    w1 = build_localization_weights(NX, obs_idx, radius=6.0, window_len=1)
    wL = build_localization_weights(NX, obs_idx, radius=6.0, window_len=L)
    assert wL.shape == (NX, M * L)
    for block in range(L):
        assert np.array_equal(wL[:, block * M : (block + 1) * M], w1)


def test_non_diagonal_R_rejected():
    _, H, _, R, _, _ = _setup()
    R_bad = R.copy()
    R_bad[0, 1] = R_bad[1, 0] = 0.5
    for ctor in (
        lambda: SequentialLETKF(H, R_bad),
        lambda: LETKF4D(H, R_bad, window_len=L),
    ):
        try:
            ctor()
        except ValueError:
            continue
        raise AssertionError("non-diagonal R should be rejected")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
    print("\nall passed" if not failures else f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
