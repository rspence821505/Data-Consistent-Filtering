"""
Covariance localization utilities for the localized ensemble transform filters.

The LETKF variants of Algorithms 3 and 4 localize by tapering the observation
precision: for local analysis region ``q`` the algorithm uses
``R_q^{-1} = diag(rho_q) R^{-1}``, where ``rho_q`` is a compactly supported
correlation function of the distance between the region and each observation
(Hunt, Kostelich & Szunyogh, 2007).  An observation whose taper weight is zero
contributes nothing to that region's transform, so a full-length weighted
formulation is numerically identical to selecting a local subset -- and it keeps
every region's system the same shape, which is what allows the analysis to be
batched.
"""

import numpy as np


def gaspari_cohn(distance, radius):
    """Gaspari--Cohn (1999) fifth-order piecewise rational correlation function.

    Parameters
    ----------
    distance : array_like
        Separation distances.
    radius : float
        Cutoff distance at which the taper reaches zero.  In the Gaspari--Cohn
        parameterization this is ``2c``, so the half-width is ``c = radius / 2``.

    Returns
    -------
    weights : ndarray
        Taper in [0, 1], exactly zero beyond ``radius``.
    """
    distance = np.abs(np.asarray(distance, dtype=float))
    if radius <= 0:
        raise ValueError("radius must be positive.")

    c = radius / 2.0
    z = distance / c
    weights = np.zeros_like(z)

    near = z <= 1.0
    mid = (z > 1.0) & (z <= 2.0)

    zn = z[near]
    weights[near] = (
        -0.25 * zn**5 + 0.5 * zn**4 + 0.625 * zn**3 - (5.0 / 3.0) * zn**2 + 1.0
    )

    zm = z[mid]
    weights[mid] = (
        (1.0 / 12.0) * zm**5
        - 0.5 * zm**4
        + 0.625 * zm**3
        + (5.0 / 3.0) * zm**2
        - 5.0 * zm
        + 4.0
        - (2.0 / 3.0) / zm
    )

    return np.clip(weights, 0.0, 1.0)


def periodic_distance(i, j, n):
    """Shortest separation between indices on a periodic ring of length ``n``."""
    d = np.abs(np.asarray(i, dtype=float) - np.asarray(j, dtype=float))
    return np.minimum(d, n - d)


def build_localization_weights(n, obs_indices, radius, window_len=1):
    """Taper weights linking each state index to each (stacked) observation.

    Parameters
    ----------
    n : int
        State dimension; the domain is treated as a periodic ring, as for
        Lorenz-96.
    obs_indices : array_like of int
        State index observed by each row of ``H``.
    radius : float or None
        Gaspari--Cohn cutoff in grid units.  ``None`` disables localization and
        returns an array of ones, which reduces the filters to the global ETKF.
    window_len : int, optional
        Number of observation times stacked per window.  The stacked vector
        repeats the same spatial layout ``window_len`` times, so the weights are
        tiled to match (default: 1, the sequential case).

    Returns
    -------
    weights : ndarray, shape (n, len(obs_indices) * window_len)
        Row ``i`` holds the taper between state ``i`` and every stacked
        observation.
    """
    obs_indices = np.asarray(obs_indices, dtype=int)
    m = obs_indices.size

    if radius is None:
        return np.ones((n, m * window_len))

    state = np.arange(n)[:, None]
    distance = periodic_distance(state, obs_indices[None, :], n)
    weights = gaspari_cohn(distance, radius)

    # The stacked observation vector is [time 1 (m), time 2 (m), ...] and every
    # time shares the same spatial layout, so tile along the observation axis.
    return np.tile(weights, (1, window_len))
