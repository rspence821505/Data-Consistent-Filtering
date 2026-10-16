"""Validate the realization-batched kernels against the single-realization path.

The batched kernels change two things that can move results:

* the forecast advances all realizations in one ``rk4_step`` call -- this is
  exactly the same elementwise arithmetic per column and is checked here to be
  bit-identical;
* the gain uses the low-rank identity instead of ``pinv``, and the stochastic
  member updates are done as one matrix product instead of a loop -- both are
  exact in exact arithmetic but reassociate floating-point operations, so they
  are checked against a tolerance rather than for bit-identity.

Run with:  python -m pytest tests/test_batched_kernels.py -v
       or:  python tests/test_batched_kernels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import bias_variance_mc as bv  # noqa: E402


def _config(n_windows: int, n_mc: int) -> dict:
    return {
        "experiment": {"seeds": [42, 123]},
        "model": {"n": 40, "F": 8.0, "dt": 0.01, "spinup_steps": 0},
        "observations": {
            "m": 20,
            "observe_every": 2,
            "noise_std": 1.5,
            "steps_per_obs": 10,
        },
        "assimilation": {
            "window_length": 5,
            "n_windows": n_windows,
            "ensemble_size": 10,
        },
        "bias_variance": {"n_monte_carlo": n_mc, "spinup_windows": 0},
        "filters": {"qpca_endcf": {"k": 1}},
    }


def test_truth_batch_is_bit_identical():
    """generate_truth_batch must reproduce the per-seed helper exactly."""
    config = _config(n_windows=20, n_mc=2)
    seeds = [42, 123, 456]
    batch = bv.generate_truth_batch(config, seeds)

    for seed in seeds:
        truth, obs, H = bv.generate_truth_and_observations(config, seed)
        b_truth, b_obs, b_H = batch[seed]
        assert np.array_equal(truth, b_truth), f"truth differs for seed {seed}"
        assert np.array_equal(obs, b_obs), f"observations differ for seed {seed}"
        assert np.array_equal(H, b_H)


def test_random_state_matches_legacy_global_seeding():
    """RandomState(s) must reproduce the stream np.random.seed(s) produces."""
    for seed in (10000, 10007):
        np.random.seed(seed)
        reference = [np.random.randn(40), np.random.randn(10, 20)]
        stream = np.random.RandomState(seed)
        assert np.array_equal(reference[0], stream.randn(40))
        assert np.array_equal(reference[1], stream.randn(10, 20))


def test_batched_forecast_is_bit_identical():
    """Stacked ensembles advance exactly as separately advanced ones."""
    from src.models.lorenz96 import l96_rhs, rk4_step

    rng = np.random.default_rng(0)
    n, N, M = 40, 10, 8
    X0 = rng.standard_normal((n, M * N))

    batched = X0.copy()
    for _ in range(50):
        batched = rk4_step(l96_rhs, batched, 0.01, F=8.0)

    separate = np.empty_like(X0)
    for i in range(M):
        block = X0[:, i * N : (i + 1) * N].copy()
        for _ in range(50):
            block = rk4_step(l96_rhs, block, 0.01, F=8.0)
        separate[:, i * N : (i + 1) * N] = block

    assert np.array_equal(batched, separate)


def _reference_analyses(config, truth, obs, H, filter_name, n_mc):
    """Single-realization path, called with the same truth/obs as the batched one.

    ``_run_one_detailed`` regenerates truth from a seed; the kernels are invoked
    directly here so both paths see byte-identical inputs.
    """
    analyses, spectrals = [], []
    for r in range(n_mc):
        seed = bv.ENSEMBLE_SEED_BASE + r
        if filter_name == "QPCA-EnDCF":
            a, s = bv.run_qpca_endcf(
                config, truth, obs, H, seed, kappa=1, return_diagnostics=True
            )
            spectrals.append(s)
        else:
            a = bv.get_filter_kernel(filter_name, config)(config, truth, obs, H, seed)
        analyses.append(a)
    return np.array(analyses), (np.array(spectrals) if spectrals else None)


def test_batched_kernels_match_reference_short_horizon():
    """Over a few windows, before chaos amplifies roundoff, the paths agree.

    Tolerances reflect the conditioning of each gain: the 4D/sequential systems
    are well conditioned (eps = sigma^2), whereas the QPCA gain carries a 1e-6
    ridge on a rank-9 matrix, putting cond(S) near 1.6e8 -- so 1e-8 relative is
    the accuracy the problem itself allows, not a defect of either method.
    """
    config = _config(n_windows=3, n_mc=4)
    truth, obs, H = bv.generate_truth_and_observations(config, 42)
    scale = np.abs(truth).max()

    tolerances = {"Seq-EnKF": 1e-9, "4D-EnKF": 1e-9, "QPCA-EnDCF": 1e-5}
    for filter_name, tol in tolerances.items():
        reference, ref_spectral = _reference_analyses(
            config, truth, obs, H, filter_name, config["bias_variance"]["n_monte_carlo"]
        )
        batched, spectral = bv.run_filter_batched(
            filter_name,
            config,
            truth,
            obs,
            H,
            range(config["bias_variance"]["n_monte_carlo"]),
        )
        rel = np.abs(reference - batched).max() / scale
        assert rel < tol, f"{filter_name}: relative difference {rel:.3e} exceeds {tol:.0e}"

        if filter_name == "QPCA-EnDCF":
            assert spectral is not None and ref_spectral is not None
            assert spectral.shape == ref_spectral.shape
            diag = np.abs(spectral - ref_spectral.astype(np.float32))
            assert np.nanmax(diag) < 1e-4, f"spectral diagnostics differ by {np.nanmax(diag):.3e}"


def test_batched_summary_statistics_match_over_long_horizon():
    """At the full horizon the trajectories decorrelate, but the statistics agree.

    Chaos turns any roundoff difference into an O(1) trajectory difference, so
    this checks the quantities the experiment actually reports rather than the
    paths themselves.
    """
    config = _config(n_windows=150, n_mc=40)
    truth, obs, H = bv.generate_truth_and_observations(config, 42)
    L = config["assimilation"]["window_length"]
    truth_windows = truth[::L]
    n_mc = config["bias_variance"]["n_monte_carlo"]

    def _attempt(run):
        """Return analyses, or None if the filter diverged on this trajectory."""
        try:
            out = run()
        except (FloatingPointError, np.linalg.LinAlgError):
            return None
        return out if np.isfinite(out).all() else None

    for filter_name in bv.FILTER_NAMES:
        reference = _attempt(
            lambda: _reference_analyses(config, truth, obs, H, filter_name, n_mc)[0]
        )
        batched = _attempt(
            lambda: bv.run_filter_batched(
                filter_name, config, truth, obs, H, range(n_mc)
            )[0]
        )

        if reference is None or batched is None:
            # Some filters here are genuinely unstable at this configuration: an
            # unlocalized, uninflated ETKF at N=10 on n=40 spans 9 of 40
            # dimensions and can collapse over a long horizon.
            #
            # The two paths need NOT agree on which realizations blow up.
            # _ensemble_transform and _batched_ensemble_transform are
            # algebraically identical and share the same 1e-12 eigenvalue floor,
            # but they contract their einsums in a different order, so they
            # follow different floating-point trajectories of a chaotic system.
            # For a marginally stable filter, which path diverges first is a
            # coin flip, and demanding agreement would make this test flaky for
            # a reason that says nothing about the batching.
            #
            # What must hold is that only the known-unstable filters do this,
            # and that the batched kernel fails LOUDLY rather than returning
            # silent NaNs for the driver to average into a result.
            assert filter_name in ("Seq-ETKF", "4D-ETKF", "Seq-LETKF", "4D-LETKF"), (
                f"{filter_name} diverged at this configuration and is not one of "
                "the ensemble transform filters known to be marginal here"
            )
            if batched is None:
                with pytest.raises(FloatingPointError, match="diverged"):
                    bv.run_filter_batched(
                        filter_name, config, truth, obs, H, range(n_mc)
                    )
            continue

        ref_summary = bv._summarize_detailed_pair(reference, truth_windows, 0)
        new_summary = bv._summarize_detailed_pair(batched, truth_windows, 0)

        for key in ("mse_trace", "variance_trace", "bias_squared_trace"):
            a = ref_summary[key].mean()
            b = new_summary[key].mean()
            # Both are M=40 Monte Carlo estimates of the same quantity from
            # decorrelated trajectories, so they agree to sampling error.
            assert abs(a - b) / a < 0.10, (
                f"{filter_name} {key}: reference {a:.4f} vs batched {b:.4f}"
            )


# --------------------------------------------------------------------------- #
# Ensemble transform variants (Seq-ETKF / 4D-ETKF)
# --------------------------------------------------------------------------- #


def test_filter_registry_is_self_consistent():
    """Every filter the driver offers must have a config key and a kernel."""
    for name in bv.FILTER_NAMES:
        assert name in bv.AVAILABLE_FILTER_NAMES, f"{name} missing from AVAILABLE"
    for name in bv.AVAILABLE_FILTER_NAMES:
        assert name in bv.FILTER_CONFIG_KEYS, f"{name} has no config key"
        assert bv.get_filter_kernel(name, _config(1, 1)) is not None


def test_etkf_variants_are_the_unlocalized_transform_filters():
    """Seq-ETKF / 4D-ETKF must be Algorithms 3 and 4 with no taper.

    A null radius makes build_localization_weights return an array of ones, so
    the localized kernel reduces to the global ETKF. If that ever stopped being
    true the ETKF columns would silently become localized.
    """
    config = _config(n_windows=2, n_mc=2)
    config["filters"]["seq_etkf"] = {"localization_radius": None, "inflation": 1.0}
    config["filters"]["etkf_4d"] = {"localization_radius": None, "inflation": 1.0}

    for name in ("Seq-ETKF", "4D-ETKF"):
        weights, inflation = bv.letkf_settings(config, name)
        assert np.array_equal(weights, np.ones_like(weights)), f"{name} is tapered"
        assert inflation == 1.0

    # A sequential filter's taper spans one observation time; a 4D one spans the
    # stacked window. Getting this wrong fails loudly at the first update.
    m = config["observations"]["m"]
    L = config["assimilation"]["window_length"]
    assert bv.letkf_settings(config, "Seq-ETKF")[0].shape[1] == m
    assert bv.letkf_settings(config, "4D-ETKF")[0].shape[1] == m * L


def test_etkf_variants_run_and_match_their_reference_kernel():
    """The batched ETKF kernels must agree with the single-realization path."""
    config = _config(n_windows=3, n_mc=3)
    truth, obs, H = bv.generate_truth_and_observations(config, 42)
    scale = np.abs(truth).max()

    for name in ("Seq-ETKF", "4D-ETKF"):
        reference, _ = _reference_analyses(config, truth, obs, H, name, 3)
        batched, spectral = bv.run_filter_batched(name, config, truth, obs, H, range(3))
        assert spectral is None
        assert np.isfinite(batched).all(), f"{name} produced non-finite output"
        rel = np.abs(reference - batched).max() / scale
        assert rel < 1e-9, f"{name}: relative difference {rel:.3e}"


def test_adding_filters_does_not_disturb_the_existing_three():
    """Ensemble seeds carry no filter key, so the published filters are untouched.

    This is what makes extending FILTER_NAMES safe here and unsafe in
    baseline.py, where per-filter streams are spawned in list order. It is
    checked directly: running a filter alone must equal running it alongside the
    others.
    """
    config = _config(n_windows=3, n_mc=3)
    truth, obs, H = bv.generate_truth_and_observations(config, 42)

    for name in ("Seq-EnKF", "4D-EnKF", "QPCA-EnDCF"):
        alone, _ = bv.run_filter_batched(name, config, truth, obs, H, range(3))
        for other in ("Seq-ETKF", "4D-ETKF"):
            bv.run_filter_batched(other, config, truth, obs, H, range(3))
        after, _ = bv.run_filter_batched(name, config, truth, obs, H, range(3))
        assert np.array_equal(alone, after), (
            f"{name} changed once the ETKF variants were run alongside it"
        )

    # And the initial ensembles themselves are filter-independent by construction.
    streams_a = bv._batched_initial_ensembles(truth[0], 40, 10, bv._make_streams(range(3)))
    streams_b = bv._batched_initial_ensembles(truth[0], 40, 10, bv._make_streams(range(3)))
    assert np.array_equal(streams_a, streams_b)


def test_uniform_taper_short_circuit_is_exact_and_leaves_localization_alone():
    """A uniform taper solves one system; a real taper must still solve n.

    ``build_localization_weights`` returns an array of ones for a null radius,
    so an unlocalized ETKF asks for n_regions bit-identical N x N systems. The
    transform detects that and solves one. Two things have to hold: the single
    system must equal every system the full computation would have produced,
    and a genuinely localized taper must be untouched by the optimization.
    """
    n, d, N, M = 40, 100, 10, 6
    rng = np.random.default_rng(4)
    A_z = rng.standard_normal((d, M, N))
    innovation = rng.standard_normal((d, M))

    uniform = np.ones((n, d)) / 1.5**2
    W = bv._batched_ensemble_transform(A_z, innovation, uniform, N)
    assert W.shape == (M, 1, N, N), "the uniform case should collapse to one region"

    # The collapsed transform must equal what a per-region solve would give.
    # Feeding two identical rows exercises the same code path at n_regions=2.
    W_two = bv._batched_ensemble_transform(A_z, innovation, uniform[:2], N)
    assert np.array_equal(W, W_two)

    # A real Gaspari-Cohn taper is region-dependent and must NOT be collapsed.
    from src.utils.localization import build_localization_weights

    taper = build_localization_weights(n, np.arange(0, n, 2), 1.5, window_len=5)
    W_loc = bv._batched_ensemble_transform(A_z, innovation, taper / 1.5**2, N)
    assert W_loc.shape == (M, n, N, N)
    assert not np.allclose(W_loc, W_loc[:, :1]), "a localized taper collapsed"

    # _apply_transform must handle both shapes and agree with the tiled form.
    Xv = rng.standard_normal((n, M, N))
    a, b = Xv.copy(), Xv.copy()
    bv._apply_transform(a, W, 1.0)
    bv._apply_transform(b, np.repeat(W, n, axis=1), 1.0)
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_localized_filters_are_unaffected_by_the_short_circuit():
    """Seq-LETKF and 4D-LETKF must be bit-identical to the pre-optimization path."""
    config = _config(n_windows=20, n_mc=4)
    config["filters"]["seq_letkf"] = {"localization_radius": 1.5, "inflation": 1.0}
    config["filters"]["letkf_4d"] = {"localization_radius": 1.5, "inflation": 1.0}
    truth, obs, H = bv.generate_truth_and_observations(config, 42)

    def reference_transform(A_z, innovation, precision, N):
        """The transform as it stood before the uniform short-circuit."""
        C = np.einsum("din,qd,dim->iqnm", A_z, precision, A_z, optimize=True)
        b = np.einsum("din,qd,di->iqn", A_z, precision, innovation, optimize=True)
        target = C + (N - 1) * np.eye(N)
        target = 0.5 * (target + np.swapaxes(target, -1, -2))
        mu, V = np.linalg.eigh(target)
        mu = np.maximum(mu, 1e-12)
        Vt = np.swapaxes(V, -1, -2)
        P = np.einsum("iqnk,iqk,iqkm->iqnm", V, 1.0 / mu, Vt, optimize=True)
        Wt = np.einsum(
            "iqnk,iqk,iqkm->iqnm", V, np.sqrt((N - 1) / mu), Vt, optimize=True
        )
        wb = np.einsum("iqnm,iqm->iqn", P, b, optimize=True)
        return wb[..., None] + Wt

    current = bv._batched_ensemble_transform
    try:
        for name in ("Seq-LETKF", "4D-LETKF"):
            new, _ = bv.run_filter_batched(name, config, truth, obs, H, range(4))
            bv._batched_ensemble_transform = reference_transform
            old, _ = bv.run_filter_batched(name, config, truth, obs, H, range(4))
            bv._batched_ensemble_transform = current
            assert np.array_equal(new, old), f"{name} changed"
    finally:
        bv._batched_ensemble_transform = current


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
