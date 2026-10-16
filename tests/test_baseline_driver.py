"""Contract checks for the baseline driver's data generation and time alignment.

Both properties below were broken at some point and neither is visible in any
settled statistic, which is exactly why they are pinned here.

  1. The observation errors must be independent of the truth. `run_seed` used to
     hand `generate_observations` the same integer it had just used to seed the
     global legacy stream for the initial condition, which made the first
     ceil(n/m) observation-error vectors *equal* to that initial condition.

  2. The sequential and 4D filters must be comparable on a shared window axis.
     `run_filter` scores a sequential sample k against truth[k+1] and a 4D
     window w against truth[(w+1)*L], so the sequential sample that shares a
     verification time with window w is w*L + (L-1) -- not w*L. The notebooks
     subsample with `[L-1::L]` on the strength of that.

Run:  python tests/test_baseline_driver.py   (or: python -m pytest tests/test_baseline_driver.py -v)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.baseline import _STREAM_ORDER, load_config, run_seed  # noqa: E402
from src.utils.observations import (  # noqa: E402
    build_obs_operator,
    generate_observations,
    generate_truth,
)


def _small_config(n_windows=6):
    """The real config, shortened to a handful of windows."""
    config = load_config()
    config["assimilation"]["n_windows"] = n_windows
    config["derived"]["total_observations"] = (
        n_windows * config["assimilation"]["window_length"]
    )
    return config


def test_observation_errors_are_independent_of_the_initial_condition():
    """The regression: obs error at k=0 must not be the initial condition."""
    config = _small_config()
    model_cfg, obs_cfg = config["model"], config["observations"]
    n, m = model_cfg["n"], obs_cfg["m"]
    sigma = obs_cfg["noise_std"]

    for seed in (42, 123, 456):
        np.random.seed(seed)
        x0 = np.random.randn(n)
        H, _ = build_obs_operator(n, observe_every=obs_cfg.get("observe_every", 2))
        truth = generate_truth(x0, 4, obs_cfg["steps_per_obs"], model_cfg["dt"],
                               model_cfg["F"], model_cfg.get("spinup_steps", 0))

        # The defect, reproduced: seeding re-starts the stream that made x0.
        legacy = generate_observations(truth, H, sigma, seed=seed)
        assert np.allclose((legacy[0] - H @ truth[1]) / sigma, x0[:m]), (
            "expected the legacy seed= path to still reproduce the old draw order"
        )

        # The fix: a dedicated stream is not the initial condition.
        children = np.random.SeedSequence(seed).spawn(len(_STREAM_ORDER) + 1)
        fixed = generate_observations(
            truth, H, sigma, rng=np.random.default_rng(children[-1])
        )
        err0 = (fixed[0] - H @ truth[1]) / sigma
        assert not np.allclose(err0, x0[:m]), (
            f"seed {seed}: observation error at k=0 is still the initial condition"
        )
        assert abs(np.corrcoef(err0, x0[:m])[0, 1]) < 0.75, (
            f"seed {seed}: observation error at k=0 still tracks the initial condition"
        )


def test_generate_observations_rejects_both_sources():
    config = _small_config()
    H, _ = build_obs_operator(config["model"]["n"], 2)
    truth = np.zeros((3, config["model"]["n"]))
    try:
        generate_observations(truth, H, 1.5, seed=1, rng=np.random.default_rng(1))
    except ValueError:
        return
    raise AssertionError("passing both seed and rng should raise")


def test_adding_the_observation_stream_left_the_filter_streams_alone():
    """`spawn` is prefix-stable, so the appended stream repaints nothing."""
    for seed in (42, 123):
        before = np.random.SeedSequence(seed).spawn(len(_STREAM_ORDER))
        after = np.random.SeedSequence(seed).spawn(len(_STREAM_ORDER) + 1)
        assert [s.spawn_key for s in before] == [
            s.spawn_key for s in after[: len(_STREAM_ORDER)]
        ], "the observation stream must be appended, never inserted"


def test_sequential_sample_L_minus_1_verifies_against_the_4d_window_end():
    """The phase the notebooks' `window_axis` relies on."""
    config = _small_config(n_windows=6)
    L = config["assimilation"]["window_length"]
    n_windows = config["assimilation"]["n_windows"]

    out = run_seed(42, config, keep_ensemble=True,
                   filters=["Seq-EnKF", "4D-EnKF"])
    seq_truth = np.asarray(out["Seq-EnKF"]["truth"])      # (n_windows*L, n)
    fourd_truth = np.asarray(out["4D-EnKF"]["truth"])     # (n_windows, n)

    assert seq_truth.shape[0] == n_windows * L
    assert fourd_truth.shape[0] == n_windows

    aligned = seq_truth[L - 1 :: L]
    assert np.array_equal(aligned, fourd_truth), (
        "sequential samples [L-1::L] must be verified against the same truth "
        "states as the 4D window ends"
    )

    # And the phase the code used to take is demonstrably a different time.
    assert not np.array_equal(seq_truth[::L], fourd_truth), (
        "phase 0 should NOT coincide with the window ends (L > 1)"
    )


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
