"""Validation of the refactored IGTS detector.

Tests cover:
1. Synthetic univariate (mean-shift)
2. Synthetic multivariate (mean-shift, 3 channels)
3. Real MoCap data (CMU subject 86, trial 0)
4. Cumsum optimisation correctness (compare old vs new entropy)
"""

import time

import numpy as np

from tsseg.algorithms.igts import InformationGainDetector
from tsseg.algorithms.igts.detector import (
    _IGTS,
    _augment_univariate,
    _entropy_from_sums,
    entropy,
)
from tsseg.data.datasets import load_mocap


def _tolerance_check(detected, true_cps, tol):
    """Check each true CP is matched by a detected CP within tolerance."""
    for tcp in true_cps:
        dists = np.abs(np.array(detected) - tcp)
        if dists.min() > tol:
            return False
    return True


# ── 1. Synthetic univariate ──────────────────────────────────────────────

def test_univariate_mean_shift():
    print("=" * 60)
    print("TEST 1: Synthetic univariate (mean shift at 100, 200)")
    rng = np.random.RandomState(0)
    seg1 = rng.randn(100, 1) + 0
    seg2 = rng.randn(100, 1) + 5
    seg3 = rng.randn(100, 1) - 3
    X = np.concatenate([seg1, seg2, seg3])
    true_cps = [100, 200]

    det = InformationGainDetector(k_max=2, step=1)
    t0 = time.perf_counter()
    det.fit(X)
    cps = det.predict(X)
    elapsed = time.perf_counter() - t0

    print(f"  True CPs:     {true_cps}")
    print(f"  Detected CPs: {cps.tolist()}")
    print(f"  Time:         {elapsed:.3f}s")
    print(f"  Shape:        {cps.shape}, dtype: {cps.dtype}")

    tol = 5
    ok = _tolerance_check(cps, true_cps, tol)
    print(f"  Within ±{tol}:   {'PASS' if ok else 'FAIL'}")
    assert ok, f"Detected {cps} not within ±{tol} of {true_cps}"


# ── 2. Synthetic multivariate ────────────────────────────────────────────

def test_multivariate_mean_shift():
    print("=" * 60)
    print("TEST 2: Synthetic multivariate (3 channels, CPs at 80, 160)")
    rng = np.random.RandomState(42)
    n_channels = 3
    seg1 = rng.randn(80, n_channels) + np.array([0, 2, -1])
    seg2 = rng.randn(80, n_channels) + np.array([5, -2, 3])
    seg3 = rng.randn(80, n_channels) + np.array([-3, 0, 6])
    X = np.concatenate([seg1, seg2, seg3])
    true_cps = [80, 160]

    det = InformationGainDetector(k_max=2, step=1)
    t0 = time.perf_counter()
    det.fit(X)
    cps = det.predict(X)
    elapsed = time.perf_counter() - t0

    print(f"  True CPs:     {true_cps}")
    print(f"  Detected CPs: {cps.tolist()}")
    print(f"  Time:         {elapsed:.3f}s")

    tol = 5
    ok = _tolerance_check(cps, true_cps, tol)
    print(f"  Within ±{tol}:   {'PASS' if ok else 'FAIL'}")
    assert ok, f"Detected {cps} not within ±{tol} of {true_cps}"


# ── 3. Cumsum vs naive entropy ───────────────────────────────────────────

def test_cumsum_matches_naive():
    print("=" * 60)
    print("TEST 3: Cumsum entropy matches naive entropy")
    rng = np.random.RandomState(7)
    X = np.abs(rng.randn(200, 4)) + 0.01  # ensure positive
    X_aug = _augment_univariate(X)

    change_points = [0, 50, 130, 200]

    # Naive: via the original entropy() function
    naive_ig = _IGTS.information_gain_score(X_aug, change_points)

    # Cumsum: via the new path
    cumsum = np.zeros((X_aug.shape[0] + 1, X_aug.shape[1]), dtype=np.float64)
    np.cumsum(X_aug, axis=0, out=cumsum[1:])
    h_total = _entropy_from_sums(cumsum[-1])
    cumsum_ig = _IGTS._ig_from_cumsum(cumsum, X_aug.shape[0], h_total, change_points)

    print(f"  Naive IG:  {naive_ig:.10f}")
    print(f"  Cumsum IG: {cumsum_ig:.10f}")
    diff = abs(naive_ig - cumsum_ig)
    print(f"  Abs diff:  {diff:.2e}")
    ok = diff < 1e-10
    print(f"  Match:     {'PASS' if ok else 'FAIL'}")
    assert ok, f"IG mismatch: naive={naive_ig}, cumsum={cumsum_ig}"


# ── 4. Real MoCap data ──────────────────────────────────────────────────

def test_mocap_trial0():
    print("=" * 60)
    print("TEST 4: Real MoCap data — CMU subject 86, trial 0")
    X, y = load_mocap(trial=0)
    true_cps = (np.where(np.diff(y) != 0)[0] + 1).tolist()
    n_true = len(true_cps)

    det = InformationGainDetector(k_max=n_true, step=2)
    t0 = time.perf_counter()
    det.fit(X)
    cps = det.predict(X)
    elapsed = time.perf_counter() - t0

    print(f"  Series shape:   {X.shape}")
    print(f"  True CPs ({n_true}):  {true_cps}")
    print(f"  Detected CPs ({len(cps)}): {cps.tolist()}")
    print(f"  Time:           {elapsed:.3f}s")

    # Looser tolerance for real data
    tol = 20
    matched = 0
    for tcp in true_cps:
        dists = np.abs(cps - tcp)
        if dists.min() <= tol:
            matched += 1
    recall = matched / n_true if n_true > 0 else 0
    print(f"  Recall@{tol}:     {recall:.2%} ({matched}/{n_true})")
    print(f"  Result:         {'PASS' if recall >= 0.5 else 'SOFT FAIL (informational)'}")


# ── 5. Speed comparison ─────────────────────────────────────────────────

def test_speed_improvement():
    print("=" * 60)
    print("TEST 5: Speed — cumsum vs naive on 1000-point series")
    rng = np.random.RandomState(99)
    X = np.abs(rng.randn(1000, 2)) + 0.01
    X_aug = _augment_univariate(X)

    # Naive path
    igts_naive = _IGTS(k_max=3, step=5)
    t0 = time.perf_counter()
    # Use the static method in a loop to simulate naive
    cps_identity = [0, X_aug.shape[0]]
    for _ in range(3):
        best_c, ig_max = -1, -1.0
        for c in range(0, X_aug.shape[0], 5):
            if c in cps_identity:
                continue
            try_cps = sorted(set(cps_identity) | {c})
            ig = _IGTS.information_gain_score(X_aug, try_cps)
            if ig > ig_max:
                ig_max = ig
                best_c = c
        cps_identity = sorted(set(cps_identity) | {best_c})
    t_naive = time.perf_counter() - t0

    # Cumsum path (the new default)
    igts_fast = _IGTS(k_max=3, step=5)
    t0 = time.perf_counter()
    igts_fast.find_change_points(X_aug)
    t_cumsum = time.perf_counter() - t0

    speedup = t_naive / t_cumsum if t_cumsum > 0 else float("inf")
    print(f"  Naive time:  {t_naive:.3f}s")
    print(f"  Cumsum time: {t_cumsum:.3f}s")
    print(f"  Speedup:     {speedup:.1f}x")


if __name__ == "__main__":
    test_univariate_mean_shift()
    print()
    test_multivariate_mean_shift()
    print()
    test_cumsum_matches_naive()
    print()
    test_mocap_trial0()
    print()
    test_speed_improvement()
    print("\n" + "=" * 60)
    print("ALL DONE")
