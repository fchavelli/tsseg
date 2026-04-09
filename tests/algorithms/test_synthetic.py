"""Synthetic smoke tests for segmentation detectors.

Shared helpers + behavioural tests that run on easy piecewise-constant
signals.  Each algorithm class gets its own ``Test*Synthetic`` group.
"""

import numpy as np
import pytest

from tsseg.algorithms.vsax.detector import VSAXDetector

# ---------------------------------------------------------------------------
# Helpers (reusable across algorithm test classes)
# ---------------------------------------------------------------------------


def _make_piecewise_constant(
    segment_lengths, means, noise_std=0.1, rng=None, n_channels=1
):
    """Generate a piecewise-constant signal with Gaussian noise.

    Parameters
    ----------
    segment_lengths : list[int]
    means : list[float] | list[array-like]
        Per-segment mean.  Scalars for univariate, arrays for multivariate.
    noise_std : float
    rng : int | None
    n_channels : int
        If > 1 *and* means are scalars, tile the scalar across channels.
    """
    rng = np.random.default_rng(rng)
    parts = []
    for length, mean in zip(segment_lengths, means):
        mean = np.atleast_1d(mean)
        if mean.shape[0] == 1 and n_channels > 1:
            mean = np.full(n_channels, mean[0])
        parts.append(
            rng.normal(loc=mean, scale=noise_std, size=(length, mean.shape[0]))
        )
    out = np.concatenate(parts, axis=0)
    if out.shape[1] == 1:
        return out.ravel()
    return out


def _ari(labels_true, labels_pred):
    """Adjusted Rand Index (sklearn-free, for a quick sanity check)."""
    from itertools import combinations

    n = len(labels_true)
    assert n == len(labels_pred)
    # Build pair-level agreement
    tp = fp = fn = tn = 0
    for i, j in combinations(range(n), 2):
        same_true = labels_true[i] == labels_true[j]
        same_pred = labels_pred[i] == labels_pred[j]
        tp += same_true and same_pred
        fp += (not same_true) and same_pred
        fn += same_true and (not same_pred)
        tn += (not same_true) and (not same_pred)
    # ARI formula
    denom = (tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)
    if denom == 0:
        return 1.0
    # Hubert-Arabie formulation
    ri = (tp + tn) / (tp + fp + fn + tn)
    eri = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (tp + fp + fn + tn) ** 2
    if 1 - eri == 0:
        return 1.0
    return (ri - eri) / (1 - eri)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVSAXSynthetic:
    """Behavioural tests on easy synthetic series."""

    def test_univariate_three_states(self):
        """Three clearly separated plateaux → should recover 3 states."""
        X = _make_piecewise_constant(
            segment_lengths=[100, 100, 100],
            means=[0.0, 5.0, -5.0],
            noise_std=0.2,
            rng=42,
        )
        det = VSAXDetector(
            min_segment_length=20,
            max_segment_length=150,
            penalty=0.5,
            alphabet_size=6,
            paa_segments=4,
        )
        det.fit(X)
        labels = det.predict(X)

        assert labels.shape == (300,)
        n_states = len(np.unique(labels))
        # Should find exactly 3 or very close
        assert 2 <= n_states <= 5, f"expected ~3 states, got {n_states}"

    def test_univariate_returning_state(self):
        """A-B-A pattern → first and last segment should share a state."""
        X = _make_piecewise_constant(
            segment_lengths=[100, 100, 100],
            means=[0.0, 8.0, 0.0],
            noise_std=0.15,
            rng=7,
        )
        det = VSAXDetector(
            min_segment_length=20,
            max_segment_length=120,
            penalty=0.3,
            symbol_merge_threshold=0.3,
        )
        det.fit(X)
        labels = det.predict(X)

        # The first and last segments should be assigned the same state
        state_first = int(np.median(labels[:50]))
        state_last = int(np.median(labels[250:]))
        assert state_first == state_last, (
            f"A-B-A: first state={state_first}, last state={state_last} — "
            "should be equal"
        )

    def test_multivariate_two_states(self):
        """Two-channel signal with anti-correlated shifts."""
        rng = np.random.default_rng(99)
        n_seg = 150
        ch1 = np.concatenate(
            [
                rng.normal(0, 0.2, n_seg),
                rng.normal(3, 0.2, n_seg),
            ]
        )
        ch2 = np.concatenate(
            [
                rng.normal(3, 0.2, n_seg),
                rng.normal(0, 0.2, n_seg),
            ]
        )
        X = np.column_stack([ch1, ch2])

        det = VSAXDetector(
            min_segment_length=30,
            max_segment_length=200,
            penalty=0.5,
            paa_segments=4,
            alphabet_size=4,
        )
        det.fit(X)
        labels = det.predict(X)

        assert labels.shape == (300,)
        n_states = len(np.unique(labels))
        assert 2 <= n_states <= 4, f"expected ~2 states, got {n_states}"

    def test_multivariate_channels_distinguished(self):
        """Two segments identical on ch1 but different on ch2 → distinct states."""
        rng = np.random.default_rng(123)
        n_seg = 120
        ch1 = rng.normal(0, 0.1, 2 * n_seg)  # same for both halves
        ch2 = np.concatenate(
            [
                rng.normal(0, 0.1, n_seg),
                rng.normal(4, 0.1, n_seg),
            ]
        )
        X = np.column_stack([ch1, ch2])

        det = VSAXDetector(
            min_segment_length=20,
            max_segment_length=180,
            penalty=0.4,
            paa_segments=4,
            alphabet_size=5,
        )
        det.fit(X)
        labels = det.predict(X)

        state_first = np.median(labels[:60]).astype(int)
        state_second = np.median(labels[180:]).astype(int)
        assert state_first != state_second, (
            "Segments differ on ch2 but got same state — "
            "multivariate symbols not working"
        )

    def test_constant_signal_single_state(self):
        """Flat signal → should produce a single state."""
        X = np.ones(200) * 3.0
        det = VSAXDetector(min_segment_length=20, max_segment_length=200, penalty=1.0)
        det.fit(X)
        labels = det.predict(X)

        assert labels.shape == (200,)
        assert len(np.unique(labels)) == 1

    def test_exact_matching_mode(self):
        """symbol_merge_threshold=0 should still run without errors."""
        X = _make_piecewise_constant([80, 80, 80], [0, 3, 0], rng=0)
        det = VSAXDetector(
            min_segment_length=15,
            max_segment_length=120,
            penalty=0.5,
            symbol_merge_threshold=0.0,
        )
        det.fit(X)
        labels = det.predict(X)
        assert labels.shape == (240,)
        assert len(np.unique(labels)) >= 2

    def test_gaussian_breakpoints_mode(self):
        """adaptive_breakpoints=False should still produce valid output."""
        X = _make_piecewise_constant([100, 100], [0, 4], rng=1)
        det = VSAXDetector(
            min_segment_length=20,
            max_segment_length=150,
            penalty=0.5,
            adaptive_breakpoints=False,
        )
        det.fit(X)
        labels = det.predict(X)
        assert labels.shape == (200,)
        assert len(np.unique(labels)) >= 2

    def test_short_signal(self):
        """Signal shorter than min_segment_length should not crash."""
        X = np.array([1.0, 2.0, 3.0])
        det = VSAXDetector(min_segment_length=20, max_segment_length=100, penalty=0.5)
        det.fit(X)
        labels = det.predict(X)
        assert labels.shape == (3,)


# ===========================================================================
# VQTSSDetector tests
# ===========================================================================

torch = pytest.importorskip("torch")
from tsseg.algorithms.vqtss.detector import VQTSSDetector  # noqa: E402

# Small config shared by all VQTSS tests to keep them fast (<5 s each)
_VQTSS_FAST = dict(
    window_size=32,
    stride=4,
    hidden_dim=16,
    num_embeddings=8,
    epochs=5,
    batch_size=16,
    learning_rate=1e-3,
    random_state=42,
)


class TestVQTSSSynthetic:
    """Behavioural tests for VQTSSDetector on synthetic data."""

    def test_univariate_two_states(self):
        """Two clearly separated plateaux → at least 2 distinct states."""
        X = _make_piecewise_constant(
            segment_lengths=[100, 100],
            means=[0.0, 5.0],
            noise_std=0.2,
            rng=42,
        )
        det = VQTSSDetector(**_VQTSS_FAST)
        det.fit(X.reshape(-1, 1))
        labels = det.predict(X.reshape(-1, 1))

        assert labels.shape == (200,)
        assert len(np.unique(labels)) >= 2, (
            f"expected ≥2 states, got {len(np.unique(labels))}"
        )

    def test_multivariate_two_states(self):
        """Two-channel signal with anti-correlated shifts."""
        rng = np.random.default_rng(99)
        n = 100
        ch1 = np.concatenate([rng.normal(0, 0.2, n), rng.normal(3, 0.2, n)])
        ch2 = np.concatenate([rng.normal(3, 0.2, n), rng.normal(0, 0.2, n)])
        X = np.column_stack([ch1, ch2])

        det = VQTSSDetector(**_VQTSS_FAST)
        det.fit(X)
        labels = det.predict(X)

        assert labels.shape == (200,)
        assert len(np.unique(labels)) >= 2

    def test_constant_signal_few_states(self):
        """Flat signal → should produce very few distinct codes."""
        X = np.ones((200, 1)) * 3.0
        det = VQTSSDetector(**_VQTSS_FAST)
        det.fit(X)
        labels = det.predict(X)

        assert labels.shape == (200,)
        # A constant signal should collapse to very few codes
        n_states = len(np.unique(labels))
        assert n_states <= 4, f"expected ≤4 states for constant signal, got {n_states}"

    def test_output_length_matches_input(self):
        """Labels array must have exactly n_timepoints entries."""
        rng = np.random.default_rng(0)
        for n in [100, 150, 200]:
            X = rng.normal(size=(n, 2))
            det = VQTSSDetector(**_VQTSS_FAST)
            det.fit(X)
            labels = det.predict(X)
            assert labels.shape == (n,), (
                f"n={n}: expected shape ({n},), got {labels.shape}"
            )

    def test_deterministic_with_seed(self):
        """Same random_state → identical predictions."""
        X = _make_piecewise_constant([80, 80], [0, 4], noise_std=0.2, rng=1)
        X = X.reshape(-1, 1)

        labels_a = VQTSSDetector(**_VQTSS_FAST).fit(X).predict(X)
        labels_b = VQTSSDetector(**_VQTSS_FAST).fit(X).predict(X)
        np.testing.assert_array_equal(labels_a, labels_b)

    def test_multivariate_channels_distinguished(self):
        """Segments identical on ch1 but different on ch2 → distinct codes."""
        rng = np.random.default_rng(123)
        n = 100
        ch1 = rng.normal(0, 0.1, 2 * n)
        ch2 = np.concatenate([rng.normal(0, 0.1, n), rng.normal(5, 0.1, n)])
        X = np.column_stack([ch1, ch2])

        det = VQTSSDetector(**_VQTSS_FAST, smoothness_weight=0.01)
        det.fit(X)
        labels = det.predict(X)

        # The dominant code in each half should differ
        code_first = int(np.median(labels[:40]))
        code_second = int(np.median(labels[160:]))
        assert code_first != code_second, (
            "Segments differ on ch2 but got the same dominant code"
        )
