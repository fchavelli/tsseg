"""Tests of LaplacianDetector and of its ported graph helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import median_filter
from sklearn.metrics import adjusted_rand_score

from tsseg.algorithms import LaplacianDetector
from tsseg.algorithms.laplacian._louvain import (
    louvain_communities,
    louvain_from_affinity,
)
from tsseg.algorithms.laplacian._recurrence import (
    recurrence_affinity,
    timelag_median_filter,
)
from tsseg.algorithms.laplacian.laplacian import laplacian_embedding


def _two_regimes(n_blocks=6, block=500, seed=0):
    """Alternating sines of period 50 and 12 (states 0 and 1), plus noise."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_blocks) % 2, block)
    t = np.arange(len(y))
    x = np.where(y == 0, np.sin(2 * np.pi * t / 50), np.sin(2 * np.pi * t / 12))
    return x + 0.1 * rng.normal(size=len(y)), y


def _three_states_multivariate(seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2, 0, 1, 2, 0], 600)
    t = np.arange(len(y))
    periods = np.array([40.0, 15.0, 7.0])[y]
    X = np.stack(
        [np.sin(2 * np.pi * t / periods + j) for j in range(3)], axis=1
    ) + 0.2 * rng.normal(size=(len(y), 3))
    return X, y


# ----------------------------------------------------------------------
# recurrence affinity and diagonal median filter
# ----------------------------------------------------------------------


def _reference_affinity(F, width, k):
    """The definition, by brute force: mutual k-NN outside the band."""
    U = F / np.linalg.norm(F, axis=1, keepdims=True)
    D = np.clip(1.0 - U @ U.T, 0.0, 2.0)
    t = len(F)
    band = np.abs(np.arange(t)[:, None] - np.arange(t)[None, :]) < width
    D_out = np.where(band, np.inf, D)
    knn = np.zeros((t, t), dtype=bool)
    for i in range(t):
        knn[i, np.argsort(D_out[i])[:k]] = True
    mutual = knn & knn.T
    far = np.array([D[i, mutual[i]].max() for i in range(t) if mutual[i].any()])
    return np.where(mutual, np.exp(-D / np.median(far)), 0.0)


def test_recurrence_affinity_is_the_mutual_knn_affinity():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(60, 5))
    R = recurrence_affinity(F, width=3, k=8)
    np.testing.assert_allclose(R, _reference_affinity(F, 3, 8), atol=1e-12)
    np.testing.assert_array_equal(R, R.T)
    assert np.all(np.diagonal(R) == 0)
    assert np.all(R[np.abs(np.subtract.outer(range(60), range(60))) < 3] == 0)


def test_recurrence_affinity_default_neighbours_is_librosa_rule():
    # k = 2 ceil(sqrt(t - 2 width + 1)): at most k links per frame
    rng = np.random.default_rng(1)
    F = rng.normal(size=(100, 4))
    R = recurrence_affinity(F, width=2)
    k = int(2 * np.ceil(np.sqrt(100 - 4 + 1)))
    assert (R > 0).sum(axis=1).max() <= k
    np.testing.assert_allclose(R, _reference_affinity(F, 2, k), atol=1e-12)


def test_recurrence_affinity_rejects_a_too_wide_band():
    with pytest.raises(ValueError, match="width"):
        recurrence_affinity(np.ones((10, 2)), width=4)


def _reference_timelag(R, size):
    """librosa's formulation: pad, shear to lag coordinates, filter, shear back."""
    t = R.shape[0]
    padded = np.pad(R, [(0, t), (0, 0)])
    lag = np.empty_like(padded)
    for i in range(t):
        lag[:, i] = np.roll(padded[:, i], -i)
    lag = median_filter(lag, size=(1, size))
    back = np.empty_like(lag)
    for i in range(t):
        back[:, i] = np.roll(lag[:, i], i)
    return back[:t]


@pytest.mark.parametrize("size", [1, 3, 9])
def test_timelag_median_filter_matches_the_lag_formulation(size):
    rng = np.random.default_rng(2)
    R = rng.random((40, 40)) * (rng.random((40, 40)) < 0.3)
    R = np.maximum(R, R.T)
    out = timelag_median_filter(R, size)
    np.testing.assert_array_equal(out, _reference_timelag(R, size))
    if size > 1:
        # the zero padding breaks the symmetry near the ends of the diagonals
        assert not np.allclose(out, out.T)


# ----------------------------------------------------------------------
# Louvain
# ----------------------------------------------------------------------


def test_louvain_separates_two_cliques_joined_by_one_edge():
    edges = [(u, v, 1.0) for u in range(5) for v in range(u + 1, 5)]
    edges += [(u, v, 1.0) for u in range(5, 10) for v in range(u + 1, 10)]
    edges += [(4, 5, 0.1)]
    comms = louvain_communities(edges, seed=0)
    assert sorted(map(sorted, comms)) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_louvain_of_an_empty_graph():
    assert louvain_communities([], seed=0) == []
    assert louvain_from_affinity(np.zeros((5, 5)), seed=0) == []


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_louvain_matches_networkx(seed):
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(seed)
    A = rng.random((80, 80)) * (rng.random((80, 80)) < 0.08)
    A = np.triu(A, 1)
    A = A + A.T
    i, j = np.nonzero(np.triu(A, 1))
    G = nx.Graph()
    G.add_weighted_edges_from(
        zip(i.tolist(), j.tolist(), A[i, j].tolist(), strict=True)
    )
    expected = nx.community.louvain_communities(G, weight="weight", seed=seed)
    assert louvain_from_affinity(A, seed=seed) == expected


# ----------------------------------------------------------------------
# embedding
# ----------------------------------------------------------------------


def test_embedding_spectrum_is_a_normalised_laplacian_spectrum():
    X, _ = _three_states_multivariate()
    emb = laplacian_embedding(X, n_eig=10)
    ev = emb.eigenvalues
    assert ev.shape == (10,)
    assert ev.min() > -1e-8 and ev.max() < 2 + 1e-8
    assert np.all(np.diff(ev) >= -1e-12)
    assert emb.eigenvectors.shape == (emb.n_frames, 10)
    # frames every 16 points, window 128, exclusion ceil(128 / 32) = 4 frames
    assert emb.n_frames == (len(X) - 128) // 16 + 1
    assert emb.exclusion == 4
    assert emb.frame_of_point.shape == (len(X),)
    assert emb.frame_of_point[0] == 0
    assert emb.frame_of_point[-1] == emb.n_frames - 1


def test_window_is_shortened_on_short_series():
    emb = laplacian_embedding(np.sin(np.arange(200.0)), window_size=128)
    assert emb.window_size == 50  # a quarter of the series


# ----------------------------------------------------------------------
# detector
# ----------------------------------------------------------------------


def test_guided_recovers_two_alternating_regimes():
    x, y = _two_regimes()
    det = LaplacianDetector(n_states=2, random_state=0)
    labels = det.fit_predict(x)
    assert labels.shape == y.shape and labels.dtype == np.int64
    assert det.n_states_ == 2
    assert adjusted_rand_score(y, labels) > 0.9


@pytest.mark.parametrize("feature", ["spec", "raw"])
def test_guided_multivariate(feature):
    X, y = _three_states_multivariate()
    labels = LaplacianDetector(n_states=3, feature=feature, random_state=0).fit_predict(
        X
    )
    assert len(np.unique(labels)) == 3
    assert adjusted_rand_score(y, labels) > 0.8


def test_unsupervised_combines_louvain_and_silhouette():
    X, y = _three_states_multivariate()
    det = LaplacianDetector(random_state=0)
    labels = det.fit_predict(X)
    k = int(round(np.sqrt(det.k_louvain_ * det.k_silhouette_)))
    assert det.n_states_ == k
    assert len(np.unique(labels)) == k
    assert 2 <= det.k_silhouette_ <= 20 and 1 <= det.k_louvain_ <= 20
    assert adjusted_rand_score(y, labels) > 0.5


def test_max_states_caps_both_estimates():
    X, _ = _three_states_multivariate()
    det = LaplacianDetector(max_states=2, random_state=0).fit(X)
    det.predict(X)
    assert det.k_louvain_ <= 2 and det.k_silhouette_ == 2


def test_seeded_runs_are_identical():
    X, _ = _three_states_multivariate()
    a = LaplacianDetector(random_state=3).fit_predict(X)
    b = LaplacianDetector(random_state=3).fit_predict(X)
    np.testing.assert_array_equal(a, b)
    assert LaplacianDetector().get_tag("non_deterministic") is True
    assert LaplacianDetector(random_state=0).get_tag("non_deterministic") is False


def test_scale_of_the_channels_does_not_matter():
    # the detector standardises each channel
    X, _ = _three_states_multivariate()
    a = LaplacianDetector(n_states=3, random_state=0).fit_predict(X)
    b = LaplacianDetector(n_states=3, random_state=0).fit_predict(
        X * np.array([1000.0, 0.01, 1.0]) + 5.0
    )
    assert adjusted_rand_score(a, b) > 0.99


def test_single_state_cases():
    x, _ = _two_regimes()
    assert np.all(LaplacianDetector(n_states=1).fit_predict(x) == 0)
    # fewer than 6 frames: one state
    det = LaplacianDetector(random_state=0)
    assert np.all(det.fit_predict(np.sin(np.arange(60.0))) == 0)
    assert det.n_states_ == 1 and det.k_louvain_ is None


def test_constant_stretches_and_constant_series():
    x, y = _two_regimes()
    x[:500] = 0.0  # a constant first block
    labels = LaplacianDetector(n_states=2, random_state=0).fit_predict(x)
    assert len(np.unique(labels)) == 2
    det = LaplacianDetector(random_state=0)
    labels = det.fit_predict(np.zeros(2000))
    assert np.all(labels == 0) and labels.shape == (2000,)
    assert det.n_states_ == 1 and det.k_louvain_ is None


def test_parameter_schema_rejects_invalid_values():
    x, _ = _two_regimes()
    with pytest.raises(ValueError):
        LaplacianDetector(feature="mfcc").fit_predict(x)
    with pytest.raises(ValueError):
        LaplacianDetector(n_states=0).fit_predict(x)


# ----------------------------------------------------------------------
# parity with librosa (not a dependency: skipped without it)
# ----------------------------------------------------------------------


def _mocap_frames():
    from tsseg import load_mocap

    X, _ = load_mocap(trial=0)
    X = (X - X.mean(0)) / X.std(0)
    return laplacian_embedding(X[:3000], n_eig=5)


def test_recurrence_affinity_matches_librosa():
    librosa = pytest.importorskip("librosa")
    emb = _mocap_frames()
    F, width = emb.features, emb.exclusion
    for k in (None, 12):
        expected = librosa.segment.recurrence_matrix(
            F.T, k=k, width=width, mode="affinity", metric="cosine", sym=True
        )
        np.testing.assert_array_equal(
            recurrence_affinity(F, width=width, k=k), expected
        )


def test_timelag_median_filter_matches_librosa():
    librosa = pytest.importorskip("librosa")
    emb = _mocap_frames()
    R = recurrence_affinity(emb.features, width=emb.exclusion)
    expected = librosa.segment.timelag_filter(median_filter)(R, size=(1, 9))
    np.testing.assert_array_equal(timelag_median_filter(R, 9), expected)
