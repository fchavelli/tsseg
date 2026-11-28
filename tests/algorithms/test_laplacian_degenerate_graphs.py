"""Exact-duplicate and zero-bandwidth limits of the Laplacian graphs."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from tsseg.algorithms import LaplacianDetector
from tsseg.algorithms.laplacian._recurrence import recurrence_affinity


def _duplicate_affinity_oracle(features, metric):
    """All three copies tie at distance zero and fit in each k=3 list.

    Enumerate all pairs rather than using a nearest-neighbour query. Two
    extra, mutually close unique rows can have a positive-distance link,
    but their distances do not change the zero median bandwidth.
    """
    distance = pairwise_distances(features, metric=metric)
    np.fill_diagonal(distance, np.inf)
    neighbours = np.zeros(distance.shape, dtype=bool)
    for row in range(len(features)):
        neighbours[row, np.argsort(distance[row])[:3]] = True
    mutual = neighbours & neighbours.T
    farthest = [
        distance[row, mutual[row]].max()
        for row in range(len(features))
        if mutual[row].any()
    ]
    assert np.median(farthest) == 0
    return (mutual & (distance == 0)).astype(float)


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
@pytest.mark.parametrize("positive_pair", [False, True])
def test_exact_duplicate_links_follow_the_zero_bandwidth_limit(metric, positive_pair):
    features = np.tile(np.eye(4), (4, 1))
    if positive_pair:
        features = np.r_[features, [[1.0, 1.0, 1.0, 1.0], [1.1, 1.0, 1.0, 1.0]]]
    affinity = recurrence_affinity(features, width=1, k=3, metric=metric)
    expected = _duplicate_affinity_oracle(features, metric)
    np.testing.assert_array_equal(affinity, expected)
    assert np.count_nonzero(affinity) == 48  # four complete, four-node groups
    assert np.isfinite(affinity).all()


@pytest.mark.parametrize("width", [1, 2, 3])
def test_duplicate_ties_respect_mutuality_neighbour_count_and_exclusion(width):
    features = np.tile(np.eye(4), (5, 1))
    affinity = recurrence_affinity(features, width=width, k=2)
    np.testing.assert_array_equal(affinity, affinity.T)
    assert np.count_nonzero(affinity) > 0
    assert (affinity > 0).sum(axis=1).max() <= 2
    distance = pairwise_distances(features, metric="cosine")
    assert np.all(distance[affinity > 0] == 0)
    excluded = (
        np.abs(np.subtract.outer(np.arange(len(features)), np.arange(len(features))))
        < width
    )
    assert np.all(affinity[excluded] == 0)
    np.testing.assert_array_equal(
        affinity, recurrence_affinity(features, width=width, k=2)
    )


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
@pytest.mark.parametrize("width", [1, 3])
def test_zero_distance_link_with_positive_bandwidth_matches_pairwise_oracle(
    metric, width
):
    features = np.random.default_rng(42).normal(size=(18, 4))
    features[0] = features[7] = [1.0, 0.0, 0.0, 0.0]
    distance = pairwise_distances(features, metric=metric)
    allowed = np.abs(np.subtract.outer(np.arange(18), np.arange(18))) >= width
    bandwidth = np.median(np.where(allowed, distance, 0).max(axis=1))
    assert bandwidth > 0
    expected = np.where(allowed, np.exp(-distance / bandwidth), 0)
    affinity = recurrence_affinity(features, width=width, k=17, metric=metric)
    np.testing.assert_allclose(affinity, expected, rtol=1e-14, atol=1e-14)
    assert affinity[0, 7] == 1.0


@pytest.mark.parametrize(
    "feature_values",
    [np.repeat([1.0, 2.0, 3.0, 4.0], 5), np.arange(1.0, 21.0)],
    ids=["zero-median", "positive-median"],
)
def test_embedding_path_weights_use_the_bandwidth_limit(monkeypatch, feature_values):
    from tsseg.algorithms.laplacian import laplacian as module

    features = feature_values[:, None]
    captured = []
    original_laplacian = module.scipy.sparse.csgraph.laplacian

    def capture_graph(graph, **kwargs):
        captured.append(graph.copy())
        return original_laplacian(graph, **kwargs)

    monkeypatch.setattr(
        module, "frame_features", lambda windows, feature: features.copy()
    )
    monkeypatch.setattr(
        module, "recurrence_affinity", lambda F, **kw: np.zeros((len(F),) * 2)
    )
    monkeypatch.setattr(module.scipy.sparse.csgraph, "laplacian", capture_graph)
    embedding = module.laplacian_embedding(
        np.ones((160, 1)), window_size=8, step=8, n_eig=4, evec_smooth=1
    )
    squared_distance = np.diff(feature_values) ** 2
    bandwidth = np.median(squared_distance)
    expected = (
        np.exp(-squared_distance / bandwidth)
        if bandwidth > 0
        else (squared_distance == 0).astype(float)
    )
    np.testing.assert_array_equal(
        captured[0], np.diag(expected, k=1) + np.diag(expected, k=-1)
    )
    assert np.isfinite(embedding.eigenvalues).all()
    assert np.isfinite(embedding.eigenvectors).all()
    if bandwidth == 0:
        assert np.count_nonzero(embedding.eigenvalues < 1e-10) == 4


@pytest.mark.parametrize("n_states", [None, 4])
@pytest.mark.parametrize("feature", ["raw", "spec"])
def test_plateau_graphs_return_finite_labels(n_states, feature):
    X = np.repeat(np.tile(np.eye(4), (2, 1)), 64, axis=0)
    detector = LaplacianDetector(
        n_states=n_states,
        feature=feature,
        window_size=8,
        step=8,
        rec_smooth=1,
        evec_smooth=1,
        random_state=0,
        max_states=5,
    )
    labels = detector.fit_predict(X)
    assert labels.shape == (len(X),)
    assert np.issubdtype(labels.dtype, np.integer)
    assert 1 <= len(np.unique(labels)) <= (n_states or 5)
    assert np.isfinite(detector.eigenvalues_).all()
