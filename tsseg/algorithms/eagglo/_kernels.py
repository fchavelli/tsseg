"""Numba kernels of E-Agglo (numba is an optional dependency of tsseg)."""

import numpy as np
from numba import njit

__all__ = [
    "euclidean",
    "euclidean_matrix_to_matrix",
    "get_distance_matrix",
    "get_distance_single",
]


@njit(fastmath=True, cache=True)
def _update_distances(
    i, j, K, merged_, n_cluster, distances, left, right, open, sizes, progression, lm
) -> None:
    """Update distance from new cluster to other clusters, store to self."""
    # which clusters were merged_, info only
    merged_[K - n_cluster + 1, 0] = -i if i <= n_cluster else i - n_cluster
    merged_[K - n_cluster + 1, 1] = -j if j <= n_cluster else j - n_cluster

    # update left and right neighbors
    ll = left[i]
    rr = right[j]
    left[K + 1] = ll
    right[K + 1] = rr
    right[ll] = K + 1
    left[rr] = K + 1

    # update information about which clusters have been merged_
    open[i] = False
    open[j] = False

    # assign size to newly created cluster
    n1 = sizes[i]
    n2 = sizes[j]
    sizes[K + 1] = n1 + n2

    # update set of change points
    progression[K - n_cluster + 2, :] = progression[K - n_cluster + 1,]
    progression[K - n_cluster + 2, lm[j]] = np.nan
    lm[K + 1] = lm[i]

    # update distances
    for k in range(K + 1):
        if open[k]:
            n3 = sizes[k]
            n = n1 + n2 + n3
            val = (
                (n - n2) * distances[i, k]
                + (n - n1) * distances[j, k]
                - n3 * distances[i, j]
            ) / n
            distances[K + 1, k] = val
            distances[k, K + 1] = val


@njit(fastmath=True, cache=True)
def _gof_update(i, gof_, left, right, distances, sizes):
    """Compute the updated goodness-of-fit statistic, left cluster given by i."""
    fit = gof_[-1]
    j = right[i]

    # get new left and right clusters
    rr = right[j]
    ll = left[i]

    # remove unneeded values in the gof_
    fit -= 2 * (distances[i, j] + distances[i, ll] + distances[j, rr])

    # get cluster sizes
    n1 = sizes[i]
    n2 = sizes[j]

    # add distance to new left cluster
    n3 = sizes[ll]
    k = (
        (n1 + n3) * distances[i, ll]
        + (n2 + n3) * distances[j, ll]
        - n3 * distances[i, j]
    ) / (n1 + n2 + n3)
    fit += 2 * k

    # add distance to new right
    n3 = sizes[rr]
    k = (
        (n1 + n3) * distances[i, rr]
        + (n2 + n3) * distances[j, rr]
        - n3 * distances[i, j]
    ) / (n1 + n2 + n3)
    fit += 2 * k

    return fit


@njit(fastmath=True, cache=True)
def get_distance_matrix(X, Y, alpha) -> float:
    """Calculate cluster distance."""
    dist = euclidean_matrix_to_matrix(X, Y)
    return np.power(dist, alpha).mean()


@njit(fastmath=True, cache=True)
def get_distance_single(X, Y, alpha) -> float:
    dist = euclidean(X, Y)
    return np.power(dist, alpha)  # .mean()


@njit(fastmath=True, cache=True)
def euclidean_matrix_to_matrix(a, b):
    """Compute the Euclidean distances between the rows of two matrices."""
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = euclidean(a[i], b[j])
    return out


@njit(fastmath=True, cache=True)
def euclidean(u, v):
    buf = u - v
    return np.sqrt(buf @ buf)
