"""Offline Gaussian partitions under a Normal-inverse-Wishart prior."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, multigammaln


class _GaussianSegments:
    """Stable segment evidence retaining all observations on a boundary grid.

    The prior is NIW(mu0, kappa0, nu0=2*alpha0+d-1, Psi0=2*beta0*I).
    In one dimension it equals the Normal-Gamma model. Block moments are
    combined by Welford's recurrence instead of squared-prefix subtraction.
    """

    def __init__(self, data, *, mu0, kappa0, alpha0, beta0, max_candidates):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        self.n, self.d = data.shape
        step = (
            1
            if max_candidates is None
            else max(1, (self.n + max_candidates) // (max_candidates + 1))
        )
        self.positions = np.r_[np.arange(0, self.n, step), self.n]
        self.kappa, self.beta = kappa0, beta0
        self.nu = 2 * alpha0 + self.d - 1
        self.prior_offset = data[0] - mu0
        counts = np.arange(self.n + 1)
        self._normalisers = (
            -0.5 * counts * self.d * np.log(np.pi)
            + 0.5 * self.d * (np.log(kappa0) - np.log(kappa0 + counts))
            + 0.5 * self.nu * self.d * np.log(2 * beta0)
            + multigammaln(0.5 * (self.nu + counts), self.d)
            - multigammaln(0.5 * self.nu, self.d)
        )
        n_blocks = len(self.positions) - 1
        self._counts = np.diff(self.positions)
        self._means = np.empty((n_blocks, self.d))
        self._scatter = np.empty((n_blocks, self.d, self.d))
        for j, (start, end) in enumerate(
            zip(self.positions[:-1], self.positions[1:], strict=True)
        ):
            block = data[start:end] - data[0]
            mean = block.mean(axis=0)
            centered = block - mean
            self._means[j] = mean
            self._scatter[j] = centered.T @ centered

    def _log_marginals(self, means, scatter, counts):
        deviation = means + self.prior_offset
        weight = self.kappa * counts / (self.kappa + counts)
        if self.d == 1:
            scale = 2 * self.beta + scatter[:, 0, 0]
            logdet = np.log(scale) + np.log1p(weight * deviation[:, 0] ** 2 / scale)
        else:
            # The determinant lemma keeps large mean offsets out of the
            # covariance matrix. Cholesky is faster than eigenvectors when
            # the prior ridge is resolved at the scatter's numerical scale.
            resolution = (
                np.finfo(float).eps
                * self.d
                * np.max(np.abs(scatter), axis=(1, 2), initial=0.0)
            )
            try:
                if np.any(2 * self.beta <= resolution):
                    raise np.linalg.LinAlgError(
                        "Prior ridge below numerical resolution"
                    )
                covariance = scatter + 2 * self.beta * np.eye(self.d)
                chol = np.linalg.cholesky(covariance)
                solved = np.linalg.solve(chol, deviation[:, :, None])[:, :, 0]
                quadratic = np.sum(solved**2, axis=1)
                logdet = 2 * np.sum(
                    np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=1
                ) + np.log1p(weight * quadratic)
            except np.linalg.LinAlgError:
                # Preserve a very small ridge on a deficient-rank scatter.
                values, vectors = np.linalg.eigh(scatter)
                cutoff = (
                    np.finfo(float).eps
                    * self.d
                    * np.max(np.abs(values), axis=1, initial=0.0)
                )
                values = np.where(values <= cutoff[:, None], 0.0, values)
                values += 2 * self.beta
                projections = np.einsum("nji,nj->ni", vectors, deviation)
                quadratic = np.sum(projections**2 / values, axis=1)
                logdet = np.sum(np.log(values), axis=1) + np.log1p(weight * quadratic)
                # Welford may leave roundoff-level correlations in an otherwise
                # diagonal scatter. Evaluate these rows directly: a spectral
                # cutoff would erase resolved variances in smaller channels.
                diagonal = np.diagonal(scatter, axis1=1, axis2=2)
                scales = np.sqrt(np.maximum(diagonal, 0.0))
                tolerance = (
                    np.finfo(float).eps
                    * self.d
                    * scales[:, :, None]
                    * scales[:, None, :]
                )
                is_diagonal = np.all(
                    (np.abs(scatter) <= tolerance) | np.eye(self.d, dtype=bool),
                    axis=(1, 2),
                )
                scale = diagonal[is_diagonal] + 2 * self.beta
                quadratic = np.sum(deviation[is_diagonal] ** 2 / scale, axis=1)
                logdet[is_diagonal] = np.sum(np.log(scale), axis=1) + np.log1p(
                    weight[is_diagonal] * quadratic
                )
        return self._normalisers[counts] - 0.5 * (self.nu + counts) * logdet

    def rows(self, min_size: int, *, reverse: bool = False):
        """Yield an endpoint, eligible neighbours, and segment log evidence."""
        n_blocks = len(self._counts)
        means = np.zeros_like(self._means)
        scatter = np.zeros_like(self._scatter)
        order = range(n_blocks - 1, -1, -1) if reverse else range(n_blocks)
        for block in order:
            if reverse:
                active = slice(block, None)
                counts = self.positions[block + 1 :] - self.positions[block]
            else:
                active = slice(None, block + 1)
                counts = self.positions[block + 1] - self.positions[: block + 1]
            old_counts = counts - self._counts[block]
            delta = self._means[block] - means[active]
            weight = old_counts * self._counts[block] / counts
            scatter[active] += self._scatter[block] + (
                delta[:, :, None] * delta[:, None, :] * weight[:, None, None]
            )
            means[active] += delta * (self._counts[block] / counts)[:, None]
            valid = counts >= min_size
            evidence = self._log_marginals(
                means[active][valid], scatter[active][valid], counts[valid]
            )
            if reverse:
                yield block, np.arange(block + 1, n_blocks + 1)[valid], evidence
            else:
                yield block + 1, np.arange(block + 1)[valid], evidence


def _edge_prior(positions, starts, ends, hazard):
    """Geometric duration mass, with right censoring of the last segment."""
    lengths = positions[ends] - positions[starts]
    return (lengths - 1) * np.log1p(-hazard) + np.where(
        np.asarray(ends) == len(positions) - 1, 0.0, np.log(hazard)
    )


def _map_partition(segments, hazard, min_size, max_cps):
    """Joint MAP partition with an optional upper bound on the change count."""
    positions = segments.positions
    n_vertices = len(positions)
    max_possible = min(n_vertices - 2, max(0, segments.n // min_size - 1))
    if max_cps is None or max_cps >= max_possible:
        scores = np.full(n_vertices, -np.inf)
        scores[0] = 0.0
        parent = np.full(n_vertices, -1, dtype=int)
        for end, starts, evidence in segments.rows(min_size):
            if not starts.size:
                continue
            candidates = (
                scores[starts] + evidence + _edge_prior(positions, starts, end, hazard)
            )
            best = int(np.argmax(candidates))
            scores[end] = candidates[best]
            parent[end] = starts[best]
        cuts = []
        point = parent[-1]
        while point > 0:
            cuts.append(positions[point])
            point = parent[point]
    else:
        # State = endpoint and segment count. This constrains the partition
        # itself, instead of trimming dates from an unconstrained fit.
        n_segments = max_cps + 1
        scores = np.full((n_vertices, n_segments + 1), -np.inf)
        scores[0, 0] = 0.0
        parent = np.full(scores.shape, -1, dtype=int)
        for end, starts, evidence in segments.rows(min_size):
            if not starts.size:
                continue
            edge = evidence + _edge_prior(positions, starts, end, hazard)
            candidates = scores[starts, :n_segments] + edge[:, None]
            best = np.argmax(candidates, axis=0)
            scores[end, 1:] = candidates[best, np.arange(n_segments)]
            parent[end, 1:] = starts[best]
        count = int(np.argmax(scores[-1, 1:])) + 1
        cuts = []
        point = parent[-1, count]
        while point > 0:
            cuts.append(positions[point])
            count -= 1
            point = parent[point, count]
    return np.asarray(cuts[::-1], dtype=np.int64)


def _boundary_marginals(segments, hazard, min_size, truncate):
    """Boundary marginals conditional on the grid and minimum segment length."""
    positions = segments.positions
    n_vertices = len(positions)
    backward = np.full(n_vertices, -np.inf)
    backward[-1] = 0.0
    last_end = np.full(n_vertices, n_vertices - 2, dtype=int)
    for start, ends, evidence in segments.rows(min_size, reverse=True):
        if not ends.size:
            continue
        terms = evidence + _edge_prior(positions, start, ends, hazard) + backward[ends]
        keep = len(ends) - 1
        if truncate is not None and keep:
            cumulative = np.logaddexp.accumulate(terms[:-1])
            with np.errstate(invalid="ignore"):
                small = np.flatnonzero(terms[:-1] - cumulative < truncate)
            if small.size:
                keep = int(small[0]) + 1
                last_end[start] = ends[keep - 1]
        backward[start] = np.logaddexp(logsumexp(terms[:keep]), terms[-1])

    forward = np.full(n_vertices, -np.inf)
    forward[0] = 0.0
    for end, starts, evidence in segments.rows(min_size):
        if not starts.size:
            continue
        terms = forward[starts] + evidence + _edge_prior(positions, starts, end, hazard)
        retained = (end <= last_end[starts]) | (end == n_vertices - 1)
        forward[end] = logsumexp(terms[retained])
    probabilities = np.zeros(segments.n - 1)
    probabilities[positions[1:-1] - 1] = np.exp(
        np.minimum(forward[1:-1] + backward[1:-1] - backward[0], 0.0)
    )
    return probabilities
