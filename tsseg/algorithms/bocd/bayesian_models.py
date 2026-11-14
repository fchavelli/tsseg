from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _offline_changepoint_probabilities(
    data: np.ndarray, hazard: float, likelihood, truncate: float = -40
) -> np.ndarray:
    """Boundary marginals for the Gaussian product-partition model.

    A completed segment of length ``l`` has prior mass
    ``hazard * (1 - hazard)**(l - 1)``. The last segment is right censored
    and contributes only ``(1 - hazard)**(l - 1)``. Forward and backward
    messages sum over segment positions, without tracking the ordinal
    number of each boundary. Time is O(n**2) and working memory O(n).

    As in Fearnhead (2006), equation (3), a backward sum stops after a
    term whose fraction of the accumulated sum is below ``exp(truncate)``.
    The final, censored segment is always included. The forward pass uses
    exactly the same retained edges, so both passes describe one model.
    ``truncate=-np.inf`` disables truncation for numerical comparisons.
    """
    n = len(data)
    if n < 2:
        return np.empty(0, dtype=float)

    log_hazard = np.log(hazard)
    log_survival = np.arange(n) * np.log1p(-hazard)
    lengths = np.arange(1, n + 1)
    likelihood._prepare(n)

    # Center before accumulating moments: unlike differences of squared
    # prefix sums, Welford updates also work for a large observation offset.
    centered = np.asarray(data, dtype=float) - data[0]
    prior_offset = data[0] - likelihood.mu0
    means = np.zeros(n)
    sums_of_squares = np.zeros(n)
    backward = np.zeros(n + 1)
    last_end = np.full(n, n - 1, dtype=int)

    for start in range(n - 1, -1, -1):
        count = n - start
        delta = centered[start] - means[start:]
        means[start:] += delta / lengths[:count]
        sums_of_squares[start:] += delta * (centered[start] - means[start:])
        terms = (
            likelihood._log_marginals(
                means[start:] + prior_offset, sums_of_squares[start:], lengths[:count]
            )
            + log_survival[:count]
        )
        terms[:-1] += log_hazard + backward[start + 1 : n]

        # Remember the retained edges in O(n) space; the forward pass must
        # not reintroduce segments excluded by the backward truncation.
        keep = count - 1
        if keep:
            cumulative = np.logaddexp.accumulate(terms[:-1])
            small = np.flatnonzero(terms[:-1] - cumulative < truncate)
            if small.size:
                keep = int(small[0]) + 1
                last_end[start] = start + keep
        backward[start] = np.logaddexp(logsumexp(terms[:keep]), terms[-1])

    means.fill(0)
    sums_of_squares.fill(0)
    forward = np.full(n, -np.inf)
    forward[0] = 0.0
    for end in range(1, n):
        counts = lengths[:end][::-1]
        delta = centered[end - 1] - means[:end]
        means[:end] += delta / counts
        sums_of_squares[:end] += delta * (centered[end - 1] - means[:end])
        terms = (
            forward[:end]
            + likelihood._log_marginals(
                means[:end] + prior_offset, sums_of_squares[:end], counts
            )
            + log_survival[counts - 1]
            + log_hazard
        )
        forward[end] = logsumexp(terms[end <= last_end[:end]])

    # Each entry is P(boundary at t | all data). Their sum is the posterior
    # expected count, not one. Clip only floating-point overshoots of one.
    return np.exp(np.minimum(forward[1:] + backward[1:n] - backward[0], 0.0))


def _log1mexp(a: float) -> float:
    """Compute log(1 - exp(a)) for a <= 0 in a numerically stable way."""
    if a >= 0:
        return -np.inf
    if a > -np.log(2):
        return np.log(-np.expm1(a))
    return np.log1p(-np.exp(a))


def offline_changepoint_detection(
    data, prior_function, log_likelihood_class, truncate: int = -40
):
    """
    Compute the likelihood of changepoints on data.

    Parameters:
    data    -- the time series data
    truncate  -- the cutoff probability 10^truncate to stop computation for that changepoint log likelihood

    Outputs:
        P  -- the log-likelihood of a datasequence [t, s], given there is no changepoint between t and s
        Q -- the log-likelihood of data
        Pcp --  the log-likelihood that the i-th changepoint is at time step t. To actually get the probility of a changepoint at time step t sum the probabilities.
    """

    # Set up the placeholders for each parameter
    n = len(data)
    Q = np.zeros((n,))
    g = np.zeros((n,))
    G = np.zeros((n,))
    P = np.ones((n, n)) * -np.inf

    # save everything in log representation
    for t in range(n):
        g[t] = prior_function(t)
        if t == 0:
            G[t] = g[t]
        else:
            G[t] = np.logaddexp(G[t - 1], g[t])

    P[n - 1, n - 1] = log_likelihood_class.pdf(data, t=n - 1, s=n)
    Q[n - 1] = P[n - 1, n - 1]

    for t in reversed(range(n - 1)):
        P_next_cp = -np.inf  # == log(0)
        for s in range(t, n - 1):
            P[t, s] = log_likelihood_class.pdf(data, t=t, s=s + 1)

            # compute recursion
            summand = P[t, s] + Q[s + 1] + g[s + 1 - t]
            P_next_cp = np.logaddexp(P_next_cp, summand)

            # truncate sum to become approx. linear in time (see
            # Fearnhead, 2006, eq. (3))
            if summand - P_next_cp < truncate:
                break

        P[t, n - 1] = log_likelihood_class.pdf(data, t=t, s=n)

        # Compute log(1 - G) in a numerically stable way.
        antiG = _log1mexp(G[n - 1 - t])

        Q[t] = np.logaddexp(P_next_cp, P[t, n - 1] + antiG)

    Pcp = np.ones((n - 1, n - 1)) * -np.inf
    for t in range(n - 1):
        Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0]
        if np.isnan(Pcp[0, t]):
            Pcp[0, t] = -np.inf
    for j in range(1, n - 1):
        for t in range(j, n - 1):
            tmp_cond = (
                Pcp[j - 1, j - 1 : t]
                + P[j : t + 1, t]
                + Q[t + 1]
                + g[0 : t - j + 1]
                - Q[j : t + 1]
            )
            Pcp[j, t] = logsumexp(tmp_cond)
            if np.isnan(Pcp[j, t]):
                Pcp[j, t] = -np.inf

    return Q, P, Pcp


def online_changepoint_detection(data, hazard_function, log_likelihood_class):
    """
    Use online bayesian changepoint detection
    https://scientya.com/bayesian-online-change-point-detection-an-intuitive-understanding-b2d2b9dc165b

    Parameters:
    data    -- the time series data

    Outputs:
        R  -- is the probability at time step t that the last sequence is already s time steps long
        maxes -- the argmax on column axis of matrix R (growth probability value) for each time step
    """
    maxes = np.zeros(len(data) + 1)

    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = log_likelihood_class.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_function(np.array(range(t + 1)))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1 : t + 2, t + 1] = R[0 : t + 1, t] * predprobs * (1 - H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t + 1] = np.sum(R[0 : t + 1, t] * predprobs * H)

        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])

        # Update the parameter sets for each possible run length.
        log_likelihood_class.update_theta(x, t=t)

        maxes[t] = R[:, t].argmax()

    return R, maxes
