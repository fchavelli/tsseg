"""AmocDetector: the cumulative-sum objective matches the direct one."""

import numpy as np

from tsseg.algorithms import AmocDetector


def _direct(x, min_size):
    n = x.shape[0]
    sse = np.full(n - 1, np.inf)
    for t in range(min_size, n - min_size + 1):
        left, right = x[:t], x[t:]
        sse[t - 1] = ((left - left.mean(0)) ** 2).sum() + (
            (right - right.mean(0)) ** 2
        ).sum()
    return int(np.argmin(sse)) + 1


def test_same_split_as_the_direct_objective():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, d, min_size = (
            int(rng.integers(20, 400)),
            int(rng.integers(1, 5)),
            int(rng.integers(1, 8)),
        )
        x = rng.normal(rng.normal(0, 100), rng.uniform(0.01, 3), (n, d))
        x[int(rng.integers(min_size, n - min_size)) :] += rng.normal(0, 1, d)
        pred = AmocDetector(min_size=min_size).fit_predict(x)
        assert list(pred) == [_direct(x, min_size)]


def test_long_series_is_fast():
    x = np.concatenate([np.zeros((200_000, 3)), np.ones((100_000, 3))])
    x += np.random.default_rng(1).normal(0, 0.1, x.shape)
    assert list(AmocDetector().fit_predict(x)) == [200_000]
