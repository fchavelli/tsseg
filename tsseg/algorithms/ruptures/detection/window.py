"""Window-based change point detection."""

from __future__ import annotations

import numpy as np

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import argrelmax_1d, sanity_check, unzip


class Window(BaseEstimator):
    """Sliding-window change point detection."""

    def __init__(
        self,
        width: int = 100,
        model: str = "l2",
        custom_cost: BaseCost | None = None,
        min_size: int = 2,
        jump: int = 5,
        params: dict | None = None,
    ) -> None:
        self.min_size = min_size
        self.jump = jump
        self.width = 2 * (width // 2)
        self.n_samples: int | None = None
        self.signal: np.ndarray | None = None
        self.inds: np.ndarray | None = None
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.score: np.ndarray = np.array([])

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        if self.n_samples is None or self.inds is None:
            raise RuntimeError("Estimator not fitted")
        bkps = [self.n_samples]
        stop = False
        error = self.cost.sum_of_costs(bkps)
        order = max(max(self.width, 2 * self.min_size) // (2 * self.jump), 1)
        peak_inds_shifted = argrelmax_1d(self.score, order=order)
        if peak_inds_shifted.size == 0:
            return bkps
        gains = self.score[peak_inds_shifted]
        peak_inds = self.inds[peak_inds_shifted]
        gain_peak_pairs = sorted(zip(gains, peak_inds))
        while not stop and gain_peak_pairs:
            _, bkp = gain_peak_pairs.pop()
            stop = True
            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                gain = error - self.cost.sum_of_costs(sorted([bkp] + bkps))
                if gain > pen:
                    stop = False
            elif epsilon is not None:
                if error > epsilon:
                    stop = False
            if not stop:
                bkps.append(int(bkp))
                bkps.sort()
                error = self.cost.sum_of_costs(bkps)
        return bkps

    def fit(self, signal) -> "Window":
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples = self.signal.shape[0]
        self.inds = np.arange(self.n_samples, step=self.jump)
        keep = (self.inds >= self.width // 2) & (self.inds < self.n_samples - self.width // 2)
        self.inds = self.inds[keep]
        self.cost.fit(signal)
        scores = []
        for idx in self.inds:
            start, end = idx - self.width // 2, idx + self.width // 2
            gain = self.cost.error(start, end)
            if np.isinf(gain) and gain < 0:
                scores.append(0.0)
                continue
            gain -= self.cost.error(start, idx) + self.cost.error(idx, end)
            scores.append(gain)
        self.score = np.asarray(scores, dtype=float)
        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        if all(param is None for param in (n_bkps, pen, epsilon)):
            raise AssertionError("Provide at least one stopping criterion")
        return self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
