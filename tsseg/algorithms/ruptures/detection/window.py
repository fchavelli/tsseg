"""Window-based change point detection."""

from __future__ import annotations

import numpy as np

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import argrelmax_1d, pairwise, sanity_check, tie_tol, tie_unit


class Window(BaseEstimator):
    """Sliding-window change point detection.

    ``backend``: ``"auto"`` computes the costs ``l1``, ``l2``, ``rbf`` and
    ``cosine`` with numba (``ruptures.accel``) when it is installed, the kernel
    costs without the n x n Gram matrix; ``"numba"`` and ``"python"`` force
    either path. Both return the same change points.

    Scores closer than the tie gap of their windows' costs (``utils.tie_tol``)
    are equal: a peak must exceed its neighbours by more than that, the latest
    of tied peaks is taken first, as the sorted (score, index) pairs order
    exact ties, and a gain tied with ``pen`` stops.
    """

    def __init__(
        self,
        width: int = 100,
        model: str = "l2",
        custom_cost: BaseCost | None = None,
        min_size: int = 2,
        jump: int = 5,
        params: dict | None = None,
        backend: str = "auto",
    ) -> None:
        self.min_size = min_size
        self.jump = jump
        self.width = 2 * (width // 2)
        self.backend = backend
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

    def _sum_of_costs(self, bkps: list[int]) -> float:
        """``cost.sum_of_costs``, through the backend's ``error``."""
        return sum(self._error(start, end) for start, end in pairwise([0] + bkps))

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        if self.n_samples is None or self.inds is None:
            raise RuntimeError("Estimator not fitted")
        bkps = [self.n_samples]
        stop = False
        error = self._sum_of_costs(bkps)
        order = max(max(self.width, 2 * self.min_size) // (2 * self.jump), 1)
        tol = np.array([tie_tol(cost, self.width * self._unit) for cost in self._window_costs])
        peak_inds_shifted = argrelmax_1d(self.score, order=order, tol=tol)
        if peak_inds_shifted.size == 0:
            return bkps
        peaks = list(
            zip(self.score[peak_inds_shifted], self.inds[peak_inds_shifted], tol[peak_inds_shifted])
        )
        total_tol = self.n_samples * self._unit
        while not stop and peaks:
            best_gain, _, best_tol = max(peaks, key=lambda peak: peak[0])
            peak = max(
                (peak for peak in peaks if peak[0] >= best_gain - max(peak[2], best_tol)),
                key=lambda peak: peak[1],
            )
            peaks.remove(peak)
            bkp = peak[1]
            stop = True
            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                gain = error - self._sum_of_costs(sorted([bkp] + bkps))
                if gain > pen + tie_tol(error, total_tol):
                    stop = False
            elif epsilon is not None:
                if error > epsilon + tie_tol(error, total_tol):
                    stop = False
            if not stop:
                bkps.append(int(bkp))
                bkps.sort()
                error = self._sum_of_costs(bkps)
        return bkps

    def fit(self, signal) -> "Window":
        from .. import accel  # numba, imported on first use

        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples = self.signal.shape[0]
        self.inds = np.arange(self.n_samples, step=self.jump)
        keep = (self.inds >= self.width // 2) & (self.inds < self.n_samples - self.width // 2)
        self.inds = self.inds[keep]
        self.cost.fit(signal)
        self._unit = tie_unit(self.cost)
        if accel.use_numba(self.cost, self.backend):
            fast = accel.Signal(self.cost)
            self._error = fast.error
            self.score, self._window_costs = fast.window_scores(self.inds, self.width)
            return self
        self._error = self.cost.error
        scores = []
        window_costs = []
        for idx in self.inds:
            start, end = idx - self.width // 2, idx + self.width // 2
            gain = self.cost.error(start, end)
            window_costs.append(gain)
            if np.isinf(gain) and gain < 0:
                scores.append(0.0)
                continue
            gain -= self.cost.error(start, idx) + self.cost.error(idx, end)
            scores.append(gain)
        self.score = np.asarray(scores, dtype=float)
        self._window_costs = np.asarray(window_costs, dtype=float)
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
