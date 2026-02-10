"""TGLAD change point detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..base import BaseSegmenter
from .vendor import ensure_vendor_imports


@dataclass
class _WindowSummary:
    """Cached results for a processed sliding window."""

    start: int
    stop: int
    precision: np.ndarray
    score: float


def _optional_dependencies() -> List[str]:
    missing: List[str] = []
    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover - torch is optional
        missing.append("torch")
    try:
        import networkx  # noqa: F401
    except ImportError:  # pragma: no cover - optional extra
        missing.append("networkx")
    return missing


class TGLADDetector(BaseSegmenter):
    """Graph-difference based change point detector using uGLAD.

    The detector follows the batching procedure described in the original
    `tGLAD <https://github.com/DanielTao/tGLAD>`_ repository. The time-series is
    converted to overlapping windows, each window yields a precision matrix via
    the :class:`uGLAD` multitask solver.

    The change point score is calculated based on the second derivative of the
    sum of partial correlations derived from the precision matrices, faithfully
    mirroring the original implementation. Change points are emitted whenever
    the score exceeds ``threshold``.

    Parameters
    ----------
    window_size : int, default=512
        Number of time steps per uGLAD window.
    stride : int, default=128
        Step between successive windows.
    batch_size : int, default=8
        Number of windows processed together by the multitask solver.
    threshold : float, default=0.5
        Minimum Frobenius distance between adjacent precision matrices that
        triggers a change point.
    min_spacing : int, optional
        Minimum distance (in samples) required between two emitted change
        points. Defaults to the stride when ``None``.
    epochs : int, default=2000
        Number of uGLAD training epochs per batch.
    learning_rate : float, default=0.001
        Optimiser learning rate.
    glad_iterations : int, default=5
        Number of unrolled GLAD iterations (``L`` in the paper).
    eval_offset : float, default=0.1
        Eigenvalue regularisation applied to batch covariance matrices.
    verbose : bool, default=False
        Forward progress flag passed through to the uGLAD routines.
    axis : int, default=0
        Time axis of ``X``. ``0`` assumes ``(n_timepoints, n_features)``.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "python_dependencies": "torch,networkx",
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
    }

    def __init__(
        self,
        *,
        window_size: int = 512,
        stride: int = 128,
        batch_size: int = 8,
        threshold: float = 0.5,
        min_spacing: int | None = None,
        epochs: int = 2000,
        learning_rate: float = 0.001,
        glad_iterations: int = 5,
        eval_offset: float = 0.1,
        verbose: bool = False,
        axis: int = 0,
    ) -> None:
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.threshold = float(threshold)
        self.min_spacing = None if min_spacing is None else int(min_spacing)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.glad_iterations = int(glad_iterations)
        self.eval_offset = float(eval_offset)
        self.verbose = bool(verbose)

        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.glad_iterations <= 0:
            raise ValueError("glad_iterations must be a positive integer")

        ensure_vendor_imports()
        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # BaseSegmenter hooks
    # ------------------------------------------------------------------

    def _fit(self, X: np.ndarray, y=None):
        missing = _optional_dependencies()
        if missing:
            extras = ", ".join(missing)
            raise ModuleNotFoundError(
                "TGLADDetector requires optional dependencies. "
                f"Install them via `pip install tsseg[tglad]` (missing: {extras})."
            )

        X2d = self._ensure_2d(X)
        windows, _ = self._build_windows(X2d)
        summaries = self._compute_precision_batch(windows)

        change_points = self._scores_to_cps(summaries, X2d.shape[0])

        self._window_summaries = summaries
        self._change_points = change_points
        return self

    def _predict(self, X: np.ndarray):
        self._check_is_fitted()
        return self._change_points.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return X[:, np.newaxis]
        if X.ndim != 2:
            raise ValueError(
                "TGLADDetector expects 1D or 2D input arrays after preprocessing."
            )
        return X

    def _build_windows(self, X: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
        n_samples = X.shape[0]
        if n_samples < self.window_size:
            raise ValueError(
                "window_size must not exceed the number of time points. "
                f"Received window_size={self.window_size}, n_samples={n_samples}."
            )

        stride = max(self.stride, 1)
        windows: list[np.ndarray] = []
        starts: list[int] = []
        for start in range(0, n_samples - self.window_size + 1, stride):
            stop = start + self.window_size
            windows.append(X[start:stop])
            starts.append(start)
        # Ensure final tail window is included for coverage
        if starts[-1] + self.window_size < n_samples:
            start = n_samples - self.window_size
            windows.append(X[start:n_samples])
            starts.append(start)
        self._window_starts = np.asarray(starts, dtype=int)
        return windows, starts

    def _get_partial_correlation_sum(self, precision: np.ndarray) -> float:
        """Compute sum of upper triangle of partial correlation matrix."""
        # Formula: rho_ij = -p_ij / sqrt(p_ii * p_jj)
        d = precision.shape[0]
        diag = np.diag(precision)
        # Outer product to get denominator matrix
        denom = np.sqrt(np.outer(diag, diag))
        # Avoid division by zero
        denom[denom < 1e-10] = 1e-10

        rho = -precision / denom

        # We only care about upper triangle, i < j
        mask = np.triu(np.ones((d, d), dtype=bool), k=1)
        weights = rho[mask]

        # Round to 5 decimal places as in original code
        weights = np.round(weights, 5)

        return float(np.sum(weights))

    def _compute_precision_batch(self, windows: list[np.ndarray]) -> list[_WindowSummary]:
        from .vendor.uGLAD.uglad import uGLAD_multitask

        summaries: list[_WindowSummary] = []
        batch: list[np.ndarray] = []

        for window in windows:
            batch.append(window)
            if len(batch) == self.batch_size:
                summaries.extend(self._run_single_batch(batch))
                batch = []
        if batch:
            summaries.extend(self._run_single_batch(batch))

        # Compute partial correlation sums (W)
        # This mirrors the 'distance_graph' logic in the original code which sums edge weights.
        # Since original code uses sparsity=1, all edges are included.
        W = [self._get_partial_correlation_sum(s.precision) for s in summaries]

        # Compute scores based on 2nd derivative of W
        # score[k] = |(W_{k+1} - W_k) - (W_k - W_{k-1})|
        for idx, summary in enumerate(summaries):
            summary.start = int(self._window_starts[idx])
            summary.stop = summary.start + self.window_size

            if idx == 0 or idx >= len(summaries) - 1:
                summary.score = 0.0
            else:
                diff_next = W[idx+1] - W[idx]
                diff_prev = W[idx] - W[idx-1]
                summary.score = abs(diff_next - diff_prev)

        return summaries

    def _run_single_batch(self, batch: list[np.ndarray]) -> list[_WindowSummary]:
        from .vendor.uGLAD.uglad import uGLAD_multitask

        model = uGLAD_multitask()
        model.fit(
            batch,
            eval_offset=self.eval_offset,
            centered=False,
            epochs=self.epochs,
            lr=self.learning_rate,
            L=self.glad_iterations,
            verbose=self.verbose,
        )
        precisions = model.precision_
        return [_WindowSummary(start=0, stop=0, precision=precision, score=0.0) for precision in precisions]

    def _scores_to_cps(self, summaries: list[_WindowSummary], n_samples: int) -> np.ndarray:
        change_points: list[int] = []
        min_spacing = self.min_spacing or self.stride
        last_cp = -np.inf
        for summary in summaries:
            if summary.score < self.threshold:
                continue
            cp = min(summary.start, n_samples - 1)
            if cp - last_cp >= min_spacing:
                change_points.append(int(cp))
                last_cp = cp
        return np.asarray(sorted(set(change_points)), dtype=int)

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    @property
    def change_points_(self) -> np.ndarray:
        """Return change points detected during :meth:`fit`."""

        self._check_is_fitted()
        return self._change_points.copy()

    @property
    def change_scores_(self) -> np.ndarray:
        """Frobenius distances for each window."""

        self._check_is_fitted()
        return np.asarray([summary.score for summary in self._window_summaries])