"""TGLAD change point detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    DependsOn,
    HasType,
    Interval,
    ParamDef,
)
from .vendor import ensure_vendor_imports


@dataclass
class _WindowSummary:
    """Cached results for a processed sliding window."""

    start: int
    stop: int
    precision: np.ndarray
    score: float


class TGLADDetector(BaseSegmenter):
    """Graph-difference based change point detector using uGLAD.

    The detector follows the batching procedure described in the original
    `tGLAD <https://github.com/Harshs27/tGLAD>`_ repository. The time-series is
    converted to overlapping windows, each window yields a precision matrix via
    the :class:`uGLAD` multitask solver.

    The change point score is calculated based on the second derivative of the
    sum of partial correlations derived from the precision matrices, faithfully
    mirroring the original implementation. Change points are emitted whenever
    the score reaches ``threshold``.

    Parameters
    ----------
    window_size : int, default=512
        Number of time steps per uGLAD window.
    stride : int, default=128
        Step between successive windows.
    batch_size : int, default=8
        Number of windows processed together by the multitask solver.
    threshold : float, default=0.5
        Minimum window score (absolute second difference, across windows, of
        the sum of partial correlations) that triggers a change point.
    min_spacing : int, optional
        Minimum distance (in samples) required between two emitted change
        points. Defaults to the stride when ``None``.
    epochs : int, default=2000
        Number of uGLAD training epochs per batch, as in ``main.py`` of the
        authors' repository. Fewer epochs run faster but can leave every
        partial correlation at zero, and then no change point is found (see
        Notes).
    learning_rate : float, default=0.001
        Optimiser learning rate.
    glad_iterations : int, default=5
        Number of unrolled GLAD iterations (``L`` in the paper).
    eval_offset : float, default=0.1
        Eigenvalue regularisation applied to batch covariance matrices.
    verbose : bool, default=False
        If true, uGLAD prints its progress on stdout for each batch: input
        checks, the loss every tenth of the epochs and the runtime. If false,
        nothing is printed.
    axis : int, default=0
        Time axis of ``X``. ``0`` assumes ``(n_timepoints, n_features)``.

    Notes
    -----
    Each batch of windows trains a new uGLAD model, and uGLAD scales each
    channel of each window to ``[0, 1]`` (min-max) before it estimates the
    precision matrix.

    - Training length: GLAD shrinks the off-diagonal entries of the
      precision matrix by a soft threshold that it learns. After too few
      epochs, this threshold can set every partial correlation of a window
      to zero; when it does so in every window, all summaries are 0 and no
      change point is found. With 100 epochs this happened on a two-channel
      series with two clear mean shifts, and on 96 to 100 % of the windows
      of the first three MoCap trials (four channels, default
      ``window_size``). 2000 epochs take about 20 times longer.
    - Several change points per transition: every window whose score
      reaches ``threshold`` gives a change point at its start, at least
      ``min_spacing`` apart, so the windows around one transition can give
      one each. The authors' code instead marks the ``window_size`` points
      on each side of each such window as a transition, so that the marked
      intervals merge.
    - Univariate series: there is no partial correlation, and the summary
      of a window falls back to the precision of its single channel (this
      fallback is not in the authors' code). The min-max scaling removes the
      scale of the window, so this precision does not follow the variance of
      the signal, only the shape of the distribution of the window's
      values. After 100 epochs it is far from converged and jumps from one
      batch to the next; after 2000 it varies from one window to the next,
      within a stationary segment too, by far more than the default
      ``threshold``, so that most windows give a change point.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "python_dependencies": ["torch", "networkx"],
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
        "non_deterministic": True,
    }

    _parameter_schema = {
        "window_size": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of time steps per uGLAD window.",
            group="windowing",
        ),
        "stride": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Step between successive windows.",
            group="windowing",
        ),
        "batch_size": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Windows processed together by the multitask solver.",
            group="training",
        ),
        "threshold": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Window score threshold for emitting a change point.",
        ),
        "min_spacing": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Minimum distance between emitted change points.",
            nullable=True,
        ),
        "epochs": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of uGLAD training epochs per batch.",
            group="training",
        ),
        "learning_rate": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Optimiser learning rate.",
            group="training",
        ),
        "glad_iterations": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of unrolled GLAD iterations.",
            group="training",
        ),
        "eval_offset": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Eigenvalue regularisation for batch covariance.",
        ),
        "verbose": ParamDef(
            constraint=HasType((bool,)),
            description="Print progress information.",
            ui_hidden=True,
        ),
        "_cross_constraints": [
            DependsOn(
                "stride <= window_size",
                "stride must be <= window_size",
            ),
            DataDependent(
                "window_size <= n_samples",
                "Series must have at least window_size samples",
            ),
        ],
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

        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # BaseSegmenter hooks
    # ------------------------------------------------------------------

    def _fit(self, X: np.ndarray, y=None):
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
        """Compute sum of upper triangle of partial correlation matrix.

        For univariate data (D=1) the partial-correlation matrix has no
        off-diagonal entries, so we fall back to the scalar precision value.
        uGLAD min-max scales each window, so this value does not track the
        variance of the signal, only the shape of the distribution of the
        window's values (see the Notes of the class).
        """
        d = precision.shape[0]

        if d == 1:
            # Univariate fallback, not in the authors' code: the precision of
            # the min-max scaled window, blind to a change of scale alone.
            return float(precision[0, 0])

        # Formula: rho_ij = -p_ij / sqrt(p_ii * p_jj)
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

    def _compute_precision_batch(
        self, windows: list[np.ndarray]
    ) -> list[_WindowSummary]:
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
                diff_next = W[idx + 1] - W[idx]
                diff_prev = W[idx] - W[idx - 1]
                summary.score = abs(diff_next - diff_prev)

        return summaries

    def _run_single_batch(self, batch: list[np.ndarray]) -> list[_WindowSummary]:
        ensure_vendor_imports()  # torch is optional: checked at construction
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
        return [
            _WindowSummary(start=0, stop=0, precision=precision, score=0.0)
            for precision in precisions
        ]

    def _scores_to_cps(
        self, summaries: list[_WindowSummary], n_samples: int
    ) -> np.ndarray:
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
        """Score of each window (0 for the first and the last)."""

        self._check_is_fitted()
        return np.asarray([summary.score for summary in self._window_summaries])
