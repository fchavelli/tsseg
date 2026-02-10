"""Legacy pyhsmm-backed HDP-HSMM detector kept for reference."""

from __future__ import annotations

import numpy as np

from ..base import BaseSegmenter

_HAS_PYHSMM = True
try:  # pragma: no cover - optional dependency
    import tsseg  # noqa: F401  # apply numpy compatibility patches
    from pyhsmm import distributions, models
    from pyhsmm.util.text import progprint_xrange
except ImportError:  # pragma: no cover - optional dependency missing
    _HAS_PYHSMM = False


class LegacyHdpHsmmDetector(BaseSegmenter):
    """Original implementation relying on :mod:`pyhsmm`.

    The class is retained for reference and comparison purposes. Instantiating
    it without the optional ``pyhsmm`` dependency will raise an informative
    error. New code should use :class:`~tsseg.algorithms.hdp_hsmm.detector.HdpHsmmDetector`.
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "state_detection",
    }

    def __init__(self, axis=0, n_segments=None, alpha=1.0, beta=1.0, n_iter=20, n_max_states=60):
        if not _HAS_PYHSMM:  # pragma: no cover - guard when dependency missing
            raise ImportError(
                "LegacyHdpHsmmDetector requires the optional 'pyhsmm' dependency. "
                "Please install pyhsmm or use HdpHsmmDetector instead."
            )
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.n_max_states = n_max_states
        self.n_segments = n_segments
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        X = self._prepare_signal(X)
        obs_dim = X.shape[1]
        obs_hypparams = {
            "mu_0": np.zeros(obs_dim),
            "sigma_0": np.eye(obs_dim),
            "kappa_0": 0.25,
            "nu_0": obs_dim + 2,
        }
        dur_hypparams = {"alpha_0": self.alpha, "beta_0": self.beta}

        obs_distns = [
            distributions.Gaussian(**obs_hypparams) for _ in range(self.n_max_states)
        ]
        dur_distns = [
            distributions.PoissonDuration(**dur_hypparams)
            for _ in range(self.n_max_states)
        ]

        self.model_ = models.WeakLimitHDPHSMM(
            alpha=6.0,
            gamma=6.0,
            init_state_concentration=6.0,
            obs_distns=obs_distns,
            dur_distns=dur_distns,
        )
        self.model_.add_data(X, trunc=600)

        for _ in progprint_xrange(self.n_iter):
            self.model_.resample_model()
        return self

    def _predict(self, X):
        X = self._prepare_signal(X)
        state_sequence = self.model_.stateseqs[0]
        if self.get_tag("returns_dense"):
            change_points = []
            for i in range(1, len(state_sequence)):
                if state_sequence[i] != state_sequence[i - 1]:
                    change_points.append(i)
            return np.array(change_points, dtype=int)
        return state_sequence

    def _prepare_signal(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 2 and self.axis == 1:
            X = X.T
        elif X.ndim != 2:
            raise ValueError(f"Input X must be 1D or 2D, got {X.ndim}D")
        return X
