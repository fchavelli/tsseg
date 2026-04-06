"""SDAR (Sequential Discounting AR) learning algorithm.

Implements the online discounting AR model from:
    Takeuchi & Yamanishi, "A Unifying Framework for Detecting Outliers and
    Change Points from Time Series", IEEE TKDE, 2006.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve, toeplitz


class SDAR:
    """Sequential Discounting AutoRegressive model.

    Learns an AR(k) model incrementally with exponential discounting so that
    older observations are gradually forgotten.  For each new observation the
    model produces a predicted mean and variance which can be used to compute
    an outlier score.

    Parameters
    ----------
    order : int
        AR order (*k* in the paper).
    discount : float
        Discounting rate *r* in (0, 1).  Larger values forget faster.
    """

    def __init__(self, order: int, discount: float) -> None:
        self.order = order
        self.discount = discount
        # State variables initialised on first call to ``update``.
        self._mu: float = 0.0
        self._C: np.ndarray | None = None      # autocovariances C_1..C_k
        self._C0: float = 1.0                   # variance C_0
        self._A: np.ndarray | None = None       # AR coefficients
        self._sigma: float = 1.0                # residual variance
        self._buf: np.ndarray | None = None     # ring buffer of last k values
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Initialisation from a batch of data (Yule-Walker)
    # ------------------------------------------------------------------

    def _init_from_batch(self, x: np.ndarray) -> None:
        """Bootstrap AR parameters via standard Yule-Walker equations."""
        k = self.order
        n = len(x)
        mu = float(np.mean(x))
        self._mu = mu

        centered = x - mu
        # Autocovariances C_0 .. C_k
        c = np.array(
            [np.dot(centered[: n - j], centered[j:]) / n for j in range(k + 1)]
        )
        self._C0 = float(c[0])
        self._C = c[1:].copy()  # C_1 .. C_k

        # Solve Toeplitz system for AR coefficients
        T = toeplitz(c[:k])
        # Regularise slightly to avoid singular systems
        T += np.eye(k) * 1e-10 * max(abs(self._C0), 1e-10)
        self._A = solve(T, self._C, assume_a="sym")

        self._sigma = max(self._C0 - float(np.dot(self._A, self._C)), 1e-10)

        # Initialise ring buffer with last k values
        self._buf = x[-k:].copy()
        self._initialised = True

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, x_new: float) -> tuple[float, float]:
        """Ingest a new observation and return (predicted_mean, predicted_var).

        Parameters
        ----------
        x_new : float
            New observation.

        Returns
        -------
        x_hat : float
            One-step-ahead prediction for *x_new* from the previous model.
        sigma : float
            Current residual variance estimate.
        """
        if not self._initialised:
            raise RuntimeError("Call _init_from_batch before update.")

        k = self.order
        r = self.discount

        # --- prediction with current model ---
        # buf stores the last k observations, newest last
        past = self._buf[::-1]  # x_{t-1}, x_{t-2}, ..., x_{t-k}
        x_hat = float(np.dot(self._A, past - self._mu)) + self._mu

        # --- discounted parameter update ---
        mu_new = (1.0 - r) * self._mu + r * x_new

        C_new = np.empty(k)
        for j in range(k):
            C_new[j] = (1.0 - r) * self._C[j] + r * (x_new - mu_new) * (
                self._buf[k - 1 - j] - mu_new
            )
        C0_new = (1.0 - r) * self._C0 + r * (x_new - mu_new) * (x_new - mu_new)

        # Solve updated Toeplitz system
        T = toeplitz(np.r_[C0_new, C_new[:-1]])
        T += np.eye(k) * 1e-10 * max(abs(C0_new), 1e-10)
        try:
            A_new = solve(T, C_new, assume_a="sym")
        except np.linalg.LinAlgError:
            A_new = self._A  # keep previous if singular

        # Updated residual variance (discounted)
        sigma_new = (1.0 - r) * self._sigma + r * (x_new - x_hat) ** 2
        sigma_new = max(sigma_new, 1e-10)

        # --- commit state ---
        self._mu = mu_new
        self._C = C_new
        self._C0 = C0_new
        self._A = A_new
        self._sigma = sigma_new

        # Shift buffer
        self._buf = np.roll(self._buf, -1)
        self._buf[-1] = x_new

        return x_hat, sigma_new

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

    def log_loss(self, x_new: float, x_hat: float, sigma: float) -> float:
        """Logarithmic loss (negative log-likelihood under Gaussian)."""
        return 0.5 * np.log(2 * np.pi * sigma) + 0.5 * (x_new - x_hat) ** 2 / sigma

    def quadratic_loss(self, x_new: float, x_hat: float) -> float:
        """Quadratic loss (squared prediction error)."""
        return (x_new - x_hat) ** 2
