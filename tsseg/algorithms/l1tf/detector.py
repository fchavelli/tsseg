"""l1 trend filtering change point detector (Kim, Koh, Boyd & Gorinevsky, 2009).

The detector minimises the penalised objective of the paper (eq. 5) on every
channel,

    minimize (1/2)||y - x||^2 + lambda ||D x||_1,

with D the second-order difference matrix, and reports the kinks (changes of
slope) of the estimated piecewise-linear trend as change points. The convex
problem is solved exactly on its dual (the box-constrained QP of the paper's
section 5.2) by OSQP with polishing, which recovers the exact KKT solution of
the identified active set. For multivariate series the penalty is the sum of
the per-channel l1 norms (paper section 7.5); the channels are then estimated
separately and the change points are the union of the channel kinks.
"""

from __future__ import annotations

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    HasType,
    Interval,
    ParamDef,
)

__all__ = ["TrendFilteringDetector"]


def _per_channel_dual_dynamics(n: int):
    """D (second differences) and H = D D^T as scipy sparse matrices."""
    from scipy import sparse

    D = sparse.diags(
        [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    return D, (D @ D.T).tocsc()


def _active_set_polish(H, Dy, lam, nu0, max_passes=12):
    """Exact active-set polish of the dual box-QP from a warm start nu0.

    Given the guessed active set (|nu| = lam), the free coordinates solve the
    reduced system (H nu)_free = (Dy)_free exactly, which forces beta = 0
    there; feasibility and the sign of beta along the bounds are then the
    only remaining KKT conditions, and any violated coordinate moves to the
    other side of the partition. H is positive definite, so a partition that
    satisfies all KKT conditions is the unique global optimum.
    Returns (nu, beta, exact) with exact=True when all KKT conditions hold.
    """
    from scipy import sparse
    from scipy.sparse import linalg as spla

    m = Dy.size
    lam = float(lam)
    nu = np.clip(np.asarray(nu0, dtype=float), -lam, lam)
    active = np.abs(nu) >= lam * (1 - 1e-6)
    exact = False
    for _ in range(max_passes):
        free = np.nonzero(~active)[0]
        bound = np.nonzero(active)[0]
        nu = np.zeros(m)
        # saturated coordinates sit exactly at the bound, with the sign of
        # the warm start; the reduced system solves the free block exactly
        nu[bound] = lam * np.sign(nu0[bound])
        if free.size:
            sub = sparse.csc_matrix(H[free][:, free])
            rhs = Dy[free] - H[free][:, bound] @ nu[bound]
            nu[free] = spla.spsolve(sub, rhs)
        beta = Dy - H @ nu
        scale = max(1.0, np.abs(Dy).max())
        beta_zero = np.abs(beta) <= 1e-9 * scale
        # KKT checks: free coordinates feasible ...
        bad_free = free[np.abs(nu[free]) >= lam]
        # ... saturated coordinates with the wrong beta sign move to free
        wrong_sign = bound[
            ~beta_zero[bound] & (np.sign(beta[bound]) != np.sign(nu[bound]))
        ]
        if bad_free.size == 0 and wrong_sign.size == 0:
            exact = True
            break
        active[bad_free] = True
        active[wrong_sign] = False
    return nu, beta, exact


def _solve_channel(D, H, y, lam, *, eps_abs, eps_rel, max_iter, polish):
    """Solve eq. (5) for one channel through its dual box-QP (paper eq. 13):

        minimize (1/2) nu^T H nu - nu^T D y   s.t.  |nu| <= lam

    x = y - D^T nu (paper eq. 12); beta = D x = Dy - H nu. OSQP identifies
    the active set and an exact active-set polish recovers the unique KKT
    solution; the true optimum satisfies ||(D D^T)^-1 D y||_inf <= lam, the
    banded unconstrained solve, as a fallback.
    """
    import osqp
    from scipy import sparse

    m = y.size - 2
    Dy = D @ y
    nu = _banded_solve_gram(Dy)
    if np.abs(nu).max() <= lam:
        # box constraint inactive: the unconstrained dual optimum is optimal
        beta = Dy - H @ nu
        x = y - (D.T @ nu)
        return x, beta, nu, "solved (above lambda_max)"

    model = osqp.OSQP()
    model.setup(
        P=H,
        q=-Dy,
        A=sparse.eye(m).tocsc(),
        l=-lam * np.ones(m),
        u=lam * np.ones(m),
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        polish=polish,
        verbose=False,
    )
    res = model.solve()
    status = res.info.status
    nu0 = np.clip(res.x, -lam, lam)
    nu, beta, exact = _active_set_polish(H, Dy, lam, nu0)
    if not exact:
        # one escalation with tighter tolerances before giving up
        model.update_settings(
            eps_abs=eps_abs * 1e-3, eps_rel=eps_rel * 1e-3, max_iter=max_iter * 4
        )
        res = model.solve()
        status = res.info.status
        nu0 = np.clip(res.x, -lam, lam)
        nu, beta, exact = _active_set_polish(H, Dy, lam, nu0)
        if not exact:
            raise RuntimeError(
                "failed to certify the KKT solution of the dual box-QP"
                f" (OSQP status: {status})"
            )
    x = y - (D.T @ nu)
    return x, beta, nu, status


class TrendFilteringDetector(BaseSegmenter):
    """l1 trend filtering change point detector.

    Estimates, on every channel, the piecewise-linear trend that minimises
    the penalised objective of Kim, Koh, Boyd and Gorinevsky (2009, eq. 5),

    .. math:: \\min_x \\; \\frac12\\|y-x\\|_2^2 + \\lambda\\|D x\\|_1,

    where :math:`D` is the second-order difference matrix. The kinks of the
    estimate (its nonzero second differences) are reported as change points:
    a piecewise-linear trend has an exact zero second difference between two
    kinks. The problem is convex but not smooth; it is solved through its
    dual, the box-constrained QP (paper, section 5.2, eq. 13),

    .. math:: \\min_\\nu \\; \\tfrac12\\nu^\\top D D^\\top \\nu -
              \\nu^\\top D y \\quad \\text{s.t. } |\\nu| \\le \\lambda,

    by OSQP with polishing (the KKT system of the identified active set is
    then solved exactly). The milder :data:`penalty`, the more change points;
    above :math:`\\lambda_{max} = \\|(D D^\\top)^{-1} D y\\|_\\infty` (paper,
    eq. 7, exposed as the attribute ``lambda_max_``) the trend estimate is the best
    affine fit and no change point is reported.

    Multivariate series use the "sum of l1 norms" extension of the paper
    (section 7.5): each channel is filtered separately with the same
    penalty, and the change points are the union of the channel kinks. The
    group-penalty variant of section 7.5 (common kinks across channels,
    sum of l2 norms) requires a conic solver and is not implemented.

    Parameters
    ----------
    penalty : float, default=1.0
        Smoothing parameter :math:`\\lambda > 0`: the l1 penalty on the
        second differences of the trend. The values of interest depend on
        the scale of the series (compare with the attribute ``lambda_max_``).
    knot_tol : float, default=1e-3
        A coordinate counts as a kink (change point) when its dual is
        saturated, ``|dual| >= (1 - knot_tol) * penalty``, and its second
        difference exceeds ``knot_tol`` times the largest second difference.
        The solver accuracy must be significantly tighter than ``knot_tol``.
    eps_abs, eps_rel : float, default=1e-9
        Absolute and relative tolerances of OSQP.
    max_iter : int, default=200000
        Maximum number of OSQP iterations.
    polish : bool, default=True
        Whether OSQP polishes its solution (solves the KKT system of the
        identified active set exactly). Keep it on: the polish is what makes
        the exact zeros of the second differences reliable.
    axis : int, default=0
        Axis of ``X`` that represents time.

    Attributes
    ----------
    change_points_ : np.ndarray
        Sorted indices (one per channel union) where a new segment starts,
        the kinks of the estimated trends.
    trend_ : np.ndarray
        The estimated piecewise-linear trends, same shape as the input.
    second_diff_ : np.ndarray
        The second differences of the estimated trends (exact zeros between
        kinks).
    lambda_max_ : float
        :math:`\\|(D D^\\top)^{-1} D y\\|_\\infty` (norm over channels):
        above this penalty the estimate would be the best affine fit.

    Notes
    -----
    The solver differs from the paper's specialized interior-point method
    (its section 6): same strictly convex problem, so the minimiser is the
    same; OSQP returns it up to its tolerances, and the polished second
    differences are exact zeros between kinks. The ``lambda_max_`` formula is
    the paper's eq. 7; the fitting bound ``||y - x||_inf <= 4 lambda`` is
    its property (6).

    References
    ----------
    .. [1] Kim, S.-J., Koh, K., Boyd, S. and Gorinevsky, D., "l1 Trend
       Filtering", SIAM Review 51(2):339-360, 2009.
    .. [2] Stellato, B. et al., "OSQP: an operator splitting solver for
       quadratic programs", Mathematical Programming Computation 12(4), 2020.

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms.l1tf.detector import TrendFilteringDetector
    >>> t = np.arange(500)
    >>> x = 2.0 + 0.01 * t + 0.05 * np.maximum(t - 250, 0) + 0.1 * np.sin(t)
    >>> detector = TrendFilteringDetector(penalty=5.0)
    >>> detector.fit(x)
    TrendFilteringDetector(penalty=5.0)
    >>> detector.predict(x)  # doctest: +SKIP
    array([250])
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
        "python_dependencies": ["osqp"],
    }

    _parameter_schema = {
        "penalty": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="l1 penalty on the second differences (lambda > 0).",
        ),
        "knot_tol": ParamDef(
            constraint=Interval(float, 0, 1, Closed.LEFT),
            description=(
                "Tolerance of the kink tests (dual saturation and relative "
                "second difference)."
            ),
        ),
        "eps_abs": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="OSQP absolute tolerance.",
        ),
        "eps_rel": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="OSQP relative tolerance.",
        ),
        "max_iter": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="OSQP iteration limit.",
        ),
        "polish": ParamDef(
            constraint=HasType(bool),
            description="Whether OSQP polishes the solution.",
        ),
        "_cross_constraints": [
            DataDependent(
                "n_samples >= 3",
                "Series must have at least 3 samples "
                "(the second difference needs them).",
            ),
        ],
    }

    def __init__(
        self,
        *,
        penalty: float = 1.0,
        knot_tol: float = 1e-3,
        eps_abs: float = 1e-9,
        eps_rel: float = 1e-9,
        max_iter: int = 200000,
        polish: bool = True,
        axis: int = 0,
    ) -> None:
        self.penalty = float(penalty)
        self.knot_tol = float(knot_tol)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.max_iter = int(max_iter)
        self.polish = bool(polish)
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        # nothing to learn before predict: the engine works on the data
        return self

    def _predict(self, X):
        signal = np.asarray(X, dtype=float)
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]

        n = signal.shape[0]
        D, H = _per_channel_dual_dynamics(n)

        trend = np.empty_like(signal)
        second_diff = np.empty((signal.shape[0] - 2, signal.shape[1]))
        duals = np.empty_like(second_diff)
        for c in range(signal.shape[1]):
            y_c = signal[:, c]
            lam_c = _lambda_max_channel(y_c)
            if self.penalty > lam_c:
                # paper eq. (7): above lambda_max the box constraint is
                # inactive, the dual optimum is the exact banded solve and
                # the trend is the best (least squares) affine fit; the QP
                # solver would only grind on this regime.
                nu_c = _banded_solve_gram(D @ y_c)
                status_c = "solved (above lambda_max)"
            else:
                x_c, beta_c, nu_c, status_c = _solve_channel(
                    D,
                    H,
                    y_c,
                    self.penalty,
                    eps_abs=self.eps_abs,
                    eps_rel=self.eps_rel,
                    max_iter=self.max_iter,
                    polish=self.polish,
                )
            duals[:, c] = nu_c
            trend[:, c] = y_c - (D.T @ nu_c)
            second_diff[:, c] = D @ trend[:, c]

        # Union of the channel kinks. By the optimality conditions a kink is a
        # coordinate whose dual is saturated (|nu_t| = lambda, paper eq. 12)
        # AND whose second difference is nonzero; both tests are needed
        # because degenerate coordinate can touch the box boundary with an
        # exactly zero second difference (observed, confirmed by CVXPY).
        sat = np.abs(duals) >= (1.0 - self.knot_tol) * self.penalty
        beta_scale = float(np.abs(second_diff[sat]).max()) if sat.any() else 0.0
        if beta_scale > 0:
            kink = sat & (np.abs(second_diff) > self.knot_tol * beta_scale)
            # row t of the second difference carries its kink at index t + 1
            change_points = np.unique(np.nonzero(kink)[0] + 1)
        else:
            change_points = np.zeros(0, dtype=int)

        self.trend_ = trend
        self.second_diff_ = second_diff
        self.change_points_ = change_points
        # paper eq. 7, per channel: ||(D D^T)^-1 D y||_inf
        self.lambda_max_ = float(
            max(_lambda_max_channel(signal[:, c]) for c in range(signal.shape[1]))
        )
        return self.change_points_


def _banded_solve_gram(rhs):
    """Solve (D D^T) z = rhs: 5-diagonal Toeplitz system with bands 6, -4, 1."""
    from scipy.linalg import solveh_banded

    m = rhs.size
    ab = np.zeros((3, m))
    ab[0] = 6.0
    ab[1, : m - 1] = -4.0
    ab[2, : m - 2] = 1.0
    return solveh_banded(ab, rhs, lower=True)


def _lambda_max_channel(y):
    """lambda_max of the paper (eq. 7): ||(D D^T)^-1 D y||_inf.

    D D^T is the 5-diagonal Toeplitz matrix with bands 6, -4, 1 (every row
    of the second-difference matrix is [1 -2 1]), so the solve is banded.
    """
    Dy = y[:-2] - 2.0 * y[1:-1] + y[2:]
    w = _banded_solve_gram(Dy)
    return float(np.abs(w).max())
