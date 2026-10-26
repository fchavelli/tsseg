# Adapted from https://github.com/davidhallac/TICC (BSD 2-Clause, see LICENSE).
#
# Changes from the original solver, none of which changes its output:
# - the random draws use a local ``RandomState(102)`` instead of reseeding
#   NumPy's global generator, and the global print options of NumPy and pandas
#   are left untouched;
# - the solver runs in the calling process when ``num_proc == 1``;
# - the Z-update of the ADMM, the log-likelihoods of the assignment step and
#   its dynamic programme are vectorised over the clusters;
# - the confusion matrices, F1 scores and k-means clustering that the original
#   computed at each iteration without using them are removed.
# ``log_likelihood`` and ``n_free_parameters`` are additions.
"""Core solver of TICC (Toeplitz inverse covariance-based clustering)."""

from __future__ import annotations

import collections
import math
from multiprocessing import Pool

import numpy as np
from sklearn import mixture


def getTrainTestSplit(m, num_blocks, num_stacked, rng=None):
    """Compute the training indices for the TICC model.

    The training fraction is fixed at 1: with ``num_blocks = num_stacked + 1``,
    as TICC calls it, the indices are all of ``0, ..., m - num_stacked``, and
    the random draw only advances ``rng``.

    Parameters
    ----------
    m : int
        Number of observations.
    num_blocks : int
        ``window_size + 1``.
    num_stacked : int
        ``window_size``.
    rng : numpy.random.RandomState or None, default=None
        Random generator; ``None`` uses NumPy's global one.

    Returns
    -------
    list[int]
        Sorted list of training indices.
    """
    choice = np.random.choice if rng is None else rng.choice
    # Now splitting up stuff
    # split1 : Training and Test
    # split2 : Training and Test - different clusters
    training_percent = 1
    # list of training indices
    training_idx = choice(
        m - num_blocks + 1,
        size=int((m - num_stacked) * training_percent),
        replace=False,
    )
    # Ensure that the first and the last few points are in
    training_idx = list(training_idx)
    if 0 not in training_idx:
        training_idx.append(0)
    if m - num_stacked not in training_idx:
        training_idx.append(m - num_stacked)
    training_idx = np.array(training_idx)
    return sorted(training_idx)


def upperToFull(a, eps=0):
    """Symmetric matrix from the vector of its upper triangle.

    Parameters
    ----------
    a : np.ndarray
        Entries on and above the diagonal of a symmetric ``n x n`` matrix,
        row by row (the order of ``np.triu_indices(n)``), of length
        ``n * (n + 1) / 2``. Modified in place by the thresholding.
    eps : float, default=0
        Entries of absolute value below ``eps`` are set to 0.

    Returns
    -------
    np.ndarray
        The symmetric matrix, of shape ``(n, n)``.
    """
    ind = (a < eps) & (a > -eps)
    a[ind] = 0
    n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
    A = np.zeros([n, n])
    A[np.triu_indices(n)] = a
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))
    return A


def updateClusters(LLE_node_vals, switch_penalty=1):
    """Assignment step: the sequence of states of least total cost.

    Dynamic programming (Problem (3) of Hallac et al., 2017): a backward pass
    computes, for each window and state, the least cost of the windows that
    follow; a forward pass then picks the states. Ties go to the smallest
    state index.

    Parameters
    ----------
    LLE_node_vals : np.ndarray
        Cost of each window under each state, of shape
        ``(n_windows, n_states)``. TICC passes twice the negative
        log-likelihood up to a constant (see ``TICC.smoothen_clusters``),
        not the log-likelihood.
    switch_penalty : float, default=1
        Cost added each time two consecutive windows are in different
        states.

    Returns
    -------
    np.ndarray
        State of each window, of shape ``(n_windows,)``, as floats.
    """
    (T, num_clusters) = LLE_node_vals.shape
    future_cost_vals = np.zeros(LLE_node_vals.shape)
    stay = np.eye(num_clusters, dtype=bool)

    # compute future costs: row ``cluster`` of ``total_vals`` is the cost of
    # each next cluster, minus the switch penalty for staying in ``cluster``
    for i in range(T - 2, -1, -1):
        j = i + 1
        switch_vals = future_cost_vals[j, :] + LLE_node_vals[j, :] + switch_penalty
        stay_vals = switch_vals - switch_penalty
        total_vals = np.where(stay, stay_vals[:, None], switch_vals[None, :])
        future_cost_vals[i, :] = total_vals.min(axis=1)

    # compute the best path
    path = np.zeros(T)

    # the first location
    curr_location = np.argmin(future_cost_vals[0, :] + LLE_node_vals[0, :])
    path[0] = curr_location

    # compute the path
    for i in range(T - 1):
        j = i + 1
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        total_vals = future_costs + lle_vals + switch_penalty
        total_vals[int(path[i])] -= switch_penalty

        path[i + 1] = np.argmin(total_vals)

    # return the computed path
    return path


def computeBIC(K, T, clustered_points, inverse_covariances, empirical_covariances):
    """BIC of the reference implementation, not used by ``TiccDetector``.

    Parameters
    ----------
    K : int
        Number of states; unused.
    T : int
        Number of time points.
    clustered_points : array-like
        State of each window.
    inverse_covariances : dict
        Precision matrix :math:`\\Theta` of each state.
    empirical_covariances : dict
        Empirical covariance :math:`S` of the windows of each state.

    Returns
    -------
    float
        :math:`k \\log T - 2 L`, where :math:`L` sums
        :math:`\\log\\det\\Theta - \\mathrm{tr}(S\\Theta)` over the states of
        ``inverse_covariances``, without weighting them by their numbers of
        windows, and :math:`k` adds the number of entries of :math:`\\Theta`
        above ``2e-5`` in absolute value once for each run of consecutive
        windows in the same state.
    """
    mod_lle = 0

    threshold = 2e-5
    clusterParams = {}
    for cluster, clusterInverse in inverse_covariances.items():
        mod_lle += np.log(np.linalg.det(clusterInverse)) - np.trace(
            np.dot(empirical_covariances[cluster], clusterInverse)
        )
        clusterParams[cluster] = np.sum(np.abs(clusterInverse) > threshold)
    curr_val = -1
    non_zero_params = 0
    for val in clustered_points:
        if val != curr_val:
            non_zero_params += clusterParams[val]
            curr_val = val
    return non_zero_params * np.log(T) - 2 * mod_lle


def n_free_parameters(theta, window_size, n_channels, threshold=2e-5):
    """Number of free entries of a block Toeplitz precision above a threshold.

    A symmetric block Toeplitz matrix is defined by its diagonal block
    :math:`A^{(0)}` (symmetric, :math:`d(d+1)/2` free entries) and the blocks
    :math:`A^{(1)}, \\dots, A^{(w-1)}` below it (:math:`d^2` each). The ADMM
    returns the dense variable of the splitting, whose entries are never
    exactly zero: an entry counts when its absolute value exceeds
    ``threshold`` (the threshold of ``computeBIC``).

    Parameters
    ----------
    theta : np.ndarray
        Precision matrix of shape ``(window_size * n_channels,) * 2``.
    window_size : int
        Number of blocks :math:`w`.
    n_channels : int
        Size :math:`d` of each block.
    threshold : float, default=2e-5
        Entries whose absolute value is at most ``threshold`` count as zero.

    Returns
    -------
    int
        Number of free entries above the threshold.
    """
    d = n_channels
    nonzero = np.abs(theta[:, :d]) > threshold
    count = int(np.triu(nonzero[:d]).sum())
    return count + int(nonzero[d : window_size * d].sum())


class _Result:
    """Result of a call run in the calling process (``Pool.apply_async`` API)."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _SerialPool:
    """Stand-in for ``multiprocessing.Pool`` that runs each call immediately."""

    def apply_async(self, func, args=()):
        return _Result(func(*args))

    def close(self):
        pass

    def join(self):
        pass


class ADMMSolver:
    """ADMM for the Toeplitz graphical lasso, the update step of TICC.

    Solves

    .. math::

        \\min_{\\Theta} \\; -\\log\\det\\Theta + \\mathrm{tr}(S\\Theta)
        + \\|\\lambda \\circ \\Theta\\|_1

    over the symmetric block Toeplitz matrices :math:`\\Theta` (Problem (4)
    of Hallac et al., 2017, with the factor :math:`1/|P_i|` absorbed into
    :math:`\\lambda`). The variable ``x`` takes the log-determinant step, its
    copy ``z`` the block Toeplitz constraint and the :math:`\\ell_1` penalty,
    and ``u`` is the scaled dual variable. All three hold the entries on and
    above the diagonal, row by row.

    Parameters
    ----------
    lamb : np.ndarray
        Penalty weights :math:`\\lambda`, of shape ``(p, p)`` with
        ``p = num_stacked * size_blocks``.
    num_stacked : int
        Number of blocks :math:`w` (the window size).
    size_blocks : int
        Size :math:`d` of each block (the number of channels).
    rho : float
        Penalty parameter :math:`\\rho` of the augmented Lagrangian.
    S : np.ndarray
        Empirical covariance, of shape ``(p, p)``.
    rho_update_func : callable or None, default=None
        Called after each iteration from the second on, as
        ``rho_update_func(rho, res_pri, e_pri, res_dual, e_dual)``, to give a
        new :math:`\\rho`. As in the reference code, only ``u`` is rescaled
        by ``rho / new_rho``: ``rho`` itself keeps its value. TICC passes
        ``None``.

    Attributes
    ----------
    x, z, u : np.ndarray
        The ADMM variables, of length ``p * (p + 1) / 2``.
    status : str
        ``"Optimal"`` when the stopping criterion was met,
        ``"Incomplete: max iterations reached"`` otherwise.
    """

    def __init__(self, lamb, num_stacked, size_blocks, rho, S, rho_update_func=None):
        self.lamb = lamb
        self.numBlocks = num_stacked
        self.sizeBlocks = size_blocks
        probSize = num_stacked * size_blocks
        self.length = int(probSize * (probSize + 1) / 2)
        self.x = np.zeros(self.length)
        self.z = np.zeros(self.length)
        self.u = np.zeros(self.length)
        self.rho = float(rho)
        self.S = S
        self.status = "initialized"
        self.rho_update_func = rho_update_func
        self._triu = np.triu_indices(probSize)
        self._triu_size = probSize
        self._build_z_groups()

    def ij2symmetric(self, i, j, size):
        """Position of entry ``(i, j)``, ``i <= j``, in the upper triangle.

        Parameters
        ----------
        i, j : int
            Row and column of the entry, with ``i <= j``.
        size : int
            Size of the square matrix.

        Returns
        -------
        float
            Index of the entry in the vector of the entries on and above the
            diagonal, row by row.
        """
        return (size * (size + 1)) / 2 - (size - i) * (size - i + 1) / 2 + j - i

    def _build_z_groups(self):
        """Index the entries of ``z`` that the Z-update sets to a common value.

        Each group gathers the occurrences of one free entry of the block
        Toeplitz matrix (one entry of :math:`A^{(0)}` on or above the diagonal,
        or of a block :math:`A^{(m)}` below it), in the order in which the
        original loop of the Z-update visits them. The sums of ``lamb`` over each
        group do not change between iterations and are computed once.
        """
        probSize = self.numBlocks * self.sizeBlocks
        entries, groups, lam_sums, elems_list = [], [], [], []
        for i in range(self.numBlocks):
            elems = (
                self.numBlocks if i == 0 else (2 * self.numBlocks - 2 * i) / 2
            )  # i=0 is diagonal
            for j in range(self.sizeBlocks):
                startPoint = j if i == 0 else 0
                for k in range(startPoint, self.sizeBlocks):
                    locList = [
                        ((l + i) * self.sizeBlocks + j, l * self.sizeBlocks + k)
                        for l in range(int(elems))
                    ]
                    if i == 0:
                        lamSum = sum(self.lamb[loc1, loc2] for (loc1, loc2) in locList)
                        indices = [
                            self.ij2symmetric(loc1, loc2, probSize)
                            for (loc1, loc2) in locList
                        ]
                    else:
                        lamSum = sum(self.lamb[loc2, loc1] for (loc1, loc2) in locList)
                        indices = [
                            self.ij2symmetric(loc2, loc1, probSize)
                            for (loc1, loc2) in locList
                        ]
                    groups.extend([len(lam_sums)] * len(indices))
                    entries.extend(int(index) for index in indices)
                    lam_sums.append(lamSum)
                    elems_list.append(elems)
        self._z_entries = np.asarray(entries, dtype=np.intp)
        self._z_groups = np.asarray(groups, dtype=np.intp)
        self._z_lam_sums = np.asarray(lam_sums, dtype=float)
        self._z_elems = np.asarray(elems_list, dtype=float)

    def upper2Full(self, a):
        """Symmetric matrix from the vector of its upper triangle.

        Parameters
        ----------
        a : np.ndarray
            Entries on and above the diagonal, row by row.

        Returns
        -------
        np.ndarray
            The symmetric matrix.
        """
        n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
        A = np.zeros([n, n])
        A[self._triu if n == self._triu_size else np.triu_indices(n)] = a
        temp = A.diagonal()
        A = (A + A.T) - np.diag(temp)
        return A

    def Prox_logdet(self, S, A, eta):
        """Log-determinant step of the ADMM, in closed form.

        Minimises
        :math:`-\\log\\det X + \\mathrm{tr}(SX) + \\frac{\\eta}{2}\\|X - A\\|_F^2`
        from the eigendecomposition :math:`Q D Q^\\top` of
        :math:`\\eta A - S`:
        :math:`X = Q \\, \\frac{D + \\sqrt{D^2 + 4\\eta}}{2\\eta} \\, Q^\\top`.

        Parameters
        ----------
        S : np.ndarray
            Empirical covariance, of shape ``(p, p)``.
        A : np.ndarray
            Symmetric matrix of shape ``(p, p)``, here ``z - u``.
        eta : float
            Weight of the quadratic term, here :math:`\\rho`.

        Returns
        -------
        np.ndarray
            Entries of :math:`X` on and above the diagonal, row by row.
        """
        d, q = np.linalg.eigh(eta * A - S)
        X_var = (
            ((1 / (2 * float(eta))) * q)
            @ np.diag(d + np.sqrt(np.square(d) + (4 * eta) * np.ones(d.shape)))
        ) @ q.T
        # extract upper triangular part as update variable
        n = S.shape[1]
        return X_var[self._triu if n == self._triu_size else np.triu_indices(n)]

    def ADMM_x(self):
        """Update ``x``: log-determinant step at ``z - u``."""
        a = self.z - self.u
        A = self.upper2Full(a)
        eta = self.rho
        self.x = self.Prox_logdet(self.S, A, eta)

    def ADMM_z(self, index_penalty=1):
        """Soft-threshold the mean of each group of tied entries of ``x + u``.

        Each free entry of the block Toeplitz matrix gets the mean of its
        occurrences in ``x + u``, soft-thresholded at the mean of their
        weights in ``lamb`` divided by ``rho``; ``z`` repeats it at every
        occurrence.

        Parameters
        ----------
        index_penalty : int, default=1
            Unused.
        """
        a = self.x + self.u
        n_groups = self._z_lam_sums.size
        # np.bincount adds the weights in their order, from 0.0, as the
        # original loop did with ``sum``
        pointSum = np.bincount(
            self._z_groups, weights=a[self._z_entries], minlength=n_groups
        )
        rhoPointSum = self.rho * pointSum
        lamSum = self._z_lam_sums
        denom = self.rho * self._z_elems

        # Calculate soft threshold
        ans = np.zeros(n_groups)
        # If answer is positive
        positive = rhoPointSum > lamSum
        negative = ~positive & (rhoPointSum < -1 * lamSum)
        ans[positive] = np.maximum(
            (rhoPointSum[positive] - lamSum[positive]) / denom[positive], 0
        )
        ans[negative] = np.minimum(
            (rhoPointSum[negative] + lamSum[negative]) / denom[negative], 0
        )

        z_update = np.zeros(self.length)
        z_update[self._z_entries] = ans[self._z_groups]
        self.z = z_update

    def ADMM_u(self):
        """Update the scaled dual variable: ``u + x - z``."""
        u_update = self.u + self.x - self.z
        self.u = u_update

    def CheckConvergence(self, z_old, e_abs, e_rel, verbose):
        """Stopping criterion of the ADMM, from the primal and dual residuals.

        The primal residual is :math:`\\|x - z\\|`, the dual residual
        :math:`\\|\\rho (z - z_{old})\\|`; their thresholds are
        :math:`\\sqrt{L} e_{abs} + e_{rel} \\max(\\|x\\|, \\|z\\|) + 10^{-4}`
        and :math:`\\sqrt{L} e_{abs} + e_{rel} \\|\\rho u\\| + 10^{-4}`, with
        :math:`L` the length of the variables.

        Parameters
        ----------
        z_old : np.ndarray
            ``z`` before the current iteration.
        e_abs, e_rel : float
            Absolute and relative tolerances.
        verbose : bool
            Whether to print the residuals and thresholds.

        Returns
        -------
        tuple
            ``(stop, res_pri, e_pri, res_dual, e_dual)``: whether both
            residuals are at most their thresholds, then the primal residual
            and threshold, and the dual residual and threshold.
        """
        norm = np.linalg.norm
        r = self.x - self.z
        s = self.rho * (self.z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = (
            math.sqrt(self.length) * e_abs
            + e_rel * max(norm(self.x), norm(self.z))
            + 0.0001
        )
        e_dual = (
            math.sqrt(self.length) * e_abs + e_rel * norm(self.rho * self.u) + 0.0001
        )
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        if verbose:
            # Debugging information to print(convergence criteria values)
            print("  r:", res_pri)
            print("  e_pri:", e_pri)
            print("  s:", res_dual)
            print("  e_dual:", e_dual)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)

    def __call__(self, maxIters, eps_abs, eps_rel, verbose):
        """Run the ADMM.

        Parameters
        ----------
        maxIters : int
            Maximum number of iterations.
        eps_abs, eps_rel : float
            Tolerances of ``CheckConvergence``, which runs from the second
            iteration on.
        verbose : bool
            Whether to print the iteration numbers and the residuals.

        Returns
        -------
        np.ndarray
            ``x``, the entries of the estimated precision matrix on and above
            the diagonal, row by row. It is positive definite, but only
            approximately block Toeplitz and sparse: in general, none of its
            entries is exactly zero.
        """
        self.status = "Incomplete: max iterations reached"
        for i in range(maxIters):
            z_old = np.copy(self.z)
            self.ADMM_x()
            self.ADMM_z()
            self.ADMM_u()
            if i != 0:
                stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(
                    z_old, eps_abs, eps_rel, verbose
                )
                if stop:
                    self.status = "Optimal"
                    break
                new_rho = self.rho
                if self.rho_update_func:
                    new_rho = self.rho_update_func(
                        self.rho, res_pri, e_pri, res_dual, e_dual
                    )
                scale = self.rho / new_rho
                self.u = scale * self.u
            if verbose:
                # Debugging information prints current iteration #
                print("Iteration %d" % i)
        return self.x


class TICC:
    """TICC solver: alternate update and assignment steps on stacked windows.

    Parameters
    ----------
    window_size : int, default=10
        Number :math:`w` of consecutive time points stacked into each window.
    number_of_clusters : int, default=5
        Number of states.
    lambda_parameter : float, default=11e-2
        Weight of the :math:`\\ell_1` penalty on every entry of the precision
        matrix of each state, diagonal included.
    beta : float, default=400
        Switching penalty of the assignment step, stored as
        ``switch_penalty``.
    maxIters : int, default=1000
        Maximum number of iterations; ``fit_transform`` raises ``ValueError``
        if it is not positive.
    threshold : float, default=2e-5
        Stored, never read.
    write_out_file : bool, default=False
        Unused, kept for compatibility.
    prefix_string : str, default=""
        Unused, kept for compatibility.
    num_proc : int, default=1
        Number of processes of the pool that solves the update steps; ``1``
        solves them in the calling process.
    compute_BIC : bool, default=False
        Whether ``fit_transform`` also returns the value of ``computeBIC``.
    cluster_reassignment : int, default=20
        Number of consecutive windows moved to a state left without windows.
    biased : bool, default=False
        Whether the empirical covariances divide by the number of windows of
        the state (``True``) or by that number minus one (``False``).

    Attributes
    ----------
    clustered_points_ : np.ndarray
        State of each window after the last iteration, set by
        ``fit_transform``.
    train_cluster_inverse_ : dict
        Precision matrix of each state, set by ``fit_transform``.
    trained_model : dict
        Means, covariances and stacked windows read by ``predict_clusters``
        and ``log_likelihood``.
    """

    def __init__(
        self,
        window_size=10,
        number_of_clusters=5,
        lambda_parameter=11e-2,
        beta=400,
        maxIters=1000,
        threshold=2e-5,
        write_out_file=False,
        prefix_string="",
        num_proc=1,
        compute_BIC=False,
        cluster_reassignment=20,
        biased=False,
    ):
        self.window_size = window_size
        self.number_of_clusters = number_of_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.maxIters = maxIters
        self.threshold = threshold
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.num_proc = num_proc
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.biased = biased

    def fit_transform(self, series):
        """Fit the model and return the state of each window.

        Parameters
        ----------
        series : np.ndarray
            The time series, of shape ``(n_timepoints, n_channels)``.

        Returns
        -------
        clustered_points : np.ndarray
            State of each of the ``n_timepoints - window_size + 1`` windows,
            as floats. Window ``t`` holds the points ``t`` to
            ``t + window_size - 1``; the last ``window_size - 1`` windows are
            padded with zeros and keep the state of the last full window when
            ``beta > 0``.
        train_cluster_inverse : dict
            Precision matrix of each state, of shape
            ``(window_size * n_channels,) * 2``.
        bic : float
            Only when ``compute_BIC`` is true: the value of ``computeBIC``.

        Raises
        ------
        ValueError
            If ``maxIters`` is not positive.
        KeyError
            If the initial Gaussian mixture gives a state fewer than two
            windows.
        """
        if self.maxIters <= 0:
            raise ValueError(f"maxIters must be > 0, got {self.maxIters}")
        # The seed of the original code, which reseeded NumPy's global generator
        rng = np.random.RandomState(102)

        # Get data into proper format
        times_series_arr = series
        time_series_rows_size, time_series_col_size = times_series_arr.shape

        # Train test split
        training_indices = getTrainTestSplit(
            time_series_rows_size, self.num_blocks, self.window_size, rng
        )  # indices of the training samples
        num_train_points = len(training_indices)

        # Stack the training data
        complete_D_train = self.stack_training_data(
            times_series_arr, time_series_col_size, num_train_points, training_indices
        )

        # Initialization
        # Gaussian Mixture
        gmm = mixture.GaussianMixture(
            n_components=self.number_of_clusters,
            covariance_type="full",
            random_state=rng,
        )
        gmm.fit(complete_D_train)
        clustered_points = gmm.predict(complete_D_train)

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = Pool(processes=self.num_proc) if self.num_proc > 1 else _SerialPool()
        for iters in range(self.maxIters):
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(
                list
            )  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {
                k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)
            }

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(
                cluster_mean_info,
                cluster_mean_stacked_info,
                complete_D_train,
                empirical_covariances,
                len_train_clusters,
                time_series_col_size,
                pool,
                train_clusters_arr,
            )

            self.optimize_clusters(
                computed_covariance,
                len_train_clusters,
                log_det_values,
                opt_res,
                train_cluster_inverse,
            )

            # update old computed covariance
            old_computed_covariance = computed_covariance

            self.trained_model = {
                "cluster_mean_info": cluster_mean_info,
                "computed_covariance": computed_covariance,
                "cluster_mean_stacked_info": cluster_mean_stacked_info,
                "complete_D_train": complete_D_train,
                "time_series_col_size": time_series_col_size,
            }
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(
                list
            )  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {
                k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)
            }

            before_empty_cluster_assign = clustered_points.copy()

            if iters != 0:
                cluster_norms = [
                    (
                        np.linalg.norm(
                            old_computed_covariance[self.number_of_clusters, i]
                        ),
                        i,
                    )
                    for i in range(self.number_of_clusters)
                ]
                norms_sorted = sorted(cluster_norms, reverse=True)
                # clusters that are not 0 as sorted by norm
                valid_clusters = [
                    cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0
                ]

                # Add a point to the empty clusters
                # assuming more non empty clusters than empty ones
                counter = 0
                for cluster_num in range(self.number_of_clusters):
                    if len_new_train_clusters[cluster_num] == 0:
                        cluster_selected = valid_clusters[
                            counter
                        ]  # a cluster that is not len 0
                        counter = (counter + 1) % len(valid_clusters)
                        start_point = rng.choice(
                            new_train_clusters[cluster_selected]
                        )  # random point number from that cluster
                        for i in range(0, self.cluster_reassignment):
                            # put cluster_reassignment points from point_num in this cluster
                            point_to_move = start_point + i
                            if point_to_move >= len(clustered_points):
                                break
                            clustered_points[point_to_move] = cluster_num
                            computed_covariance[
                                self.number_of_clusters, cluster_num
                            ] = old_computed_covariance[
                                self.number_of_clusters, cluster_selected
                            ]
                            cluster_mean_stacked_info[
                                self.number_of_clusters, cluster_num
                            ] = complete_D_train[point_to_move, :]
                            cluster_mean_info[self.number_of_clusters, cluster_num] = (
                                complete_D_train[point_to_move, :][
                                    (self.window_size - 1)
                                    * time_series_col_size : self.window_size
                                    * time_series_col_size
                                ]
                            )

            if np.array_equal(old_clustered_points, clustered_points):
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training
        pool.close()
        pool.join()

        self.clustered_points_ = clustered_points
        self.train_cluster_inverse_ = train_cluster_inverse
        if self.compute_BIC:
            bic = computeBIC(
                self.number_of_clusters,
                time_series_rows_size,
                clustered_points,
                train_cluster_inverse,
                empirical_covariances,
            )
            return clustered_points, train_cluster_inverse, bic

        return clustered_points, train_cluster_inverse

    def log_likelihood(self):
        """Gaussian log-likelihood of the stacked windows under the fitted model.

        Sum over the windows of Eq. (2) of the paper, each window under the
        precision matrix and the mean of the cluster it is assigned to after
        the last iteration. The last ``window_size - 1`` windows, which the
        solver pads with zeros and leaves out of the assignment step, are left
        out.

        Returns
        -------
        float
            The log-likelihood.
        """
        D = self.trained_model["complete_D_train"]
        n_windows, dim = D.shape
        valid = n_windows - (self.window_size - 1)
        labels = np.asarray(self.clustered_points_[:valid], dtype=int)
        total = 0.0
        for cluster in np.unique(labels):
            theta = self.train_cluster_inverse_[cluster]
            mean = self.trained_model["cluster_mean_stacked_info"][
                self.number_of_clusters, cluster
            ]
            xc = D[:valid][labels == cluster] - mean
            quad = np.einsum("ij,ij->i", xc @ theta, xc)
            _, logdet = np.linalg.slogdet(theta)
            total += float(
                np.sum(-0.5 * quad + 0.5 * logdet - 0.5 * dim * np.log(2 * np.pi))
            )
        return total

    def smoothen_clusters(
        self,
        cluster_mean_info,
        computed_covariance,
        cluster_mean_stacked_info,
        complete_D_train,
        n,
    ):
        """Cost of each window under each state, for the assignment step.

        Parameters
        ----------
        cluster_mean_info : dict
            Unused.
        computed_covariance : dict
            Covariance :math:`\\Sigma = \\Theta^{-1}` of each state, keyed by
            ``(number_of_clusters, state)``.
        cluster_mean_stacked_info : dict
            Mean window :math:`\\mu` of each state, with the same keys.
        complete_D_train : np.ndarray
            Stacked windows, of shape ``(n_windows, window_size * n)``.
        n : int
            Number of channels.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_windows, number_of_clusters)`` holding
            :math:`(X_t - \\mu)^\\top \\Sigma^{-1} (X_t - \\mu) + \\log\\det\\Sigma`,
            twice the negative Gaussian log-likelihood of window
            :math:`X_t` up to a constant; 0 for the last ``window_size - 1``
            windows, which are padded with zeros.

        Raises
        ------
        KeyError
            If a state has no covariance yet, i.e. if it had fewer than two
            windows at the first iteration.
        """
        clustered_points_len = len(complete_D_train)
        # For each point compute the LLE; the last window_size - 1 points keep 0
        valid = max(clustered_points_len - (self.window_size - 1), 0)
        LLE_all_points_clusters = np.zeros(
            [clustered_points_len, self.number_of_clusters]
        )
        for cluster in range(self.number_of_clusters):
            cov_matrix = computed_covariance[self.number_of_clusters, cluster][
                0 : (self.num_blocks - 1) * n, 0 : (self.num_blocks - 1) * n
            ]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            cluster_mean_stacked = cluster_mean_stacked_info[
                self.number_of_clusters, cluster
            ]
            x = (
                complete_D_train[:valid]
                - cluster_mean_stacked[0 : (self.num_blocks - 1) * n]
            )
            # row t: x_t^T (inv_cov_matrix x_t)
            lle = np.einsum("ij,ij->i", x @ inv_cov_matrix.T, x) + log_det_cov
            LLE_all_points_clusters[:valid, cluster] = lle

        return LLE_all_points_clusters

    def optimize_clusters(
        self,
        computed_covariance,
        len_train_clusters,
        log_det_values,
        optRes,
        train_cluster_inverse,
    ):
        """Store the precision and covariance matrices computed by the ADMM.

        For each state whose update step ran, the vector returned by the ADMM
        becomes the precision matrix ``train_cluster_inverse[state]``, its
        inverse the covariance ``computed_covariance[number_of_clusters,
        state]`` and the log-determinant of that covariance
        ``log_det_values[number_of_clusters, state]``. States skipped by
        ``train_clusters`` keep their previous values.

        Parameters
        ----------
        computed_covariance : dict
            Covariance of each state; updated in place.
        len_train_clusters : dict
            Unused.
        log_det_values : dict
            Log-determinant of the covariance of each state; updated in place.
        optRes : list
            Output of ``train_clusters``: for each state, the result of its
            solver (an object with a ``get`` method), or ``None``.
        train_cluster_inverse : dict
            Precision matrix of each state; updated in place.
        """
        for cluster in range(self.number_of_clusters):
            if optRes[cluster] is None:
                continue
            val = optRes[cluster].get()
            # THIS IS THE SOLUTION
            S_est = upperToFull(val, 0)
            X2 = S_est
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[self.number_of_clusters, cluster] = np.log(
                np.linalg.det(cov_out)
            )
            computed_covariance[self.number_of_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2

    def train_clusters(
        self,
        cluster_mean_info,
        cluster_mean_stacked_info,
        complete_D_train,
        empirical_covariances,
        len_train_clusters,
        n,
        pool,
        train_clusters_arr,
    ):
        """Start the update step of every state with at least two windows.

        For each such state, store its mean window, the mean of its last
        block and the empirical covariance of its windows, then submit to
        ``pool`` an ``ADMMSolver`` with :math:`\\lambda` =
        ``lambda_parameter`` for every entry and :math:`\\rho = 1`, run for at
        most 1000 iterations with tolerances ``1e-6``.

        Parameters
        ----------
        cluster_mean_info : dict
            Mean of the last block of the windows of each state, of shape
            ``(1, n)``, keyed by ``(number_of_clusters, state)``; updated in
            place.
        cluster_mean_stacked_info : dict
            Mean window of each state, with the same keys; updated in place.
        complete_D_train : np.ndarray
            Stacked windows, of shape ``(n_windows, window_size * n)``.
        empirical_covariances : dict
            Empirical covariance of the windows of each state; updated in
            place.
        len_train_clusters : dict
            Number of windows of each state.
        n : int
            Number of channels.
        pool : multiprocessing.pool.Pool or _SerialPool
            Runs the solvers.
        train_clusters_arr : dict
            Indices of the windows of each state.

        Returns
        -------
        list
            For each state, the result of its solver (an object with a
            ``get`` method), or ``None`` when the state has fewer than two
            windows.
        """
        optRes = [None for i in range(self.number_of_clusters)]
        for cluster in range(self.number_of_clusters):
            cluster_length = len_train_clusters[cluster]
            # A covariance matrix cannot be computed for a single point.
            # Skip this cluster for this iteration if it has fewer than 2 points.
            if cluster_length < 2:
                continue

            size_blocks = n
            indices = train_clusters_arr[cluster]
            D_train = complete_D_train[indices, :]

            cluster_mean_info[self.number_of_clusters, cluster] = np.mean(
                D_train, axis=0
            )[(self.window_size - 1) * n : self.window_size * n].reshape([1, n])
            cluster_mean_stacked_info[self.number_of_clusters, cluster] = np.mean(
                D_train, axis=0
            )
            ##Fit a model - OPTIMIZATION
            probSize = self.window_size * size_blocks
            lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
            S = np.cov(np.transpose(D_train), bias=self.biased)
            empirical_covariances[cluster] = S

            solver = ADMMSolver(lamb, self.window_size, size_blocks, 1, S)
            # apply to process pool
            optRes[cluster] = pool.apply_async(
                solver,
                (
                    1000,
                    1e-6,
                    1e-6,
                    False,
                ),
            )
        return optRes

    def stack_training_data(self, Data, n, num_train_points, training_indices):
        """Stack the windows of the series.

        Parameters
        ----------
        Data : np.ndarray
            The time series, of shape ``(n_timepoints, n_channels)``.
        n : int
            Number of channels.
        num_train_points : int
            Number of windows, the length of ``training_indices``.
        training_indices : list[int]
            Time points of the windows, from ``getTrainTestSplit``.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_train_points, window_size * n)``: row ``i``
            concatenates the points ``training_indices[i + k]`` for
            ``k = 0, ..., window_size - 1``, with zeros where ``i + k``
            reaches ``num_train_points``.
        """
        complete_D_train = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            for k in range(self.window_size):
                if i + k < num_train_points:
                    idx_k = training_indices[i + k]
                    complete_D_train[i][k * n : (k + 1) * n] = Data[idx_k][0:n]
        return complete_D_train

    def predict_clusters(self, test_data=None):
        """Assignment step: state of each window under the current model.

        Parameters
        ----------
        test_data : np.ndarray or None, default=None
            Stacked windows, of shape ``(n_windows, window_size * n_channels)``
            as returned by ``stack_training_data``, not the raw series;
            ``None`` uses the training windows. The last ``window_size - 1``
            rows are treated as padding (cost 0 for every state).

        Returns
        -------
        np.ndarray
            State of each window, as floats (see ``updateClusters``).

        Raises
        ------
        TypeError
            If ``test_data`` is given and is not a NumPy array.
        """
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            test_data = self.trained_model["complete_D_train"]

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(
            self.trained_model["cluster_mean_info"],
            self.trained_model["computed_covariance"],
            self.trained_model["cluster_mean_stacked_info"],
            test_data,
            self.trained_model["time_series_col_size"],
        )

        # Update cluster points - using NEW smoothening
        clustered_points = updateClusters(
            lle_all_points_clusters, switch_penalty=self.switch_penalty
        )

        return clustered_points
