import numpy as np
from ..base import BaseSegmenter
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import pairwise_distances

class ICIDDetector(BaseSegmenter):
    """
    Isolation Distributional Kernel Change Interval Detection (iCID).

    This is a Python implementation of the iCID algorithm, adapted for the aeon
    framework, from the original MATLAB code by Yang Cao et al. [1]_.

    iCID detects change intervals by transforming the time series into a high-
    dimensional distributional feature space and then identifying dissimilarities
    between consecutive windows.

    The algorithm works in four main steps:
    1.  **aNNEspace Transformation**: Projects the data into a feature space
        using random subsampling to create a distributional kernel.
    2.  **Dissimilarity Scoring**: Computes a point-wise dissimilarity score
        by calculating the cosine distance between the mean embeddings of
        consecutive windows in the transformed space.
    3.  **Automatic `psi` Selection**: The granularity parameter `psi` is
        automatically selected by finding the one that minimizes the
        approximate entropy of the resulting dissimilarity score series.
    4.  **Adaptive Thresholding**: Change points are detected by applying an
        adaptive threshold to the final dissimilarity score.

    Parameters
    ----------
    window_size : int, default=50
        The size of the window_size_size for computing the dissimilarity score.
    alpha : float, default=0.5
        The sensitivity factor for the detection threshold. A higher value
        makes the detection less sensitive.
    t : int, default=200
        The number of iterations for the aNNEspace transformation, controlling
        the dimensionality of the feature space.
    psi_list : list of int, default=[2, 4, 8, 16, 32, 64]
        The list of `psi` values to test for granularity. The best one is
        selected automatically.
    axis : int, default=0
        The axis to segment along if passed a multivariate series.

    References
    ----------
    .. [1] Yang Cao, Ye Zhu, Kai Ming Ting, et al. "Detecting Change Intervals
       with Isolation Distributional Kernel." Journal of Artificial Intelligence
       Research, 2024.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
    }

    def __init__(self, window_size=50, alpha=0.5, t=200, psi_list=None, axis=0):
        self.window_size = window_size
        self.alpha = alpha
        self.t = t
        self.psi_list = psi_list if psi_list is not None else [2, 4, 8, 16, 32, 64]
        super().__init__(axis=axis)

    def _predict(self, X: np.ndarray):
        """
        Find change points in the time series X.

        Parameters
        ----------
        X : np.ndarray
            Time series to be segmented.

        Returns
        -------
        np.ndarray
            Array of indices corresponding to the found change points.
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Normalize data
        Y = minmax_scale(X)
        Y[np.isnan(Y)] = 0.5

        all_pscores = []
        entropies = []

        for psi in self.psi_list:
            pscore = self._point_score(Y, psi, self.window_size, self.t)
            all_pscores.append(pscore)
            entropy = self._approximate_entropy(pscore)
            entropies.append(entropy)

        best_idx = np.argmin(entropies)
        best_pscore_raw = all_pscores[best_idx]

        # Normalize the best pscore before thresholding
        best_pscore = minmax_scale(best_pscore_raw)

        threshold = np.mean(best_pscore) + self.alpha * np.std(best_pscore)
        binary_result = (best_pscore > threshold).astype(int)

        # Find the start of each sequence of '1's
        change_points = np.where(np.diff(binary_result, prepend=0) == 1)[0]

        return change_points.astype(np.int64)

    def _aNNEspace(self, Sdata, data, psi, t):
        """
        Transforms data into the aNNE (approximate Nearest Neighbour Embedding) space.
        Translation of aNNEspace.m.
        """
        sn, _ = Sdata.shape
        n, _ = data.shape
        
        # Ensure psi is not larger than the number of samples
        if psi > sn:
            psi = sn

        ndata_parts = []
        for _ in range(t):
            subIndex = np.random.choice(sn, psi, replace=False)
            tdata = Sdata[subIndex, :]
            
            dis = pairwise_distances(tdata, data, metric='euclidean')
            centerIdx = np.argmin(dis, axis=0)
            
            z = np.zeros((psi, n))
            z[centerIdx, np.arange(n)] = 1
            ndata_parts.append(z.T)
            
        return np.hstack(ndata_parts)

    def _point_score(self, Y, psi, window_size, t):
        """
        Calculates the dissimilarity score for each point.
        Translation of point_score.m.
        """
        ndata = self._aNNEspace(Y, Y, psi, t)
        
        n_samples = Y.shape[0]
        index = np.arange(0, n_samples, window_size)
        if index[-1] != n_samples:
            index = np.append(index, n_samples)

        mdata = []
        for i in range(len(index) - 1):
            cdata = ndata[index[i]:index[i+1], :]
            mdata.append(np.mean(cdata, axis=0))
        mdata = np.array(mdata)

        k = 1  # knn parameter from original code, effectively just comparing to previous
        scores = np.zeros(len(mdata))
        for i in range(k, len(mdata)):
            # Direct comparison with the previous window_size's embedding
            norm_i = np.linalg.norm(mdata[i, :])
            norm_prev = np.linalg.norm(mdata[i-1, :])
            if norm_i == 0 or norm_prev == 0:
                cos_sim = 1 # Treat as identical if one is a zero vector
            else:
                cos_sim = np.dot(mdata[i, :], mdata[i-1, :]) / (norm_i * norm_prev)
            
            scores[i] = 1 - cos_sim

        Pscore = np.zeros(n_samples)
        for i in range(len(index) - 1):
            Pscore[index[i]:index[i+1]] = scores[i]
            
        # Normalize the score before returning, as in the original MATLAB
        return minmax_scale(Pscore)

    def _approximate_entropy(self, x, m=2, r_factor=0.2):
        """
        A direct implementation of Approximate Entropy.
        This avoids the dependency on the `nolds` library.
        """
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            x = np.asarray(x, dtype=float)

        N = len(x)
        r = r_factor * np.std(x)

        def _phi(m_val):
            if N <= m_val:
                return 0
            x_m = np.array([x[i : i + m_val] for i in range(N - m_val + 1)])
            C_m = np.zeros(N - m_val + 1)
            for i in range(len(x_m)):
                # Chebychev distance
                dist = np.max(np.abs(x_m - x_m[i]), axis=1)
                C_m[i] = np.sum(dist <= r) / (N - m_val + 1.0)
            
            # Avoid log(0) by adding a small epsilon
            return np.mean(np.log(C_m + 1e-10))

        phi_m = _phi(m)
        phi_m_plus_1 = _phi(m + 1)

        return phi_m - phi_m_plus_1

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"window_size": 20, "alpha": 0.5, "t": 50, "psi_list": [2, 4]}
