import numpy as np
from numpy.fft import fft, ifft
from scipy.signal import find_peaks, savgol_filter

from ..base import BaseSegmenter


class EspressoDetectorOld(BaseSegmenter):
    """Segment time series using the ESPRESSO change-point detection algorithm.

    Parameters
    ----------
    window_size : int
        Length of the subsequences used to compute the matrix profile; must be at least 4.
    chain_len : int
        Number of iterations used when expanding arc sets to build the semantic density matrix.
    n_segments : int, optional
        Target number of segments to produce during prediction. Must be supplied
        and be greater than or equal to 2 when calling ``predict``.
    axis : int, default=0
        Axis corresponding to the time dimension in the input array.
    random_state : int, optional
        Seed for the internal random number generator used when sampling subsequences.
    peak_distance_fraction : float, default=0.01
        Fraction of the time series length that determines the minimum distance
        between detected peaks.

    Raises
    ------
    ValueError
        If ``window_size`` is less than 4.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
    }

    def __init__(
        self,
        window_size: int = 64,
        chain_len: int = 3,
        *,
        n_segments: int | None = None,
        axis: int = 0,
        random_state: int | None = None,
        peak_distance_fraction: float = 0.01,
    ):
        if window_size < 4:
            raise ValueError("window_size must be >= 4")
        self.window_size = int(window_size)
        self.chain_len = int(chain_len)
        self.peak_distance_fraction = float(peak_distance_fraction)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.n_segments = n_segments
        super().__init__(axis=axis)

    def _reset_rng(self) -> None:
        self._rng = np.random.default_rng(self.random_state)

    def _fit(self, X, y=None):
        self._reset_rng()
        return self

    def _predict(self, X, axis=None):
        X = np.asarray(X, dtype=float)
        if axis is not None and axis != self.axis:
            X = np.moveaxis(X, axis, self.axis)
        if self.axis != 0:
            X = np.moveaxis(X, self.axis, 0)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.ndim != 2:
            raise ValueError("Input time series must be 1D or 2D array-like")
        if self.n_segments is None or self.n_segments < 2:
            raise ValueError("n_segments must be provided and >= 2 for prediction")

        data = X.T  # shape (n_channels, n_timepoints)
        change_points = self._run_espresso(self.n_segments, data)
        change_points = np.array(sorted(set(int(cp) for cp in change_points)), dtype=int)
        change_points = change_points[(change_points > 0) & (change_points < X.shape[0])]
        return change_points

    def _fast_find_nn_pre(self, x: np.ndarray):
        n = len(x)
        padded = np.concatenate([x, np.zeros(n)])
        X = fft(padded)

        cum_sumx = np.cumsum(x)
        cum_sumx2 = np.cumsum(x**2)

        sumx2 = cum_sumx2[self.window_size - 1 : n] - np.concatenate(
            ([0], cum_sumx2[: n - self.window_size])
        )
        sumx = cum_sumx[self.window_size - 1 : n] - np.concatenate(
            ([0], cum_sumx[: n - self.window_size])
        )
        meanx = sumx / self.window_size
        sigmax2 = (sumx2 / self.window_size) - (meanx**2)
        sigmax2 = np.maximum(sigmax2, np.finfo(float).eps)
        sigmax = np.sqrt(sigmax2)

        return X, n, sumx2, sumx, meanx, sigmax2, sigmax

    def _fast_find_nn(
        self,
        X,
        y,
        n,
        m,
        sumx2,
        sumx,
        meanx,
        sigmax2,
        sigmax,
    ):
        y = np.asarray(y, dtype=float)
        y = (y - np.mean(y)) / (np.std(y, ddof=1) or 1.0)
        y = y[::-1]
        y = np.concatenate([y, np.zeros(2 * n - len(y))])

        Y = fft(y)
        Z = X * Y
        z = np.real(ifft(Z))

        sumy = np.sum(y[:m])
        sumy2 = np.sum(y[:m] ** 2)

        safe_sigmax2 = np.maximum(sigmax2, np.finfo(float).eps)
        safe_sigmax = np.maximum(sigmax, np.finfo(float).eps)

        dist = (
            (sumx2 - 2 * sumx * meanx + m * (meanx**2)) / safe_sigmax2
            - 2 * (z[m - 1 : n] - sumy * meanx) / safe_sigmax
            + sumy2
        )
        dist = np.sqrt(np.abs(dist))
        return dist

    def _timeseries_self_join_fast(self, series: np.ndarray):
        if self.window_size > len(series) / 2:
            raise ValueError("Time series is too short for the chosen subsequence length")
        if self.window_size < 4:
            raise ValueError("subsequence length must be at least 4")

        series = np.asarray(series, dtype=float)
        if series.ndim != 1:
            raise ValueError("Input series must be one-dimensional")

        exclusion_zone = round(self.window_size / 4)
        matrix_profile_length = len(series) - self.window_size + 1
        matrix_profile = np.full(matrix_profile_length, np.inf)
        mp_index = np.zeros(matrix_profile_length, dtype=int)

        X, n, sumx2, sumx, meanx, sigmax2, sigmax = self._fast_find_nn_pre(series)

        picked_idx = self._rng.permutation(matrix_profile_length)
        for idx in picked_idx:
            subsequence = series[idx : idx + self.window_size]
            distance_profile = self._fast_find_nn(
                X, subsequence, n, self.window_size, sumx2, sumx, meanx, sigmax2, sigmax
            )
            distance_profile = np.abs(distance_profile)

            exclusion_start = max(0, idx - exclusion_zone)
            exclusion_end = min(matrix_profile_length, idx + exclusion_zone)
            distance_profile[exclusion_start : exclusion_end + 1] = np.inf

            update_mask = distance_profile < matrix_profile
            mp_index[update_mask] = idx
            matrix_profile[update_mask] = distance_profile[update_mask]

        return matrix_profile, mp_index

    def _compute_matrix_profile(self, integ_ts: np.ndarray):
        data = np.asarray(integ_ts, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = data

        mp_rows = []
        mpi_rows = []
        for row in data:
            mp, mpi = self._timeseries_self_join_fast(row)
            mp_rows.append(mp)
            mpi_rows.append(mpi)
        return np.array(mp_rows), np.array(mpi_rows)

    def _extract_new_arc_set(
        self,
        matrix_profile_row,
        mp_index_row,
        arc_set,
        arc_cost,
        threshold,
        last_arc_set,
        last_arc_cost,
    ):
        dontcare = len(matrix_profile_row)
        initial_arcs = last_arc_set.copy()
        initial_arcs[last_arc_cost > threshold] = dontcare

        temp = np.append(initial_arcs, dontcare)
        new_arcs = temp[initial_arcs]

        temp_arc_cost = np.full(dontcare, threshold + 1.0)
        temp_arc_set = np.full(dontcare, dontcare, dtype=int)

        for i in range(dontcare):
            new_arc = int(new_arcs[i]) if i < len(new_arcs) else dontcare
            if new_arc == dontcare:
                continue
            if abs(new_arc - i) <= self.window_size / 4:
                continue
            neighbour = int(last_arc_set[i]) if i < len(last_arc_set) else dontcare
            if neighbour == dontcare or neighbour >= dontcare:
                continue
            arc_set[i] = np.append(arc_set[i], new_arc)
            candidate_cost = last_arc_cost[i] + last_arc_cost[neighbour]
            arc_cost[i] = np.append(arc_cost[i], candidate_cost)
            temp_arc_cost[i] = candidate_cost
            temp_arc_set[i] = new_arc

        return arc_set, arc_cost, temp_arc_set, temp_arc_cost

    def _calculate_semantic_density_matrix(self, matrix_profile_row, mp_index_row):
        matrix_profile_row = np.asarray(matrix_profile_row, dtype=float)
        mp_index_row = np.asarray(mp_index_row, dtype=int)
        dontcare = len(matrix_profile_row) + 1
        threshold = 2 * np.max(matrix_profile_row)

        arc_set = [np.array([idx]) for idx in mp_index_row]
        arc_cost = [np.array([cost]) for cost in matrix_profile_row]
        last_arc_set = mp_index_row.astype(int)
        last_arc_cost = matrix_profile_row.astype(float)

        for _ in range(self.chain_len):
            arc_set, arc_cost, last_arc_set, last_arc_cost = self._extract_new_arc_set(
                matrix_profile_row,
                mp_index_row,
                arc_set,
                arc_cost,
                threshold,
                last_arc_set,
                last_arc_cost,
            )
            if np.sum(last_arc_set < dontcare) == 0:
                break

        nnmark = np.zeros(len(matrix_profile_row))
        min_vals = np.array([np.min(arc) for arc in arc_set])
        max_vals = np.array([np.max(arc) for arc in arc_set])
        totmin = np.min(np.abs(min_vals - np.arange(len(min_vals))))
        totmax = np.max(np.abs(max_vals - np.arange(len(max_vals))))
        denom = max(totmax - totmin, 1.0)

        for j, arc_list in enumerate(arc_set):
            for arc in arc_list:
                small = min(j, arc)
                large = max(j, arc)
                length = large - small
                nnmark[small:large] += 1 - ((length - totmin) / denom)
        return nnmark

    def _clean_ts(self, series: np.ndarray, mode: int):
        integ_ts = np.array(series, dtype=float)
        if integ_ts.ndim == 1:
            integ_ts = integ_ts[:, np.newaxis]
        n, m = integ_ts.shape

        for i in range(m):
            column = integ_ts[:, i]
            min_val = np.min(column) if mode != 2 else 0.0
            column = column - min_val
            if mode != 2:
                total = np.sum(column)
                if total:
                    column = column / (total / 1000)
            if mode == 1:
                max_val = np.max(column)
                new_row = max_val - column
                total = np.sum(new_row)
                if total:
                    new_row = new_row / (total / 1000)
                integ_ts = np.vstack((integ_ts, new_row))
            integ_ts[:, i] = column

        return np.cumsum(integ_ts, axis=1)

    def _smooth_data(self, data, method="gaussian", window_length=5, polyorder=2):
        data = np.asarray(data)
        if method == "gaussian":
            return savgol_filter(data, window_length, polyorder, mode="nearest", axis=-1)
        if method == "movmean":
            kernel = np.ones(window_length) / window_length
            if data.ndim == 1:
                return np.convolve(data, kernel, mode="same")
            return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=-1, arr=data)
        if method == "savgol":
            return savgol_filter(data, window_length, polyorder, mode="nearest", axis=-1)
        raise ValueError("Unsupported smoothing method")

    def _find_peaks(self, data, *, min_peak_distance=1, height=None, prominence=None, width=None):
        peaks, properties = find_peaks(
            data,
            distance=min_peak_distance,
            height=height,
            prominence=prominence,
            width=width,
        )
        if "prominences" in properties:
            order = np.argsort(-properties["prominences"])
        elif "peak_heights" in properties:
            order = np.argsort(-properties["peak_heights"])
        else:
            order = np.arange(len(peaks))
        peaks = peaks[order]
        for key, value in properties.items():
            properties[key] = np.asarray(value)[order]
        return peaks, properties

    def _separate_greedy_ig(self, ts, num_segs, cc):
        ts = np.asarray(ts, dtype=float)
        if ts.ndim == 1:
            ts = ts[:, np.newaxis]
        else:
            ts = ts.T

        data_length, number_of_ts = ts.shape
        integ_ts = self._clean_ts(ts, 0)
        pdist = max(int(np.floor(data_length * self.peak_distance_fraction)), 1)
        cc = -1 * self._smooth_data(cc)

        best_tt = []
        best_ig = 0.0
        for d in range(number_of_ts):
            peaks, properties = self._find_peaks(
                cc[d, :],
                min_peak_distance=pdist,
                prominence=True,
            )
            prominences = properties.get("prominences", np.array([]))
            locs = peaks[np.argsort(-prominences)] if len(prominences) else peaks
            tt: list[int] = []
            max_ig = np.zeros(len(locs))
            remain_locs = locs.tolist()

            for i in range(len(locs)):
                temp_tt = None
                for loc in remain_locs:
                    candidate_tt = sorted(tt + [loc, data_length])
                    ig = self._ig_cal(integ_ts, candidate_tt, i + 1)
                    if ig > max_ig[i]:
                        max_ig[i] = ig
                        temp_tt = loc
                if temp_tt is not None:
                    tt.append(temp_tt)
                    remain_locs.remove(temp_tt)

            if len(max_ig) == 0:
                continue
            t = min(num_segs - 1, len(max_ig))
            if t <= 0:
                continue
            if max_ig[t - 1] > best_ig:
                best_ig = max_ig[t - 1]
                best_tt = tt[:t]
        return best_tt, best_ig

    def _ig_cal(self, integ_ts, pos_tt1, k):
        integ_ts = np.asarray(integ_ts, dtype=float)
        le_ts, nu_ts = integ_ts.shape
        pos_tt = np.sort(pos_tt1)
        ts_dist = np.array([integ_ts[:, i] for i in range(nu_ts)])
        ig = self._sh_entropy(ts_dist)

        last_id = 0
        for boundary in pos_tt:
            boundary = int(boundary)
            boundary = np.clip(boundary, 0, le_ts - 1)
            seg = np.array([integ_ts[boundary - 1, j] - integ_ts[last_id, j] for j in range(nu_ts)])
            segment_length = boundary - last_id
            if segment_length > 0:
                ig -= (segment_length / le_ts) * self._sh_entropy(seg)
            last_id = boundary
        return ig

    @staticmethod
    def _sh_entropy(x):
        x = np.array(x, dtype=float)
        x = np.abs(x)
        x = x[x > 0]
        total = np.sum(x)
        if total <= 0:
            return 0.0
        probabilities = x / total
        return -np.sum(probabilities * np.log(probabilities))

    def _run_espresso(self, n_segments: int, data: np.ndarray):
        MP, MPI = self._compute_matrix_profile(data)
        wcac = np.zeros_like(MP)
        for i in range(MP.shape[0]):
            wcac[i, :] = self._calculate_semantic_density_matrix(MP[i], MPI[i])
        esp_tt, _ = self._separate_greedy_ig(data, n_segments, wcac)
        return esp_tt
