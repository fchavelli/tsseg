"""
AutoPlait Python detector for time series segmentation.
Clean implementation following sklearn/aeon standards.
"""

import numpy as np
from copy import deepcopy
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
from ..base import BaseSegmenter


class AutoPlaitDetector(BaseSegmenter):
    """
    AutoPlait segmentation using pure Python implementation.
    
    AutoPlait automatically determines the optimal number of segments using
    the Minimum Description Length (MDL) criterion with Hidden Markov Models.
    
    Parameters
    ----------
    n_segments : int, default=None
        Number of segments to split the time series into. If None, then the number of
        segments is determined automatically using MDL criterion.
    min_components : int, default=1
        Minimum number of HMM components per regime.
    max_components : int, default=8
        Maximum number of HMM components per regime.
    max_segments : int, default=100
        Maximum number of segments allowed.
    n_iter_hmm : int, default=1
        Number of EM iterations for HMM fitting.
    n_iter_min : int, default=2
        Minimum iterations for regime splitting.
    n_iter_max : int, default=10
        Maximum iterations for regime splitting.
    segment_penalty : float, default=1e-2
        Penalty factor for segment complexity.
    regime_penalty : float, default=3e-2
        Penalty factor for regime complexity.
    n_samples : int, default=10
        Number of samples for centroid search.
    local_search_ratio : float, default=0.1
        Ratio of data length for local search window.
    remove_noise : bool, default=True
        Whether to remove noise segments.
    n_jobs : int, default=1
        Number of parallel jobs for centroid search.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    _tags = {
        "X_inner_type": "np.ndarray",
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:multivariate": True,
    }
    
    def __init__(
        self,
        n_segments=None,
        min_components=1,
        max_components=8,
        max_segments=100,
        n_iter_hmm=1,
        n_iter_min=2,
        n_iter_max=10,
        segment_penalty=1e-2,
        regime_penalty=3e-2,
        n_samples=10,
        local_search_ratio=0.1,
        remove_noise=True,
        n_jobs=1,
        random_state=None
    ):
        self.min_components = min_components
        self.max_components = max_components
        self.max_segments = max_segments
        self.n_iter_hmm = n_iter_hmm
        self.n_iter_min = n_iter_min
        self.n_iter_max = n_iter_max
        self.segment_penalty = segment_penalty
        self.regime_penalty = regime_penalty
        self.n_samples = n_samples
        self.local_search_ratio = local_search_ratio
        self.remove_noise = remove_noise
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        super().__init__(n_segments=n_segments, axis=0)
    
    def _fit(self, X, y=None):
        """
        Fit the AutoPlait model to the data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The time series data to segment.
        y : None
            Ignored. For API compatibility.
        """
        X = self._validate_data(X)
        self.n_timepoints_, self.n_features_ = X.shape
        
        # Standardize the data
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Initialize with single regime covering all data
        initial_regime = _Regime(max_segments=self.max_segments)
        initial_regime.add_segment(0, self.n_timepoints_)
        self._estimate_hmm(X_scaled, initial_regime)
        
        candidates = [initial_regime]
        self.regimes_ = []
        
        # Recursive regime splitting
        while candidates:
            current_cost = self._compute_total_mdl(self.regimes_, candidates)
            regime = candidates.pop()
            
            # Try to split the regime
            regime1, regime2 = self._split_regime(X_scaled, regime)
            
            if regime1.n_segments > 0 and regime2.n_segments > 0:
                split_cost = regime1.cost_total + regime2.cost_total + \
                           self.regime_penalty * regime.cost_total
                
                if split_cost < regime.cost_total:
                    candidates.append(regime1)
                    candidates.append(regime2)
                    continue
            
            # Keep the original regime
            self.regimes_.append(regime)
        
        return self
    
    def _predict(self, X):
        """
        Predict segment labels for the time series.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The time series data to segment.
            
        Returns
        -------
        np.ndarray
            If returns_dense is True (default): array of change point locations.
            If returns_dense is False: array of segment labels for each time point.
        """
        X = self._validate_data(X)
        
        if not hasattr(self, 'regimes_'):
            raise ValueError("Model must be fitted before prediction.")
        
        # Create segment labels
        labels = np.zeros(X.shape[0], dtype=int)
        
        for regime_id, regime in enumerate(self.regimes_):
            for start, length in regime.segments[:regime.n_segments]:
                labels[start:start + length] = regime_id
        
        # Convert to change points (dense representation)
        if self.get_tag("returns_dense"):
            change_points = []
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    change_points.append(i)
            return np.array(change_points, dtype=int)
        else:
            return labels
    
    def _validate_data(self, X):
        """Validate and reshape input data."""
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"Input X must be 1D or 2D, got {X.ndim}D")
        
        return X
    
    def _split_regime(self, X, regime):
        """Split a regime into two sub-regimes."""
        optimal_regime1 = _Regime(max_segments=self.max_segments)
        optimal_regime2 = _Regime(max_segments=self.max_segments)
        
        seed_length = max(1, int(self.n_timepoints_ * self.local_search_ratio))
        
        # Find optimal centroids
        regime1, regime2 = self._find_centroids(X, regime, seed_length)
        
        if regime1.n_segments == 0 or regime2.n_segments == 0:
            return optimal_regime1, optimal_regime2
        
        # Iterative refinement
        for iteration in range(self.n_iter_max):
            self._select_largest_segment(regime1)
            self._select_largest_segment(regime2)
            
            self._estimate_hmm(X, regime1)
            self._estimate_hmm(X, regime2)
            
            self._search_optimal_cuts(X, regime, regime1, regime2)
            
            if regime1.n_segments == 0 or regime2.n_segments == 0:
                break
            
            # Check for improvement
            current_cost = regime1.cost_total + regime2.cost_total
            optimal_cost = optimal_regime1.cost_total + optimal_regime2.cost_total
            
            if current_cost < optimal_cost:
                self._copy_regime(regime1, optimal_regime1)
                self._copy_regime(regime2, optimal_regime2)
            elif iteration >= self.n_iter_min:
                break
        
        # Final estimation
        if optimal_regime1.n_segments > 0 and optimal_regime2.n_segments > 0:
            self._estimate_hmm(X, optimal_regime1)
            self._estimate_hmm(X, optimal_regime2)
        
        return optimal_regime1, optimal_regime2
    
    def _find_centroids(self, X, regime, seed_length):
        """Find optimal initial centroids for regime splitting."""
        uniform_samples = self._create_uniform_samples(regime, seed_length)
        
        if uniform_samples.n_segments < 2:
            return self._fixed_sampling(regime, seed_length)
        
        # Parallel search for best centroid pair
        if self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_centroid_pair)(
                    X, regime, seed_length, i, j, uniform_samples
                ) for i, j in combinations(range(uniform_samples.n_segments), 2)
            )
        else:
            results = [
                self._evaluate_centroid_pair(X, regime, seed_length, i, j, uniform_samples)
                for i, j in combinations(range(uniform_samples.n_segments), 2)
            ]
        
        if not results or all(cost == np.inf for cost, _, _ in results):
            return self._fixed_sampling(regime, seed_length)
        
        # Select best centroid pair
        best_idx = np.argmin([cost for cost, _, _ in results])
        _, seg1, seg2 = results[best_idx]
        
        regime1 = _Regime(max_segments=self.max_segments)
        regime2 = _Regime(max_segments=self.max_segments)
        regime1.add_segment(seg1[0], seg1[1])
        regime2.add_segment(seg2[0], seg2[1])
        
        return regime1, regime2
    
    def _evaluate_centroid_pair(self, X, regime, seed_length, idx1, idx2, samples):
        """Evaluate a specific centroid pair."""
        regime1 = _Regime(max_segments=self.max_segments)
        regime2 = _Regime(max_segments=self.max_segments)
        
        start1 = samples.segments[idx1, 0]
        start2 = samples.segments[idx2, 0]
        
        if abs(start1 - start2) < seed_length:
            return np.inf, None, None
        
        regime1.add_segment(start1, seed_length)
        regime2.add_segment(start2, seed_length)
        
        self._estimate_hmm_k(X, regime1, self.min_components)
        self._estimate_hmm_k(X, regime2, self.min_components)
        
        self._search_optimal_cuts(X, regime, regime1, regime2)
        
        if regime1.n_segments == 0 or regime2.n_segments == 0:
            return np.inf, None, None
        
        cost = regime1.cost_total + regime2.cost_total
        return cost, (start1, seed_length), (start2, seed_length)
    
    def _create_uniform_samples(self, regime, seed_length):
        """Create uniform sampling points within regime segments."""
        samples = _Regime(max_segments=self.max_segments)
        
        total_length = sum(regime.segments[i, 1] for i in range(regime.n_segments))
        step_size = max(1, (total_length - seed_length) // self.n_samples)
        
        for i in range(regime.n_segments):
            start, length = regime.segments[i]
            end = start + length
            
            current_pos = start
            while current_pos + seed_length <= end and samples.n_segments < self.n_samples:
                samples.add_segment(current_pos, seed_length)
                current_pos += step_size
        
        return samples
    
    def _fixed_sampling(self, regime, seed_length):
        """Fallback sampling when uniform sampling fails."""
        regime1 = _Regime(max_segments=self.max_segments)
        regime2 = _Regime(max_segments=self.max_segments)
        
        if regime.n_segments > 0:
            start = regime.segments[0, 0]
            length = min(seed_length, regime.segments[0, 1])
            regime1.add_segment(start, length)
            
            if regime.n_segments > 1:
                start = regime.segments[1, 0]
                length = min(seed_length, regime.segments[1, 1])
                regime2.add_segment(start, length)
            else:
                mid_point = start + regime.segments[0, 1] // 2
                regime2.add_segment(mid_point, min(seed_length, regime.segments[0, 1] - mid_point))
        
        return regime1, regime2
    
    def _search_optimal_cuts(self, X, parent_regime, regime1, regime2):
        """Search for optimal cut points between two regimes."""
        regime1.clear_segments()
        regime2.clear_segments()
        
        for i in range(parent_regime.n_segments):
            start, length = parent_regime.segments[i]
            self._segment_viterbi_search(X[start:start+length], start, regime1, regime2)
        
        if self.remove_noise:
            self._remove_noise_segments(X, parent_regime, regime1, regime2)
        
        self._compute_regime_cost(X, regime1)
        self._compute_regime_cost(X, regime2)
    
    def _segment_viterbi_search(self, X_segment, offset, regime1, regime2):
        """Apply Viterbi-like search for optimal segmentation of a single segment."""
        if len(X_segment) == 0:
            return
        
        # Simple greedy assignment based on likelihood
        model1 = regime1.hmm_model
        model2 = regime2.hmm_model
        
        if model1 is None or model2 is None:
            # Fallback: assign to first regime
            regime1.add_segment(offset, len(X_segment))
            return
        
        try:
            score1 = model1.score(X_segment)
            score2 = model2.score(X_segment)
            
            if score1 > score2:
                regime1.add_segment(offset, len(X_segment))
            else:
                regime2.add_segment(offset, len(X_segment))
        except:
            # Fallback on error
            regime1.add_segment(offset, len(X_segment))
    
    def _remove_noise_segments(self, X, parent_regime, regime1, regime2):
        """Remove noise segments by reassigning small segments."""
        if regime1.n_segments <= 1 and regime2.n_segments <= 1:
            return
        
        threshold = self.segment_penalty * parent_regime.cost_total
        
        # Simple noise removal: reassign very small segments
        for regime in [regime1, regime2]:
            other_regime = regime2 if regime is regime1 else regime1
            
            i = 0
            while i < regime.n_segments:
                start, length = regime.segments[i]
                if length < threshold:
                    # Remove from current regime and add to other
                    regime.remove_segment(i)
                    other_regime.add_segment(start, length)
                else:
                    i += 1
    
    def _estimate_hmm(self, X, regime):
        """Estimate optimal HMM for a regime."""
        if regime.n_segments == 0:
            regime.cost_total = np.inf
            return
        
        best_cost = np.inf
        best_k = self.min_components
        
        for k in range(self.min_components, self.max_components + 1):
            prev_cost = regime.cost_total
            self._estimate_hmm_k(X, regime, k)
            self._compute_regime_cost(X, regime)
            
            if regime.cost_total < best_cost:
                best_cost = regime.cost_total
                best_k = k
            elif regime.cost_total > prev_cost:
                break
        
        # Fit with optimal k
        self._estimate_hmm_k(X, regime, best_k)
        self._compute_regime_cost(X, regime)
    
    def _estimate_hmm_k(self, X, regime, n_components):
        """Estimate HMM with specific number of components."""
        if regime.n_segments == 0:
            return
        
        X_concat, lengths = self._prepare_hmm_data(X, regime)
        
        if len(X_concat) == 0:
            return
        
        try:
            regime.hmm_model = GaussianHMM(
                n_components=n_components,
                covariance_type='diag',
                n_iter=self.n_iter_hmm,
                random_state=self.random_state
            )
            regime.hmm_model.fit(X_concat, lengths=lengths)
            regime.delta = regime.n_segments / sum(lengths)
        except:
            regime.hmm_model = None
            regime.delta = 0.0
    
    def _prepare_hmm_data(self, X, regime):
        """Prepare data for HMM fitting."""
        if regime.n_segments == 0:
            return np.array([]).reshape(0, X.shape[1]), []
        
        segments = []
        lengths = []
        
        for i in range(regime.n_segments):
            start, length = regime.segments[i]
            segment_data = X[start:start + length]
            segments.append(segment_data)
            lengths.append(length)
        
        if not segments:
            return np.array([]).reshape(0, X.shape[1]), []
        
        X_concat = np.vstack(segments)
        return X_concat, lengths
    
    def _compute_regime_cost(self, X, regime):
        """Compute MDL cost for a regime."""
        if regime.n_segments == 0 or regime.hmm_model is None:
            regime.cost_coding = np.inf
            regime.cost_total = np.inf
            return
        
        # Data coding cost
        regime.cost_coding = 0.0
        for i in range(regime.n_segments):
            start, length = regime.segments[i]
            try:
                segment_data = X[start:start + length]
                log_likelihood = regime.hmm_model.score(segment_data)
                regime.cost_coding += -log_likelihood / np.log(2)
            except:
                regime.cost_coding = np.inf
                break
        
        # Model cost (MDL)
        k = regime.hmm_model.n_components
        d = regime.hmm_model.n_features
        m = regime.n_segments
        
        model_cost = self._compute_hmm_cost(k, d)
        length_cost = sum(np.log2(regime.segments[i, 1]) for i in range(m))
        length_cost += m * np.log2(k)
        
        regime.cost_total = regime.cost_coding + model_cost + length_cost
        
        # Avoid overfitting
        if regime.cost_total < 0:
            regime.cost_total = np.inf
    
    def _compute_hmm_cost(self, n_components, n_features):
        """Compute HMM model complexity cost."""
        # Model parameters: start probabilities + transition matrix + emission parameters
        n_params = n_components + n_components**2 + 2 * n_components * n_features
        return n_params * 4 * 8  # Assuming 4 bytes per float, 8 bits per byte
    
    def _compute_total_mdl(self, regimes, candidates):
        """Compute total MDL cost."""
        n_regimes = len(regimes) + len(candidates)
        n_segments = sum(r.n_segments for r in regimes) + sum(r.n_segments for r in candidates)
        
        data_cost = sum(r.cost_total for r in regimes) + sum(r.cost_total for r in candidates)
        model_cost = self._log_star(n_regimes) + self._log_star(n_segments)
        model_cost += n_segments * np.log2(n_regimes) if n_regimes > 0 else 0
        model_cost += 4 * 8 * n_regimes**2  # Regime complexity
        
        return data_cost + model_cost
    
    def _log_star(self, x):
        """Universal code length."""
        return 2 * np.log2(max(1, x)) + 1
    
    def _select_largest_segment(self, regime):
        """Keep only the largest segment in a regime."""
        if regime.n_segments == 0:
            return
        
        largest_idx = np.argmax(regime.segments[:regime.n_segments, 1])
        start, length = regime.segments[largest_idx]
        
        regime.clear_segments()
        regime.add_segment(start, length)
    
    def _copy_regime(self, source, target):
        """Copy regime data from source to target."""
        target.segments = deepcopy(source.segments)
        target.n_segments = source.n_segments
        target.total_length = source.total_length
        target.cost_total = source.cost_total
        target.cost_coding = source.cost_coding
        target.delta = source.delta
        target.hmm_model = source.hmm_model


class _Regime:
    """Internal class representing a regime (collection of segments)."""
    
    def __init__(self, max_segments=100):
        self.max_segments = max_segments
        self.segments = np.zeros((max_segments, 2), dtype=int)  # [start, length]
        self.n_segments = 0
        self.total_length = 0
        self.cost_coding = np.inf
        self.cost_total = np.inf
        self.delta = 0.0
        self.hmm_model = None
    
    def add_segment(self, start, length):
        """Add a new segment to the regime."""
        if length <= 0 or self.n_segments >= self.max_segments:
            return
        
        # Insert in sorted order
        insert_idx = 0
        while insert_idx < self.n_segments and self.segments[insert_idx, 0] < start:
            insert_idx += 1
        
        # Shift existing segments
        if insert_idx < self.n_segments:
            self.segments[insert_idx + 1:self.n_segments + 1] = \
                self.segments[insert_idx:self.n_segments]
        
        self.segments[insert_idx] = [start, length]
        self.n_segments += 1
        self.total_length += length
        self._update_delta()
        
        # Handle overlaps (simplified)
        self._merge_overlapping_segments()
    
    def remove_segment(self, index):
        """Remove segment at given index."""
        if 0 <= index < self.n_segments:
            length = self.segments[index, 1]
            self.segments[index:-1] = self.segments[index + 1:]
            self.n_segments -= 1
            self.total_length -= length
            self._update_delta()
    
    def clear_segments(self):
        """Clear all segments."""
        self.n_segments = 0
        self.total_length = 0
        self.delta = 0.0
    
    def _update_delta(self):
        """Update delta parameter."""
        self.delta = self.n_segments / self.total_length if self.total_length > 0 else 0.0
    
    def _merge_overlapping_segments(self):
        """Merge overlapping segments."""
        if self.n_segments <= 1:
            return
        
        i = 0
        while i < self.n_segments - 1:
            start1, length1 = self.segments[i]
            start2, length2 = self.segments[i + 1]
            end1 = start1 + length1
            
            if end1 > start2:  # Overlap detected
                # Merge segments
                new_end = max(end1, start2 + length2)
                self.segments[i, 1] = new_end - start1
                
                # Remove the second segment
                self.segments[i + 1:-1] = self.segments[i + 2:]
                self.n_segments -= 1
                self.total_length = sum(self.segments[j, 1] for j in range(self.n_segments))
            else:
                i += 1
        
        self._update_delta()
