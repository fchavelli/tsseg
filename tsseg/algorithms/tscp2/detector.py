"""TensorFlow reimplementation of the TS-CP2 contrastive change point detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, TYPE_CHECKING

import numpy as np

from ..base import BaseSegmenter

try:  # pragma: no cover - optional dependency guidance
    import tensorflow as _tf
except ImportError:  # pragma: no cover - allow module import without TensorFlow
    _tf = None

if TYPE_CHECKING:  # pragma: no cover - link real types for static analysis
    import tensorflow as tf
else:
    tf = _tf

from . import losses
from .network import TemporalEncoder

__all__ = ["TSCP2Detector"]


LossFn = Callable[..., tuple[Any, Any, Any]]


_LOSS_REGISTRY: Dict[str, LossFn] = {
    "nce": losses.info_nce_loss,
    "info_nce": losses.info_nce_loss,
    "dcl": losses.debiased_contrastive_loss,
    "fc": losses.focal_contrastive_loss,
    "harddcl": losses.hard_contrastive_loss,
}

_SIMILARITY_REGISTRY: Dict[str, Callable[[Any, Any], Any]] = {
    "cosine": losses.cosine_similarity_dim2,
    "dot": losses.dot_similarity_dim2,
    "euclidean": losses.euclidean_similarity_dim2,
    "edit": losses.edit_similarity_dim2,
}

_SIMILARITY_DIAG_REGISTRY: Dict[str, Callable[[Any, Any], Any]] = {
    "cosine": losses.cosine_similarity_dim1,
    "dot": losses.dot_similarity_dim1,
    "euclidean": losses.euclidean_similarity_dim1,
    "edit": losses.edit_similarity_dim1,
}


@dataclass
class _TrainingStats:
    pos_sim: float
    neg_sim: float


class TSCP2Detector(BaseSegmenter):
    """Time Series Change Point Detection with contrastive predictive coding.

    This implementation mirrors the original TensorFlow TS-CP2 reference
    (Deldari et al., WWW'21, https://doi.org/10.1145/3442381.3449903).
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "python_dependencies": ["tensorflow", "tcn"],
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    def __init__(
        self,
        *,
        window_size: int = 128,
        n_cps: int | None = None,
        similarity_threshold: float | None = None,
        stride: int = 5,
        code_size: int = 32,
        nb_filters: int = 64,
        kernel_size: int = 4,
        dilations: tuple[int, ...] | None = (1, 2, 4, 8),
        nb_stacks: int = 2,
        dropout_rate: float = 0.0,
        dense_units: tuple[int, ...] | None = None,
        batch_size: int = 64,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        loss: str = "nce",
        temperature: float = 0.1,
        tau: float = 0.1,
        beta: float = 0.1,
        similarity: str = "cosine",
        refit_on_predict: bool = False,
        axis: int = 0,
    ) -> None:
        if tf is None or not hasattr(TemporalEncoder, "call"):
            raise ImportError(
                "TSCP2Detector requires the optional 'tscp2' extra (tensorflow>=2.13 and tcn)."
            )
        if n_cps is not None and similarity_threshold is not None:
            similarity_threshold = None
        if loss not in _LOSS_REGISTRY:
            raise ValueError(f"Unknown loss '{loss}'. Choose from {sorted(_LOSS_REGISTRY)}")
        if similarity not in _SIMILARITY_REGISTRY:
            raise ValueError(
                f"Unknown similarity '{similarity}'. Choose from {sorted(_SIMILARITY_REGISTRY)}"
            )
        self.loss = loss
        self.similarity = similarity
        self.window_size = int(window_size)
        self.n_cps = None if n_cps is None else int(n_cps)
        self.similarity_threshold = None if similarity_threshold is None else float(similarity_threshold)
        self._auto_threshold = self.n_cps is None and self.similarity_threshold is None
        self.stride = max(int(stride), 1)
        self.code_size = int(code_size)
        if dilations is None:
            dilations = (1, 2, 4, 8)
        self.dilations = tuple(int(d) for d in dilations)
        self.nb_filters = int(nb_filters)
        self.kernel_size = int(kernel_size)
        self.nb_stacks = int(nb_stacks)
        self.dropout_rate = float(dropout_rate)
        self.dense_units = None if dense_units is None else tuple(int(u) for u in dense_units)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.loss_name = loss
        self.temperature = float(temperature)
        self.tau = float(tau)
        self.beta = float(beta)
        self.similarity_name = similarity
        self.refit_on_predict = refit_on_predict
        self._encoder: TemporalEncoder | None = None
        self._train_stats: _TrainingStats | None = None
        self._train_signal: np.ndarray | None = None
        self._loss_fn: LossFn = _LOSS_REGISTRY[self.loss_name]
        self._similarity_matrix_fn = _SIMILARITY_REGISTRY[self.similarity_name]
        self._similarity_diag_fn = _SIMILARITY_DIAG_REGISTRY[self.similarity_name]
        super().__init__(axis=axis)

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X[:, np.newaxis]
        if X.ndim != 2:
            raise ValueError("TSCP2Detector expects 1D or 2D arrays")
        return X

    def _build_pairs(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, _ = signal.shape

        while n_samples <= 2 * self.window_size:
            if self.window_size <= 1:
                raise ValueError(
                    f"Input series (length={n_samples}) is too short for analysis. "
                    "Cannot reduce window_size further."
                )
            print(
                f"Warning: Input series ({n_samples}) shorter than 2*window_size ({self.window_size}). "
                f"Reducing window_size to {self.window_size // 2}."
            )
            self.window_size //= 2

        total = n_samples - 2 * self.window_size
        histories: list[np.ndarray] = []
        futures: list[np.ndarray] = []
        centers: list[int] = []
        for start in range(0, total + 1, self.stride):
            hist = signal[start : start + self.window_size]
            fut = signal[start + self.window_size : start + 2 * self.window_size]
            histories.append(hist)
            futures.append(fut)
            centers.append(start + self.window_size)
        history_arr = np.stack(histories)
        future_arr = np.stack(futures)
        center_arr = np.asarray(centers, dtype=int)
        return history_arr, future_arr, center_arr

    def _make_dataset(self, history: np.ndarray, future: np.ndarray) -> tf.data.Dataset:
        drop_last = history.shape[0] >= self.batch_size
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor(history.astype(np.float32)),
                tf.convert_to_tensor(future.astype(np.float32)),
            )
        )
        buffer = max(history.shape[0], self.batch_size * 4)
        dataset = dataset.shuffle(buffer_size=buffer, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=drop_last)
        return dataset

    def _loss_kwargs(self) -> Dict[str, float]:
        params: Dict[str, float] = {"temperature": self.temperature}
        if self.loss_name in {"dcl", "harddcl"}:
            params["tau_plus"] = self.tau
        if self.loss_name == "fc":
            params["elimination_topk"] = self.beta
        if self.loss_name == "harddcl":
            params["beta"] = self.beta
        return params

    def _fit(self, X, y=None):  # noqa: D401 - docstring inherited
        signal = self._ensure_2d(X)
        history, future, _ = self._build_pairs(signal)
        if history.shape[0] < 2:
            raise ValueError("Not enough windowed samples for contrastive training")
        n_features = signal.shape[1]
        encoder = TemporalEncoder(
            input_features=n_features,
            window_size=self.window_size,
            code_size=self.code_size,
            nb_filters=self.nb_filters,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
            nb_stacks=self.nb_stacks,
            dropout_rate=self.dropout_rate,
            dense_units=self.dense_units,
        )
        optimiser = tf.keras.optimizers.Adam(self.learning_rate)
        dataset = self._make_dataset(history, future)
        loss_kwargs = self._loss_kwargs()
        last_pos, last_neg = 0.0, 0.0

        for _ in range(self.epochs):
            for hist_batch, fut_batch in dataset:
                with tf.GradientTape() as tape:
                    z_hist = encoder(hist_batch, training=True)
                    z_fut = encoder(fut_batch, training=True)
                    z_hist = tf.math.l2_normalize(z_hist, axis=1)
                    z_fut = tf.math.l2_normalize(z_fut, axis=1)
                    loss, pos_sim, neg_sim = self._loss_fn(
                        z_hist,
                        z_fut,
                        self._similarity_matrix_fn,
                        **loss_kwargs,
                    )
                grads = tape.gradient(loss, encoder.trainable_variables)
                optimiser.apply_gradients(zip(grads, encoder.trainable_variables))
                last_pos = float(pos_sim.numpy())
                last_neg = float(neg_sim.numpy())

        self._encoder = encoder
        self._train_stats = _TrainingStats(pos_sim=last_pos, neg_sim=last_neg)
        self._train_signal = signal
        if self._auto_threshold and self._train_stats is not None:
            pos = self._train_stats.pos_sim
            neg = self._train_stats.neg_sim
            self.similarity_threshold = pos - ((pos - neg) / 3.0)
        return self

    def _predict(self, X):  # noqa: D401 - docstring inherited
        if self._encoder is None:
            raise RuntimeError("TSCP2Detector must be fitted before predict")
        signal = self._ensure_2d(X)
        if self._train_signal is not None and signal.shape[1] != self._train_signal.shape[1]:
            raise ValueError("Input feature dimension differs from the fitted data")
        if self.refit_on_predict and (self._train_signal is None or not np.array_equal(signal, self._train_signal)):
            self._fit(signal)
            signal = self._train_signal
        history, future, centers = self._build_pairs(signal)
        history_tensor = tf.convert_to_tensor(history.astype(np.float32))
        future_tensor = tf.convert_to_tensor(future.astype(np.float32))
        z_hist = self._encoder(history_tensor, training=False)
        z_fut = self._encoder(future_tensor, training=False)
        similarities = self._similarity_diag_fn(z_hist, z_fut).numpy()
        change_scores = 1.0 - similarities
        if self.n_cps is not None:
            n_cps = min(self.n_cps, centers.shape[0])
            if n_cps == 0:
                return np.array([], dtype=int)
            idx = np.argpartition(change_scores, -n_cps)[-n_cps:]
            bkps = centers[idx]
        elif self.similarity_threshold is not None:
            mask = similarities < self.similarity_threshold
            bkps = centers[mask]
        else:
            raise RuntimeError("No selection criterion available. Fit the detector before predict or set n_cps/similarity_threshold.")
        bkps = np.unique(bkps.astype(int))
        bkps = bkps[(bkps > 0) & (bkps < signal.shape[0])]
        return bkps
