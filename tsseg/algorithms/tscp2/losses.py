"""Loss utilities adapted from the original TensorFlow TS-CP2 implementation."""

from __future__ import annotations

from typing import Callable, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional dependency guard
    import tensorflow as _tf
except ImportError:  # pragma: no cover - enable package import without TensorFlow
    _tf = None

if TYPE_CHECKING:  # pragma: no cover - for static analyzers only
    import tensorflow as tf
else:
    tf = _tf  # type: ignore[assignment]


def _missing_dependency(*_args, **_kwargs):  # pragma: no cover - helper for stubs
    raise ImportError("TensorFlow is required for TS-CP2 losses. Install the 'tscp2' extra.")


if tf is None:  # pragma: no cover - export stubs when TensorFlow is absent
    cosine_similarity_vector = cosine_similarity_matrix = _missing_dependency  # type: ignore[assignment]
    dot_similarity_vector = dot_similarity_matrix = _missing_dependency  # type: ignore[assignment]
    euclidean_similarity_vector = euclidean_similarity_matrix = _missing_dependency  # type: ignore[assignment]
    edit_similarity_vector = edit_similarity_matrix = _missing_dependency  # type: ignore[assignment]
    info_nce_loss = debiased_contrastive_loss = focal_contrastive_loss = hard_contrastive_loss = _missing_dependency  # type: ignore[assignment]
else:  # pragma: no cover - TensorFlow is available, expose real implementations

    def _l2_normalize(x: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(x, axis=1)


    def cosine_similarity_vector(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Cosine similarity for aligned pairs (shape: [batch])."""

        return tf.reduce_sum(_l2_normalize(x) * _l2_normalize(y), axis=1)


    def cosine_similarity_matrix(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """Cosine similarity matrix between two batches (shape: [batch, batch])."""

        return tf.matmul(_l2_normalize(x), _l2_normalize(y), transpose_b=True)


    def dot_similarity_vector(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(x * y, axis=1)


    def dot_similarity_matrix(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, y, transpose_b=True)


    def euclidean_similarity_vector(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        distance = tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=-1))
        return 1.0 / (1.0 + distance)


    def euclidean_similarity_matrix(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        x_exp = tf.expand_dims(x, 1)
        y_exp = tf.expand_dims(y, 0)
        distance = tf.sqrt(tf.reduce_sum(tf.square(x_exp - y_exp), axis=-1))
        return 1.0 / (1.0 + distance)


    edit_similarity_vector = euclidean_similarity_vector
    edit_similarity_matrix = euclidean_similarity_matrix


    def _mask_off_diagonal(sim: tf.Tensor) -> tf.Tensor:
        n = tf.shape(sim)[0]
        mask = tf.logical_not(tf.eye(n, dtype=tf.bool))
        negatives = tf.boolean_mask(sim, mask)
        return tf.reshape(negatives, (n, n - 1))


    def info_nce_loss(
        history: tf.Tensor,
        future: tf.Tensor,
        similarity: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        *,
        temperature: float = 0.1,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """InfoNCE loss mirroring the original TensorFlow implementation."""

        sim = similarity(history, future)
        pos = tf.linalg.diag_part(sim)
        pos_sim = tf.math.exp(pos / temperature)
        all_sim = tf.math.exp(sim / temperature)

        numerator = tf.reduce_sum(pos_sim)
        denom = tf.reduce_sum(all_sim, axis=1)
        logits = tf.math.divide_no_nan(tf.broadcast_to(numerator, tf.shape(denom)), denom)

        criterion = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM
        )
        labels = tf.ones_like(logits)
        loss = criterion(y_true=labels, y_pred=logits)

        neg = _mask_off_diagonal(sim)
        mean_pos = tf.reduce_mean(pos)
        mean_neg = tf.reduce_mean(neg)
        return loss, mean_pos, mean_neg


    def debiased_contrastive_loss(
        history: tf.Tensor,
        future: tf.Tensor,
        similarity: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        *,
        temperature: float = 0.1,
        tau_plus: float = 0.1,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Debiased contrastive loss (DCL)."""

        sim = similarity(history, future)
        pos = tf.linalg.diag_part(sim)
        pos_sim = tf.math.exp(pos / temperature)
        neg = _mask_off_diagonal(sim)
        neg_sim = tf.math.exp(neg / temperature)

        n = tf.cast(tf.shape(history)[0] - 1, tf.float32)
        numerator = -tau_plus * n * pos_sim + tf.reduce_sum(neg_sim, axis=-1)
        lower_bound = n * tf.math.exp(-1.0 / temperature)
        denom = tf.maximum(numerator / (1.0 - tau_plus), lower_bound)
        prob = tf.math.divide_no_nan(pos_sim, pos_sim + denom)
        loss = -tf.reduce_mean(tf.math.log(prob + 1e-12))

        mean_pos = tf.reduce_mean(pos)
        mean_neg = tf.reduce_mean(neg)
        return loss, mean_pos, mean_neg


    def focal_contrastive_loss(
        history: tf.Tensor,
        future: tf.Tensor,
        similarity: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        *,
        temperature: float = 0.1,
        elimination_topk: float = 0.1,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Focal contrastive loss that removes the hardest negatives."""

        sim = similarity(history, future) / temperature
        pos = tf.linalg.diag_part(sim)
        pos_sim = tf.math.exp(pos)
        neg = _mask_off_diagonal(sim)

        n = tf.shape(history)[0]
        topk_float = tf.clip_by_value(elimination_topk, 0.0, 0.5) * tf.cast(n, tf.float32)
        topk = tf.maximum(1, tf.cast(tf.math.ceil(topk_float), tf.int32))

        sorted_neg = tf.sort(neg, direction="DESCENDING", axis=1)
        remaining = sorted_neg[:, topk:]

        def _safe_mean(values: tf.Tensor) -> tf.Tensor:
            return tf.cond(
                tf.equal(tf.shape(values)[1], 0),
                lambda: tf.constant(0.0, dtype=values.dtype),
                lambda: tf.reduce_mean(values),
            )

        neg_sim = tf.reduce_sum(tf.math.exp(remaining), axis=1)
        loss = -tf.reduce_mean(tf.math.log(tf.math.divide_no_nan(pos_sim, pos_sim + neg_sim) + 1e-12))

        mean_pos = tf.reduce_mean(pos) * temperature
        mean_neg = _safe_mean(remaining) * temperature
        return loss, mean_pos, mean_neg


    def hard_contrastive_loss(
        history: tf.Tensor,
        future: tf.Tensor,
        similarity: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        *,
        temperature: float = 0.1,
        tau_plus: float = 0.1,
        beta: float = 0.1,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Hard negative debiased contrastive loss."""

        sim = similarity(history, future)
        pos = tf.linalg.diag_part(sim)
        pos_sim = tf.math.exp(pos / temperature)
        neg = _mask_off_diagonal(sim)
        neg_sim = tf.math.exp(neg / temperature)

        mean_neg_sim = tf.reduce_mean(neg_sim, axis=1, keepdims=True)
        if beta == 0:
            reweight = tf.ones_like(neg_sim)
        else:
            reweight = tf.math.divide_no_nan(beta * neg_sim, mean_neg_sim)

        n = tf.cast(tf.shape(history)[0] - 1, tf.float32)
        numerator = -tau_plus * n * pos_sim + tf.reduce_sum(reweight * neg_sim, axis=-1)
        lower_bound = tf.math.exp(-1.0 / temperature)
        denom = tf.maximum(numerator / (1.0 - tau_plus), lower_bound)
        prob = tf.math.divide_no_nan(pos_sim, pos_sim + denom)
        loss = -tf.reduce_mean(tf.math.log(prob + 1e-12))

        mean_pos = tf.reduce_mean(pos)
        mean_neg = tf.reduce_mean(neg)
        return loss, mean_pos, mean_neg


# Compatibility aliases with the upstream repository naming.
cosine_similarity_dim1 = cosine_similarity_vector  # type: ignore[assignment]
cosine_similarity_dim2 = cosine_similarity_matrix  # type: ignore[assignment]
euclidean_similarity_dim1 = euclidean_similarity_vector  # type: ignore[assignment]
euclidean_similarity_dim2 = euclidean_similarity_matrix  # type: ignore[assignment]
dot_similarity_dim1 = dot_similarity_vector  # type: ignore[assignment]
dot_similarity_dim2 = dot_similarity_matrix  # type: ignore[assignment]
edit_similarity_dim1 = edit_similarity_vector  # type: ignore[assignment]
edit_similarity_dim2 = edit_similarity_matrix  # type: ignore[assignment]
