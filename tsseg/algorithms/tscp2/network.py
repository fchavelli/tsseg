"""TensorFlow encoder building blocks for the TS-CP2 detector."""

from __future__ import annotations

from typing import Iterable, Sequence, TYPE_CHECKING

try:  # pragma: no cover - optional dependency guidance
    import tensorflow as _tf
except ImportError:  # pragma: no cover - allow lazy failure
    _tf = None

try:  # pragma: no cover - optional dependency guidance
    from tcn import TCN as _TCN  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - allow lazy failure
    _TCN = None

if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    import tensorflow as tf
    from tcn import TCN  # type: ignore[import-not-found]
else:
    tf = _tf
    TCN = _TCN


if tf is None or TCN is None:  # pragma: no cover - provide a helpful stub

    class TemporalEncoder:  # type: ignore[too-few-public-methods]
        """Placeholder encoder that reports missing dependencies."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple guard
            raise ImportError(
                "TensorFlow and the 'tcn' package are required for TS-CP2. Install the 'tscp2' extra."
            )

else:

    class TemporalEncoder(tf.keras.Model):
        """Temporal convolutional encoder mirroring the original TS-CP2 design."""

        def __init__(
            self,
            *,
            input_features: int,
            window_size: int,
            code_size: int,
            nb_filters: int = 64,
            kernel_size: int = 4,
            dilations: Sequence[int] = (1, 2, 4, 8),
            nb_stacks: int = 2,
            dropout_rate: float = 0.0,
            padding: str = "causal",
            use_skip_connections: bool = True,
            dense_units: Iterable[int] | None = None,
        ) -> None:
            super().__init__(name="tscp2_temporal_encoder")
            self.window_size = int(window_size)
            self.input_features = int(input_features)
            self.code_size = int(code_size)

            if dense_units is None:
                half_steps = max(self.window_size // 2, 1)
                dense_units = (2 * half_steps, half_steps)

            self.tcn = TCN(
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                nb_stacks=nb_stacks,
                dilations=list(dilations),
                activation="relu",
                use_skip_connections=use_skip_connections,
                return_sequences=True,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
                padding=padding,
                kernel_initializer="random_normal",
            )
            self.flatten = tf.keras.layers.Flatten()
            self.dense_layers = [tf.keras.layers.Dense(int(units), activation="relu") for units in dense_units]
            self.projection = tf.keras.layers.Dense(self.code_size)

            dummy = tf.zeros((1, self.window_size, self.input_features), dtype=tf.float32)
            _ = self(dummy, training=False)

        def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
            x = self.tcn(inputs, training=training)
            x = self.flatten(x)
            for dense in self.dense_layers:
                x = dense(x, training=training)
            x = self.projection(x)
            return x
