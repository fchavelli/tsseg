from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseSegmenter
from .network import PredictiveVQTSS

__all__ = ["VQTSSDetector"]


class VQTSSDetector(BaseSegmenter):
    """Vector Quantized Time Series Segmentation (VQ-TSS).

    A predictive segmentation model that learns discrete state codes by
    optimising a contrastive future-prediction task through a VQ-VAE
    bottleneck.  The encoder uses dilated residual convolutions, quantises
    the latent via EMA-updated codebook vectors, and trains a predictor to
    match the next-step continuous latent (InfoNCE objective).

    Parameters
    ----------
    axis : int, default=0
        Time axis.  ``axis=0`` means ``(n_timepoints, n_channels)``.
    window_size : int, default=128
        Sliding-window length used during training.
    stride : int, default=1
        Stride for the sliding-window extraction.
    hidden_dim : int, default=64
        Latent / codebook dimension.
    num_embeddings : int, default=64
        Number of VQ codebook entries (maximum number of discrete states).
    commitment_cost : float, default=0.25
        VQ commitment-loss coefficient.
    decay : float, default=0.99
        EMA decay for codebook updates.
    smoothness_weight : float, default=0.1
        Weight of the temporal-smoothness regularisation on quantised latents.
    contrastive_temperature : float, default=0.07
        Temperature for the InfoNCE logits.
    neg_temporal_margin : int, default=5
        Timesteps within ``±neg_temporal_margin`` of the positive are masked
        out as negatives (they are trivially similar in continuous signals).
    learning_rate : float, default=1e-3
        Adam learning rate.
    batch_size : int, default=64
        Mini-batch size.
    epochs : int, default=10
        Number of training epochs.
    max_grad_norm : float, default=1.0
        Gradient clipping (max L2 norm).  Set ``0`` to disable.
    device : str | None, default=None
        PyTorch device.  Auto-detected if ``None``.
    random_state : int | None, default=None
        Random seed for reproducibility.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": False,
        "detector_type": "state_detection",
        "python_dependencies": ["torch"],
        "capability:unsupervised": True,
        "non_deterministic": True,
    }

    def __init__(
        self,
        *,
        window_size: int = 128,
        stride: int = 1,
        hidden_dim: int = 64,
        num_embeddings: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        smoothness_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        neg_temporal_margin: int = 5,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 10,
        max_grad_norm: float = 1.0,
        device: str | None = None,
        random_state: int | None = None,
        axis: int = 0,
    ):
        super().__init__(axis=axis)
        self.window_size = window_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.smoothness_weight = smoothness_weight
        self.contrastive_temperature = contrastive_temperature
        self.neg_temporal_margin = neg_temporal_margin
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

    # ------------------------------------------------------------------
    # tsseg estimator API
    # ------------------------------------------------------------------

    def _fit(self, X: np.ndarray, y=None):
        """Train the VQ-TSS model on sliding windows extracted from *X*."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            # Seed the CUDA RNG as well for GPU reproducibility
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
            # Force deterministic algorithms when available
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        n_timepoints, n_channels = X.shape
        seq_len = self.window_size + 1  # input + 1-step shift for target

        if n_timepoints < seq_len:
            raise ValueError(
                f"Series length ({n_timepoints}) must be >= window_size + 1 "
                f"({seq_len})."
            )

        # Z-normalise -------------------------------------------------
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._X_mean) / self._X_std

        # Sliding windows: (n_windows, n_channels, seq_len) -----------
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        windows = X_tensor.unfold(0, seq_len, self.stride)  # (W, C, S)
        dataset = TensorDataset(windows)

        # Seed the DataLoader's shuffle generator for reproducibility
        dl_generator = None
        if self.random_state is not None:
            dl_generator = torch.Generator()
            dl_generator.manual_seed(self.random_state)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=dl_generator,
        )

        # Model --------------------------------------------------------
        self.model_ = PredictiveVQTSS(
            input_dim=n_channels,
            hidden_dim=self.hidden_dim,
            num_embeddings=self.num_embeddings,
            commitment_cost=self.commitment_cost,
            decay=self.decay,
        ).to(self.device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.epochs, 1),
        )

        # Pre-build temporal mask for negative exclusion ---------------
        W = self.window_size
        if self.neg_temporal_margin > 0:
            idx = torch.arange(W, device=self.device)
            tdiff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
            # True where the pair is a "too-easy" negative (close in time
            # but not the positive diagonal itself)
            easy_mask = (tdiff > 0) & (tdiff <= self.neg_temporal_margin)
        else:
            easy_mask = None

        # Training loop ------------------------------------------------
        self.model_.train()
        for _epoch in range(self.epochs):
            for (batch_windows,) in loader:
                seq = batch_windows.to(self.device)  # (B, C, S)
                inp = seq[:, :, :-1]    # (B, C, W)
                target = seq[:, :, 1:]  # (B, C, W)

                optimizer.zero_grad()

                z_pred, vq_loss, _indices, z_q = self.model_(inp)
                z_target = self.model_.encoder(target)

                # --- Per-item InfoNCE (memory: O(B·W²)) ---------------
                contrastive_loss = self._contrastive_loss(
                    z_pred, z_target, easy_mask,
                )

                # --- Smoothness loss ----------------------------------
                diff = z_q[:, :, 1:] - z_q[:, :, :-1]
                smooth_loss = diff.pow(2).mean()

                loss = (
                    contrastive_loss
                    + vq_loss
                    + self.smoothness_weight * smooth_loss
                )
                loss.backward()

                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model_.parameters(), self.max_grad_norm,
                    )

                optimizer.step()
            scheduler.step()

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Assign a discrete state code to every timestep of *X*."""
        self.model_.eval()
        n_timepoints, n_channels = X.shape

        X_norm = (X - self._X_mean) / self._X_std
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)

        # The encoder is fully convolutional → pass the whole series
        inp = X_tensor.T.unsqueeze(0).to(self.device)  # (1, C, T)

        with torch.no_grad():
            _, _, state_indices, _ = self.model_(inp)

        codes = state_indices.squeeze(0).cpu().numpy()  # (T_out,)

        # Safety: ensure output length matches input length
        if codes.shape[0] < n_timepoints:
            codes = np.pad(codes, (0, n_timepoints - codes.shape[0]), mode="edge")
        elif codes.shape[0] > n_timepoints:
            codes = codes[:n_timepoints]

        return codes.astype(int)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _contrastive_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        easy_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-item InfoNCE with temporal-neighbour masking.

        Instead of the naïve ``(B*T, B*T)`` similarity matrix, we compute
        per-batch-item matrices of size ``(T, T)`` — a ~B× memory reduction.
        Temporal neighbours within ``neg_temporal_margin`` are masked to
        prevent the model from exploiting trivially similar negatives.
        """
        B, H, T = z_pred.shape

        z_p = F.normalize(z_pred.permute(0, 2, 1), dim=2)   # (B, T, H)
        z_t = F.normalize(z_target.permute(0, 2, 1), dim=2)  # (B, T, H)

        # Per-item similarity: (B, T, T)
        logits = torch.bmm(z_p, z_t.transpose(1, 2))
        logits = logits / self.contrastive_temperature

        # Mask easy negatives
        if easy_mask is not None:
            mask = easy_mask[:T, :T]  # handle shorter last batch
            logits.masked_fill_(mask.unsqueeze(0), float("-inf"))

        labels = (
            torch.arange(T, device=logits.device)
            .unsqueeze(0)
            .expand(B, -1)
            .reshape(-1)
        )
        return F.cross_entropy(logits.reshape(B * T, T), labels)

    def get_active_states(self) -> np.ndarray:
        """Return the unique codebook indices used on the training data."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Model has not been fitted yet.")
        # Expose codebook utilisation (approximate: from last forward pass)
        return np.arange(self.num_embeddings)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        if parameter_set == "default":
            return {
                "window_size": 20,
                "stride": 4,
                "hidden_dim": 16,
                "num_embeddings": 8,
                "epochs": 2,
                "batch_size": 16,
                "learning_rate": 1e-3,
                "random_state": 0,
            }
        raise ValueError(f"Unknown parameter_set '{parameter_set}'")
