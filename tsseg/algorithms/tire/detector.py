from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import find_peaks

from ..base import BaseSegmenter

from . import utils


@dataclass
class _TireConfig:
    window_size: int
    stride: int
    domain: Literal["TD", "FD", "both"]
    intermediate_dim_td: int
    latent_dim_td: int
    nr_shared_td: int
    nr_ae_td: int
    loss_weight_td: float
    intermediate_dim_fd: int
    latent_dim_fd: int
    nr_shared_fd: int
    nr_ae_fd: int
    loss_weight_fd: float
    nfft: int
    norm_mode: Literal["window", "timeseries"]
    peak_distance_fraction: float
    max_epochs: int
    patience: int
    learning_rate: float


class _ParallelAutoEncoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        intermediate_dim: int,
        latent_dim: int,
        nr_ae: int,
        nr_shared: int,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.nr_ae = nr_ae
        self.nr_shared = nr_shared

        input_dim = window_size
        self.encoder_hidden = (
            nn.Linear(input_dim, intermediate_dim) if intermediate_dim > 0 else None
        )
        self.encoder_activation = nn.ReLU() if intermediate_dim > 0 else None

        self.encoder_shared = nn.Linear(
            intermediate_dim if intermediate_dim > 0 else input_dim,
            nr_shared,
        )
        self.encoder_unshared = nn.Linear(
            intermediate_dim if intermediate_dim > 0 else input_dim,
            latent_dim - nr_shared if latent_dim > nr_shared else 0,
        )

        self.decoder_hidden = (
            nn.Linear(latent_dim, intermediate_dim) if intermediate_dim > 0 else None
        )
        self.decoder_activation = nn.ReLU() if intermediate_dim > 0 else None
        self.decoder_out = nn.Linear(
            intermediate_dim if intermediate_dim > 0 else latent_dim, input_dim
        )
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent representation."""
        z_shared_all = []
        z_unshared_all = []
        reconstructed = []
        for i in range(self.nr_ae):
            xi = x[:, i, :]
            h = xi
            if self.encoder_hidden is not None:
                h = self.encoder_activation(self.encoder_hidden(h))
            z_shared = self.tanh(self.encoder_shared(h))
            if self.latent_dim > self.nr_shared:
                z_unshared = self.tanh(self.encoder_unshared(h))
                z = torch.cat([z_shared, z_unshared], dim=-1)
            else:
                z_unshared = torch.empty(0, device=x.device)
                z = z_shared

            y = z
            if self.decoder_hidden is not None:
                y = self.decoder_activation(self.decoder_hidden(y))
            recon = self.tanh(self.decoder_out(y))

            z_shared_all.append(z_shared.unsqueeze(1))
            z_unshared_all.append(z_unshared.unsqueeze(1) if z_unshared.numel() else None)
            reconstructed.append(recon.unsqueeze(1))

        z_shared_all = torch.cat(z_shared_all, dim=1)
        if self.latent_dim > self.nr_shared:
            z_unshared_all = torch.cat([t for t in z_unshared_all if t is not None], dim=1)
            latent = torch.cat([z_shared_all, z_unshared_all], dim=-1)
        else:
            latent = z_shared_all
        reconstructed = torch.cat(reconstructed, dim=1)
        return reconstructed, z_shared_all, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, z_shared, latent = self.forward(x)
        return latent


class TireDetector(BaseSegmenter):
    """TIRE (Time-Invariant Representation) change point detector."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "returns_dense": True,
        "fit_is_empty": False,
        "python_dependencies": "torch",
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    def __init__(
        self,
        window_size: int = 20,
        stride: int = 1,
        domain: Literal["TD", "FD", "both"] = "both",
        intermediate_dim_td: int = 0,
        latent_dim_td: int = 1,
        nr_shared_td: int = 1,
        nr_ae_td: int = 3,
        loss_weight_td: float = 1.0,
        intermediate_dim_fd: int = 10,
        latent_dim_fd: int = 1,
        nr_shared_fd: int = 1,
        nr_ae_fd: int = 3,
        loss_weight_fd: float = 1.0,
        nfft: int = 30,
        norm_mode: Literal["window", "timeseries"] = "timeseries",
        peak_distance_fraction: float = 0.01,
        max_epochs: int = 20,
        patience: int = 5,
        learning_rate: float = 1e-3,
        *,
        n_segments: int | None = None,
        axis: int = 0,
        random_state: int | None = None,
    ) -> None:
        if window_size < 4:
            raise ValueError("window_size must be at least 4")
        super().__init__(axis=axis)

        # expose init parameters for get_params compatibility
        self.window_size = window_size
        self.stride = stride
        self.domain = domain
        self.intermediate_dim_td = intermediate_dim_td
        self.latent_dim_td = latent_dim_td
        self.nr_shared_td = nr_shared_td
        self.nr_ae_td = nr_ae_td
        self.loss_weight_td = loss_weight_td
        self.intermediate_dim_fd = intermediate_dim_fd
        self.latent_dim_fd = latent_dim_fd
        self.nr_shared_fd = nr_shared_fd
        self.nr_ae_fd = nr_ae_fd
        self.loss_weight_fd = loss_weight_fd
        self.nfft = nfft
        self.norm_mode = norm_mode
        self.peak_distance_fraction = peak_distance_fraction
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_segments = n_segments

        self.config = _TireConfig(
            window_size=window_size,
            stride=stride,
            domain=domain,
            intermediate_dim_td=intermediate_dim_td,
            latent_dim_td=latent_dim_td,
            nr_shared_td=nr_shared_td,
            nr_ae_td=nr_ae_td,
            loss_weight_td=loss_weight_td,
            intermediate_dim_fd=intermediate_dim_fd,
            latent_dim_fd=latent_dim_fd,
            nr_shared_fd=nr_shared_fd,
            nr_ae_fd=nr_ae_fd,
            loss_weight_fd=loss_weight_fd,
            nfft=nfft,
            norm_mode=norm_mode,
            peak_distance_fraction=peak_distance_fraction,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
        )
        self._rng = np.random.default_rng(random_state)
        torch.manual_seed(random_state or 0)

    # ---------------------------- helper utilities ------------------------- #
    def _prepare_windows(self, X: np.ndarray) -> np.ndarray:
        window_size = self.config.window_size
        stride = self.config.stride
        if X.ndim == 1:
            return utils.ts_to_windows(X, 0, window_size, stride)

        windows_per_channel = [
            utils.ts_to_windows(X[:, i], 0, window_size, stride) for i in range(X.shape[1])
        ]
        stacked = np.stack(windows_per_channel)
        return utils.combine_ts(stacked)

    @staticmethod
    def _prepare_input_paes(windows: np.ndarray, nr_ae: int) -> np.ndarray:
        nr_windows = windows.shape[0]
        sequences = []
        for i in range(nr_ae):
            sequences.append(windows[i : nr_windows - nr_ae + 1 + i])
        return np.transpose(np.array(sequences), (1, 0, 2))

    def _train_autoencoder(
        self,
        windows: np.ndarray,
        intermediate_dim: int,
        latent_dim: int,
        nr_shared: int,
        nr_ae: int,
        loss_weight: float,
    ) -> np.ndarray:
        if windows.shape[0] < nr_ae + 1:
            raise ValueError(
                "Not enough windows to train the parallel autoencoders. Try reducing nr_ae or window_size."
            )

        device = torch.device("cpu")
        model = _ParallelAutoEncoder(
            window_size=windows.shape[-1],
            intermediate_dim=intermediate_dim,
            latent_dim=latent_dim,
            nr_ae=nr_ae,
            nr_shared=nr_shared,
        ).to(device)

        new_windows = self._prepare_input_paes(windows, nr_ae)
        tensor = torch.from_numpy(new_windows).float().to(device)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        best_loss = math.inf
        epochs_no_improve = 0

        for epoch in range(self.config.max_epochs):
            model.train()
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon, z_shared, _ = model(batch)
                mse_loss = torch.mean((batch - recon) ** 2)
                if z_shared.shape[1] > 1:
                    diff = z_shared[:, 1:, :] - z_shared[:, :-1, :]
                    shared_loss = torch.mean(diff**2)
                else:
                    shared_loss = torch.tensor(0.0, device=device)
                loss = mse_loss + loss_weight * shared_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)

            epoch_loss = total_loss / len(dataset)
            if epoch_loss + 1e-6 < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.patience:
                    break

        if "best_state" in locals():
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            encoded = model.encode(tensor).cpu().numpy()

        # Reconstruct shared representations similar to original implementation
        nr_windows = windows.shape[0]
        head = encoded[:, 0, :nr_shared]
        tail = encoded[-nr_ae + 1 :, nr_ae - 1, :nr_shared]
        combined = np.concatenate((head, tail), axis=0)
        # Ensure combined length matches original windows count
        if combined.shape[0] > nr_windows:
            combined = combined[:nr_windows]
        elif combined.shape[0] < nr_windows:
            pad = np.repeat(combined[-1:, :], nr_windows - combined.shape[0], axis=0)
            combined = np.concatenate((combined, pad), axis=0)
        return combined

    def _compute_latent_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        windows_td = self._prepare_windows(X)
        features_td = self._train_autoencoder(
            windows_td,
            self.config.intermediate_dim_td,
            self.config.latent_dim_td,
            self.config.nr_shared_td,
            self.config.nr_ae_td,
            self.config.loss_weight_td,
        )

        features_fd = None
        if self.config.domain in {"FD", "both"}:
            windows_fd = utils.calc_fft(
                windows_td, nfft=self.config.nfft, norm_mode=self.config.norm_mode
            )
            features_fd = self._train_autoencoder(
                windows_fd,
                self.config.intermediate_dim_fd,
                self.config.latent_dim_fd,
                self.config.nr_shared_fd,
                self.config.nr_ae_fd,
                self.config.loss_weight_fd,
            )
        return features_td, features_fd

    def _run_pipeline(self, X: np.ndarray) -> np.ndarray:
        features_td, features_fd = self._compute_latent_features(X)

        if self.config.domain == "TD":
            distances = utils.distance(features_td, self.config.window_size)
        elif self.config.domain == "FD":
            distances = utils.distance(features_fd, self.config.window_size)
        else:
            beta = np.quantile(utils.distance(features_td, self.config.window_size), 0.95)
            alpha = np.quantile(utils.distance(features_fd, self.config.window_size), 0.95)
            combined = np.concatenate(
                (features_td * alpha, features_fd * beta), axis=1
            )
            combined = utils.matched_filter(combined, self.config.window_size)
            distances = utils.distance(combined, self.config.window_size)

        distances = utils.matched_filter(distances, self.config.window_size)
        scores = utils.new_peak_prominences(distances)[0]
        scores = np.array(scores)
        if scores.size == 0:
            scores = np.zeros_like(distances)
        scores = scores / (np.max(scores) + 1e-8)
        change_scores = np.concatenate(
            (
                np.zeros((self.config.window_size,)),
                scores,
                np.zeros((self.config.window_size - 1,)),
            )
        )
        return change_scores

    def _fit(self, X, y=None):
        self._rng = np.random.default_rng(self.random_state)
        self._fit_data = np.asarray(X, dtype=float)
        return self

    def _predict(self, X, axis=None):
        X = np.asarray(X, dtype=float)
        if axis is None:
            axis = self.axis
        if axis != 0:
            X = np.moveaxis(X, axis, 0)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        scores = self._run_pipeline(X)

        if self.n_segments is None or self.n_segments < 2:
            # Return peaks detected with default prominence
            peaks, _ = find_peaks(scores, distance=max(1, int(self.config.window_size * self.config.peak_distance_fraction)), prominence=0.1)
            return peaks.astype(int)

        peaks, properties = find_peaks(
            scores,
            distance=max(1, int(self.config.window_size * self.config.peak_distance_fraction)),
            prominence=0,  # request prominence computation so we can rank peaks
        )
        if peaks.size == 0:
            return peaks.astype(int)
        prominences = properties["prominences"]
        order = np.argsort(-prominences)
        top_k = peaks[order][: self.n_segments - 1]
        top_k = np.sort(top_k)
        return top_k.astype(int)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "window_size": 16,
            "stride": 4,
            "n_segments": 3,
            "max_epochs": 5,
            "patience": 2,
            "intermediate_dim_td": 8,
            "latent_dim_td": 2,
            "nr_shared_td": 1,
            "nr_ae_td": 2,
            "intermediate_dim_fd": 8,
            "latent_dim_fd": 2,
            "nr_shared_fd": 1,
            "nr_ae_fd": 2,
            "domain": "TD",
        }
