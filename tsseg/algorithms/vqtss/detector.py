from __future__ import annotations

import numpy as np
import pandas as pd
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

    A predictive segmentation model that learns discrete states (codes) 
    by optimizing a future prediction task through a VQ-VAE bottleneck.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": False,
        "detector_type": "state_detection",
        "python_dependencies": ["torch"],
        "capability:unsupervised": True,
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
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 10,
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
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        
        self.model_ = None
        self._active_codes = None

    def _fit(self, X: np.ndarray, y=None) -> VQTSSDetector:
        # X shape: (n_timepoints, n_channels)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        n_timepoints, n_channels = X.shape
        
        # Prepare data: Sliding windows
        # We want to predict X[t+1], so we need windows of size window_size + 1
        # Input: X[t : t+window_size]
        # Target: X[t+1 : t+window_size+1]
        
        windows = []
        targets = []
        
        # Simple sliding window generation
        # Note: For very large datasets, a custom Dataset class is better to avoid memory duplication
        # Here we assume it fits in memory for simplicity as per typical tsseg usage
        
        # We need window_size + 1 to have input and shifted target
        seq_len = self.window_size + 1
        
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._X_mean) / self._X_std
        
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        
        # Create windows
        # Unfold: (n_timepoints - seq_len + 1, seq_len, n_channels)
        if n_timepoints < seq_len:
             raise ValueError(f"Data length {n_timepoints} is smaller than window_size+1 {seq_len}")

        # (Batch, Channels, Time)
        windows_tensor = X_tensor.unfold(0, seq_len, self.stride)
        # Permute is not needed if X is (Time, Channels) and we unfold dim 0
        # unfold puts the window dimension at the end: (Batch, Channels, WindowTime)

        dataset = TensorDataset(windows_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Init model
        self.model_ = PredictiveVQTSS(
            input_dim=n_channels,
            hidden_dim=self.hidden_dim,
            num_embeddings=self.num_embeddings,
            commitment_cost=self.commitment_cost,
            decay=self.decay
        ).to(self.device)
        
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                # batch[0] shape: (Batch, Channels, Time=window_size+1)
                seq = batch[0].to(self.device)
                
                # Input: first window_size
                inp = seq[:, :, :-1]
                # Target: last window_size (shifted by 1)
                target = seq[:, :, 1:]
                
                optimizer.zero_grad()
                
                # Forward pass on input
                # z_pred: Predicted latent for t+1 based on quantized t
                z_pred, vq_loss, _, z_q = self.model_(inp)
                
                # Forward pass on target to get ground truth latents
                # We use the encoder to get the continuous latent of the target
                z_target = self.model_.encoder(target)
                
                # InfoNCE Loss (Contrastive)
                # We want z_pred[b, t] to be close to z_target[b, t]
                # and far from other z_target[b', t']
                
                # Flatten batch and time dimensions for contrastive learning
                # (B, H, T) -> (B*T, H)
                z_pred_flat = z_pred.transpose(1, 2).reshape(-1, self.hidden_dim)
                z_target_flat = z_target.transpose(1, 2).reshape(-1, self.hidden_dim)
                
                # Normalize embeddings
                z_pred_norm = F.normalize(z_pred_flat, dim=1)
                z_target_norm = F.normalize(z_target_flat, dim=1)
                
                # Compute similarity matrix (B*T, B*T)
                temperature = 0.1
                
                logits = torch.matmul(z_pred_norm, z_target_norm.T) / temperature
                
                # Labels are the diagonal indices (0, 1, 2, ...)
                labels = torch.arange(logits.shape[0], device=self.device)
                
                contrastive_loss = F.cross_entropy(logits, labels)
                
                # Smoothness loss: penalize rapid changes in the latent code
                # z_q shape: (Batch, Hidden, Time)
                diff = z_q[:, :, 1:] - z_q[:, :, :-1]
                smooth_loss = torch.mean(diff ** 2)
                
                # Total Loss
                loss = contrastive_loss + vq_loss + self.smoothness_weight * smooth_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("The model has not been fitted yet.")
            
        self.model_.eval()
        n_timepoints, n_channels = X.shape
        X_norm = (X - self._X_mean) / self._X_std
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        
        # Input shape: (1, Channels, Time)
        inp = X_tensor.unsqueeze(0).permute(0, 2, 1).to(self.device)
        
        with torch.no_grad():
            # We only care about the state indices
            # Forward returns: x_pred, vq_loss, state_indices, z_q
            _, _, state_indices, _ = self.model_(inp)
        
        codes = state_indices.squeeze(0).cpu().numpy()
        
        return codes.astype(int)

    def get_active_states(self):
        """Returns the set of unique codes used by the model."""
        pass
