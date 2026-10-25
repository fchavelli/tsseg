"""Parallel autoencoders of TIRE (torch is an optional dependency of tsseg)."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["_ParallelAutoEncoder"]


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
            z_unshared_all.append(
                z_unshared.unsqueeze(1) if z_unshared.numel() else None
            )
            reconstructed.append(recon.unsqueeze(1))

        z_shared_all = torch.cat(z_shared_all, dim=1)
        if self.latent_dim > self.nr_shared:
            z_unshared_all = torch.cat(
                [t for t in z_unshared_all if t is not None], dim=1
            )
            latent = torch.cat([z_shared_all, z_unshared_all], dim=-1)
        else:
            latent = z_shared_all
        reconstructed = torch.cat(reconstructed, dim=1)
        return reconstructed, z_shared_all, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, z_shared, latent = self.forward(x)
        return latent
