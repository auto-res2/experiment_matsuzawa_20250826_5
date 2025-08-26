"""
train.py – training / core algorithm implementations for HADuS++
This file contains all neural-network modules and the code that is
actually "expensive" in the real system.  For the toy reproduction we
only implement the tiny auto-encoders that are used in Experiment-2.

The file purposefully has **no** file-system side-effects – everything
that produces figures / logs is placed in evaluate.py so we can unit-
test the training code independently.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------
# 1.  Encoders / Decoder used in the paper (scaled-down for toy run)
# ---------------------------------------------------------------------

class EucEncoder(nn.Module):
    """128-d Euclidean latent encoder (baseline)."""

    def __init__(self, in_dim: int = 384, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,384) → (B,128)
        return self.net(x)


class HypEncoder(nn.Module):
    """8-d Hyperbolic (actually Euclidean in toy) encoder."""

    def __init__(self, in_dim: int = 384, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,384) → (B,8)
        return self.net(x)


class SharedDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 384)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B,d) → (B,384)
        return self.net(z)


# ---------------------------------------------------------------------
# 2.  Utility: tiny auto-encoder training loop
# ---------------------------------------------------------------------

def train_autoencoder(
    encoder: nn.Module,
    decoder: nn.Module,
    x_full: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Tuple[nn.Module, nn.Module, float]:
    """Return (encoder, decoder, final_MSE)"""
    dataset = TensorDataset(x_full)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for _ in range(epochs):
        for (x,) in loader:
            z = encoder(x)
            recon = decoder(z)
            loss = F.mse_loss(recon, x)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = encoder(x_full)
        recon = decoder(z)
        mse = F.mse_loss(recon, x_full).item()
    return encoder, decoder, mse
