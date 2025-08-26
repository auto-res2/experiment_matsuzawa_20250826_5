"""
preprocess.py – in the full system this would download datasets and
convert raw trajectories to the format that the diffusion models
expect.  For our toy example we only provide a helper that samples
random high-dimensional SE(3) vectors so the other modules can stay
clean.
"""
from __future__ import annotations

import torch


def sample_se3_vectors(n: int) -> torch.Tensor:
    """Return (n,384) tensor used as toy proxy for SE(3) trajectories."""
    return torch.randn(n, 384)
