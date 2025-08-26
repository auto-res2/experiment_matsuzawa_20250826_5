import os
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# Synthetic data generator (multiple patterns)
# -----------------------------------------------------------------------------

class SyntheticPatterns(Dataset):
    """Generates small images with several patterns to stress different feature statistics.
    Patterns: noise, stripes, checkerboard, radial, gradients.
    """
    def __init__(self, n: int = 512, res: int = 32, num_classes: int = 4, seed: int = 123):
        super().__init__()
        self.n = n
        self.res = res
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        H = W = self.res
        typ = idx % 5
        img = self._make_pattern(H, W, typ)
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        return {"img": img, "label": label}

    def _make_pattern(self, H, W, typ: int) -> torch.Tensor:
        if typ == 0:  # Gaussian noise
            arr = self.rng.randn(H, W, 3)
        elif typ == 1:  # Vertical stripes
            x = np.arange(W)[None, :]  # shape (1, W)
            stripe = np.sin(2 * np.pi * x / 4)  # (1, W)
            stripe = np.repeat(stripe, H, axis=0)  # (H, W)
            arr = np.repeat(stripe[:, :, None], 3, axis=2)  # (H, W, 3)
        elif typ == 2:  # Checkerboard
            yy, xx = np.mgrid[:H, :W]
            arr = (((yy // 4 + xx // 4) % 2)[..., None] * np.ones(3)).astype(np.float32)
        elif typ == 3:  # Radial bullseye
            yy, xx = np.mgrid[:H, :W]
            rr = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
            arr = (np.sin(rr / 2)[..., None] * np.ones(3))
        else:  # Smooth gradient
            yy, xx = np.mgrid[:H, :W]
            arr = (yy[..., None] / H) * (xx[..., None] / W) * np.ones(3)
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        img = torch.from_numpy(arr).permute(2, 0, 1)
        return img


def get_synthetic_dataloader(n: int = 512, res: int = 32, batch: int = 16, shuffle: bool = True,
                             num_workers: int = 0, seed: int = 123) -> DataLoader:
    ds = SyntheticPatterns(n=n, res=res, seed=seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return dl
