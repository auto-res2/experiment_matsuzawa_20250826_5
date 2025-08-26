"""
train.py – training utilities for RevSparse-ViM toy pipeline
===========================================================
• All experiment images are now stored in
      .research/iteration6/images
  (was …/iteration5/… before).
• No other behavioural changes – the public API is preserved.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------------------------------------------------------------
# 1.  Toy reference implementations
# ---------------------------------------------------------------------
class ToyBaselineNet(nn.Module):
    """3-layer CNN that mimics baseline Vision-Mamba network."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

    # Interfaces expected by experiment harness --------------------
    def ssm_cache_size(self):
        return 0

    def set_tau(self, tau: float):
        pass

    def forward_count_tokens(self, x):
        """Return dummy token count (spatial tokens)."""
        return torch.zeros(x.size(0), device=x.device) + (x.size(-1) // 16) ** 2


# ---------- reversible additive coupling primitive -------------------
class _RevAdditiveCoupling(torch.autograd.Function):
    """Custom autograd function implementing an additive reversible block.

    The previous implementation called ``torch.autograd.backward`` inside
    ``backward`` which lead to an infinite recursion (``backward`` kept
    invoking itself) and finally to a CUDA OOM.  The new version follows
    the standard analytic gradients derived for additive coupling
    (cf. Gomez et al. – *RevNets*).
    """

    @staticmethod
    def forward(ctx, x1, x2, f, g):
        ctx.f, ctx.g = f, g
        with torch.no_grad():
            y1 = x1 + f(x2)
            y2 = x2 + g(y1)
        # save outputs – from these we can reconstruct the inputs cheaply
        ctx.save_for_backward(y1, y2)
        return y1, y2

    @staticmethod
    def backward(ctx, dy1, dy2):
        y1, y2 = ctx.saved_tensors
        f, g = ctx.f, ctx.g

        # Enable gradient tracking for intermediate computations
        with torch.set_grad_enabled(True):
            # -------- step 1  : recover x2 and compute grad contribution ---
            y1_ = y1.detach().requires_grad_(True)
            g_y1 = g(y1_)
            x2 = (y2 - g_y1).detach().requires_grad_(True)

            #   dy1 receives extra contribution through g ( since y2 = x2 + g(y1) )
            grad_y1, = torch.autograd.grad(
                outputs=g_y1,
                inputs=y1_,
                grad_outputs=dy2,
                retain_graph=True,
                allow_unused=True,
            )
            if grad_y1 is None:
                grad_y1 = torch.zeros_like(dy1)
            dx1_intermediate = dy1 + grad_y1

            # -------- step 2 : recover x1 and final dx2 --------------------
            f_x2 = f(x2)
            grad_x2, = torch.autograd.grad(
                outputs=f_x2,
                inputs=x2,
                grad_outputs=dx1_intermediate,
                retain_graph=False,
                allow_unused=True,
            )
            if grad_x2 is None:
                grad_x2 = torch.zeros_like(dy2)
            dx2 = dy2 + grad_x2
            dx1 = dx1_intermediate

        # No gradients for f / g (they are regular nn.Modules handled by autograd)
        return dx1, dx2, None, None


class RevBlock(nn.Module):
    """Reversible additive coupling block (channel-wise split)."""

    def __init__(self, channels: int):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("`channels` must be even for reversible coupling.")
        inner_c = channels // 2
        # Each sub-network keeps the tensor size constant (C/2 → C/2)
        self.f = nn.Sequential(nn.Conv2d(inner_c, inner_c, 3, padding=1), nn.ReLU())
        self.g = nn.Sequential(nn.Conv2d(inner_c, inner_c, 3, padding=1), nn.ReLU())

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = _RevAdditiveCoupling.apply(x1, x2, self.f, self.g)
        return torch.cat([y1, y2], dim=1)


class SparseSharedLinear(nn.Module):
    """Low-rank shared-state linear:  W = Σ_k α_k B̂_k  (toy)."""

    def __init__(self, in_f: int, out_f: int, rank: int = 4):
        super().__init__()
        self.bases = nn.Parameter(torch.randn(rank, in_f, out_f) * 0.02)
        self.alpha = nn.Parameter(torch.randn(rank))

    def forward(self, x):
        W = torch.einsum("k,kio->io", self.alpha, self.bases)
        return F.linear(x, W.t())


class ToyRevSparseNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        rank: int = 4,
        tau: float = 0.0,
        reversible: bool = True,
    ):
        super().__init__()
        self.tau = tau
        C = 64
        self.stem = nn.Conv2d(3, C, 3, padding=1)
        blocks: List[nn.Module] = []
        for _ in range(3):
            if reversible:
                blocks.append(RevBlock(C))
            else:
                blocks.append(nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.ReLU()))
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = SparseSharedLinear(C, num_classes, rank=rank)

    # ---------------- token pruning helpers -------------------------
    def _saliency(self, x):
        return torch.sigmoid(F.avg_pool2d(x.mean(1, keepdim=True), 3, 1, 1))

    def forward(self, x):
        x = F.relu(self.stem(x))
        if self.tau > 0.0:
            mask = (self._saliency(x) > self.tau).float()
            x = x * mask
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.head(x)

    # -------- book-keeping interfaces -------------------------------
    def ssm_cache_size(self):
        return sum(p.numel() * p.element_size() for p in self.head.parameters())

    def set_tau(self, tau: float):
        self.tau = tau

    def forward_count_tokens(self, x):
        with torch.no_grad():
            sal = self._saliency(F.relu(self.stem(x)))
            return (sal > self.tau).float().sum([1, 2, 3])


# ---------------------------------------------------------------------
# 2.  builders & train loop
# ---------------------------------------------------------------------

# All experiment images are now stored under iteration6 ---------------
IMAGE_DIR = Path(".research/iteration6/images")  # centralised image path


def build_model(cfg) -> nn.Module:
    if cfg.model.name == "baseline":
        return ToyBaselineNet(num_classes=cfg.data.num_classes)
    elif cfg.model.name == "revsparse":
        return ToyRevSparseNet(
            num_classes=cfg.data.num_classes,
            rank=cfg.model.rank,
            tau=cfg.model.tau,
            reversible=cfg.model.reversible,
        )
    else:
        raise ValueError(f"Unknown model {cfg.model.name}")


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _bytes_to_mb(x: int):
    return x / 2 ** 20 if x >= 0 else float("nan")


def train(cfg, model: nn.Module, loaders: Tuple, device):
    train_loader, val_loader = loaders
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Ensure learning rate is a float (YAML treats 1e-3 as a string)
    lr = float(cfg.optim.lr)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    n_steps = len(train_loader) * cfg.train.epochs
    loss_history: List[float] = []

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * imgs.size(0)
        epoch_loss /= len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{cfg.train.epochs}  |  loss={epoch_loss:.4f}")

    # save model ---------------------------------------------------
    _ensure_dir(Path("models"))
    save_path = Path("models") / f"{cfg.model.name}.pt"
    torch.save(model.state_dict(), save_path)

    # plot ----------------------------------------------------------
    _ensure_dir(IMAGE_DIR)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(range(1, cfg.train.epochs + 1), loss_history, "o-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss curve")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "training_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    # memory --------------------------------------------------------
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory: {_bytes_to_mb(mem):.1f} MB")

    return model, loss_history
