import os
import math
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .preprocess import get_synthetic_dataloader

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def pick_device(user_choice: str = "auto") -> str:
    if user_choice.lower() == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return user_choice


def set_torch_seed(seed: int = 123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Tiny UNet Teacher + DF-Diff components (toy but faithful to method)
# -----------------------------------------------------------------------------

class TinyBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(1, c)

    def forward(self, x):
        h = self.act(self.norm(self.conv1(x)))
        h = self.norm(self.conv2(h))
        return self.act(h + x)


def sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1 + 1e-8))
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TinyUNetTeacher(nn.Module):
    def __init__(self, c=32, time_dim=64):
        super().__init__()
        self.c = c
        self.time_dim = time_dim
        # Time embedding MLP
        self.time_mlp1 = nn.Linear(time_dim, c)
        self.time_mlp2 = nn.Linear(c, c)
        # Encoder levels
        self.enc0 = TinyBlock(c)
        self.down = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.enc1 = TinyBlock(c)
        # Decoder and head
        self.up = nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
        self.dec = TinyBlock(c)
        self.head = nn.Conv2d(c, 3, 1)
        self.inp = nn.Conv2d(3, c, 1)

    @property
    def time_embed_dim(self):
        return self.time_dim

    def time_embed(self, tvec: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.time_mlp1(tvec))
        return self.time_mlp2(h)

    def forward_with_features(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None):
        B = x.size(0)
        tvec = sinusoidal_time_embed(t, self.time_dim)
        temb = self.time_embed(tvec)
        h = self.inp(x)
        h = h + temb[:, :, None, None]
        f0 = self.enc0(h)
        h1 = self.down(f0)
        h1 = h1 + temb[:, :, None, None]
        f1 = self.enc1(h1)
        # Decode
        u = self.up(f1)
        u = u + f0
        u = u + temb[:, :, None, None]
        d = self.dec(u)
        y = torch.sigmoid(self.head(d))
        return y, [f0, f1]

    # For DF-Diff wrapper API compatibility
    def forward_block_i(self, i: int, Fi_prev: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor:
        if i == 0:
            return self.enc0(Fi_prev)
        elif i == 1:
            return self.enc1(Fi_prev)
        else:
            raise IndexError("Invalid feature level")

    def final_from_features(self, feats: List[torch.Tensor], t: torch.Tensor, cond=None) -> torch.Tensor:
        f0, f1 = feats[0], feats[1]
        u = self.up(f1)
        u = u + f0
        tvec = sinusoidal_time_embed(t, self.time_dim)
        temb = self.time_embed(tvec)
        u = u + temb[:, :, None, None]
        d = self.dec(u)
        y = torch.sigmoid(self.head(d))
        return y


class PerChannelQuant:
    def __init__(self, nbits=8, eps=1e-8):
        assert nbits in [4, 6, 8]
        self.nbits = nbits
        self.eps = eps
        self.maxq = float(2 ** (nbits - 1) - 1)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C = x.shape[:2]
        x_ = x.view(B, C, -1)
        scale = x_.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps) / self.maxq
        q = torch.clamp((x_ / scale).round(), -self.maxq, self.maxq)
        q = q.to(torch.int8)
        return q.view_as(x), scale.view(B, C, 1, 1)

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q.float() * scale


class DepthwiseSeparable(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.pw(self.act(self.dw(x)))


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 2.0):
        super().__init__()
        h = max(4, int(dim * hidden_mult))
        self.fc1 = nn.Conv2d(dim, h, 1)
        self.fc2 = nn.Conv2d(h, dim * 2, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        d_g = self.fc2(h)
        d, g = d_g.chunk(2, dim=1)
        g = torch.sigmoid(g)
        return d * g


class DeltaUpdater(nn.Module):
    def __init__(self, dim: int, mlp_mult: float = 2.0):
        super().__init__()
        self.conv = DepthwiseSeparable(dim)
        self.mlp = GatedMLP(dim, hidden_mult=mlp_mult)

    def forward(self, x, t_embed):
        y = self.conv(x)
        y = y + t_embed[:, :, None, None]
        return self.mlp(y)


class BinaryRouter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, 1, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, t_embed, step_idx: int, total_steps: int, training: bool = True, tau: float = 1.0):
        logits = (self.proj(x) + t_embed[:, :1, None, None]).mean(dim=(2, 3))
        frac = step_idx / max(1, total_steps - 1)
        schedule = 2.0 * (0.5 - frac)
        logits = logits + self.bias + schedule
        p = torch.sigmoid(logits / tau)
        if training:
            z = (p > 0.5).float() + (p - p.detach())
        else:
            z = (p > 0.5).float()
        return z.squeeze(1), p.squeeze(1)


class DFDiffUNetWrapper(nn.Module):
    def __init__(self, base_unet: TinyUNetTeacher, feature_dims: List[int], delta_mult=1.0, feature_bits=8):
        super().__init__()
        self.base = base_unet
        self.feature_dims = feature_dims
        self.delta_blocks = nn.ModuleList([DeltaUpdater(d, mlp_mult=delta_mult) for d in feature_dims])
        self.routers = nn.ModuleList([BinaryRouter(d) for d in feature_dims])
        self.t_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(self.base.time_embed_dim, d), nn.SiLU(), nn.Linear(d, d))
            for d in feature_dims
        ])
        self.quant = PerChannelQuant(nbits=feature_bits)
        self.register_buffer("have_cache", torch.tensor(0, dtype=torch.uint8), persistent=False)
        self.cache_q: List[Optional[torch.Tensor]] = [None for _ in feature_dims]
        self.cache_s: List[Optional[torch.Tensor]] = [None for _ in feature_dims]

    def clear_cache(self):
        self.have_cache.zero_()
        self.cache_q = [None for _ in self.feature_dims]
        self.cache_s = [None for _ in self.feature_dims]

    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        tvec = sinusoidal_time_embed(t, self.base.time_embed_dim)
        return self.base.time_embed(tvec)

    @torch.no_grad()
    def forward_first_step(self, x, t, cond=None):
        y, feats = self.base.forward_with_features(x, t, cond)
        self.cache_q, self.cache_s = [], []
        for Fi in feats:
            q, s = self.quant.quantize(Fi)
            self.cache_q.append(q)
            self.cache_s.append(s)
        self.have_cache.fill_(1)
        return y

    def forward_with_delta(self, x, t, step_idx: int, total_steps: int, cond=None,
                           router_sparsity_lambda: float = 0.0, router_stats: Optional[dict] = None,
                           training: bool = False):
        assert self.have_cache.item() == 1, "Cache not initialized; call forward_first_step first."
        B = x.size(0)
        t_embed = self.encode_time(t)
        frac_full_accum = 0.0
        for i, (dU, R, tproj) in enumerate(zip(self.delta_blocks, self.routers, self.t_proj)):
            Fi_prev = self.quant.dequantize(self.cache_q[i], self.cache_s[i])
            t_e = tproj(t_embed)
            z, p = R(Fi_prev, t_e, step_idx=step_idx, total_steps=total_steps, training=training)
            # Always compute Fi_full for simplicity/robustness
            Fi_full = self.base.forward_block_i(i, Fi_prev, t, cond)
            dFi = dU(Fi_prev, t_e)
            Fi = torch.where(z.view(B, 1, 1, 1).bool(), Fi_full, Fi_prev + dFi)
            q, s = self.quant.quantize(Fi.detach())
            self.cache_q[i], self.cache_s[i] = q, s
            frac_full_accum += z.float().mean().item()
        if router_stats is not None:
            router_stats['frac_full'].append(frac_full_accum / len(self.delta_blocks))
            router_stats['frac_delta'].append(1.0 - router_stats['frac_full'][-1])
        out = self.base.final_from_features([
            self.quant.dequantize(self.cache_q[i], self.cache_s[i]) for i in range(len(self.delta_blocks))
        ], t, cond)
        sparsity_loss = router_sparsity_lambda * (frac_full_accum / len(self.delta_blocks))
        return out, sparsity_loss


# -----------------------------------------------------------------------------
# Training loop (distillation proxy on synthetic patterns)
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    res: int = 32
    batch: int = 16
    iters: int = 200
    lr: float = 1e-3
    router_lambda: float = 1e-3
    delta_mult: float = 1.0
    feature_bits: int = 8
    seed: int = 123
    device: str = "auto"
    fp16: bool = False
    images_dir: str = ".research/iteration1/images"
    models_dir: str = "models"


def build_models(device: str = "auto", delta_mult: float = 1.0, feature_bits: int = 8) -> Tuple[TinyUNetTeacher, DFDiffUNetWrapper]:
    device = pick_device(device)
    teacher = TinyUNetTeacher(c=32, time_dim=64).to(device)
    student_wrap = DFDiffUNetWrapper(teacher, feature_dims=[32, 32], delta_mult=delta_mult, feature_bits=feature_bits).to(device)
    return teacher, student_wrap


def _ensure_dirs(cfg: TrainConfig):
    os.makedirs(cfg.images_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)


def train_distill_proxy(cfg: TrainConfig) -> Dict[str, Any]:
    _ensure_dirs(cfg)
    device = pick_device(cfg.device)
    set_torch_seed(cfg.seed)

    # Models
    teacher, student_wrap = build_models(device=device, delta_mult=cfg.delta_mult, feature_bits=cfg.feature_bits)
    if cfg.fp16 and device.startswith("cuda"):
        teacher.half(); student_wrap.half()

    opt = torch.optim.AdamW(student_wrap.parameters(), lr=cfg.lr)

    # Data
    dl = get_synthetic_dataloader(n=512, res=cfg.res, batch=cfg.batch, shuffle=True, num_workers=0, seed=cfg.seed)

    losses = []
    dl_it = iter(dl)
    teacher.eval()
    student_wrap.train()

    for it in range(cfg.iters):
        try:
            batch = next(dl_it)
        except StopIteration:
            dl_it = iter(dl)
            batch = next(dl_it)
        x = batch["img"].to(device)
        # Random timestep per batch
        t = torch.randint(0, 16, (x.size(0),), device=device)
        with torch.no_grad():
            y_teacher, feats_teacher = teacher.forward_with_features(x, t)
        # Student first step caches
        y_student = student_wrap.forward_first_step(x, t)
        loss_pred = F.mse_loss(y_student, y_teacher)
        # One delta update step
        y_student2, sparsity_loss = student_wrap.forward_with_delta(
            x, t, step_idx=1, total_steps=2, router_sparsity_lambda=cfg.router_lambda, router_stats=None, training=True
        )
        loss_feat = 0.0
        for i in range(len(student_wrap.feature_dims)):
            f_stu = student_wrap.quant.dequantize(student_wrap.cache_q[i], student_wrap.cache_s[i])
            f_teach = feats_teacher[i].detach()
            loss_feat = loss_feat + F.mse_loss(f_stu, f_teach)
        loss = loss_pred + 0.1 * loss_feat + sparsity_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_wrap.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
        if (it + 1) % max(1, cfg.iters // 10) == 0:
            print(f"[train] iter {it+1}/{cfg.iters}  loss={losses[-1]:.4f}")

    # Save training loss plot (PDF)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4.2, 3.2))
        plt.plot(losses, label="train_loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("DF-Diff distillation proxy loss")
        plt.legend()
        fig_path = os.path.join(cfg.images_dir, "training_loss_df-diff.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {fig_path}")
    except Exception as e:
        print(f"Warning: failed to save training loss figure: {e}")

    # Save checkpoints
    teacher_path = os.path.join(cfg.models_dir, "teacher_tinyunet.pth")
    student_path = os.path.join(cfg.models_dir, "student_dfdiff.pth")
    torch.save({"state_dict": teacher.state_dict()}, teacher_path)
    torch.save({"state_dict": student_wrap.state_dict()}, student_path)
    print(f"Saved models to: {teacher_path} and {student_path}")

    return {
        "loss_last": losses[-1] if len(losses) > 0 else None,
        "loss_mean": float(np.mean(losses)) if len(losses) > 0 else None,
        "student": student_wrap,
        "teacher": teacher,
        "losses": losses,
    }
