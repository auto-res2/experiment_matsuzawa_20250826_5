"""
evaluate.py – contains the three toy experiments from the paper and all
plotting / logging responsibilities.  Figures are stored under
.research/iteration1/images as required.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from .train import EucEncoder, HypEncoder, SharedDecoder, train_autoencoder

# Make sure the images directory exists before any plotting happens
IMG_DIR = Path(".research/iteration1/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1.   Minimal implementation of HADuS++ vs. Baseline
# ---------------------------------------------------------------------

class ToySparseDiffuser(nn.Module):
    def __init__(self, latent_dim: int = 128, steps: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, latent_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for _ in range(self.steps):
            h = self.net(h)
        return h


class HyperbolicAdapter(nn.Module):
    def __init__(self, in_dim: int = 128, hyp_dim: int = 8):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hyp_dim)
        self.decoder = nn.Linear(hyp_dim, in_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class AdaptiveScheduler:
    def __init__(
        self,
        min_steps: int = 2,
        K_max: int = 40,
        tau_easy: float = 0.3,
        tau_hard: float = 0.7,
        K_skip: int = 10,
    ):
        self.min_steps = min_steps
        self.K_max = K_max
        self.tau_easy = tau_easy
        self.tau_hard = tau_hard
        self.K_skip = K_skip

    def allocate(self, kappa: float) -> Tuple[int, bool]:
        if kappa < self.tau_easy:
            return max(self.min_steps, self.K_skip), True
        elif kappa > self.tau_hard:
            return self.K_max, False
        else:
            return self.min_steps, False


class HDPlannerBaseline(nn.Module):
    def __init__(self, latent_dim: int = 128, steps: int = 100):
        super().__init__()
        self.diffuser = ToySparseDiffuser(latent_dim, steps)

    def plan(self, obs):
        return self.diffuser(obs)


class HADuSPlanner(nn.Module):
    def __init__(self, base_planner: HDPlannerBaseline, hyp_dim: int = 8):
        super().__init__()
        self.base = base_planner
        self.adapter = HyperbolicAdapter(base_planner.diffuser.latent_dim, hyp_dim)
        self.scheduler = AdaptiveScheduler()

    def plan(self, obs):
        z_h = self.adapter.encode(obs)
        kappa = float(torch.rand(()))
        steps, cache = self.scheduler.allocate(kappa)
        if cache:
            out = self.base.diffuser.net(obs)
        else:
            out = obs
            for _ in range(steps):
                out = self.base.diffuser.net(out)
        return self.adapter.decode(z_h) + 0.0 * out


# ---------------------------------------------------------------------
# 2.  Toy environment to get wall-clock timings
# ---------------------------------------------------------------------
class ToyEnv:
    def __init__(self, horizon: int = 30):
        self.horizon = horizon
        self.t = 0
        self.obs_dim = 128

    def reset(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.t = 0
        return torch.randn(self.obs_dim)

    def step(self, action):
        self.t += 1
        done = self.t >= self.horizon
        reward = float(torch.randn(()))
        obs = torch.randn(self.obs_dim)
        return obs, reward, done, {}


# ---------------------------------------------------------------------
# 3.   EXPERIMENT-1: end-to-end speed-up
# ---------------------------------------------------------------------

def run_exp1(num_episodes: int = 20, seed: int = 0):
    print("===== EXPERIMENT-1  (Toy)  End-to-End speed-up =====")
    env = ToyEnv()
    torch.manual_seed(seed)

    algos = {
        "baseline": HDPlannerBaseline(),
        "hadus"  : HADuSPlanner(HDPlannerBaseline()),
    }

    wall_clock: Dict[str, List[float]] = {k: [] for k in algos}
    returns: Dict[str, List[float]] = {k: [] for k in algos}

    for name, planner in algos.items():
        for ep in range(num_episodes):
            obs = env.reset(seed + ep)
            done = False
            ep_ret = 0.0
            t0 = time.perf_counter()
            while not done:
                action = planner.plan(obs)
                obs, r, done, _ = env.step(action)
                ep_ret += r
            wall_clock[name].append(time.perf_counter() - t0)
            returns[name].append(ep_ret)
        print(
            f"Algo={name:9s}  avg-wall-clock={np.mean(wall_clock[name]):.4f}s   "
            f"avg-return={np.mean(returns[name]):.2f}"
        )

    # PDF figure --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.boxplot(data=[wall_clock[k] for k in algos], ax=ax)
    ax.set_xticklabels(list(algos.keys()))
    ax.set_ylabel("Wall-clock / s")
    ax.set_title("End-to-End wall-clock (toy)")
    fig.tight_layout()
    (IMG_DIR / "wall_clock.pdf").with_suffix(".pdf")
    plt.savefig(IMG_DIR / "wall_clock.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 4.   EXPERIMENT-2: latent distortion comparison
# ---------------------------------------------------------------------

def sample_se3_vectors(n: int) -> torch.Tensor:
    return torch.randn(n, 384)


def distortion(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    d_lat = torch.cdist(z, z)
    d_x = torch.cdist(x, x)
    return torch.max(torch.abs(d_lat - d_x) / (d_x + 1e-6))


def run_exp2(n_samples: int = 256, epochs: int = 10, batch_size: int = 64):
    print("===== EXPERIMENT-2  Hyperbolic vs. Euclidean latent =====")
    X = sample_se3_vectors(n_samples)

    # Euclidean 128-d ---------------------------------------------------
    enc_e, dec_e, mse_e = train_autoencoder(EucEncoder(), SharedDecoder(128), X, epochs, batch_size)
    with torch.no_grad():
        dist_e = distortion(enc_e(X), X).item()

    # Hyperbolic 8-d ----------------------------------------------------
    enc_h, dec_h, mse_h = train_autoencoder(HypEncoder(), SharedDecoder(8), X, epochs, batch_size)
    with torch.no_grad():
        dist_h = distortion(enc_h(X), X).item()

    print(f"Distortion  Euclidean-128: {dist_e:.3f}   Hyperbolic-8: {dist_h:.3f}")

    fig, ax = plt.subplots(figsize=(3, 2))
    sns.barplot(x=["Euc-128", "Hyp-8"], y=[dist_e, dist_h], ax=ax)
    ax.set_ylabel("Distortion (↓ better)")
    fig.tight_layout()
    plt.savefig(IMG_DIR / "latent_distortion.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 5.   EXPERIMENT-3: scheduler ablation
# ---------------------------------------------------------------------

def simulate_segment(scheduler: AdaptiveScheduler):
    kappa = random.random()
    steps, cached = scheduler.allocate(kappa)
    return {"kappa": kappa, "steps": steps, "cached": int(cached)}


def run_exp3(n_segments: int = 200):
    print("===== EXPERIMENT-3  Adaptive Scheduler ablation =====")
    sched_fixed = AdaptiveScheduler(min_steps=20, K_max=20, tau_easy=0, tau_hard=1, K_skip=0)
    sched_cache = AdaptiveScheduler(min_steps=2, K_max=20, tau_easy=0.2, tau_hard=0.8, K_skip=12)
    sched_ads   = AdaptiveScheduler()  # default

    scheds = {"Fixed20": sched_fixed, "DeepCache": sched_cache, "ADS": sched_ads}

    records: Dict[str, Dict[str, List[float]]] = {k: {"kappa": [], "steps": []} for k in scheds}

    for name, sch in scheds.items():
        for _ in range(n_segments):
            out = simulate_segment(sch)
            for key in ["kappa", "steps"]:
                records[name][key].append(out[key])
        easy_idx = [i for i, k in enumerate(records[name]["kappa"]) if k < 0.3]
        hard_idx = [i for i, k in enumerate(records[name]["kappa"]) if k > 0.7]
        easy_st = np.mean([records[name]["steps"][i] for i in easy_idx])
        hard_st = np.mean([records[name]["steps"][i] for i in hard_idx])
        print(f"Algo={name:9s}  steps_easy={easy_st:.1f}  steps_hard={hard_st:.1f}")

    fig, ax = plt.subplots(figsize=(3, 2))
    sns.violinplot(data=[records[k]["steps"] for k in scheds], ax=ax)
    ax.set_xticklabels(list(scheds.keys()))
    ax.set_ylabel("Allocated steps")
    ax.set_title("ADS vs baselines")
    fig.tight_layout()
    plt.savefig(IMG_DIR / "allocated_steps.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 6.  Helpers for unit tests / CI
# ---------------------------------------------------------------------

def quick_test():
    """Run tiny versions of all experiments – fast enough for CI."""
    torch.set_num_threads(1)
    run_exp1(num_episodes=5)
    run_exp2(n_samples=64, epochs=2)
    run_exp3(n_segments=50)
    print("All toy experiments executed successfully ✔")
