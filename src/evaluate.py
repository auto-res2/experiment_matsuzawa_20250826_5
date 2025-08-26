import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid

from .train import (
    pick_device,
    TinyUNetTeacher,
    DFDiffUNetWrapper,
    TrainConfig,
    build_models,
)

sns.set_context("paper")
sns.set_style("whitegrid")

# -----------------------------------------------------------------------------
# Timing / tracing utils
# -----------------------------------------------------------------------------

class StepTrace:
    def __init__(self):
        self.step_ms: List[float] = []
        self.step_mem_mb: List[float] = []
        self.frac_full: List[float] = []
        self.frac_delta: List[float] = []


class CudaTimer:
    def __init__(self, device="cuda"):
        self.device = device
        self.events = []
        self.cpu_times = []
        self.use_cuda = torch.cuda.is_available() and device.startswith("cuda")

    def mark(self):
        if self.use_cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.events.append(e)
        else:
            import time as _time
            self.cpu_times.append(_time.perf_counter())

    def durations_ms(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            return [self.events[i].elapsed_time(self.events[i + 1]) for i in range(len(self.events) - 1)]
        else:
            return [
                (self.cpu_times[i + 1] - self.cpu_times[i]) * 1000.0 for i in range(len(self.cpu_times) - 1)
            ]


def torch_max_mem_gb(device: str) -> float:
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def reset_torch_mem(device: str):
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------

class SamplerAPI:
    def __init__(self, name: str):
        self.name = name
        self.router_log = StepTrace()
        self._router_hook = None
        self.use_fp16 = False

    def to(self, device):
        return self

    def eval(self):
        return self

    def enable_fp16(self):
        self.use_fp16 = True

    def set_feature_quant_bits(self, nbits: int):
        pass

    def set_router_sparsity(self, lam: float):
        pass

    def set_delta_capacity_mult(self, mult: float):
        pass

    @torch.no_grad()
    def sample(self, num_images: int, resolution: int, steps: int, device: str = "cuda",
               return_trace: bool = False) -> Tuple[torch.Tensor, Optional[StepTrace]]:
        raise NotImplementedError


class BaselineSampler(SamplerAPI):
    def __init__(self, name: str, teacher: TinyUNetTeacher):
        super().__init__(name)
        self.teacher = teacher

    def to(self, device):
        self.teacher.to(device)
        return self

    def enable_fp16(self):
        super().enable_fp16()
        self.teacher.half()

    @torch.no_grad()
    def sample(self, num_images: int, resolution: int, steps: int, device: str = "cuda",
               return_trace: bool = False):
        self.teacher.eval()
        dtype = torch.float16 if (self.use_fp16 and device.startswith("cuda")) else torch.float32
        trace = StepTrace()
        timer = CudaTimer(device)
        x = torch.rand(num_images, 3, resolution, resolution, device=device, dtype=dtype)
        timer.mark()
        for si in range(steps):
            t = torch.full((num_images,), si, device=device, dtype=torch.long)
            _y, _feats = self.teacher.forward_with_features(x, t)
            if torch.cuda.is_available() and device.startswith("cuda"):
                trace.step_mem_mb.append(torch.cuda.memory_allocated() / (1024 ** 2))
            frac_full = 1.0
            frac_delta = 0.0
            trace.frac_full.append(frac_full)
            trace.frac_delta.append(frac_delta)
            timer.mark()
        step_ms = timer.durations_ms()
        trace.step_ms = step_ms
        imgs = _y.detach().clamp(0, 1)
        return (imgs, trace) if return_trace else imgs


class DFDiffSampler(SamplerAPI):
    def __init__(self, name: str, student: DFDiffUNetWrapper, router_lambda: float = 0.0):
        super().__init__(name)
        self.student = student
        self.router_lambda = router_lambda

    def to(self, device):
        self.student.to(device)
        return self

    def enable_fp16(self):
        super().enable_fp16()
        self.student.half()

    def set_feature_quant_bits(self, nbits: int):
        # Recreate quantizer with new bitwidth
        from .train import PerChannelQuant
        self.student.quant = PerChannelQuant(nbits=nbits)

    def set_router_sparsity(self, lam: float):
        self.router_lambda = lam

    def set_delta_capacity_mult(self, mult: float):
        from .train import DeltaUpdater
        dims = self.student.feature_dims
        device = next(self.student.parameters()).device
        self.student.delta_blocks = nn.ModuleList([DeltaUpdater(d, mlp_mult=mult) for d in dims]).to(device)

    @torch.no_grad()
    def sample(self, num_images: int, resolution: int, steps: int, device: str = "cuda",
               return_trace: bool = False):
        self.student.eval()
        dtype = torch.float16 if (self.use_fp16 and device.startswith("cuda")) else torch.float32
        trace = StepTrace()
        timer = CudaTimer(device)
        x = torch.rand(num_images, 3, resolution, resolution, device=device, dtype=dtype)
        self.student.clear_cache()
        timer.mark()
        # First step: full to cache
        t0 = torch.zeros((num_images,), device=device, dtype=torch.long)
        _y0 = self.student.forward_first_step(x, t0)
        if torch.cuda.is_available() and device.startswith("cuda"):
            trace.step_mem_mb.append(torch.cuda.memory_allocated() / (1024 ** 2))
        trace.frac_full.append(1.0)
        trace.frac_delta.append(0.0)
        timer.mark()
        last_y = _y0
        # Subsequent steps
        router_stats = {"frac_full": [], "frac_delta": []}
        for si in range(1, steps):
            t = torch.full((num_images,), si, device=device, dtype=torch.long)
            _y, _ = self.student.forward_with_delta(
                x, t, step_idx=si, total_steps=steps, router_sparsity_lambda=self.router_lambda,
                router_stats=router_stats, training=False
            )
            last_y = _y
            if torch.cuda.is_available() and device.startswith("cuda"):
                trace.step_mem_mb.append(torch.cuda.memory_allocated() / (1024 ** 2))
            f_full = router_stats['frac_full'][-1]
            f_delta = router_stats['frac_delta'][-1]
            trace.frac_full.append(f_full)
            trace.frac_delta.append(f_delta)
            timer.mark()
        trace.step_ms = timer.durations_ms()
        imgs = last_y.detach().clamp(0, 1)
        return (imgs, trace) if return_trace else imgs

# -----------------------------------------------------------------------------
# Configs for evaluation runs
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    res: int = 32
    steps: int = 16
    batch: int = 1
    fp16: bool = False
    device: str = "auto"
    images_dir: str = ".research/iteration3/images"
    results_dir: str = ".research/iteration2"
    # DF-Diff specific
    feature_bits: int = 8
    router_lambda: float = 1e-3
    delta_mult: float = 1.0


@dataclass
class QualityCfg:
    res: int = 32
    num_samples: int = 128
    steps: int = 16
    batch: int = 16
    device: str = "auto"
    fp16: bool = False
    feature_bits: int = 8
    router_lambda: float = 1e-3
    delta_mult: float = 1.0
    images_dir: str = ".research/iteration3/images"
    results_dir: str = ".research/iteration2"


@dataclass
class RobustCfg:
    res: int = 32
    steps: int = 16
    batch: int = 1
    device: str = "auto"
    fp16: bool = False
    images_dir: str = ".research/iteration3/images"
    results_dir: str = ".research/iteration2"

# -----------------------------------------------------------------------------
# Plots helpers (save as high-quality PDF)
# -----------------------------------------------------------------------------

def _ensure_dirs(images_dir: str, results_dir: str):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)


def _save_grid_pdf(tensor_img: torch.Tensor, path_pdf: str, nrow: int = 4):
    grid = make_grid(tensor_img.cpu(), nrow=nrow)
    npimg = grid.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.figure(figsize=(4.0, 4.0))
    plt.axis('off')
    plt.imshow(npimg)
    plt.tight_layout(pad=0)
    plt.savefig(path_pdf, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# Runs
# -----------------------------------------------------------------------------

def run_memory_latency(cfg: RunConfig, student: Optional[DFDiffUNetWrapper] = None, teacher: Optional[TinyUNetTeacher] = None) -> Dict[str, Any]:
    _ensure_dirs(cfg.images_dir, cfg.results_dir)
    device = pick_device(cfg.device)

    # Models
    if teacher is None or student is None:
        teacher, student = build_models(device=device, delta_mult=cfg.delta_mult, feature_bits=cfg.feature_bits)

    baseline = BaselineSampler("baseline", teacher).to(device)
    ours = DFDiffSampler("df-diff", student, router_lambda=cfg.router_lambda).to(device)

    if cfg.fp16 and device.startswith("cuda"):
        baseline.enable_fp16(); ours.enable_fp16()

    ours.set_feature_quant_bits(cfg.feature_bits)

    # Warmup
    reset_torch_mem(device)
    _ = baseline.sample(num_images=1, resolution=cfg.res, steps=min(4, cfg.steps), device=device)
    reset_torch_mem(device)
    _ = ours.sample(num_images=1, resolution=cfg.res, steps=min(4, cfg.steps), device=device)
    reset_torch_mem(device)

    # Measure Baseline
    t0 = time.perf_counter()
    imgs_b, trace_b = baseline.sample(num_images=cfg.batch, resolution=cfg.res, steps=cfg.steps, device=device, return_trace=True)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak_b = torch_max_mem_gb(device)

    # Measure DF-Diff
    reset_torch_mem(device)
    t2 = time.perf_counter()
    imgs_o, trace_o = ours.sample(num_images=cfg.batch, resolution=cfg.res, steps=cfg.steps, device=device, return_trace=True)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    peak_o = torch_max_mem_gb(device)

    # Aggregate
    res_rec = {
        "resolution": cfg.res,
        "steps": cfg.steps,
        "batch": cfg.batch,
        "fp16": cfg.fp16,
        "feature_bits": cfg.feature_bits,
        "router_lambda": cfg.router_lambda,
        "delta_mult": cfg.delta_mult,
        "baseline_total_s": round(t1 - t0, 4),
        "baseline_avg_step_ms": round(float(np.mean(trace_b.step_ms)) if trace_b.step_ms else float('nan'), 3),
        "baseline_peak_mem_gb": round(peak_b, 4),
        "dfdiff_total_s": round(t3 - t2, 4),
        "dfdiff_avg_step_ms": round(float(np.mean(trace_o.step_ms)) if trace_o.step_ms else float('nan'), 3),
        "dfdiff_peak_mem_gb": round(peak_o, 4),
        "dfdiff_avg_frac_full": round(float(np.mean(trace_o.frac_full[1:])) if trace_o.frac_full else float('nan'), 4),
    }
    print("Memory/Latency Summary:")
    for k, v in res_rec.items():
        print(f"  {k}: {v}")

    # Save CSV
    df = pd.DataFrame([res_rec])
    csv_path = os.path.join(cfg.results_dir, f"memory_latency_df-diff_r{cfg.res}_s{cfg.steps}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plots: step time curves and router curve
    # Baseline step time
    plt.figure(figsize=(4.2, 3.2))
    plt.plot(trace_b.step_ms, label="baseline")
    plt.xlabel("Step")
    plt.ylabel("Time (ms)")
    plt.title("Per-step latency (baseline)")
    plt.legend()
    fig_path = os.path.join(cfg.images_dir, "step_time_baseline.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # DF-Diff step time
    plt.figure(figsize=(4.2, 3.2))
    plt.plot(trace_o.step_ms, label="df-diff")
    plt.xlabel("Step")
    plt.ylabel("Time (ms)")
    plt.title("Per-step latency (DF-Diff)")
    plt.legend()
    fig_path = os.path.join(cfg.images_dir, "step_time_df-diff.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # Router curve
    plt.figure(figsize=(4.2, 3.2))
    plt.plot(trace_o.frac_full, label="full (fallback)")
    plt.plot(trace_o.frac_delta, label="delta")
    plt.xlabel("Step")
    plt.ylabel("Fraction")
    plt.title("Router usage over steps (DF-Diff)")
    plt.legend()
    fig_path = os.path.join(cfg.images_dir, "router_fraction_df-diff.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # Memory bars
    plt.figure(figsize=(4.2, 3.2))
    methods = ["baseline", "df-diff"]
    mem = [peak_b, peak_o]
    sns.barplot(x=methods, y=mem, palette="deep")
    plt.ylabel("Peak GPU mem (GB)")
    plt.title("Peak memory during sampling")
    fig_path = os.path.join(cfg.images_dir, "memory_latency_df-diff.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # Example output images (save as PDF grids)
    img_b_pdf = os.path.join(cfg.images_dir, "baseline_grid.pdf")
    img_o_pdf = os.path.join(cfg.images_dir, "df-diff_grid.pdf")
    _save_grid_pdf(imgs_b[: min(8, cfg.batch)], img_b_pdf, nrow=4)
    _save_grid_pdf(imgs_o[: min(8, cfg.batch)], img_o_pdf, nrow=4)

    return res_rec


def run_quality_proxy(cfg: QualityCfg, teacher: Optional[TinyUNetTeacher] = None,
                      student: Optional[DFDiffUNetWrapper] = None) -> Dict[str, Any]:
    _ensure_dirs(cfg.images_dir, cfg.results_dir)
    device = pick_device(cfg.device)

    if teacher is None or student is None:
        teacher, student = build_models(device=device, delta_mult=cfg.delta_mult, feature_bits=cfg.feature_bits)

    teacher.eval(); student.eval()

    # Synthetic dataset for proxy quality
    from .preprocess import SyntheticPatterns
    ds = SyntheticPatterns(n=cfg.num_samples, res=cfg.res)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch, shuffle=False, num_workers=0)

    mses = []
    with torch.no_grad():
        for batch in dl:
            x = batch["img"].to(device)
            B = x.size(0)
            t = torch.zeros((B,), device=device, dtype=torch.long)
            y_t, _ = teacher.forward_with_features(x, t)
            student.clear_cache()
            y_s = student.forward_first_step(x, t)
            mse = F.mse_loss(y_s, y_t).item()
            mses.append(mse)
    mean_mse = float(np.mean(mses)) if len(mses) > 0 else float('nan')
    print(f"Quality proxy (teacher vs. DF-Diff first-step MSE): {mean_mse:.6f}")

    # Plot distribution
    plt.figure(figsize=(4.2, 3.2))
    sns.histplot(mses, bins=20, kde=True)
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.title("Teacher vs DF-Diff output MSE (proxy)")
    path = os.path.join(cfg.images_dir, "quality_proxy_df-diff.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # CSV
    df = pd.DataFrame({"mse": mses})
    csv_path = os.path.join(cfg.results_dir, f"quality_proxy_mse_r{cfg.res}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return {"mean_mse": mean_mse}


def run_robustness(cfg: RobustCfg, teacher: Optional[TinyUNetTeacher] = None,
                   student: Optional[DFDiffUNetWrapper] = None) -> Dict[str, Any]:
    _ensure_dirs(cfg.images_dir, cfg.results_dir)
    device = pick_device(cfg.device)

    if teacher is None or student is None:
        teacher, student = build_models(device=device, delta_mult=1.0, feature_bits=8)

    df_sampler = DFDiffSampler("df-diff", student, router_lambda=1e-3).to(device)

    # Ablations: feature bits, router lambda, delta capacity
    feature_bits_list = [8, 6, 4]
    router_lambda_list = [0.0, 1e-3, 1e-2]
    delta_mult_list = [0.5, 1.0, 1.5]

    recs = []
    for fb in feature_bits_list:
        for lam in router_lambda_list:
            for dm in delta_mult_list:
                df_sampler.set_feature_quant_bits(fb)
                df_sampler.set_router_sparsity(lam)
                df_sampler.set_delta_capacity_mult(dm)
                reset_torch_mem(device)
                t0 = time.perf_counter()
                _, trace = df_sampler.sample(num_images=cfg.batch, resolution=cfg.res, steps=cfg.steps,
                                             device=device, return_trace=True)
                if torch.cuda.is_available() and device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                peak = torch_max_mem_gb(device)
                rec = {
                    "feature_bits": fb,
                    "router_lambda": lam,
                    "delta_mult": dm,
                    "avg_step_ms": float(np.mean(trace.step_ms)) if trace.step_ms else float('nan'),
                    "avg_frac_full": float(np.mean(trace.frac_full[1:])) if trace.frac_full else float('nan'),
                    "peak_mem_gb": peak,
                    "total_time_s": elapsed,
                }
                recs.append(rec)
                print(f"Ablation fb={fb}, lam={lam}, dm={dm}: mem={peak:.4f}GB, step_ms={rec['avg_step_ms']:.2f}, frac_full={rec['avg_frac_full']:.3f}")

    # Save CSV
    df = pd.DataFrame(recs)
    csv_path = os.path.join(cfg.results_dir, f"ablation_df-diff_r{cfg.res}_s{cfg.steps}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot: memory vs. bits, colored by delta_mult, style by router_lambda
    plt.figure(figsize=(4.6, 3.2))
    sns.scatterplot(data=df, x="feature_bits", y="peak_mem_gb", hue="delta_mult", style="router_lambda", s=60)
    plt.title("Peak memory vs. feature quantization bits")
    plt.ylabel("Peak GPU mem (GB)")
    plt.xlabel("Feature bits")
    path = os.path.join(cfg.images_dir, "ablation_memory_bits_df-diff.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Plot: avg frac full vs avg step ms
    plt.figure(figsize=(4.6, 3.2))
    sns.scatterplot(data=df, x="avg_frac_full", y="avg_step_ms", hue="feature_bits", style="delta_mult", s=60)
    plt.title("Latency vs. router full fraction")
    plt.xlabel("Avg fraction of full fallbacks")
    plt.ylabel("Avg step time (ms)")
    path = os.path.join(cfg.images_dir, "latency_vs_router_df-diff.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return {"records": recs}
