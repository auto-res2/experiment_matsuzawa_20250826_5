import argparse
import os
import yaml

from .train import TrainConfig, train_distill_proxy, build_models, pick_device
from .evaluate import RunConfig, QualityCfg, RobustCfg, run_memory_latency, run_quality_proxy, run_robustness

IMAGES_DIR = ".research/iteration4/images"
RESULTS_DIR = ".research/iteration2"
MODELS_DIR = "models"


def load_config(path: str) -> dict:
    if path is None or (not os.path.exists(path)):
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def merge_args_with_cfg(args, cfg: dict):
    # Simple overlay: args override cfg
    out = dict(cfg)
    for k, v in vars(args).items():
        if v is not None:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="DF-Diff experimental runner")
    parser.add_argument("--plan", type=str, default="test", choices=[
        "test", "train_only", "memory_latency", "quality_proxy", "robustness", "all"
    ])
    parser.add_argument("--config", type=str, default="config/dfdiff.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--res", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--feature_bits", type=int, default=None)
    parser.add_argument("--router_lambda", type=float, default=None)
    parser.add_argument("--delta_mult", type=float, default=None)

    args = parser.parse_args()

    cfg_file = load_config(args.config)
    cfg = merge_args_with_cfg(args, cfg_file)

    # Ensure dirs
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Common hyperparams with defaults
    device = cfg.get("device", "auto") if cfg.get("device", None) is not None else "auto"
    fp16 = bool(cfg.get("fp16", False))
    res = int(cfg.get("res", 32))
    steps = int(cfg.get("steps", 12))
    batch = int(cfg.get("batch", 1))
    iters = int(cfg.get("iters", 120))
    feature_bits = int(cfg.get("feature_bits", 8))
    router_lambda = float(cfg.get("router_lambda", 1e-3))
    delta_mult = float(cfg.get("delta_mult", 1.0))

    if args.plan == "test":
        # Quick end-to-end sanity run
        tcfg = TrainConfig(res=res, batch=16, iters=min(120, iters), lr=1e-3, router_lambda=router_lambda,
                           delta_mult=delta_mult, feature_bits=feature_bits, device=device, fp16=fp16,
                           images_dir=IMAGES_DIR, models_dir=MODELS_DIR)
        train_res = train_distill_proxy(tcfg)

        rcfg = RunConfig(res=res, steps=min(12, steps), batch=batch, fp16=fp16, device=device,
                         images_dir=IMAGES_DIR, results_dir=RESULTS_DIR,
                         feature_bits=feature_bits, router_lambda=router_lambda, delta_mult=delta_mult)
        run_memory_latency(rcfg, student=train_res["student"], teacher=train_res["teacher"])

        rocfg = RobustCfg(res=res, steps=min(10, steps), batch=batch, device=device, fp16=fp16,
                          images_dir=IMAGES_DIR, results_dir=RESULTS_DIR)
        run_robustness(rocfg, teacher=train_res["teacher"], student=train_res["student"])

        print("\nQuick test completed. Inspect .research/iteration4/images for results and images.")
        return

    if args.plan == "train_only":
        tcfg = TrainConfig(res=res, batch=16, iters=iters, lr=1e-3, router_lambda=router_lambda,
                           delta_mult=delta_mult, feature_bits=feature_bits, device=device, fp16=fp16,
                           images_dir=IMAGES_DIR, models_dir=MODELS_DIR)
        train_distill_proxy(tcfg)
        return

    if args.plan == "memory_latency":
        tcfg = TrainConfig(res=res, batch=16, iters=min(200, iters), lr=1e-3, router_lambda=router_lambda,
                           delta_mult=delta_mult, feature_bits=feature_bits, device=device, fp16=fp16,
                           images_dir=IMAGES_DIR, models_dir=MODELS_DIR)
        train_res = train_distill_proxy(tcfg)
        rcfg = RunConfig(res=res, steps=steps, batch=batch, fp16=fp16, device=device,
                         images_dir=IMAGES_DIR, results_dir=RESULTS_DIR,
                         feature_bits=feature_bits, router_lambda=router_lambda, delta_mult=delta_mult)
        run_memory_latency(rcfg, student=train_res["student"], teacher=train_res["teacher"])
        return

    if args.plan == "quality_proxy":
        tcfg = TrainConfig(res=res, batch=16, iters=min(200, iters), lr=1e-3, router_lambda=router_lambda,
                           delta_mult=delta_mult, feature_bits=feature_bits, device=device, fp16=fp16,
                           images_dir=IMAGES_DIR, models_dir=MODELS_DIR)
        train_res = train_distill_proxy(tcfg)
        qcfg = QualityCfg(res=res, num_samples=256, steps=steps, batch=16, device=device, fp16=fp16,
                          feature_bits=feature_bits, router_lambda=router_lambda, delta_mult=delta_mult,
                          images_dir=IMAGES_DIR, results_dir=RESULTS_DIR)
        run_quality_proxy(qcfg, teacher=train_res["teacher"], student=train_res["student"])
        return

    if args.plan == "robustness":
        tcfg = TrainConfig(res=res, batch=16, iters=min(120, iters), lr=1e-3, router_lambda=router_lambda,
                           delta_mult=delta_mult, feature_bits=feature_bits, device=device, fp16=fp16,
                           images_dir=IMAGES_DIR, models_dir=MODELS_DIR)
        train_res = train_distill_proxy(tcfg)
        rocfg = RobustCfg(res=res, steps=steps, batch=batch, device=device, fp16=fp16,
                          images_dir=IMAGES_DIR, results_dir=RESULTS_DIR)
        run_robustness(rocfg, teacher=train_res["teacher"], student=train_res["student"])
        return

    if args.plan == "all":
        tcfg = TrainConfig(res=res, batch=16, iters=min(200, iters), lr=1e-3, router_lambda=router_lambda,
                           delta_mult=delta_mult, feature_bits=feature_bits, device=device, fp16=fp16,
                           images_dir=IMAGES_DIR, models_dir=MODELS_DIR)
        train_res = train_distill_proxy(tcfg)
        rcfg = RunConfig(res=res, steps=steps, batch=batch, fp16=fp16, device=device,
                         images_dir=IMAGES_DIR, results_dir=RESULTS_DIR,
                         feature_bits=feature_bits, router_lambda=router_lambda, delta_mult=delta_mult)
        run_memory_latency(rcfg, student=train_res["student"], teacher=train_res["teacher"])
        qcfg = QualityCfg(res=res, num_samples=256, steps=steps, batch=16, device=device, fp16=fp16,
                          feature_bits=feature_bits, router_lambda=router_lambda, delta_mult=delta_mult,
                          images_dir=IMAGES_DIR, results_dir=RESULTS_DIR)
        run_quality_proxy(qcfg, teacher=train_res["teacher"], student=train_res["student"])
        rocfg = RobustCfg(res=res, steps=steps, batch=batch, device=device, fp16=fp16,
                          images_dir=IMAGES_DIR, results_dir=RESULTS_DIR)
        run_robustness(rocfg, teacher=train_res["teacher"], student=train_res["student"])
        print("\nAll experiments completed. Inspect .research/iteration4/images for results and images.")
        return


if __name__ == "__main__":
    main()
