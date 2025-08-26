

"""
src/main.py
====================================================
Project entry-point (`python -m src.main`).  It loads the
YAML configuration, executes training, then the evaluation
suite.  Results and plots are written to the directory
.research/iteration7/images as required.
"""
from __future__ import annotations
import yaml, pathlib

from .train import run_training
from .evaluate import SeedStabilityEval, LatencyPrivacyEval
from .utils import ensure_dir, IMG_DIR

###########################################################################
#                       Load configuration                                #
###########################################################################

def _load_cfg() -> dict:
    cfg_path = pathlib.Path("config/config.yaml")
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
    else:
        # fall-back defaults
        cfg = {"n_tasks": 3,
               "classes_per_task": 5,
               "dataset_size": 5000,
               "batch_size": 64,
               "epochs": 2,
               "lr": 1e-3,
               "seed": 2025,
               "max_bytes": 128*1024}
    print("[MAIN] Loaded configuration", cfg)
    return cfg

###########################################################################
#                               main                                       #
###########################################################################

def main():
    cfg = _load_cfg()
    ensure_dir(IMG_DIR)

    # ---------------- training -----------------
    results = run_training(cfg)

    # ---------------- evaluation --------------
    seed_eval = SeedStabilityEval(); seed_eval.run(n_rounds=3)
    lat_eval  = LatencyPrivacyEval(); lat_eval.run(n_samples=100)

    print("[MAIN] Done.  Accuracy history:", results["accuracy"])

if __name__ == "__main__":
    main()
