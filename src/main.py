"""
main.py – orchestrates the whole experiment
==========================================
Run from project root with
    python -m src.main
"""
from __future__ import annotations
import yaml
from pathlib import Path
import torch

from . import preprocess as prep
from . import train as tr
from . import evaluate as ev


def load_cfg() -> dict:
    """Load YAML config and return a SimpleNamespace tree."""
    # Updated to load the existing config file.
    cfg_path = Path('config/config.yaml')
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path.resolve()}")

    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f)
    # convert to simple namespace-like object for dot access
    from types import SimpleNamespace
    def _rec(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _rec(v) for k, v in d.items()})
        else:
            return d
    return _rec(cfg)


def main():
    cfg = load_cfg()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running on device: {device}")

    # data ----------------------------------------------------------
    train_loader, val_loader = prep.get_loaders(cfg)

    # model ---------------------------------------------------------
    model = tr.build_model(cfg)

    # training ------------------------------------------------------
    model, _ = tr.train(cfg, model, (train_loader, val_loader), device)

    # evaluation ----------------------------------------------------
    ev.evaluate(cfg, model, val_loader, device)


if __name__ == '__main__':
    main()
