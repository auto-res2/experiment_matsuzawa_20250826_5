
"""
src/train.py
====================================================
Light-weight training pipeline for the SGHR-Lite-v2
continual–learning experiment.  All heavy components
(diffusion decode, BCH, LoRA, …) are replaced by
computationally cheap stubs so that the whole experiment
runs in <30 s on CPU and in <10 s on a single T4.

The module exposes one public entry‐point:
    run_training(cfg: dict) -> dict
which is imported and executed by src.main.

The code uses only relative imports and depends only on
packages listed in requirements.txt.
"""
from __future__ import annotations
import time, math, json, pathlib
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import numpy as np

from .preprocess import get_tasks_datasets
from .utils import seed_everything, SGHRBuffer, SimpleHead, save_pdf, ensure_dir, IMG_DIR

###########################################################################
#                       Training routine                                  #
###########################################################################

def _train_one_task(task_id: int,
                    dataset: Subset,
                    buffer: SGHRBuffer,
                    head: nn.Module | None,
                    cfg: dict,
                    device: torch.device):
    """Trains on a single task and updates the SGHR buffer."""

    loader = DataLoader(dataset,
                        batch_size=cfg["batch_size"],
                        shuffle=True,
                        num_workers=0)

    # grow classifier to accommodate new classes ( +5 per task )
    n_classes = (task_id + 1) * cfg["classes_per_task"]
    if head is None:
        head = SimpleHead(in_dim=3072, n_classes=n_classes).to(device)
    else:
        # enlarge last Linear layer if needed
        old_state = head.state_dict()
        head = SimpleHead(in_dim=3072, n_classes=n_classes).to(device)
        head.load_state_dict({k: v for k, v in old_state.items()
                              if k in head.state_dict() and
                                 head.state_dict()[k].shape == old_state[k].shape})
    opt = torch.optim.Adam(head.parameters(), lr=cfg["lr"])

    clip_stub = nn.Flatten()   # cheap stand-in encoder: [B,3,32,32] → [B, 3072]

    # --------------------   train   ----------------------------------
    for epoch in range(cfg["epochs"]):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # replay half batch from buffer
            replay_imgs, replay_labels = buffer.sample(k=len(imgs)//2)
            if len(replay_imgs):
                replay_imgs, replay_labels = replay_imgs.to(device), replay_labels.to(device)
                imgs_cat  = torch.cat([imgs, replay_imgs])
                labels_cat = torch.cat([labels, replay_labels % n_classes])
            else:
                imgs_cat, labels_cat = imgs, labels % n_classes

            feats  = clip_stub(imgs_cat)
            logits = head(feats)
            loss   = F.cross_entropy(logits, labels_cat)
            opt.zero_grad(); loss.backward(); opt.step()

    # --------------------   evaluate current task   ------------------
    head.eval();  total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            feats = clip_stub(imgs.to(device))
            preds = head(feats).argmax(1)
            correct += (preds.cpu() == (labels % n_classes)).sum().item()
            total   += len(labels)
    acc = correct / total * 100.0

    # --------------------   store into buffer   ----------------------
    for imgs, labels in loader:
        buffer.encode_and_store(imgs, labels)

    mem_kb = buffer.bytes_used / 1024
    print(f"[TRAIN] Task {task_id} finished – acc {acc:.2f}% – memory {mem_kb:.1f} kB")
    return head, acc, mem_kb

###########################################################################
#                       Public API                                        #
###########################################################################

def run_training(cfg: dict) -> dict:
    seed_everything(cfg.get("seed", 2025))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = get_tasks_datasets(n_tasks=cfg["n_tasks"], total_size=cfg["dataset_size"])
    buffer = SGHRBuffer(max_bytes=cfg["max_bytes"])

    acc_hist, mem_hist = [], []
    head = None
    for t, subset in enumerate(tasks):
        head, acc, mem = _train_one_task(t, subset, buffer, head, cfg, device)
        acc_hist.append(acc)
        mem_hist.append(mem)

    # plots -----------------------------------------------------------
    ensure_dir(IMG_DIR)
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set_theme()

    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(x=list(range(1,cfg["n_tasks"]+1)), y=acc_hist, marker="o", ax=ax)
    ax.set_xlabel("Task"); ax.set_ylabel("Accuracy (%)")
    save_pdf(fig, f"{IMG_DIR}/accuracy_curve.pdf"); plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.lineplot(x=list(range(1,cfg["n_tasks"]+1)), y=mem_hist, marker="o", ax=ax2)
    ax2.set_xlabel("Task"); ax2.set_ylabel("Memory (kB)")
    save_pdf(fig2, f"{IMG_DIR}/memory_growth.pdf"); plt.close(fig2)

    return {"accuracy": acc_hist, "memory_kb": mem_hist}
