"""
evaluate.py – evaluation utilities for RevSparse-ViM toy pipeline
=================================================================
Implements  evaluate(model, loader, device)
"""
from __future__ import annotations
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

from .train import _ensure_dir, _bytes_to_mb


def evaluate(cfg, model, loader, device):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            pred   = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    acc = 100*correct/total
    print(f"Evaluation accuracy: {acc:.2f} %")

    # bar plot ------------------------------------------------------
    _ensure_dir(Path('.research/iteration1/images'))
    fig, ax = plt.subplots(figsize=(3,3))
    ax.bar(['RevSparse'], [acc], color='tab:green')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation accuracy')
    plt.tight_layout()
    plt.savefig('.research/iteration1/images/val_accuracy.pdf', bbox_inches='tight')
    plt.close(fig)

    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU memory during evaluation: {_bytes_to_mb(mem):.1f} MB")

    return acc
