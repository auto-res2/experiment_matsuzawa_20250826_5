"""
src/preprocess.py
====================================================
Data preparation utilities.  Because the demonstration
uses torchvision's FakeData we don't need heavy
pre-processing, but the file keeps the structure ready
for real datasets.
"""
from __future__ import annotations
from typing import List
import numpy as np
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as T

############################################################################
#                      Public helper                                       #
############################################################################

def get_tasks_datasets(n_tasks: int = 3, total_size: int = 5000) -> List[Subset]:
    """Splits a FakeData dataset into `n_tasks` disjoint subsets."""
    transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.FakeData(size=total_size,
                                            image_size=(3,32,32),
                                            num_classes=100,
                                            transform=transform)
    indices = np.array_split(np.arange(total_size), n_tasks)
    return [Subset(dataset, idx.tolist()) for idx in indices]
