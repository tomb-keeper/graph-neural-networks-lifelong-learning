import open_learning
import numpy as np
import torch

def test_evaluation():
    # Perfect prediction with no unseen
    labels = torch.tensor([0,1,2,3,5,6])
    predictions = torch.tensor([0,1,2,3,5,6])
    unseen_classes = set()
    reject_mask = tor