import open_learning
import numpy as np
import torch

def test_evaluation():
    # Perfect prediction with no unseen
    labels = torch.tensor([0,1,2,3,5,6])
    predictions = torch.tensor([0,1,2,3,5,6])
    unseen_classes = set()
    reject_mask = torch.zeros(labels.size(0), dtype=torch.bool)
    scores = open_learning.evaluate(labels, unseen_classes,
                                    predictions, reject_mask)

    assert scores['open_mcc'] == 0.0
    assert scores['open_f1_macro'] == 1.0

    # Perfect prediction and perfect rejection
    labels = torch.tensor([0,1,2,3,5,6])
    predictions = torch.tensor([0,1,2,3,5,6])
    unseen_classes = set([5,6])
    reject_mask = torch.zeros(labels.size(0), dtype=torch.bool)
    reject_mask[[-1,-2]] = True  # Cheatz
    scores = open_learning.evaluate(labels, unseen_classes,
                                    predictions, reject_mask)

    assert scores['open_mcc'] == 1.0
    assert scores['open_f1_macro'] == 1.0

    # Perfect prediction but imperfect rejection
    labels = torch.tensor([0,1,2,3,5,6])
 