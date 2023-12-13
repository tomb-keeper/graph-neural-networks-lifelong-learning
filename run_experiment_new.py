
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import gc

import numpy as np
import dgl
import torch_geometric as tg
import torch
# import torch.nn as nn
import torch.nn.functional as F

# Models
from sklearn.metrics import f1_score

from models import GraphSAGE
from models import GAT
from models import MLP
from models import MostFrequentClass
from models import JKNet
from models.sgnet import SGNet

from models.graphsaint import train_saint, evaluate_saint
from models import geometric as geo
from models.node2vec import (add_node2vec_args,
                             train_node2vec,
                             evaluate_node2vec)

# from datasets import load_data  # unused

from lifelong_learning import lifelong_nodeclf_identifier
from lifelong_learning import LifelongNodeClassificationDataset
from lifelong_learning import collate_tasks

from resultswriter import CSVResultsWriter

import open_learning

try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
    print("Not using weightsandbiases integration. To use `pip install wandb`")




def compute_weights(ts, exponential_decay, initial_quantity=1.0, normalize=True):
    ts = torch.as_tensor(ts)
    delta_t = ts.max() - ts
    values = initial_quantity * torch.exp(- exponential_decay * delta_t)
    if normalize:
        # When normalizing, the initial_quantity is irrelevant
        values = values / values.sum()
    return values


def train(model, optimizer, g, feats, labels, mask=None, epochs=1,
          weights=None, backend='dgl', open_learning_model=None):
    model.train()
    reduction = 'none' if weights is not None else 'mean'

    if hasattr(model, '__reset_cache__'):
        print("Resetting Model Cache")
        model.__reset_cache__()

    if mask is not None:
        # Reduce view alredy here rather than in each epoch (prevent bugs)
        labels = labels[mask]

    for epoch in range(epochs):
        inputs = (g, feats) if backend == 'dgl' else (feats, g)

        logits = model(*inputs)
        if mask is not None:
            logits = logits[mask]

        if open_learning_model is not None:
            # The open learning model defines the loss
            # print("Logits", logits.size(), logits.dtype)
            # print("Labels", labels.size(), labels.dtype)
            loss = open_learning_model.loss(logits, labels)
        else:
            # Standard cross entropy training
            loss = F.cross_entropy(logits, labels, reduction=reduction)

        if weights is not None:
            loss = (loss * weights).sum()

        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        myloss = loss.detach().item()
        myepoch = epoch + 1
        wandb.log({"epoch": myepoch, "train/loss": myloss})
        print("\rEpoch {:d} | Loss: {:.4f}".format(myepoch, myloss),
              flush=True, end='')