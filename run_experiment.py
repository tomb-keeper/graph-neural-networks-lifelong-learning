
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import gc

import numpy as np
import pandas as pd
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
from models.node2vec import add_node2vec_args, train_node2vec, evaluate_node2vec


# EvolveGCN
# from models.evolvegcn.egcn_o import EGCN
# from models.evolvegcn.models import Classifier
# import models.evolvegcn.utils as egcn_utils

from datasets import load_data


def appendDFToCSV_void(df, csvFilePath, sep=","):
    """ Safe appending of a pandas df to csv file
    Source: https://stackoverflow.com/questions/17134942/pandas-dataframe-output-end-of-csv
    """
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception(
            "Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(
                len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


def compute_weights(ts, exponential_decay, initial_quantity=1.0, normalize=True):
    ts = torch.as_tensor(ts)
    delta_t = ts.max() - ts
    values = initial_quantity * torch.exp(- exponential_decay * delta_t)
    if normalize:
        # When normalizing, the initial_quantity is irrelevant
        values = values / values.sum()
    return values


def train(model, optimizer, g, feats, labels, mask=None, epochs=1, weights=None,
          backend='dgl'):
    model.train()
    reduction = 'none' if weights is not None else 'mean'

    if hasattr(model, '__reset_cache__'):
        print("Resetting Model Cache")
        model.__reset_cache__()

    for epoch in range(epochs):
        inputs = (g, feats) if backend == 'dgl' else (feats, g)
        logits = model(*inputs)

        if mask is not None:
            loss = F.cross_entropy(logits[mask], labels[mask], reduction=reduction)
        else:
            loss = F.cross_entropy(logits, labels, reduction=reduction)

        if weights is not None:
            loss = (loss * weights).sum()

        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {:d} | Loss: {:.4f}".format(epoch + 1, loss.detach().item()))


def evaluate(model, g, feats, labels, mask=None, compute_loss=True,
             backend='dgl'):
    model.eval()

    if hasattr(model, '__reset_cache__'):
        print("Resetting Model Cache")
        model.__reset_cache__()

    with torch.no_grad():
        inputs = (g, feats) if backend == 'dgl' else (feats, g)
        logits = model(*inputs)

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        if compute_loss:
            loss = F.cross_entropy(logits, labels).item()
        else:
            loss = None

        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        __max_vals, max_indices = torch.max(logits.detach(), 1)
        acc = (max_indices == labels).sum().float() / labels.size(0)
        f1 = f1_score(labels.cpu(), max_indices.cpu(), average="macro")

    return acc.item(), f1, loss


def build_model(args, in_feats, n_hidden, n_classes, device, n_layers=1, backend='geometric',
                edge_index=None, num_nodes=None):
    if args.model == 'graphsaint':
        assert backend == 'geometric'
        model_spec = args.variant
    else:
        model_spec = args.model

    if backend == 'geometric':
        print("Using Geometric Backend")
        if model_spec == 'gs-mean':
            model = geo.GraphSAGE(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif model_spec == "gcn":
            model = geo.GCN(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif model_spec == "gat":