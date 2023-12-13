
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

    if open_learning_model is not None:
        print("Fitting Open Learning Model")
        open_learning_model.fit(logits, labels)
        print(open_learning_model)


def evaluate(model, g, feats, labels, mask=None, compute_loss=True,
             backend='dgl',
             open_learning_model=None,
             known_classes: set = None,
             unseen_classes: set = None,
             save_logits=None):
    model.eval()

    if hasattr(model, '__reset_cache__'):
        print("Resetting Model Cache")
        model.__reset_cache__()

    with torch.no_grad():
        inputs = (g, feats) if backend == 'dgl' else (feats, g)
        logits = model(*inputs)

        # Reduce view on test mask
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        if compute_loss:
            if open_learning_model is None:
                loss = F.cross_entropy(logits, labels).item()
            else:
                loss = open_learning_model.loss(logits, labels).item()

        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        __max_vals, max_indices = torch.max(logits.detach(), 1)
        acc = (max_indices == labels).sum().float() / labels.size(0)
        f1 = f1_score(labels.cpu(), max_indices.cpu(), average="macro")

        scores = {
            'accuracy': acc.item(),
            'f1_macro': f1,
            'loss': loss
        }

        if open_learning_model is not None:
            subset = torch.LongTensor(list(known_classes))
            reject_mask = open_learning_model.reject(logits, subset=subset)
            predictions = open_learning_model.predict(logits, subset=subset)
            open_scores = open_learning.evaluate(labels, unseen_classes,
                                                 predictions, reject_mask)
            scores.update(open_scores)

            if save_logits is not None:
                print("Saveing logits to", save_logits)
                os.makedirs(save_logits, exist_ok=True)
                # Save logits
                np.savetxt(os.path.join(save_logits, "logits.gz"),
                           logits.sigmoid().cpu().numpy())

                # Save targets (same way as in open_learning.evaluate)
                labels_numpy = labels.cpu().clone().numpy()
                true_reject =  np.isin(labels_numpy, list(unseen_classes))
                labels_numpy[true_reject] = -100
                np.savetxt(os.path.join(save_logits, "labels.gz"),
                           labels_numpy, fmt="%d")

                # verify that we have not modified orig labels
                assert -100 not in labels, "Data leak. Needs fix"

    # return acc.item(), f1, loss
    return scores


def build_model(args, in_feats, n_hidden, n_classes, device, n_layers=1, backend='geometric'):
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
            print("Warning, GAT doesn't respect n_layers")
            heads = [8, args.gat_out_heads]  # Fixed head config
            n_hidden_per_head = int(n_hidden / heads[0])
            model = geo.GAT(in_feats, n_hidden_per_head, n_classes, F.relu, args.dropout, 0.6, heads).to(device)
        elif model_spec == "mlp":
            model = geo.MLP(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif model_spec == 'jknet-sageconv':
            # Geometric JKNEt with SAGECOnv
            model = JKNet(tg.nn.SAGEConv, in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout,
                    mode="cat", conv_kwargs={"normalize": False}, backend="geometric").to(device)
        elif model_spec == 'jknet-graphconv':
            model = JKNet(tg.nn.GraphConv, in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout,
                          mode="cat", conv_kwargs={"aggr": "mean"}, backend="geometric").to(device)
        elif model_spec == "sgnet":
            model = geo.SGNet(in_channels=in_feats, out_channels=n_classes, K=n_layers, cached=True).to(device)
        else:
            raise NotImplementedError(f"Unknown model spec 'f{model_spec} for backend {backend}")
    elif backend == 'dgl': # DGL models
        if model_spec == 'gs-mean':
            model = GraphSAGE(in_feats, n_hidden, n_classes,
                              n_layers, F.relu, args.dropout,
                              'mean').to(device)
        elif model_spec == 'mlp':
            model = MLP(in_feats, n_hidden, n_classes,
                        n_layers, F.relu, args.dropout).to(device)
        elif model_spec == 'mostfrequent':
            model = MostFrequentClass()
        elif model_spec == 'gat':
            print("Warning, GAT doesn't respect n_layers")
            heads = [8, args.gat_out_heads]  # Fixed head config
            # Div num_hidden by heads for same capacity
            n_hidden_per_head = int(n_hidden / heads[0])
            assert n_hidden_per_head * heads[0] == n_hidden, f"{n_hidden} not divisible by {heads[0]}"
            model = GAT(1, in_feats, n_hidden_per_head, n_classes,
                        heads, F.elu, 0.6, 0.6, 0.2, False).to(device)
        elif model_spec == 'node2vec':
            raise NotImplementedError("Node2vec initializer needs to move to different location")
            # model = tg.nn.Node2Vec(
            #     edge_index,
            #     n_hidden,
            #     args.n2v_walk_length,
            #     args.n2v_context_size,
            #     walks_per_node=args.n2v_walks_per_node,
            #     p=args.n2v_p,
            #     q=args.n2v_q,
            #     num_negative_samples=args.n2v_num_negative_samples,
            #     num_nodes=num_nodes,
            #     sparse=True
            # )
        elif model_spec == 'jknet-sageconv':
            # DGL JKNet
            model = JKNet(dgl.nn.pytorch.SAGEConv,
                    in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout,
                    mode="cat", conv_args=["mean"], backend='dgl').to(device)
        elif model_spec == 'sgnet':
            model = SGNet(in_feats, n_classes, k=n_layers, cached=True, bias=True, norm=None).to(device)
        else:
            raise NotImplementedError(f"Unknown model spec 'f{model_spec} for backend {backend}")
    else:
        raise NotImplementedError(f"Unknown backend: {backend}")

    return model


def build_optimizer(args, model):
    if args.model in ['most_frequent']:
        # for models that don't need an optimizer
        return None

    if args.model == 'node2vec':
        # Use SparseAdam for node2vec to speed things up
        optimizer = torch.optim.SparseAdam(model.parameters(),
                                           lr=args.lr * args.rescale_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr * args.rescale_lr,
                                     weight_decay=args.weight_decay * args.rescale_wd)
    return optimizer

def count_params(model):
   return sum(np.product(p.size()) for p in model.parameters())

def restart(model, mode, known_classes: set, new_classes: set):
    if mode == 'cold' or (mode == 'hybrid' and new_classes):
        # NEW version, equivalent to legacy-cold, but more efficient
        model.reset_parameters()
    elif mode == 'warm':
        # Skip for first task (does not make sense and makes problem for SGNET)
        if new_classes:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~ New classes encountered... ~~~~~~")
            print("~~~~~~ doing partial warm reinit! ~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # If there are new classes:
            # 1) Save parameters of final layer
            # 2) Reinit parameters of final layer
            # 3) Copy saved parameters to new final layer
            known_class_ids = torch.LongTensor(list(known_classes))
            saved_params = [p.data.clone() for p in model.final_parameters()]
            model.reset_final_parameters()
            print("[Debug] known_class_ids during restart:", known_class_ids)
            for i, params in enumerate(model.final_parameters()):
                if params.dim() == 1:  # bias vector
                    params.data[known_class_ids] = saved_params[i][known_class_ids]
                elif params.dim() == 2:  # weight matrix
                    params.data[known_class_ids, :] = saved_params[i][known_class_ids, :]
                else:
                    NotImplementedError("Parameter dim > 2 ?")
            # del saved_params  # Explicit cleanup!?
    else:
        raise NotImplementedError("Unknown --start arg: '%s'" % mode)
    return model

def zero_unseen_classes(model, unseen_classes: set):
    print(f"Setting params to zero for {len(unseen_classes)} classes")
    unseen_class_ids = torch.LongTensor(list(unseen_classes))
    for params in model.final_parameters():
        if params.dim() == 1:  # bias vector
            params.data[unseen_class_ids] = -1e12  # big negative bias
        elif params.dim() == 2:  # weight matrix
            params.data[unseen_class_ids, :] = 0   # zero weights
        else:
            NotImplementedError("Parameter dim > 2 ?")

    return model





def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_sampling = args.model in ['gcn_cv_sc']
    backend = args.backend

    print("Using backend:", backend)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if args.model == 'mostfrequent':
        device = torch.device("cpu")

    # LEGACY CODE, not used anymore
    # if args.model in ['graphsaint']:
    #     print("///////////////////")
    #     print("//// inductive ////")
    #     print("///////////////////")
    #     # Train completely on Task t-1
    #     globals_device = torch.device("cpu")
    #     assert args.inductive
    # else:
    #     print("//////////////////////")
    #     print("//// transductive ////")
    #     print("//////////////////////")
    #     globals_device = device
    #     inductive = False

    # Assume preprocessed dataset is in subdir of dataset
    print("Expecting preprocessed data at", args.data_path)
    dataset = LifelongNodeClassificationDataset(args.data_path,
                                                inductive=args.inductive)
    print(dataset)
    print(f"[t_min, tmax] = [{dataset.t_min}, {dataset.t_max}]")
    print(f"t_zero in dataset = {dataset.t_zero} (should be the one before t_start)")
    assert dataset.t_zero == args.t_start - 1, "Supplied t_start -1 is not equal to t_zero of dataset"
    assert dataset.history_size == args.history, "History sizes do not match"
    assert dataset.backend == args.backend, "Backends do not match"

    n_classes = dataset.num_classes
    in_feats = dataset.num_features
    n_hidden = args.n_hidden

    model = build_model(args, in_feats, n_hidden, n_classes, device,
                        n_layers=args.n_layers, backend=backend)
    if args.model == 'gcn_cv_sc':
        # unzip training and inference models
        model, infer_model = model
    print(model)
    optimizer = build_optimizer(args, model)

    if USE_WANDB:
        wandb.watch(model)

    num_params = count_params(model) if optimizer is not None else 0
    print("#params:", num_params)
    if args.only_count_params:
        exit(0)


    rw = CSVResultsWriter(args)

    known_classes = set()
    all_classes = set(range(dataset.num_classes))
    taskloader = torch.utils.data.DataLoader(dataset, shuffle=False,
                                             batch_size=1,
                                             collate_fn=collate_tasks)

    if args.open_learning is not None:
        olg_model = open_learning.build(args, num_classes=n_classes)
        print("Open Learning Model:", olg_model)
    else:
        # backward compat
        olg_model = None

    for t, batch in enumerate(taskloader):
        if args.only_first_task and t > 0:
            print("Finished with first task, exiting.")
            break

        if args.inductive:
            train_task, task = batch[0]
        else:
            train_task = None
            task = batch[0]