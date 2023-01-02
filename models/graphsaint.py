
import gc
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler
from torch_geometric.data import GraphSAINTNodeSampler, Data


def cpu_fallback(model, x, edge_index):
    device = x.device
    try:
        output = model(x, edge_index)
        did_shift = False
    except RuntimeError:
        print("[CUDA MEMORY EXCEEDED] Shifting data and model to CPU")
        x = x.cpu()
        edge_index = edge_index.cpu()
        model = model.cpu()
        output = model(x, edge_index)

        did_shift = True

    return output, did_shift




def train_saint(model, optimizer, g, feats, labels, mask=None, epochs=1, weights=None, sampling='node', walk_length=2,
                coverage=200, batch_size=1000, n_jobs=1, device=None):

    if mask is not None:
        assert mask.dtype == th.bool, "Mask needs to be dtype bool for GraphSAINT"
        print("GraphSAINT mask size:", mask.size(0))
        use_mask = True
    else:
        use_mask = False

    use_norm = coverage > 0

    if weights is not None:
        raise NotImplementedError("Weights not implemented for GraphSAINT")

    num_nodes = feats.size(0)
    if isinstance(batch_size, float):
        batch_size = int(num_nodes * batch_size)

    data = Data(x=feats, edge_index=g, y=labels, mask=mask)
    sampler_args = {'data': data, 'batch_size': batch_size,
                    'num_workers': 0, 'num_steps': epochs, 'sample_coverage': coverage,
                    'pin_memory': False} # Pin memory to optimize speed
    if sampling == "node":
        sampler = GraphSAINTNodeSampler
    elif sampling == "edge":
        sampler = GraphSAINTEdgeSampler
    elif sampling == "rw":
        sampler = GraphSAINTRandomWalkSampler
        sampler_args['walk_length'] = walk_length
    elif "class_balanced" in sampling:
        from sampler_torch.sampler import ClassBalancedSampler
        sampler = ClassBalancedSampler
        sampler_args.pop('num_workers')
        sampler_args['n_jobs'] = n_jobs
    else:
        raise NotImplementedError(f"\"{sampling}\" is not a supported sampling method for GraphSAINT!")
    model.train()
    loader = sampler(**sampler_args)

    reduction = 'none' if use_norm else 'mean'

    for i, batch in enumerate(loader):
        # mask -> saint sampled subgraph
        if device is not None:
            batch = batch.to(device)

        if use_norm:
            logits = model(batch.x, batch.edge_index, edge_weight=batch.edge_norm)
        else:
            logits = model(batch.x, batch.edge_index)

        if use_mask: