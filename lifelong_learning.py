
import os
import json
import numpy as np
import torch
import torch_geometric as tg
import pickle
from tqdm import tqdm

def lifelong_nodeclf_identifier(dataset, t_zero, history, backend, label_rate=None):
    # make sure dataset is not a path
    dataset_name = os.path.basename(os.path.abspath(dataset))
    s = f"{dataset_name}-tzero{t_zero}-history{history}-{backend}"

    if label_rate is not None:
        s += "-" + str(label_rate)
    return s

class Task:
    def __init__(self, x, y, task_id=None):
        self.x = x
        self.y = y
        self.task_id = task_id

def collate_tasks(list_of_tasks):
    return list_of_tasks

class LifelongDataset(torch.utils.data.Dataset):
    """ Dataset class for Lifelong learning, yields Task(x,y) objects"""
    def __init__(self, t, x, y):
        self.task_ids = torch.as_tensor(t, dtype=torch.long)
        self.x = torch.as_tensor(x)
        self.y = torch.as_tensor(y)
        self.idx2task = np.unique(self.task_ids.numpy())
        self.task2idx = {task_id.item(): idx for  idx, task_id in enumerate(self.idx2task)}
        assert self.x.size(0) == self.y.size(0)
        assert self.x.size(0) == self.task_ids.size(0)

    def __getitem__(self, i):
        task_id = self.idx2task[i]
        return Task(self.x[self.task_ids == task_id], self.y[self.task_ids == task_id],
                task_id=task_id)

    def __len__(self):
        return len(self.idx2task)


def _check_graph_args(dgl_graph, edge_index, edge_attr):
    assert dgl_graph is not None or edge_index is not None, "Graph argument required"
    backend = ''
    if edge_index is not None:
        assert dgl_graph is None, "Supply only dgl graph or edge_index, not both!"
        backend = 'geometric'

    if dgl_graph is not None:
        assert edge_index is None, "Supply only dgl graph or edge_index, not both!"
        assert edge_attr is None, "Supply only dgl graph or edge_index, not both!"
        backend = 'dgl'
    return backend

# def _subsample_mask(self, mask, ratio):
#     """ Subsamples the training set, does not create overlap with test"""
#     subsample_mask = torch.rand(mask.size()) < ratio
#     new_mask = mask * subsample_mask
#     return new_mask

class NodeClassificationTask:
    def __init__(self, x, y, dgl_graph=None, edge_index=None, edge_attr=None, num_nodes=None,
            task_ids=None, train_mask=None, test_mask=None, task_id=None):
        self.backend = _check_graph_args(dgl_graph, edge_index, edge_attr)
        self.x = torch.as_tensor(x, dtype=torch.float)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.task_id = int(task_id) if task_id else None
        self.dgl_graph = dgl_graph # dgl (sub-)graph
        self.edge_index = edge_index  # geometric's sparse graph representation
        self.edge_attr = edge_attr   # geometric's edge attributes
        self.num_nodes = int(num_nodes) if num_nodes else self.x.size(0)
        # self.num_edges = edge_index.size(1) if edge_index is not None else dgl_graph.number_of_edges()
        self.task_ids = torch.as_tensor(task_ids, dtype=torch.long)
        self.train_mask = torch.as_tensor(train_mask, dtype=torch.bool)
        self.test_mask = torch.as_tensor(test_mask, dtype=torch.bool)

        assert self.task_ids.size(0) == self.num_nodes
        assert self.train_mask.size(0) == self.num_nodes
        assert self.test_mask.size(0) == self.num_nodes

    def set_all_train_(self):