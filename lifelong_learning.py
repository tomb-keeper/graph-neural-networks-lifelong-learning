
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
        """ Compose a subgraph on the basis of the train set"""
        self.train_mask = torch.ones(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        return self

    def to(self, device):
        """ Put all relevant data to device """
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        if self.dgl_graph is not None:
            self.dgl_graph = self.dgl_graph.to(device)
        return self

    def graph(self):
        """ Returns dglgraph or edge_index depending on backend """
        if self.backend == 'dgl':
            return self.dgl_graph
        elif self.backend == 'geometric':
            return self.edge_index
        else:
            raise ValueError("Unknown backend")

    def __repr__(self):
        return f"NodeClassificationTask(task_id={self.task_id}, num_nodes={self.num_nodes})"


    def save(self, path):
        """ Saves the task as pickle """
        with open(path, 'wb') as fhandle:
            pickle.dump(self, fhandle)

    @staticmethod
    def load(path):
        """ Loads the pickle'd task """
        with open(path, 'rb') as fhandle:
            obj = pickle.load(fhandle)
        assert isinstance(obj, NodeClassificationTask)
        return obj





def _get_node_mask(task_ids, current, cumulate=0):
    return ((task_ids <= current) & (task_ids >= (current - cumulate)))


def _make_subgraph_task_dgl(task_ids, current, x, y, dgl_graph, cumulate=0,
                            global_train_mask=None):
    subg_mask = _get_node_mask(task_ids, current, cumulate=cumulate)
    # Create subgraph
    subg_nodes = torch.arange(dgl_graph.number_of_nodes())[subg_mask]
    subg = dgl_graph.subgraph(subg_nodes)
    # Reduce view of features, labels, task_ids
    subg_features = x[subg_mask]
    subg_labels = y[subg_mask]
    subg_task_ids = task_ids[subg_mask]


    # Create masks
    train_mask = subg_task_ids < current
    test_mask = subg_task_ids == current

    if global_train_mask is not None:
        train_mask = train_mask * global_train_mask[subg_mask]

    # Number of nodes
    subg_num_nodes = subg.number_of_nodes()

    return NodeClassificationTask(
            subg_features,
            subg_labels,
            dgl_graph=subg,
            num_nodes=subg_num_nodes,
            task_ids=subg_task_ids,
            train_mask=train_mask, test_mask=test_mask,
            task_id=current)

def _make_subgraph_task_geometric(task_ids, current, x, y, edge_index, edge_attr=None, cumulate=0,
                                  global_train_mask=None):
    subg_mask = _get_node_mask(task_ids, current, cumulate=cumulate)
    subg_edge_index, subg_edge_attr = tg.utils.subgraph(subg_mask,
            edge_index, edge_attr=edge_attr,
            relabel_nodes=True,
            num_nodes=x.size(0))
    # Reduce view of features, labels, task_ids
    subg_features = x[subg_mask]
    subg_labels = y[subg_mask]
    subg_task_ids = task_ids[subg_mask]
    # Create masks
    train_mask = subg_task_ids < current
    test_mask = subg_task_ids == current

    if global_train_mask is not None:
        train_mask = train_mask * global_train_mask[subg_mask]

    subg_num_nodes = subg_features.size(0)

    return NodeClassificationTask(
            subg_features, 
            subg_labels,
            edge_index=subg_edge_index,
            edge_attr=subg_edge_attr,
            num_nodes=subg_num_nodes,
            task_ids=subg_task_ids,
            train_mask=train_mask, test_mask=test_mask,
            task_id=current)

def make_subgraph_task(task_ids, current, x, y, dgl_graph=None, edge_index=None, edge_attr=None, cumulate=0,
                       global_train_mask=None):
    backend = _check_graph_args(dgl_graph, edge_index, edge_attr)
    if backend == 'geometric':
        task = _make_subgraph_task_geometric(task_ids,
                                             current,
                                             x,
                                             y,
                                             edge_index,
                                             edge_attr=edge_attr,
                                             cumulate=cumulate,
                                             global_train_mask=global_train_mask)
    elif backend == 'dgl':
        task = _make_subgraph_task_dgl(task_ids,
                                       current,
                                       x,
                                       y,
                                       dgl_graph,
                                       cumulate=cumulate,
                                       global_train_mask=global_train_mask)
    else:
        raise ValueError("Unknown Backend")
    return task



TASK_PREFIX = "task-"

def task_path(root_dir, i):
    task_filename = f"{TASK_PREFIX}{i}.pkl"
    return os.path.join(root_dir, task_filename)


def make_lifelong_nodeclf_dataset(path, task_ids, x, y,
        dgl_graph=None,
        edge_index=None, edge_attr=None,
        t_zero=None,
        cumulate=0,
        inductive=False,
        subsample_train=None):
    task_ids = torch.as_tensor(task_ids, dtype=torch.long)
    t_numpy = task_ids.numpy()
    print("Creating lifelong node classification dataset")
    backend = _check_graph_args(dgl_graph, edge_index, edge_attr)
    print(f"...using backend: {backend}")