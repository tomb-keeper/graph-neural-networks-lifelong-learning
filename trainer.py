
import os.path

import dataclasses
from dataclasses import dataclass, field, is_dataclass

from tqdm import tqdm
import numpy as np
import pandas as pd
from enum import Enum
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

from lifelong_learning import LifelongGraphDataset

class RestartMode(Enum):
    WARM = "warm"
    COLD = "cold"


@dataclass
class TrainingArguments:
    dataset: str = field(default=None, metadata={"help": "Dataset name or path"})
    history: int = field(default=None, metadata={"help": "History size"})
    restart_mode: RestartMode = field(default="warm", metadata={"help": "Restart mode of {'warm', 'cold'}"})
    learning_rate: float = field(default=1e-3, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    num_pretrain_epochs: int = field(default=0, metadata={"help": "Number of pretraining epochs"})
    num_train_epochs: int = field(default=200, metadata={"help": "Number of training epochs per time"})


class ResultsWriter:
    """
    A stateful ResultsWriter that writes results to csv files including hyperparameters and other things
        Example:
            rw = ResultsWriter("something",
    """
    def __init__(self, path: str, state: dict = None):
        self.path = path
        self._state = dict(state) if state is not None else {}
        self._frozen_keys = None

    def _freeze(self, keys):
        """ Freezes the keys used for the first write """
        self._frozen_keys = frozenset(keys)

    @property
    def _isfrozen(self):
        return bool(self._frozen_keys)

    def _check(self, keys):
        if self._isfrozen and not set(keys).issubset(self._frozen_keys):
            raise KeyError("ResultsWriter's keys are frozen")

    def update(self, dictlike:dict=None, **kwargs):
        """
        Safely updates internal state, for example: rw.update(t=2)
        """
        if dictlike:
            kwargs = {**dictlike, **kwargs}
        self._check(kwargs.keys())

        self._state.update(kwargs)

    def add_result(self, dictlike:dict=None, **kwargs):
        """
        Writes a result to `self.path`. Arguments can be provided as dict or as keyword arguments.
        """
        if dictlike:
            kwargs = {**dictlike, **kwargs}
        self._check(kwargs.keys())

        # Create new dict to write, *dont* update state
        record = {**self._state, **kwargs}

        # Write full record to csv file
        result = pd.DataFrame.from_records([record])
        # include header only if file does not exist
        header = not os.path.isfile(self.path)
        result.to_csv(self.path, header=header, index=False, mode='a')

        if not self._isfrozen:
            # Freeze after first write
            # including result-specific columns
            self._freeze(record.keys())
            # No more updates to the state's keys are allowed


class IncrementalTrainer:
    def __init__(self, model, dataset: LifelongGraphDataset, args: TrainingArguments):
        """ Initializes a trainer """
        # TODO: Add argument for inductive
        # TODO: Add tensorboard SummaryWriter
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.args = args
        self.dataset = dataset
        self.num_nodes = features.size(0) if num_nodes is None else num_nodes
        assert args.restart_mode in ['warm', 'cold'], "Unknown restart mode: " + restart_mode
        self.results_writer = ResultsWriter("/tmp/test.csv", dataclasses.asdict(args))
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        """
        Method that returns an optimizer
        Subclasses may overwrite this to use a different optimizer
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)

    def _prepare_data_for_time(self, t, history, device=None, exclude_class=None):
        print("Preparing data for time:", t)
        # Prepare subgraph

        # Subg holds vertex ids corresponding to original graph
        subg_nodes = torch.arange(self.num_nodes)[(self.timestamps <= t) & (t >= (t - history))]
        subg_num_nodes = subg_nodes.size(0)

        # Create the subgraph (depends on backend)
        if self.backend == 'dgl':
            subg = graph.subgraph(subg_nodes)
            subg.set_n_initializer(dgl.init.zero_initializer)
        elif self.backend == 'geometric':