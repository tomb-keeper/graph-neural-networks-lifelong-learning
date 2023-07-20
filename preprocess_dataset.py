import os
import argparse
from datasets import load_data
import torch

from lifelong_learning import make_lifelong_nodeclf_dataset, lifelong_nodeclf_identifier


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("--t_zero", default=None, type=int, help="Last task identifier before the first evaluation task identifier")
    parser.add_argument("--history", default=0, type=int, help="History size")
    parser.add_argument("--backend", default="dgl", type=str, choices=["dgl", "geometric"])
    parser.add_argument("--basedir", help="Basedir for preprocessed dataset, else create subdirectory in input")
    parser.add_argument("--label_rate", help="Subsample the train nodes globally.", default=None, type=float)
    args = parser.parse_args()
    if args.label_rate is not None:
        asser