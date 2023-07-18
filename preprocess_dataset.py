import os
import argparse
from datasets import load_data
import torch

from lifelong_learning import make_lifelong_nodeclf_dataset, lifelong_nodeclf_identifier


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("--t_zero", default=