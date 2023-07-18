import os
import argparse
from datasets import load_data
import torch

from lifelong_learning import make_lifelong_nodeclf_dataset, li