import os.path as osp
import json
import argparse
import numpy as np
import pandas as pd

def resolve_classes(class_set, index2class):
    return [index2class[c] for c in list(class_set)]


def main():
    parser = argparse.Ar