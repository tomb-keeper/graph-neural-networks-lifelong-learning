""" Module holding the logic for writing results to disk in csv format """

import os
import pandas as pd


# do not change this
# unless also changing CSVResultsWriter.add_result()
RESULT_COLS = ['dataset',
               'history',
               'label_rate',
               'inductive',
               'seed',
               'backend',
               'model',
               'variant',
               'n_hidden',
               'n_layers',
               'dropout',
               'sampling',
               'batch