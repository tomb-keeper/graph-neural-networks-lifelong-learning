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
               'batch_size',
               'saint_coverage',
               'initial_epochs',
               'initial_lr',
               'initial_wd',
               'annual_epochs',
               'annual_lr',
               'annual_wd',
               'start',
               'decay',
               'year',
      