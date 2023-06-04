
""" Module for Open Learning """
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score


import torch
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy


class OpenLearning(Module, ABC):
    """ Abstract base class for open world learning """

    @abstractmethod
    def loss(self, logits, labels):
        """ Return loss score to train model """
        raise NotImplementedError("Abstract method called")

    def fit(self, logits, labels):
        """ Hook to learn additional parameters on whole training set """
        return self

    @abstractmethod
    def predict(self, logits, subset=None):
        """ Return most likely classes per instance """
        raise NotImplementedError("Abstract method called")

    @abstractmethod
    def reject(self, logits, subset=None):
        """ Return example-wise mask to emit 1 if reject and 0 otherwise """
        raise NotImplementedError("Abstract method called")

    def forward(self, logits, labels=None, subset=None):

        reject_mask = self.reject(logits, subset=subset)
        predictions = self.predict(logits, subset=subset)
        loss = self.loss(logits, labels) if labels is not None else None

        return reject_mask, predictions, loss


class DeepOpenClassification(OpenLearning):
    """
    Deep Open ClassificatioN: Sigmoidal activation + Threshold based rejection
    Inputs should *not* be activated in any way.
    This module will apply sigmoid activations.
    """
    def __init__(self, threshold: float = 0.5,