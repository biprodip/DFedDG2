"""metrics.py — Lightweight evaluation metrics for federated learning clients.

These functions operate on raw model outputs (logits) and target labels, and
return scalar tensors suitable for accumulation across batches.
"""

import torch
import torch.nn.functional as F


def mse(y_pred, y):
    """Mean squared error between predictions and targets.

    Args:
        y_pred: Predicted values tensor.
        y: Ground-truth target tensor of the same shape as ``y_pred``.

    Returns:
        torch.Tensor: Scalar MSE loss.
    """
    return F.mse_loss(y_pred, y)


def binary_accuracy(y_pred, y):
    """Number of correct predictions for a binary classifier.

    Applies sigmoid then rounds to 0/1 before comparing to ``y``.

    Args:
        y_pred: Raw logits or probabilities (before sigmoid), shape [N].
        y: Binary ground-truth labels {0, 1}, shape [N].

    Returns:
        torch.Tensor: Scalar count of correct predictions (not normalised).
    """
    y_pred = torch.round(torch.sigmoid(y_pred))  # round to nearest integer
    correct = (y_pred == y).float()
    acc = correct.sum()
    return acc


def accuracy(y_pred, y):
    """Number of correct predictions for a multi-class classifier.

    Selects the class with the highest logit as the predicted label.

    Args:
        y_pred: Class logits, shape [N, C].
        y: Ground-truth class indices, shape [N].

    Returns:
        torch.Tensor: Scalar count of correct predictions (not normalised).
    """
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).float()
    acc = correct.sum()
    return acc
