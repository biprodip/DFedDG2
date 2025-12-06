import torch
import torch.nn as nn
import torch.nn.functional as F


def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    # print('Using imbalance loss')
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        :param alpha: Weighting factor for the class. Can be a single float or a list of floats for each class.
        :param gamma: Focusing parameter to reduce the loss contribution from easy examples.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float32)

    def forward(self, inputs, targets):
        """
        :param inputs: Predictions (logits) with shape (N, C) where C = number of classes.
        :param targets: Ground truth labels with shape (N).
        """
        # Convert targets to one-hot encoding
        if inputs.dim() > 1:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        else:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Compute softmax over the classes
        probs = F.softmax(inputs, dim=1)
        probs = probs.clamp(min=1e-8, max=1.0)  # Avoid log(0)

        # Compute the focal loss
        log_p = torch.log(probs)
        focal_term = (1 - probs) ** self.gamma
        
        if self.alpha.dim() > 0:
            alpha = self.alpha.to(inputs.device)
            alpha = alpha.unsqueeze(0).expand_as(probs)
            alpha = alpha * targets_one_hot
        else:
            alpha = self.alpha

        loss = -alpha * focal_term * targets_one_hot * log_p

        if self.reduction == 'mean':
            return loss.sum() / targets.size(0)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss