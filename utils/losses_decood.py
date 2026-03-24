
"""
Loss functions for DFedDG2 decentralized federated learning.

This module provides three core loss classes used by ClientDecoodVMF:

  - ``CompLoss``: Compactness loss — pulls each sample's embedding toward its
    class prototype, encouraging intra-class clustering.

  - ``DisLoss``: Dispersion loss — pushes apart inter-class prototypes using
    Exponential Moving Average (EMA) updates. Also estimates the vMF
    concentration parameter (kappa) for each class prototype.

  - ``SupConLoss``: Supervised contrastive loss (Khosla et al., 2020).
    Supports both supervised (label-guided) and unsupervised (SimCLR) modes.

Additional utilities:
  - ``Proxy_Anchor``: Proxy-anchor metric learning loss (Kim et al., 2020).
  - ``binarize``: Converts integer labels to one-hot format.
  - ``l2_norm``: Row-wise L2 normalization.

References:
    SupCon: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
            https://arxiv.org/abs/2004.11362
    Proxy Anchor: Kim et al., "Proxy Anchor Loss for Deep Metric Learning", CVPR 2020.
    vMF kappa estimation: Hornik & Grün, "movMF: An R Package ...", JSS 2014.
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import defaultdict


def binarize(T, nb_classes):
    """
    Convert integer class labels to a one-hot binary matrix.

    Args:
        T (Tensor): 1-D integer label tensor of shape [N].
        nb_classes (int): Total number of classes.

    Returns:
        Tensor: Float tensor of shape [N, nb_classes] with one-hot rows.

    Note:
        Relies on the global ``args.device`` for output placement.
    """
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).to(args.device)
    return T

def l2_norm(input):
    """
    Apply row-wise L2 normalization to a 2-D tensor.

    Each row is divided by its L2 norm. A small epsilon (1e-12) is added
    to the squared norm before taking the square root to avoid division by zero.

    Args:
        input (Tensor): Float tensor of shape [N, D].

    Returns:
        Tensor: L2-normalized tensor of the same shape [N, D].
    """
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)  # 1e-12: numerical stability guard
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    """
    Proxy Anchor loss for deep metric learning (Kim et al., CVPR 2020).

    Each class is represented by a learnable proxy vector. The loss pulls
    samples toward their positive proxy and pushes them away from negative
    proxies using margin-based exponential terms.

    Args:
        nb_classes (int): Number of classes (one proxy per class).
        sz_embed (int): Embedding dimensionality.
        mrg (float): Margin applied to positive/negative cosine similarities.
        alpha (float): Scaling factor controlling loss sharpness.
    """
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).to(args.device))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        """
        Compute the Proxy Anchor loss for a batch of embeddings.

        Args:
            X (Tensor): Embedding matrix of shape [N, D].
            T (Tensor): Integer class labels of shape [N].

        Returns:
            Tensor: Scalar loss value.
        """
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Cosine similarity: [N, nb_classes]
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)  # Proxies with at least one positive sample in the batch
        num_valid_proxies = len(with_pos_proxies)

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class CompLoss(nn.Module):
    """
    Compactness Loss with class-conditional prototypes.

    Pulls each sample's embedding toward the prototype of its ground-truth
    class using a softmax cross-entropy over prototype similarities. This
    encourages intra-class compactness in the embedding space.

    The loss is computed as:
        L_comp = -(τ / τ_base) * mean_n [ log( exp(f_n · μ_{y_n} / τ) /
                                                 Σ_c exp(f_n · μ_c / τ) ) ]

    where f_n is the feature of sample n, μ_c is the prototype of class c,
    and τ is the temperature.

    Args:
        args: Global config. Requires ``args.num_classes`` and ``args.device``.
        temperature (float): Softmax temperature τ for logit scaling.
        base_temperature (float): Normalization temperature; typically equals ``temperature``.
    """
    def __init__(self, args, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels):
        """
        Compute the compactness loss for a batch of features.

        Expects prototypes to already be L2-normalized (as produced by DisLoss).

        Args:
            features (Tensor): Batch embeddings of shape [N, D].
            prototypes (Tensor): Class prototype matrix of shape [num_classes, D].
            labels (Tensor): Integer ground-truth labels of shape [N].

        Returns:
            Tensor: Scalar compactness loss.
        """
        proxy_labels = torch.arange(0, self.args.num_classes).to(self.args.device)  # [0, ..., num_classes-1]
        labels = labels.contiguous().view(-1, 1)  # reshape [N] → [N, 1] for broadcasting

        # print(proxy_labels.shape, (proxy_labels.T).shape, proxy_labels.permute(*torch.arange(proxy_labels.ndim - 1, -1, -1)).shape)
        mask = torch.eq(labels, proxy_labels.permute(*torch.arange(proxy_labels.ndim - 1, -1, -1))).float().to(self.args.device) #bz x cls    # [[0010000000],[0000010000],[0001000000]]

        # compute logits
        feat_dot_prototype = torch.div(
            torch.matmul(features, prototypes.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss




class DisLoss(nn.Module):
    """
    Dispersion Loss with EMA-updated class prototypes.

    Pushes apart the prototypes of different classes, promoting inter-class
    dispersion in the embedding space. Prototypes are maintained as Exponential
    Moving Averages (EMA) of per-class feature representations and updated
    in-place during each forward pass.

    Alongside prototypes, the vMF concentration parameter kappa is estimated
    using the Hornik & Grün (2014) approximation:

        R̂ = ||μ_unnorm||
        κ̂ = R̂ * (d - R̂²) / (1 - R̂²)

    where d is the feature dimension. A high κ̂ indicates a tight, reliable
    prototype; κ̂ = 0 indicates no data was seen for that class.

    The dispersion loss is:
        L_dis = (τ / τ_base) * mean_c [ log( Σ_{c' ≠ c} exp(μ_c · μ_c' / τ) /
                                              (num_cls - 1) ) ]

    Args:
        args: Global config. Requires ``num_classes``, ``feat_dim``, ``proto_m``, ``device``.
        model: Client model; uses ``model.base`` for feature extraction during initialization.
        loader: Training DataLoader used to compute initial prototype estimates.
        temperature (float): Softmax temperature τ.
        base_temperature (float): Normalization temperature; typically equals ``temperature``.
    """
    def __init__(self, args, model, loader, temperature= 0.1, base_temperature=0.1):
        super(DisLoss, self).__init__()
        self.args = args
        self.epsilon = 1e-08  # Small offset added to zero prototypes of unseen classes to avoid degenerate norms
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer("prototypes", torch.zeros(self.args.num_classes,self.args.feat_dim))
        self.model = model
        self.loader = loader
        self.kappa_hat = defaultdict(list)
        self.R_hat = defaultdict(list)
        self.init_class_prototypes()



    def forward(self, features, labels, prototypes):
        """
        Update EMA prototypes and compute the dispersion loss.

        For each sample in the batch, the prototype of its class is updated via:
            μ_c ← normalize( proto_m * μ_c + (1 - proto_m) * f )
        where proto_m is the EMA momentum from ``args.proto_m``.

        The vMF concentration κ̂ is recomputed after each EMA update using
        the pre-normalization norm R̂ of the updated prototype.

        The dispersion loss then encourages all class prototypes to be
        mutually dissimilar by maximizing the average inter-class similarity
        (acting as a repulsive objective).

        Args:
            features (Tensor): Batch embeddings of shape [N, D].
            labels (Tensor): Integer ground-truth labels of shape [N].
            prototypes (Tensor): Current prototype matrix of shape [num_classes, D].
                                 Updated in-place and stored in ``self.prototypes``.

        Returns:
            Tensor: Scalar dispersion loss.
        """
        num_cls = self.args.num_classes
        for j in range(len(features)):
            # EMA update: blend current prototype with new feature observation
            tmp_proto = prototypes[labels[j].item()] * self.args.proto_m + features[j] * (1 - self.args.proto_m)
            self.R_hat[labels[j].item()] = torch.norm(tmp_proto).detach()

            # Estimate vMF concentration κ̂ from pre-normalization norm R̂
            R_hat = self.R_hat[labels[j].item()]
            R_hat_sqr = R_hat * R_hat
            self.kappa_hat[labels[j].item()] = (
                R_hat * (self.args.feat_dim - R_hat_sqr) / (1 - R_hat_sqr)
                if R_hat < 0.999 else 1e6  # R̂ ≈ 1 makes denominator (1 - R̂²) → 0; cap at large finite value
            )

            prototypes[labels[j].item()] = F.normalize(tmp_proto, dim=0)

            # NOTE: Earlier versions used kappa_hat = R_hat directly (simpler proxy).
            # The current formula is the full Hornik & Grün (2014) approximation.

        self.prototypes = prototypes.detach()

        labels = torch.arange(0, num_cls).to(self.args.device)  # Use all class indices for prototype-level loss
        labels = labels.contiguous().view(-1, 1)

        # Off-diagonal mask: 1 for all inter-class pairs, 0 on diagonal
        mask = (1 - torch.eq(labels, labels.T).float()).to(self.args.device)

        logits = torch.div(
            torch.matmul(prototypes, prototypes.T),
            self.temperature)

        # Zero out self-similarity (diagonal) to exclude it from the mean
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_cls).view(-1, 1).to(self.args.device),
            0
        )
        mask = mask * logits_mask
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]  # Drop rows for classes with no neighbors in mask
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss



    def init_class_prototypes(self):
        """
        Initialize class prototypes by averaging features from the training loader.

        For each class, computes the mean feature vector from all training samples
        using the frozen backbone (``model.base``), then L2-normalizes the result.
        Also initializes R̂ and κ̂ per class.

        Classes absent from the local training data (OOD classes) are initialized
        to a zero vector offset by ``self.epsilon`` to avoid degenerate norms,
        with κ̂ = 0 to signal unreliable concentration.

        Side effects:
            Sets ``self.prototypes``, ``self.R_hat``, and ``self.kappa_hat``.
        """
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.args.num_classes
        with torch.no_grad():
            # Accumulate feature sums per class; classes not seen locally remain zero
            prototypes = torch.zeros(self.args.num_classes, self.args.feat_dim).to(self.args.device)
            for (input, target, _) in self.loader:
                input = input[0]
                input, target = input.to(self.args.device), target.to(self.args.device)
                features = self.model.base(input)
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            
            
            for cls in range(self.args.num_classes):
                if prototype_counts[cls] > 0:
                    prototypes[cls] /= prototype_counts[cls]
                    self.R_hat[cls] = torch.norm(prototypes[cls]).detach()

                    # Estimate vMF concentration κ̂ from pre-normalization norm R̂
                    R_hat = self.R_hat[cls]
                    R_hat_sqr = R_hat * R_hat
                    self.kappa_hat[cls] = (
                        R_hat * (self.args.feat_dim - R_hat_sqr) / (1 - R_hat_sqr)
                        if R_hat < 0.999 else 1e6  # R̂ ≈ 1 → denominator → 0; cap at large finite value
                    )
                else:
                    # OOD class: no local samples; offset from zero to give a valid norm
                    prototypes[cls] += self.epsilon
                    self.R_hat[cls] = torch.norm(prototypes[cls]).detach()
                    self.kappa_hat[cls] = 0  # κ̂ = 0 signals no reliable concentration estimate

            
            
            # NOTE: An earlier version used kappa_hat[cls] = R_hat directly as a simpler
            # proxy for concentration. The current formula is the full Hornik & Grün (2014)
            # approximation. The older variant is no longer active.

            duration = time.time() - start  # Prototype initialization wall-clock time (log if needed)

            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes