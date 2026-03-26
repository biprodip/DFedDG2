"""
vMF-likelihood-weighted prototype gossip communication for DFedDG2.

This module implements the gossip communication protocol used by
``ClientDecoodVMF`` clients. Instead of equal-weight averaging, each client
computes how well its own local features are explained by each neighbor's
von Mises-Fisher (vMF) mixture model, and uses those likelihoods as weights
when aggregating neighbor prototypes.

Key functions
-------------
``comm_vmf_gossip``
    Top-level round loop: local training → weight computation → adjacency
    update → prototype aggregation → DisLoss injection.

``neighbor_gossip_weights``
    Computes per-neighbor softmax weights from vMF-mixture log-likelihoods.

``compute_log_C``
    Numerically stable log normalizer of the vMF distribution.

``proto_aggregation``
    Aggregates weighted prototype lists and estimates kappa for each class.

``collect_features``
    Extracts a capped set of normalized features from a client's training data.

``make_doubly_stochastic``
    Iterative Sinkhorn balancing to produce a doubly stochastic weight matrix.

``receive_protos``
    Builds a weighted prototype list from selected neighbors plus self.
"""

from utils.utils_proto import *
from collections import defaultdict
import json
import logging
import time
import numpy as np
import torch.nn.functional as F
from scipy.special import ive
from comm_utils.decentralized import *

LOGGER = logging.getLogger(__name__)

def make_doubly_stochastic(W, iters=5, eps=1e-12):
    """
    Iterative Sinkhorn balancing to make a non-negative matrix doubly stochastic.

    Alternately normalizes columns then rows until both sum to 1 (up to
    numerical precision). Starting from a row-stochastic matrix converges
    quickly; ``iters=5`` is sufficient for typical mixing matrices.

    Args:
        W (Tensor): Non-negative matrix of shape [N, N].
        iters (int): Number of alternating normalization passes.
        eps (float): Minimum clamp value to prevent division by zero in
            degenerate rows/columns.

    Returns:
        Tensor: Doubly stochastic matrix of the same shape as ``W``,
            modified in-place.
    """
    for _ in range(iters):
        col_sum = W.sum(0, keepdim=True).clamp_min(eps)
        W /= col_sum   # normalize columns → each column sums to 1
        row_sum = W.sum(1, keepdim=True).clamp_min(eps)
        W /= row_sum   # normalize rows → each row sums to 1
    return W



def compute_log_C(kappa: torch.Tensor, d: int) -> torch.Tensor:
    """
    Compute the numerically stable log-normalizer log C_d(κ) of the vMF distribution.

    The vMF normalizer is defined via the modified Bessel function of the first kind:

        C_d(κ) = κ^(d/2 - 1) / ((2π)^(d/2) · I_{d/2-1}(κ))

    Taking the log and using scipy's exponentially-scaled Bessel function
    ``ive(ν, κ) = exp(-|κ|) · I_ν(κ)`` for numerical stability:

        log I_ν(κ) = |κ| + log(ive(ν, κ))
        log C_d(κ) = log I_ν(κ) + κ − ν · log(κ)

    where ν = d/2 − 1.

    Args:
        kappa (Tensor): Concentration parameters, shape [Nn, K] or any shape.
            Values are clipped to [1e-300, ∞) to avoid log(0).
        d (int): Feature dimensionality (the d in C_d).

    Returns:
        Tensor: log C_d(κ), same shape as ``kappa``, on the same device/dtype.
    """
    nu = d / 2.0 - 1.0  # Order of the Bessel function

    # Transfer to numpy; clip to avoid zeros in log
    k_np = kappa.detach().cpu().numpy()
    k_np = np.clip(k_np, 1e-300, None)
    tiny = np.finfo(np.float64).tiny

    # ive(ν, κ) = exp(-|κ|)·I_ν(κ)  →  I_ν(κ) = exp(|κ|)·ive(ν, κ)
    ive_vals = ive(nu, k_np)
    # ive_vals = np.maximum(ive_vals, 1e-300)  # Guard against underflow to zero

    # Guard against zeros / negatives / nan
    ive_vals = np.where(np.isfinite(ive_vals), ive_vals, tiny)
    ive_vals = np.clip(ive_vals, tiny, None)

    # log I_ν(κ) using the exponentially-scaled form for numerical stability
    logI = np.log(ive_vals) + np.abs(k_np)

    # Return to torch on the original device/dtype
    logI_t = torch.from_numpy(logI).to(device=kappa.device, dtype=kappa.dtype)

    # Assemble: log C_d(κ) = log I_ν + κ − ν·log κ
    logC = logI_t + kappa - nu * torch.log(kappa.clamp_min(1e-12))  # 1e-12: guard against log(0)
    return logC



def neighbor_gossip_weights(
        X:      torch.Tensor,   # [Ni, d]   — local client features
        mus:    torch.Tensor,   # [Nn, K, d] — neighbor prototype means
        kappas: torch.Tensor,   # [Nn, K]   — neighbor concentration params
        pis:    torch.Tensor    # [Nn, K]   — neighbor class mixing weights
) -> torch.Tensor:
    """
    Compute softmax-normalized gossip weights based on vMF-mixture log-likelihoods.

    For each neighbor j, evaluates how well the neighbor's vMF mixture model
    (parameterized by μ_jk, κ_jk, π_jk per class k) explains the focal
    client's local feature set X. The per-sample log-likelihood under neighbor j is:

        log q_j(x_n) = logsumexp_k [ log π_jk + log C_d(κ_jk) + κ_jk · (μ_jk · x_n) ]

    The total log-likelihood over all local samples is then softmax-normalized
    across neighbors to yield weights w_j ∈ (0, 1) summing to 1.

    Args:
        X (Tensor): Local feature matrix, shape [Ni, d]. Will be L2-normalized
            internally.
        mus (Tensor): Neighbor prototype means, shape [Nn, K, d]. Should already
            be L2-normalized (unit sphere).
        kappas (Tensor): vMF concentration parameters per neighbor/class,
            shape [Nn, K]. Higher κ = tighter cluster.
        pis (Tensor): Class mixing proportions per neighbor, shape [Nn, K].
            Typically set to the neighbor's empirical class frequencies.

    Returns:
        Tensor: Gossip weights w_j, shape [Nn], summing to 1.
    """
    X = F.normalize(X, dim=1)  # Ensure features lie on the unit sphere: [Ni, d]

    # Log-normalizer of the vMF distribution per neighbor/class: [Nn, K]
    logC = compute_log_C(kappas, X.size(1))

    # Log-likelihood terms: s[n,j,k] = log π_jk + logC_jk + κ_jk · (μ_jk · x_n)
    s = torch.einsum('id,jkd->ijk', X, mus)  # dot products μᵀx → [Ni, Nn, K]
    s = s * kappas.unsqueeze(0)              # scale by κ    → [Ni, Nn, K]
    s = s + logC.unsqueeze(0)               # add logC       → [Ni, Nn, K]
    s = s + torch.log(pis).unsqueeze(0)     # add log π      → [Ni, Nn, K]

    # Marginalize over classes: log q_j(x_n) = logsumexp_k s[n,j,k] → [Ni, Nn]
    log_q = torch.logsumexp(s, dim=2)

    # Sum log-likelihoods over all local samples → [Nn]
    log_like = log_q.sum(dim=0)

    # Softmax-normalize across neighbors so weights sum to 1
    return torch.softmax(log_like, dim=0)  # [Nn]




def receive_protos(client, sel_clients, neighbor_weights):
    """
    Build a list of weighted prototype dictionaries from neighbors and self.

    Each neighbor's prototypes are scaled by its corresponding gossip weight
    w_j before being added to the list. The focal client's own prototypes are
    appended unweighted as the last entry (weight implicitly 1.0, handled by
    ``proto_aggregation``).

    Args:
        client: The focal client object whose own prototypes are included.
        sel_clients (list): Neighbor client objects in the same order as
            ``neighbor_weights``.
        neighbor_weights (Tensor): Gossip weights w_j, shape [len(sel_clients)],
            as returned by ``neighbor_gossip_weights``.

    Returns:
        list[dict[int, Tensor]]: Length ``len(sel_clients) + 1``. Each entry is
            a dict mapping class label → weighted prototype tensor. The final
            entry is the focal client's own (unweighted) prototypes.
    """
    assert len(sel_clients) == len(neighbor_weights)

    uploaded_protos = []

    # Scale each neighbor's prototypes by its gossip weight w_j
    for cl, w in zip(sel_clients, neighbor_weights):
        weighted = {lab: proto * w for lab, proto in cl.local_protos.items()}
        uploaded_protos.append(weighted)

    # Append self prototypes unweighted (treated as weight=1 in proto_aggregation)
    self_weighted = {lab: proto for lab, proto in client.local_protos.items()}
    uploaded_protos.append(self_weighted)

    return uploaded_protos





def collect_features(client, max_pts=1024):
    """
    Extract a capped set of L2-normalized feature vectors from a client's training data.

    Iterates over the client's training loader, passes batches through the frozen
    backbone (``model.base``), L2-normalizes the output, and accumulates until
    at least ``max_pts`` samples are collected. Stops early once the cap is reached
    to avoid unnecessary computation on large datasets.

    Args:
        client: A client object with ``train_loader``, ``model``, and ``device``.
        max_pts (int): Maximum number of feature vectors to collect. Defaults to
            1024, which is sufficient for reliable vMF likelihood estimates while
            keeping computation lightweight.

    Returns:
        Tensor: L2-normalized feature matrix of shape [Ni, d], where Ni ≤ max_pts.
    """
    reps = []
    with torch.no_grad():
        for (x, _, _) in client.train_loader:
            x = x[0].to(client.device)
            r = client.model.base(x)
            reps.append(F.normalize(r, dim=1))
            if sum(t.size(0) for t in reps) >= max_pts:
                break
    return torch.cat(reps, dim=0)  # [Ni, d], Ni ≤ max_pts





def proto_aggregation(local_protos_list, num_classes, feat_dim, ood_template=None):
    """
    Aggregate weighted prototypes from multiple clients and estimate vMF concentration.

    Sums prototype tensors across all clients for each class label (prototypes
    from ``receive_protos`` are already scaled by gossip weights w_j), then
    estimates the vMF concentration parameter κ̂ from the pre-normalization norm
    R̂ of the summed vector using the Hornik & Grün (2014) formula:

        R̂ = ||Σ_j proto_j||
        κ̂ = R̂ · (d - R̂²) / (1 - R̂²)

    κ̂ is computed **before** L2-normalizing the aggregate, because normalization
    destroys the magnitude information needed for the estimate.

    Classes absent from all clients' prototype lists (e.g., globally unseen classes)
    are filled with ``ood_template`` if provided, or a zero vector otherwise,
    and assigned κ̂ = 0 to signal unreliable concentration.

    Args:
        local_protos_list (list[dict[int, Tensor]]): One dict per client, mapping
            class label → (possibly weighted) prototype tensor.
        num_classes (int): Total number of classes.
        feat_dim (int): Feature embedding dimensionality.
        ood_template (Tensor | None): Fallback prototype for completely missing
            class labels. Typically the OOD initialization prototype from the
            focal client.

    Returns:
        tuple[dict[int, Tensor], dict[int, float]]:
            - ``agg_protos_label``: L2-normalized aggregated prototype per class.
            - ``agg_kappa_label``: Estimated κ̂ per class (0 for missing classes).
    """
    agg_protos_label = defaultdict(list)
    agg_kappa_label = {}
    labels_check = [0] * num_classes  # Track which labels appear in at least one client

    # Collect all prototype tensors per label across clients
    for local_protos in local_protos_list:
        for label, proto in local_protos.items():
            agg_protos_label[label].append(proto.data)

    # Sum, estimate κ̂, then normalize
    for label, proto_list in agg_protos_label.items():
        proto_sum = sum(proto_list)

        # Estimate κ̂ from pre-normalization norm R̂ (Hornik & Grün, 2014)
        R_hat = torch.norm(proto_sum).detach()
        R_hat_sqr = R_hat ** 2
        kappa_hat = (
            R_hat * (feat_dim - R_hat_sqr) / (1 - R_hat_sqr)
            if R_hat < 0.999 else 1e6  # R̂ ≈ 1 → denominator → 0; cap at large finite value
        )
        agg_kappa_label[label] = kappa_hat

        agg_protos_label[label] = F.normalize(proto_sum, dim=0).detach()
        labels_check[label] = 1

    # Fill in missing class labels with OOD template or zero vector
    for label in range(num_classes):
        if not labels_check[label]:
            if ood_template is not None:
                agg_protos_label[label] = ood_template
            else:
                agg_protos_label[label] = torch.zeros(feat_dim).to(proto_sum.device)
            agg_kappa_label[label] = 0  # κ̂ = 0 signals no reliable concentration estimate

    return agg_protos_label, agg_kappa_label







def comm_vmf_gossip(args, adj, clients, debug=False, test_loader=None):
    """
    Run ``args.num_rounds`` rounds of vMF-likelihood-weighted prototype gossip.

    Each round proceeds in five phases:

    1. **Dynamic topology** (optional, ``args.dynamic_topo == 1``):
       Sample a new Erdős–Rényi graph each round (p=0.5) and Sinkhorn-balance
       its adjacency matrix.

    2. **Local update**: Every client trains for ``local_epochs`` epochs using
       the combined CE + DECOOD loss. Wall-clock time is recorded per client.

    3. **Weight computation**: For each client i, collect up to 1024 local
       feature vectors, then compute vMF-mixture log-likelihoods under each
       neighbor's prototype distribution → softmax-normalized weights w_j.
       The adjacency matrix is updated by blending learned weights with the
       base topology via ε-regularization (ε=0.01), then Sinkhorn-balanced.

    4. **Prototype aggregation**: Each client collects weighted neighbor
       prototypes via ``receive_protos``, then calls ``proto_aggregation``
       to produce global μ and κ estimates. Results are buffered in each
       client's ``aggregated_protos`` / ``aggregated_kappas``.

    5. **Post-gossip injection**: Buffered prototypes are committed to
       ``global_protos`` and injected into each client's DisLoss buffer so
       the next local training round starts from globally informed prototypes.

    After each round, average test accuracy is logged. At the end of all rounds,
    kappa history (currently empty placeholder) is written to
    ``global_local_kappa_history.json``.

    Args:
        args (Namespace): Experiment config. Required fields:
            ``num_rounds``, ``num_classes``, ``feat_dim``, ``dynamic_topo``,
            ``global_seed``.
        adj (np.ndarray | Tensor): Initial adjacency matrix, shape [N, N].
            Type depends on ``dynamic_topo``: numpy array for static (0),
            Tensor for dynamic (1).
        clients (list[ClientDecoodVMF]): All participating clients.
        debug (bool): Reserved for future verbose logging; currently unused.
        test_loader: DataLoader passed to ``evaluate_clients`` for per-round
            accuracy measurement. May be None if evaluation is not required.

    Returns:
        list[float]: Per-round average local test accuracy across all clients.

    Side effects:
        - Modifies ``adj`` in-place each round with updated gossip weights.
        - Writes ``global_local_kappa_history.json`` to the working directory.
    """
    avg_l_acc, results = [], []
    local_times, gossip_times = [], []

    for rnd in range(args.num_rounds):

        LOGGER.info("Round %02d", rnd)

        # --- Phase 1: Dynamic topology (resample graph each round) --------
        if args.dynamic_topo == 1:
            graph = get_communication_graph(adj.shape[0], 0.5, 3 + rnd)  # Erdős–Rényi p=0.5, deterministic seed per round
            adj_mat = nx.adjacency_matrix(graph, weight=None).todense()
            adj = torch.from_numpy(np.array(adj_mat)).float().to(clients[0].device)
            adj = make_doubly_stochastic(adj)
            # print(f'\nNew mixing mat: {adj}')

        # --- Phase 2: Local updates ----------------------------------------
        t0 = time.perf_counter()
        for cl in clients:
            start = time.perf_counter()
            cl.update()
            end = time.perf_counter()
            local_times.append(end - start)
        t1 = time.perf_counter()

        # print(adj)

        # --- Phase 3: Compute vMF-likelihood gossip weights ---------------
        for i, cli in enumerate(clients):

            # Identify graph neighbors of client i
            neigh_ids = [j for j in range(len(clients)) if adj[j][i] and j != i]
            if not neigh_ids:
                continue  # Isolated client: skip weight computation

            neighbors = [clients[j] for j in neigh_ids]

            K  = args.num_classes
            d  = args.feat_dim
            Nn = len(neighbors)

            # Build μ, κ, π tensors from neighbor prototype data
            mus_t    = torch.zeros(Nn, K, d, device=cli.device)
            kappas_t = torch.zeros(Nn, K, device=cli.device)
            pis_t    = torch.zeros(Nn, K, device=cli.device)

            # π_jk = empirical class frequency of neighbor j for class k
            for idx, nb in enumerate(neighbors):
                counts = nb.sample_per_class.to(cli.device)  # [K]
                total  = counts.sum().clamp_min(1.0)          # avoid div/0
                pis_t[idx] = counts / total

            # Populate μ and κ from each neighbor's local_protos and kappa_hats
            for idx, nb in enumerate(neighbors):
                for lab in range(K):
                    if lab in nb.local_protos:
                        mus_t[idx, lab]    = nb.local_protos[lab]
                        kappas_t[idx, lab] = nb.kappa_hats.get(lab, 0.0)

            # Collect focal client's features for likelihood evaluation (capped at 1024)
            X_i = collect_features(cli, max_pts=1024)

            # Compute softmax-normalized weights: w_j ∝ log-likelihood of X_i under neighbor j
            w_neighbors = neighbor_gossip_weights(X_i, mus_t, kappas_t, pis_t)  # [Nn]

            # Update adjacency with ε-regularized learned weights, then Sinkhorn-balance
            # ε=0.01 prevents any neighbor from getting zero weight (connectivity guarantee)
            eps = 0.01
            if args.dynamic_topo == 0:
                # Static topology: adj is a numpy array
                A = (adj > 0).astype(float)
                W = np.zeros_like(A)
                W[i, neigh_ids] = w_neighbors.cpu().numpy()
                W = (1 - eps) * W + eps * A
                W_torch = torch.from_numpy(W).to(cli.device)
                W_torch = make_doubly_stochastic(W_torch)
                adj = W_torch.cpu().numpy()
            else:
                # Dynamic topology: adj is a torch.Tensor
                A = (adj > 0).float()
                W = torch.zeros_like(A)
                W[i, neigh_ids] = w_neighbors
                W = (1 - eps) * W + eps * A
                adj = make_doubly_stochastic(W)


        # --- Phase 4: Prototype aggregation --------------------------------
        tg0 = time.perf_counter()
        for i, cli in enumerate(clients):
            rec_protos = receive_protos(cli, neighbors, w_neighbors)

            # Use OOD template if client has OOD labels, else fall back to zero vector
            ood_tmpl = cli.ood_init_proto if cli.ood_labels else None
            glob_mu, glob_kappa = proto_aggregation(
                rec_protos, args.num_classes, args.feat_dim, ood_tmpl
            )

            cli.save_aggregated_protos(glob_mu)
            cli.save_aggregated_kappas(glob_kappa)
        tg1 = time.perf_counter()
        gossip_times.append(tg1 - tg0)

        # --- Phase 5: Post-gossip injection --------------------------------
        for cl in clients:
            cl.set_global_protos()        # Commit buffered protos to global_protos
            cl.set_dis_loss_protos()      # Inject into DisLoss EMA buffer

        lacc = evaluate_clients(clients, test_loader)
        avg_l_acc.append(lacc)
        LOGGER.info("Avg test-ACC: %.4f", lacc)

        mean_local  = sum(local_times) / len(local_times)
        mean_gossip = sum(gossip_times) / len(gossip_times) / args.num_rounds
        LOGGER.info("Mean local time: %.4f s", mean_local)
        LOGGER.info("Mean gossip time: %.4f s", mean_gossip)

    # Save kappa history (results list is currently a placeholder for future logging)
    # with open("experiments/global_local_kappa_history.json", "w") as fp:
    #     json.dump(results, fp, indent=4)

    return avg_l_acc