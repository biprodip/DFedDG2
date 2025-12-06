from utils.utils_proto import *
from collections import defaultdict
import json
import time
import numpy as np
import torch.nn.functional as F
from scipy.special import ive
from comm_utils.decentralized import *

def make_doubly_stochastic(W, iters=5, eps=1e-12):
    # W: [N,N] row-stochastic (rows already sum to 1)
    for _ in range(iters):
        col_sum = W.sum(0, keepdim=True).clamp_min(eps)
        W /= col_sum                       # make columns sum to 1
        row_sum = W.sum(1, keepdim=True).clamp_min(eps)
        W /= row_sum                       # restore rows = 1
    return W



def compute_log_C(kappa: torch.Tensor, d: int) -> torch.Tensor:
    """
    Numerically stable log C_d(kappa) using scipy.special.ive:
      log I_ν(kappa) = |kappa| + log(ive(ν, kappa))
    then
      log C_d(kappa) = log I_ν + kappa − ν·log(kappa)
    """
    # ν = d/2 − 1
    nu = d/2.0 - 1.0

    # 1) pull to numpy and floor to avoid zeros
    k_np = kappa.detach().cpu().numpy()
    k_np = np.clip(k_np, 1e-300, None)

    # 2) ive returns exp(−|x|) I_ν(x), so I_ν(x) = exp(|x|)*ive(ν,x)
    ive_vals = ive(nu, k_np)
    ive_vals = np.maximum(ive_vals, 1e-300)

    # 3) log I_ν
    logI = np.log(ive_vals) + np.abs(k_np)

    # 4) back to torch
    logI_t = torch.from_numpy(logI).to(device=kappa.device, dtype=kappa.dtype)

    # 5) assemble log C_d = log I_ν + κ − ν·log κ
    logC = logI_t + kappa - nu * torch.log(kappa.clamp_min(1e-12))
    return logC



def neighbor_gossip_weights(
        X:      torch.Tensor,   # [Ni, d]
        mus:    torch.Tensor,   # [Nn, K, d]
        kappas: torch.Tensor,   # [Nn, K]
        pis:    torch.Tensor    # [Nn, K]
) -> torch.Tensor:
    """
    Returns softmax-normalised weights w_j ∈ [0,1]^Nn proportional to
    the vMF-mixture log-likelihood that neighbor j assigns to X.
    """
    # normalize data
    X = F.normalize(X, dim=1)  # [Ni, d]

    # get log C_d(kappa) per neighbor/component
    logC = compute_log_C(kappas, X.size(1))  # [Nn, K]

    # compute s[n,j,k] = log π_jk + logC_jk + κ_jk * (μ_jk · x_n)
    # use einsum with distinct indices:
    # X: [Ni, d], mus: [Nn, K, d]  → dot: [Ni, Nn, K]
    s = torch.einsum('id,jkd->ijk', X, mus)      # μᵀx   → [Ni, Nn, K]
    s = s * kappas.unsqueeze(0)                  # κ μᵀx → [Ni, Nn, K]
    s = s + logC.unsqueeze(0)                    # + logC → [Ni, Nn, K]
    s = s + torch.log(pis).unsqueeze(0)          # + logπ → [Ni, Nn, K]

    # log q_j(x_n) = logsumexpₖ s[n,j,k]  → [Ni, Nn]
    log_q = torch.logsumexp(s, dim=2)

    # total log-likelihood per neighbor: ∑ₙ log q_j(x_n)  → [Nn]
    log_like = log_q.sum(dim=0)

    # return normalized weights w_j ∝ exp(log_like_j)
    return torch.softmax(log_like, dim=0)        # [Nn], sums to 1




def receive_protos(client, sel_clients, neighbor_weights):
    """
    Build a list of weighted prototype dictionaries coming from the selected
    neighbors plus self.

    Args
    ----
    client          : the focal Client object (provides self prototypes)
    sel_clients     : list[Client]   the remote neighbors
    neighbor_weights: torch.Tensor   shape [len(sel_clients)]  — w_j

    Returns
    -------
    uploaded_protos : list[dict[label → Tensor]]
                      len = len(sel_clients) + 1   (the final entry is self)
    """
    assert len(sel_clients) == len(neighbor_weights)

    uploaded_protos = []

    # --- neighbors ---------------------------------------------------------
    for cl, w in zip(sel_clients, neighbor_weights):
        weighted = {lab: proto * w for lab, proto in cl.local_protos.items()}
        uploaded_protos.append(weighted)

    # --- self --------------------------------------------------------------
    self_weighted = {lab: proto for lab, proto in client.local_protos.items()}
    uploaded_protos.append(self_weighted)

    return uploaded_protos





def collect_features(client, max_pts=1024):
    reps = []
    with torch.no_grad():
        for (x, _, _) in client.train_loader:
            x = x[0].to(client.device)
            r = client.model.base(x)
            reps.append(F.normalize(r, dim=1))
            if sum(t.size(0) for t in reps) >= max_pts:
                break
    return torch.cat(reps, dim=0)           # [Ni, d]  (Ni ≤ max_pts)





#first compute kappa, then normalize
def proto_aggregation(local_protos_list, num_classes, feat_dim, ood_template=None):
    """
    Aggregate prototypes from multiple clients and compute their concentration parameters (kappa).
    
    Args:
        local_protos_list (list): A list of dictionaries, where each dictionary contains 
                                  prototypes of one client (key: label, value: prototype tensor).
        num_classes (int): Number of classes.
        feat_dim (int): Dimensionality of feature embeddings.
        ood_template (Tensor or None): Template prototype for out-of-distribution classes, if applicable.
    
    Returns:
        agg_protos_label (dict): Aggregated prototypes (key: label, value: normalized prototype tensor).
        agg_kappa_label (dict): Concentration parameters (kappa) for each label.
    """
    # To store aggregated prototypes and kappa values
    agg_protos_label = defaultdict(list)
    agg_kappa_label = {}

    # Check which labels are present
    labels_check = [0 for _ in range(num_classes)]  

    # Aggregate prototypes by summing across clients
    for local_protos in local_protos_list:  # Prototypes from each client
        for label, proto in local_protos.items():  # For each label in the client
            agg_protos_label[label].append(proto.data)

    # Compute the aggregated prototypes and kappa values
    for label, proto_list in agg_protos_label.items():
        # Sum all prototypes for the current label
        proto_sum = sum(proto_list)
        
        # Compute the concentration parameter (kappa) before normalization
        R_hat = torch.norm(proto_sum).detach()
        R_hat_sqr = R_hat ** 2
        kappa_hat = (
            R_hat * (feat_dim - R_hat_sqr) / (1 - R_hat_sqr)
            if R_hat < 0.999 else 1e6  # Handle edge case when R_hat ~ 1
        )
        agg_kappa_label[label] = kappa_hat

        # Normalize the aggregated prototype
        agg_proto = F.normalize(proto_sum, dim=0)
        agg_protos_label[label] = agg_proto.detach()

        labels_check[label] = 1

    # Handle missing labels by assigning OOD prototypes and default kappa
    for label in range(num_classes):
        if not labels_check[label]:
            if ood_template is not None:
                agg_protos_label[label] = ood_template
            else:
                agg_protos_label[label] = torch.zeros(feat_dim).to(proto_sum.device)
            agg_kappa_label[label] = 0  # No reliable concentration for missing labels

    return agg_protos_label, agg_kappa_label







def comm_vmf_gossip(args, adj, clients, debug=False, test_loader=None):
    """
    DECCON prototype gossip with vMF-likelihood weights.

    adj     : binary/static adjacency matrix (still used to know who is a neighbor)
    clients : list[Client]
    """

    # import random, numpy as np, json            # local import keeps header tidy

    # random.seed(args.global_seed)
    # np.random.seed(args.global_seed)


    avg_l_acc, results = [], []
    local_times, gossip_times = [], []

    for rnd in range(args.num_rounds):

        if args.dynamic_topo == 1:
            #create new graph every round
            graph = get_communication_graph(adj.shape[0], 0.5, 3+rnd) #p=0.5 seed=3
            adj_mat = nx.adjacency_matrix(graph, weight=None).todense()
            adj = torch.from_numpy(np.array(adj_mat)).float().to(clients[0].device)
            adj = make_doubly_stochastic(adj) 
            print(f'New mixing mat: {adj}')


        # ------------- LOCAL UPDATES --------------------------------------
        t0 = time.perf_counter()        
        for cl in clients:
            start = time.perf_counter()
            cl.update()  # one local epoch
            end = time.perf_counter()
            local_times.append(end - start)
        t1 = time.perf_counter()

        print(adj)

        # ------------- GOSSIP PHASE ---------------------------------------
        for i, cli in enumerate(clients):

            # neighbors according to adjacency
            neigh_ids = [j for j in range(len(clients)) if adj[j][i] and j != i]
            if not neigh_ids:
                continue

            neighbors = [clients[j] for j in neigh_ids]

            # --------- gather μ/κ tensors for likelihood weights ----------
            K   = args.num_classes
            d   = args.feat_dim
            Nn  = len(neighbors)

            # tensor placeholders
            mus_t     = torch.zeros(Nn, K, d, device=cli.device)
            kappas_t  = torch.zeros(Nn, K, device=cli.device)
            # pis_t     = torch.full((Nn, K), 1.0 / K, device=cli.device)


            # pi =local clas samples (use neighbor.sample_per_class):
            pis_t = torch.zeros(Nn, K, device=cli.device)
            for idx, nb in enumerate(neighbors):
                counts = nb.sample_per_class.to(cli.device)              # [K] counts per class
                total  = counts.sum().clamp_min(1.0)                     # avoid div0
                pis_t[idx] = counts / total

         
            for idx, nb in enumerate(neighbors):
                for lab in range(K):
                    if lab in nb.local_protos:
                        mus_t[idx, lab]    = nb.local_protos[lab]
                        kappas_t[idx, lab] = nb.kappa_hats.get(lab, 0.0)

            # print('Mu kappa computed')

            X_i = collect_features(cli, max_pts=1024)   # see helper below
            # print(X_i.shape)
            # print('Features collected')

            # --------- vMF likelihood weights -----------------------------
            w_neighbors = neighbor_gossip_weights(
                X_i,                  # [Ni, d]  #local feature###################
                mus_t,                # [Nn, K, d]
                kappas_t,             # [Nn, K]
                pis_t                 # [Nn, K]
            )                         # -> [Nn]
            

            if args.dynamic_topo == 0:
                A = (adj > 0).astype(float)
                W = np.zeros_like(A)
                W[i, neigh_ids] = w_neighbors.cpu().numpy()
                # print(f'Neighbors of client{cli.id}')
                # print(W)
                eps = 0.01
                W   = (1-eps)*W + eps*A                  # ε-regularise
                # if make_doubly_stochastic is written for torch, convert back later:
                W_torch = torch.from_numpy(W).to(cli.device)
                W_torch = make_doubly_stochastic(W_torch)
                adj = W_torch.cpu().numpy()
            else:
                # adj is a torch.Tensor of shape [N,N]
                A = (adj > 0).float()             # [N,N], FloatTensor
                W = torch.zeros_like(A)           # [N,N]
                W[i, neigh_ids] = w_neighbors     # w_neighbors is a torch.Tensor
                eps = 0.01
                W = (1 - eps) * W + eps * A
                W = make_doubly_stochastic(W)     # expects a torch.Tensor
                adj = W

            # print(f'All neighbors')
            # print(adj)

        tg0 = time.perf_counter()
        for i, cli in enumerate(clients):
            # print(f'Aggregating {i}')
            # print(adj)

            # --------- pull their protos with the new weights -------------
            rec_protos = receive_protos(cli, neighbors, w_neighbors)

            # use existing aggregation util (unchanged)
            ood_tmpl = cli.ood_init_proto if cli.ood_labels else None
            glob_mu, glob_kappa = proto_aggregation(
                rec_protos, args.num_classes, args.feat_dim, ood_tmpl
            )

            cli.save_aggregated_protos(glob_mu)
            cli.save_aggregated_kappas(glob_kappa)

        tg1 = time.perf_counter()
        gossip_times.append(tg1 - tg0)
        
        # ------------- POST-GOSSIP STEP -----------------------------------
        for cl in clients:
            cl.set_global_protos()
            cl.set_dis_loss_protos()

        lacc = evaluate_clients(clients, test_loader)
        avg_l_acc.append(lacc)
        print(f'Round {rnd:02d}  |  Avg test-ACC: {lacc:.4f}')

        mean_local  = sum(local_times) / len(local_times)
        mean_gossip = sum(gossip_times) / len(gossip_times) / args.num_rounds  # per round
        print(f'Mean local time:{mean_local}, Mean gossip time:{mean_gossip}')


    # ------------- optional: save kappa stats -----------------------------
    with open("global_local_kappa_history.json", "w") as fp:
        json.dump(results, fp, indent=4)

    return avg_l_acc