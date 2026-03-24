"""comm_gossip.py — Model-weight gossip communication for decentralised FL.

This module implements the standard (non-vMF) gossip training loop.  Unlike
comm_vmf_gossip, neighbour weights are determined by the static or dynamic
adjacency matrix rather than vMF likelihood scores, and *model parameters* are
exchanged (not just prototypes).

Key functions
-------------
make_doubly_stochastic : Sinkhorn-style balancing of a mixing matrix (tensor).
comm_gossip            : Main round loop — local update + weighted model average.

Internal helper
---------------
update_clients (nested) : thin wrapper used to print client ID before update().
"""

from utils.utils_proto import *
from collections import defaultdict
import json
import time
import random
import numpy as np
import torch.nn.functional as F
from comm_utils.decentralized import *


def make_doubly_stochastic(W, iters=5, eps=1e-12):
    """Iteratively balance a mixing matrix so both rows and columns sum to 1.

    Applies alternating column-normalisation then row-normalisation (Sinkhorn
    iterations).  Starting from a row-stochastic matrix (rows already sum to 1),
    ``iters`` passes are usually sufficient for small graphs (N ≤ 20).

    Args:
        W: [N, N] torch.Tensor that is at least approximately row-stochastic.
            Modified **in-place**.
        iters: Number of alternating normalisation passes (default: 5).
        eps: Small constant clamped to the divisor to prevent division by zero
            when a column or row sum is effectively zero (default: 1e-12).

    Returns:
        torch.Tensor: The balanced (doubly stochastic) matrix W (same object,
            modified in-place).
    """
    for _ in range(iters):
        col_sum = W.sum(0, keepdim=True).clamp_min(eps)
        W /= col_sum                       # make columns sum to 1
        row_sum = W.sum(1, keepdim=True).clamp_min(eps)
        W /= row_sum                       # restore rows = 1
    return W



def comm_gossip(args, adj, clients, debug=False, test_loader=None):
    """Main training loop for the standard (model-weight) gossip algorithm.

    Each round proceeds in three phases:

    1. **Dynamic topology** (optional) — if ``args.dynamic_topo == 1``, a new
       Erdős–Rényi random graph is sampled (p=0.5) and its adjacency matrix is
       doubly-stochastic balanced via ``make_doubly_stochastic``.

    2. **Local update** — every client runs one local training epoch
       (``client.update()``).

    3. **Weighted model average** — for each client *i*, its new model
       parameters are computed as:
           W_new[i] = adj[i][i] * W[i]  +  Σ_{j∈neighbours} adj[i][j] * W[j]
       where ``adj`` is the (possibly updated) mixing matrix.  The result is
       loaded back via ``client.avg_model``.

    Timing is recorded per-round for both local training and gossip phases.

    Args:
        args: Experiment configuration namespace.  Relevant fields:
            ``num_rounds``, ``global_seed``, ``num_clients``, ``dynamic_topo``.
        adj: [N × N] float tensor mixing matrix.  ``adj[i][j]`` is the weight
            client *i* gives to client *j*'s model.  Updated in-place when
            ``dynamic_topo == 1``.
        clients: List of client objects (one per federated node).
        debug: Unused; reserved for future verbose logging.
        test_loader: Unused in this function; each client evaluates on its own
            local test loader via ``evaluate_clients``.

    Returns:
        list[float]: Per-round average local test accuracy across all clients.

    Side effects:
        Loads new model state dicts into each client via ``client.avg_model``
        each round.  Prints per-round timing statistics.
    """
        
    print('Gossip training.')
    # torch.manual_seed(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    # torch.autograd.set_detect_anomaly(True)
    
    
    reached_consensus = False
    avg_l_acc = []
    avg_g_acc = []
    avg_l_auc = []
    avg_g_auc = []
    avg_l_unc = []
    avg_g_unc = []

    #cluster_heat = np.zeros([args.num_clusters,args.num_clusters])
    client_heat = np.zeros([args.num_clients,args.num_clients])
    
    def update_clients(c):
       print(f"Updating client {c.id}")
       c.update()
    
    local_times, gossip_times = [], []
    for e in range(args.num_rounds):

        print(f'\nRound : {e}')
        # Update each client


        if args.dynamic_topo == 1:
            #create new graph every round
            graph = get_communication_graph(adj.shape[0], 0.5, 3+e) #p=0.5 seed=3
            adj_mat = nx.adjacency_matrix(graph, weight=None).todense()
            adj = torch.from_numpy(np.array(adj_mat)).float().to(clients[0].device)
            adj = make_doubly_stochastic(adj) 
            print(f'New mixing mat: {adj}')



        for m in clients:
            print(f'Updating {m.id}')
            start = time.perf_counter()
            m.update()  # one local epoch
            end = time.perf_counter()
            local_times.append(end - start)            
            # print(f'Performance: {m.performance_test()}\n')



        req_aggregation = [True for i in range(len(clients))]
        # Set up variables where we do the averaging calculation without disturbing the previous weights
        tot_train_data = 0 #############
        new_models = []
        Ni = [1 for i in range(len(clients))] #selected neighbor length of client i every iteration(we devide by this ncorresponding umber during averaging)
        
        #copy self models
        for i in range(len(clients)):
          new_models.append(copy.deepcopy(clients[i].model.state_dict()))   
          
        tg0 = time.perf_counter()
        # Start averaging towards the goal. Here we use equal neighbor averaging method.
        for i in range(len(clients)):
          
          #No_of_sample = int(len(clients)* args.sampling)
          # random_client_ids = [random.randint(0, len(clients) - 1) for _ in range(No_of_sample)]

          # Select clients to train and participate in averaging
          adj_clients = [clients[c] for c in range(len(clients)) if (adj[c][i] and c!=i)]
          
          # Random selection
          # print('FAvg aggregation')
          # V_set = set(range(len(adj_clients)))
          # m = int(args.sub_mod_sel_ratio * len(adj_clients)) #select .7 of neighbors
          # sel_neighbor_indx = np.random.choice(list(V_set), m, replace=False)
          # Ni[i]=len(sel_neighbor_indx)+1  #including self
                  
          
          #sel_neighbors = [adj_clients[j] for j in sel_neighbor_indx] 
          neighbors = adj_clients

            
          for key in clients[i].model.state_dict().keys():
              new_models[i][key] = adj[i][i] * clients[i].model.state_dict()[key]
          
          # for sc in neighbors:
          #     for key in sc.model.state_dict().keys():
          #         new_models[i][key] += adj[i][sc.id] * sc.model.state_dict()[key]
            

          for sc in neighbors:
              if len(adj_clients)>1: # and sc.id in random_client_ids:  #sampling from adjacents     #if random selection from adjacents
                for key in sc.model.state_dict().keys():
                  new_models[i][key] += adj[i][sc.id] * sc.model.state_dict()[key]



        # average and update self
        for i in range(len(clients)):
                #clients[i].avg_model(new_models[i], Ni[i]) #N[i]: tot neighbors of client i each round
                clients[i].avg_model(new_models[i], 1) #N[i]: tot neighbors of client i each round
                print(f'Aggregated client {i}')
        
        tg1 = time.perf_counter()
        gossip_times.append(tg1 - tg0)        

       #list every round avg performance of all clients (local test data and global test data)
       #lacc, gacc, lauc, gauc, lunc, gunc = evaluate_clients(clients, test_loader)
        lacc = evaluate_clients(clients, test_loader)
        print(f'Avg ACC:{lacc}')
        avg_l_acc.append(lacc)
        # avg_g_acc.append(gacc)
        # avg_l_auc.append(lauc)
        # avg_g_auc.append(gauc)
        # avg_l_unc.append(lunc)
        # avg_g_unc.append(gunc)


        
        # Normalize the matrix
        # Client_heat = (client_heat - np.min(client_heat)) / (np.max(client_heat) - np.min(client_heat))
        # Set diagonal elements to zero
        np.fill_diagonal(client_heat, 0)

        mean_local  = sum(local_times) / len(local_times)
        mean_gossip = sum(gossip_times) / len(gossip_times) / args.num_rounds  # per round
        print(f'Mean local time:{mean_local}, Mean gossip time:{mean_gossip}')


    # for c in clients:
    #     #save client_models
    #     filename = args.params_dir + 'checkpoint_{}_{}_client_{}.pth.tar'.format(args.algorithm, args.dataset, c.id)
    #     torch.save({
    #         'model_state_dict': c.model.state_dict(),
    #         'optimizer_state_dict': c.optimizer.state_dict(),
    #         }, filename)
    #     print('Checkpoint file: ',filename)

        
    #     filename = args.params_dir + 'id_loaders_{}_{}_client_{}.pkl'.format(args.algorithm, args.dataset, c.id)
    #     with open(filename, 'wb') as f:
    #         if args.algorithm in ['FedAvg','Ditto']:
    #             pickle.dump([c.train_loader, c.test_loader, c.id_labels], f)
    #         else:
    #             pickle.dump([c.train_loader, c.test_loader, c.id_labels, c.local_protos, c.global_protos], f)
    #         print('Loaders file: ',filename)
    #         f.close()

    # print('Clients saved in file: ',filename)


  
    return avg_l_acc #, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc #client_heat 