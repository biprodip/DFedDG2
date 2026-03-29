import logging
from utils.utils_proto import *
from collections import defaultdict
import json
import time
import random
import numpy as np
import torch.nn.functional as F
from comm_utils.decentralized import *

LOGGER = logging.getLogger(__name__)


def make_doubly_stochastic(W, iters=5, eps=1e-12):
    # W: [N,N] row-stochastic (rows already sum to 1)
    for _ in range(iters):
        col_sum = W.sum(0, keepdim=True).clamp_min(eps)
        W /= col_sum                       # make columns sum to 1
        row_sum = W.sum(1, keepdim=True).clamp_min(eps)
        W /= row_sum                       # restore rows = 1
    return W



def comm_gossip(args, adj, clients, debug=False, test_loader=None):
    '''
    For a client i, Aggregates(FedAvg) all clients j if j is adjacent to i and (i!=j)
    based on the adjacency matrics adj. 

    Not random gossip (where adjacents are randomly selected).
    '''
        
    LOGGER.info("Gossip training.")
    # torch.manual_seed(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)

    
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
       LOGGER.info("Updating client %s", c.id)
       c.update()
    
    local_times, gossip_times = [], []
    for e in range(args.num_rounds):

        LOGGER.info("\nRound : %s", e)
        # Update each client


        if args.dynamic_topo == 1:
            #create new graph every round
            graph = get_communication_graph(adj.shape[0], 0.5, 3+e) #p=0.5 seed=3
            adj_mat = nx.adjacency_matrix(graph, weight=None).todense()
            adj = torch.from_numpy(np.array(adj_mat)).float().to(clients[0].device)
            adj = make_doubly_stochastic(adj) 
            LOGGER.debug("New mixing mat: %s", adj)



        for m in clients:
            LOGGER.info("Updating %s", m.id)
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

          # Select clients to train and participate in averaging
          adj_clients = [clients[c] for c in range(len(clients)) if (adj[c][i] and c!=i)]
          

          neighbors = adj_clients

            
          for key in clients[i].model.state_dict().keys():
              new_models[i][key] = adj[i][i] * clients[i].model.state_dict()[key]

            

          for sc in neighbors:
              if len(adj_clients)>1: # and sc.id in random_client_ids:  #sampling from adjacents     #if random selection from adjacents
                for key in sc.model.state_dict().keys():
                  new_models[i][key] += adj[i][sc.id] * sc.model.state_dict()[key]



        # average and update self
        for i in range(len(clients)):
                #clients[i].avg_model(new_models[i], Ni[i]) #N[i]: tot neighbors of client i each round
                clients[i].avg_model(new_models[i], 1) #N[i]: tot neighbors of client i each round
                LOGGER.info("Aggregated client %s", i)
        
        tg1 = time.perf_counter()
        gossip_times.append(tg1 - tg0)        


        lacc = evaluate_clients(clients, test_loader)
        LOGGER.info("Avg ACC: %s", lacc)
        avg_l_acc.append(lacc)

        np.fill_diagonal(client_heat, 0)

        mean_local  = sum(local_times) / len(local_times)
        mean_gossip = sum(gossip_times) / len(gossip_times) / args.num_rounds  # per round
        LOGGER.info("Mean local time: %s, Mean gossip time: %s", mean_local, mean_gossip)

    return avg_l_acc 