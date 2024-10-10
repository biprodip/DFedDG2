import copy
import torch
import numpy as np
import torch.nn as nn
from numpy import random 
from multiprocessing import pool
from collections import OrderedDict
from utils.utils import *


def comm_penz(args, adj, clients, debug=False, test_loader=None):
    '''
    For a client i, Aggregates(FedAvg) all clients j if j is adjacent to i and (i!=j)
    based on the adjacency matrics adj. 

    Not random gossip (where adjacents are randomly selected).
    '''
        
    print('Pens training.')
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
       print(f'Updating client {c.id}')
       c.update()
    

    for e in range(args.num_rounds):
        #every round 
        #First compute the neighbors and submodular selection for all clients and store in new_models[i]
        #Then aggregate for all clinets in another loop
        print(f'Round : {e}')
        # Update each client
    
        for m in clients:
            m.update()
        
        req_aggregation = [True for i in range(len(clients))]

        if reached_consensus:
          break
        
        # Set up variables where we do the averaging calculation without disturbing the previous weights
        tot_train_data = 0 #############nnnnnnnnn
        new_models = []
        Ni = [] #tot clients sel each round
        #copy self models
        for i in range(len(clients)):
          new_models.append(copy.deepcopy(clients[i].model.state_dict()))  ##copying all client models (adjacent?) 
          
          # tot_train_data +=  len(clients[i].train_loader.dataset)        #############nnnnnnnnn
          # for key in new_models[i].keys():                               #############nnnnnnnnn
          #     new_models[i][key] *= len(clients[i].train_loader.dataset) #############nnnnnnnnn
            

        #consensus checking and aggregation
        consensus_all_pair_err_min = True
        # Start averaging towards the goal. Here we use equal neighbor averaging method.
        for i in range(len(clients)):
          
          # Select clients to train and participate in averaging
          adj_clients = [clients[c] for c in range(len(clients)) if (adj[c][i] and c!=i)]
          neighbors, weights = clients_to_communicate_with(args, clients[i], adj_clients)
          
          if len(neighbors)>0:
              Ni.append(len(neighbors))
              print(f'{Ni[i]} neighbor selected for client {i}')
          else:
              Ni.append(0)
              print(f'no neighbor for client {i}.')
              req_aggregation[i] = False
              continue  #process for next client (this client should continue local learning)


          s_c = [c.id for c in neighbors]
          for peer in s_c:
              client_heat[clients[i].id][peer]+=1  #count communication
              client_heat[peer][clients[i].id]+=1
            
          #print(f'{clients[i].id} and selected peers: {s_c}')
          
          for sc in neighbors:
            # Record each key's value, while also keeping track of distance
            client_wise_dist = 0
            
            for key in sc.model.state_dict().keys():              
              #distance with parent(aggregator)
              if (consensus_all_pair_err_min==True): 
                client_wise_dist += torch.norm( sc.model.state_dict()[key].float() - clients[i].model.state_dict()[key].float() ) #diversity
              
              #new_models[i][key] += clients[j].model.state_dict()[key]*len(clients[j].train_loader.dataset)  #############nnnnnnnnn
              new_models[i][key] += sc.model.state_dict()[key]    
            
            if client_wise_dist > args.avg_error_thres:   # if client_wise_dist > avg_error_thres and reached_consensus and j>i: ****** 
                #print(f'Clients :{i} and :{j} showed there is no convergence yet')
                consensus_all_pair_err_min = False
        
        if consensus_all_pair_err_min:  
                reached_consensus = True
                print('Consensus reached (pairwise errors < avg_error_thres)')
                break
            
        # average and update self
        for i in range(len(clients)):
                if req_aggregation[i]:
                    clients[i].avg_model(new_models[i], Ni[i]) #N[i]: tot clients sel each round
                else:
                    print(f'Skipped aggregation for client {i}')


      #  #list every round avg performance of all clients (local test data and global test data)
      #   lacc, gacc, lauc, gauc, lunc, gunc = evaluate_clients(clients, test_loader)
      #   avg_l_acc.append(lacc)
      #   avg_g_acc.append(gacc)
      #   avg_l_auc.append(lauc)
      #   avg_g_auc.append(gauc)
      #   avg_l_unc.append(lunc)
      #   avg_g_unc.append(gunc)

        lacc = evaluate_clients(clients, test_loader)
        avg_l_acc.append(lacc)
      
        
        # Normalize the matrix
        client_heat = (client_heat - np.min(client_heat)) / (np.max(client_heat) - np.min(client_heat))
        # Set diagonal elements to zero
        np.fill_diagonal(client_heat, 0)

    return avg_l_acc  
    # return avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc, client_heat 