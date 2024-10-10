import copy
import torch
import numpy as np
from numpy import random 
from multiprocessing import pool
from utils.utils import evaluate_clients


def comm_gossip(args, adj, clients, debug=False, test_loader=None):
    '''
    For a client i, Aggregates(FedAvg) all clients j if j is adjacent to i and (i!=j)
    based on the adjacency matrics adj. 

    *Not random gossip (where adjacents are randomly selected).
    '''
    
    
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

    
    def update_clients(c):
       c.update()
    
    for e in range(args.num_rounds):
        
        print(f'Round : {e}')
        # Update each client
        
        
        # find adjacents
        # for m in clients:   #################################################################################pass adj to clients
        #     #adj_clients = [clients[j] for j in range(len(clients)) if (adj[m.id][j]==1 and j!=m.id)]
        #     #print(f'Total adj: {len(adj_models)}')
        #     #if args.diverse_update:
        #     m.update()
        
        
        
        
        # Create a thread pool
        
        #with pool.ThreadPool() as workhorse:
        #   workhorse.starmap(update_clients, [(m,) for m in clients])
        #print(f"All clients updated")     
        
        for m in clients:
            m.update()
            print(f'client {m.id} updated')

        if reached_consensus:
          break
        
        # Set up variables where we do the averaging calculation without disturbing the previous weights
        tot_train_data = 0 #############nnnnnnnnn
        new_models = []
        for i in range(len(clients)):
          new_models.append(copy.deepcopy(clients[i].model.state_dict()))##copying all client models (adjacent?) 
          # tot_train_data +=  len(clients[i].train_loader.dataset)        #############nnnnnnnnn
          # for key in new_models[i].keys():                               #############nnnnnnnnn
          #     new_models[i][key] *= len(clients[i].train_loader.dataset) #############nnnnnnnnn
            

        #consensus checking and aggregation
        #consensus_all_pair_err_min = True
        # Start averaging towards the goal. Here we use equal neighbor averaging method.
        for i in range(len(clients)):
          
          # Select clients to train and participate in averaging
          ########adj_clients = [clients[c] for c in range(len(clients)) and adj[i,c])]
          ########selected_clients = np.random.choice(adj_clients, args.num_sel_clients, replace=False)
          
          for j in range(len(clients)):
            # Record each key's value, while also keeping track of distance
            #client_wise_dist = 0
            
            for key in new_models[j].keys():
              
              #if (consensus_all_pair_err_min==True and j>i): #dist of i<=j has been computed
              #  client_wise_dist += torch.norm( clients[j].model.state_dict()[key] - clients[i].model.state_dict()[key] ) #diversity
              if ((i!=j) and (adj[i,j])):     #i is already copied during initialization of new_models
                #new_models[i][key] += clients[j].model.state_dict()[key]*len(clients[j].train_loader.dataset)  #############nnnnnnnnn
                new_models[i][key] += clients[j].model.state_dict()[key]    
            
            # if client_wise_dist > args.avg_error_thres:   # if client_wise_dist > avg_error_thres and reached_consensus and j>i: ****** 
            #     #print(f'Clients :{i} and :{j} showed there is no convergence yet')
            #     consensus_all_pair_err_min = False
        
        # if consensus_all_pair_err_min:  
        #         reached_consensus = True
        #         print('Consensus reached (pairwise errors < avg_error_thres)')
        #         break
            
        # Process the recorded values
        for i in range(len(clients)):
                #for FC  sum up all adjacents to get Ni
                Ni = np.sum(adj[i,:]>0)+1 #since (i,i)=0, to count self(+1)
                #for key in new_models[i].keys():
                #    new_models[i][key] /= Ni # Or use torch.div()     #average with the number of adjacents + 1(self)
                #    new_models[i][key] /= tot_train_data   #############nnnnnnnnn
                #print(f'Ni: {Ni}')
                clients[i].avg_model(new_models[i], Ni)


       #list every round avg performance
        lacc, gacc, lauc, gauc, lunc, gunc = evaluate_clients(clients, test_loader)
        avg_l_acc.append(lacc)
        avg_g_acc.append(gacc)
        avg_l_auc.append(lauc)
        avg_g_auc.append(gauc)
        avg_l_unc.append(lunc)
        avg_g_unc.append(gunc)

  
    return avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc 