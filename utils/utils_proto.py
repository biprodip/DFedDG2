import copy
import torch
import numpy as np
import torch.nn as nn
from numpy import random 
from multiprocessing import pool
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity




def evaluate_clients(clients, test_loader=None):
   round_avg_lacc = 0
   
   K = len(clients)
   # Load averaged weights in client(global models)
   for i in range(K):
       
       #local test performance
       test_acc, _, _ = clients[i].performance_test()
       #clients[i].l_test_loss_hist.append(rl_loss)
       clients[i].l_test_acc_hist.append(test_acc)
       print(f'Client id: {clients[i].id}, Lacc: {test_acc:.2f}')

       round_avg_lacc += test_acc   
       
   round_avg_lacc /= K
   
   return round_avg_lacc



def evaluate_clients_tmp(clients, test_loader):
   round_avg_lacc = 0
   round_avg_gacc = 0
   round_avg_lauc = 0
   round_avg_gauc = 0
   round_avg_lunc = 0
   round_avg_gunc = 0
   
   K = len(clients)
   # Load averaged weights in client(global models)
   for i in range(K):
       
       #local test performance
       rl_acc, rl_auc, rl_unc = clients[i].performance_test(clients[i].test_loader)
       #clients[i].l_test_loss_hist.append(rl_loss)
       clients[i].l_test_acc_hist.append(rl_acc)
       clients[i].l_test_auc_hist.append(rl_auc)
       clients[i].l_test_unc_hist.append(rl_unc)
       print(f'Clinet id: {clients[i].id},\nLacc: {rl_acc:.2f},  Lunc: {rl_unc:.2f}')

       round_avg_lacc += rl_acc  
       round_avg_lauc += rl_auc  
       round_avg_lunc += rl_unc  
       
       #clients[i].model.load_state_dict(new_models[i])
       #global test_set
       if test_loader is not None:
           rg_acc, rg_auc, rg_unc = clients[i].performance_test(test_loader)
           #clients[i].g_test_loss_hist.append(rg_loss)
           clients[i].g_test_acc_hist.append(rg_acc)
           clients[i].g_test_auc_hist.append(rg_auc)
           clients[i].g_test_unc_hist.append(rg_unc)
           #print(f'Clinet id: {clients[i].id}, Lloss: {rl_loss:.2f}  Lacc: {rl_acc:.2f}, Lunc: {rl_unc:.2f}, Gloss: {rg_loss:.2f}  Gacc: {rg_acc:.2f} Gunc: {rg_unc:.2f}')
           print(f'Gacc: {rg_acc:.2f} Gunc: {rg_unc:.2f} ')
                  
           round_avg_gacc += rg_acc
           round_avg_gauc += rg_auc
           round_avg_gunc += rg_unc
   
   round_avg_lacc /= K
   round_avg_lauc /= K
   round_avg_lunc /= K
   if test_loader is not None:  #if global test set exists
       round_avg_gacc /= K
       round_avg_gauc /= K
       round_avg_gunc /= K
   else:
       print('No global test set, so no global result.')
   
   return round_avg_lacc, round_avg_gacc, round_avg_lauc, round_avg_gauc,round_avg_lunc, round_avg_gunc





def clients_to_communicate_with(args, client, clients):
    '''
    param: args: arguments
           client: current client (c)
           clients : adjacent clients of c     

    returns: neighbors: selected adjacent clients
             weights    
              
    '''
    weights = None
    
    adj = [c.id for c in clients]
    ############################print(f'{client.id} and adjacents are {adj}')
    if args.neighbour_selection == "proto":  #loss based selection of n_neighbor and diversity based weight
        print('Proto similarity based neighbor selection')
        # Sample 10 times
        
        for _ in range(args.n_samplings):
            #clients_to_consider = np.random.choice(clients, args.n_clients_to_consider, replace=False)
            clients_to_consider = clients #adjacents

            #diversoty based weighting of selected clients
            diversity_client_metric = OrderedDict()
            
            client.collect_protos()  #update local protos on local train data
            for other_client in clients_to_consider:
                #other_client.target_protos(client.train_loader) #gt protos on clients dataset
                other_client.collect_protos()
                print(f'Dim of protos: {client.protos}')
                proto_disimilarity  = eval_protos(client.protos, other_client.protos)

                #train_loss, train_acc = evaluate(client.model, other_client.train_loader)
                diversity_client_metric[other_client] = proto_disimilarity  #+ other loss

            other_clients_sorted_by_diversity = sorted(
                diversity_client_metric, key=diversity_client_metric.get, reverse=True
            )


            if args.neighbour_exploration == "greedy":
                clients_to_consider = other_clients_sorted_by_diversity[: args.num_sel_clients]
            else:
                avg_metric = sum(diversity_client_metric.values()) / len(diversity_client_metric)
                neighbours = [k for k, v in zip(diversity_client_metric.keys(),diversity_client_metric.values()) if v > avg_metric]



    return neighbours, weights
