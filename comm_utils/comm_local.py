import copy
import torch
import numpy as np
from numpy import random 
from multiprocessing import pool
from utils.utils import evaluate_clients
from utils.utils_proto import *



def comm_local(args, adj, clients, debug=False, test_loader=None):
    '''
    Local training of every client
    No aggregation
    '''
    
    print('Local training.')
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    
    
    avg_l_acc = []
    avg_g_acc = []
    avg_l_auc = []
    avg_g_auc = []
    avg_l_unc = []
    avg_g_unc = []

    
    for e in range(args.num_rounds):
        if args.check_running:
            if e%5 == 0:
                user_in = input(f"0/1 break/cont: ")
                if user_in == '0':
                    print('Exit learning.')
                    break #model will be saved

        
        print(f'Round : {e}')
        # Update each client   
        
        for m in clients:
            print(f'Updating client {m.id}')
            m.update()

       #list every round avg performance
        lacc = evaluate_clients(clients, test_loader)
        print(f'Avg ACC:{lacc}')
        avg_l_acc.append(lacc)
        # avg_g_acc.append(gacc)
        # avg_l_auc.append(lauc)
        # avg_g_auc.append(gauc)
        # avg_l_unc.append(lunc)
        # avg_g_unc.append(gunc)

  
    return avg_l_acc #, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc

