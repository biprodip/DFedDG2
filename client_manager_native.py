#from client_fed_grad_proto import *
from clients.client_fed_avg import *
from clients.client_decood_native import *
# from clients.client_VMF import *
from clients.client_dispfl import *
from clients.client_penz import *


def get_clients(algorithm, id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test=None, user_data_idx_test=None, cluster=None):
     #print(f'Algorithm: {algorithm}')

    if algorithm in ['FedAvg','Gossip']:
          tmp_client = ClientFedAvg(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)
    elif algorithm == 'Decood':
           tmp_client = ClientDecood(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster) 
    elif algorithm == 'DecoodAvg':
           tmp_client = ClientDecoodAvg(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)  
    elif algorithm == 'DisPFL':
           tmp_client = ClientDisPFL(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)  
    elif algorithm == 'Penz':
           tmp_client = ClientPenz(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)  
#     elif algorithm == 'vmf':
#            tmp_client = ClientVMF(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)  

    else:
          print('No algorithm matched for client creation.')
    return tmp_client
