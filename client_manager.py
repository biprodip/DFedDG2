from clients.client_fed_avg import *
from clients.client_decood import *
from clients.client_decood_vmf import *
from clients.client_dispfl import *
from clients.client_penz import *
from clients.client_sparse_con import *
from clients.client_fed_proto import *



def get_clients(algorithm, id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test=None, user_data_idx_test=None, cluster=None):
     #print(f'Algorithm: {algorithm}')

    if algorithm in ['FedAvg','Gossip']:
          tmp_client = ClientFedAvg(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)
    elif algorithm == 'DecoodVMF':
           tmp_client = ClientDecoodVMF(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)
    elif algorithm == 'SparseCon':
           tmp_client = ClientSparseCon(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)
    else:
          print('No algorithm matched for client creation.')
    return tmp_client
