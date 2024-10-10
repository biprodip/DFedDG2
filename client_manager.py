#from client_fed_grad_proto import *
from clients.client_fed_avg import *
from clients.client_ditto import *
from clients.client_fed_proto import *
from clients.client_fed_GH import *
from clients.client_fed_GH_ood import *
from clients.client_fed_GH_proto import *
from clients.client_decood import *
from clients.client_decood_avg import *
# from clients.client_VMF import *
from clients.client_dispfl import *
from clients.client_penz import *


def get_clients(algorithm, id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test=None, user_data_idx_test=None, cluster=None):
     #print(f'Algorithm: {algorithm}')

    if algorithm == 'FedAvg':
          tmp_client = ClientFedAvg(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)
    elif algorithm == 'Ditto':
           tmp_client = ClientDitto(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)
    elif algorithm == 'FedRod':
           tmp_client = ClientFedRod(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)
    elif algorithm == 'FedPens':
           tmp_client = ClientFedPens(id, args, adj_mat, dataset_train, user_data_idx, cluster)
    elif algorithm == 'FedProto':
           tmp_client = ClientFedProto(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster) 
    elif algorithm == 'FedGradProto':
           tmp_client = ClientFedGradProto(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster) 
    elif algorithm == 'FedGH':
           tmp_client = ClientFedGH(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster) 
    elif algorithm == 'FedGHOOD':
           tmp_client = ClientFedGHOOD(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster) 
    elif algorithm == 'FedGHProto':
           tmp_client = ClientFedGHProto(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster) 
    elif algorithm == 'FedCon':
           tmp_client = ClientFedCon(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster) 
    elif algorithm == 'FedConCIDER':
           tmp_client = ClientFedConCIDER(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster) 
    elif algorithm == 'Decood':
           tmp_client = ClientDecood(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster) 
    elif algorithm == 'DecoodAvg':
           tmp_client = ClientDecoodAvg(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)  
    elif algorithm == 'DisPFL':
           tmp_client = ClientDisPFL(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)  
    elif algorithm == 'penz':
           tmp_client = ClientPenz(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)  
#     elif algorithm == 'vmf':
#            tmp_client = ClientVMF(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)  

    else:
          print('No algorithm matched for client creation.')
    return tmp_client
