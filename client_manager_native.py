"""client_manager_native.py — Client factory for native (non-vMF) algorithms.

Mirrors client_manager.py but supports a different set of client classes used
in earlier or baseline experiments.  The 'native' suffix distinguishes this
module from the vMF-gossip variant.

Supported algorithms
--------------------
'FedAvg' / 'Gossip' : Standard FedAvg model-weight gossip baseline.
'Decood'            : Prototype-only gossip (native DECOOD client).
'DecoodAvg'         : DECOOD variant with model averaging.
'DisPFL'            : DisPFL sparse personalised FL client.
'Penz'              : PENZ loss-based neighbour selection client.
"""

#from client_fed_grad_proto import *
from clients.client_fed_avg import *
from clients.client_decood_native import *
# from clients.client_VMF import *
from clients.client_dispfl import *
from clients.client_penz import *


def get_clients(algorithm, id, args, adj_mat, dataset_train, user_data_idx_train,
                dataset_test=None, user_data_idx_test=None, cluster=None):
    """Factory function: instantiate and return a single federated client.

    Args:
        algorithm: String identifier for the FL algorithm (see module docstring
            for supported values).
        id: Integer client ID (0-indexed).
        args: Global experiment configuration namespace.
        adj_mat: [N × N] adjacency matrix indicating graph neighbours.
        dataset_train: Full training dataset (shared; client uses subset via
            ``user_data_idx_train``).
        user_data_idx_train: Array of indices into ``dataset_train`` assigned to
            this client.
        dataset_test: Full test dataset (optional).
        user_data_idx_test: Array of indices into ``dataset_test`` for this
            client (optional).
        cluster: Cluster assignment for this client; ``None`` if not applicable.

    Returns:
        A client object of the appropriate class for the given ``algorithm``.

    Prints a warning and leaves ``tmp_client`` undefined if ``algorithm`` is
    not recognised.
    """
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
