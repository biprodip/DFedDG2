"""client_manager.py — Client factory for DFedDG2 federated algorithms.

Maps algorithm names to the corresponding client class and instantiates a
client with the provided dataset partitions and adjacency information.

Supported algorithms
--------------------
'FedAvg' / 'Gossip' : Standard FedAvg client (model-weight gossip baseline).
'DecoodVMF'         : vMF gossip client with prototype-based communication.
'SparseCon'         : Sparse contrastive learning client.
"""

from clients.client_fed_avg import *
from clients.client_decood import *
from clients.client_decood_vmf import *
from clients.client_dispfl import *
from clients.client_penz import *
from clients.client_sparse_con import *
from clients.client_fed_proto import *


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
        cluster: Cluster assignment for this client (used in clustered FL
            variants); ``None`` if not applicable.

    Returns:
        A client object of the appropriate class for the given ``algorithm``.

    Prints a warning and leaves ``tmp_client`` undefined if ``algorithm`` is
    not recognised.
    """
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
