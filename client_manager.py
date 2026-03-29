"""client_manager.py — Client factory for DFedDG2 federated algorithms.

Maps algorithm names to the corresponding client class and instantiates a
client with the provided dataset partitions and adjacency information.

Supported algorithms
--------------------
'DecoodVMF'         : vMF gossip client with prototype-based communication.
"""

import logging
from clients.client_decood_vmf import *

LOGGER = logging.getLogger(__name__)

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
        A client object for the given ``algorithm``.

    Raises:
        ValueError: If ``algorithm`` is not recognised.
    """

    if algorithm == 'DecoodVMF':
        return ClientDecoodVMF(id, args, adj_mat, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Supported: 'DecoodVMF'")
