"""topology_manager.py — Graph topology generation and adjacency-matrix caching.

Provides helpers for generating weighted mixing matrices for ring, sparse, and
fully-connected (fc) graph topologies used in decentralised FL, plus a weighted
update function for data-heterogeneity-aware adjacency weights.

Key functions
-------------
generate_asymmetric_topology      : Watts-Strogatz ring + random extra links,
                                    row-normalised (asymmetric weights).
generate_symmetric_ring_topology  : Pure ring (k=2 Watts-Strogatz), uniform
                                    row weights.
generate_symmetric_topology       : Ring + extra symmetric links, uniform
                                    row weights.
topology_manager                  : Load cached or create and cache a mixing
                                    matrix for the requested topology.
sinkhorn_normalization            : Iterative doubly-stochastic balancing
                                    (numpy version).
update_adjacency_matrix           : Re-weight edges by class-count overlap
                                    and Sinkhorn-normalise.
"""

import os
import logging
import pickle
import networkx as nx
import numpy as np
from comm_utils.mst import *

LOGGER = logging.getLogger(__name__)


def generate_asymmetric_topology(undirected_neighbor_num, num_clients):
    """Generate an asymmetrically-weighted adjacency matrix over a sparse graph.

    Construction steps:
      1. Start from a symmetric Watts-Strogatz ring with ``k`` neighbours.
      2. Overlay additional random links (also Watts-Strogatz) for density.
      3. Add a base ring topology to guarantee connectivity.
      4. Randomly drop some directed links to create asymmetry.
      5. Row-normalise so each row sums to 1 (row-stochastic, not doubly so).

    Args:
        undirected_neighbor_num: Target out-degree ``k`` for the Watts-Strogatz
            random link overlay.
        num_clients: Number of federated nodes (graph vertices).

    Returns:
        np.ndarray: [N × N] row-normalised (asymmetric) adjacency matrix.
    """
    n = num_clients
    # randomly add some links for each node (symmetric)
    k = undirected_neighbor_num
    # print("neighbors = " + str(k))
    #topology_random_link = np.array(nx.to_numpy_matrix(nx.watts_strogatz_graph(n, k, 0)), dtype=np.float32)
    topology_random_link = nx.to_numpy_array(nx.watts_strogatz_graph(n, k, 0))

    # print("randomly add some links for each node (symmetric): ")
    LOGGER.debug("Random topology: %s", topology_random_link)

    # first generate a ring topology
    topology_ring = nx.to_numpy_array(nx.watts_strogatz_graph(n, 2, 0))

    for i in range(n):
        for j in range(n):
            if topology_ring[i][j] == 0 and topology_random_link[i][j] == 1:
                topology_ring[i][j] = topology_random_link[i][j]

    np.fill_diagonal(topology_ring, 1)

    # k_d = self.out_directed_neighbor
    # Directed graph
    # Undirected graph
    # randomly delete some links
    out_link_set = set()
    for i in range(n):
        len_row_zero = 0
        for j in range(n):
            if topology_ring[i][j] == 0:
                len_row_zero += 1
        random_selection = np.random.randint(2, size=len_row_zero)
        # print(random_selection)
        index_of_zero = 0
        for j in range(n):
            out_link = j * n + i
            if topology_ring[i][j] == 0:
                if random_selection[index_of_zero] == 1 and out_link not in out_link_set:
                    topology_ring[i][j] = 1
                    out_link_set.add(i * n + j)
                index_of_zero += 1

    # print("asymmetric topology:")
    # print(topology_ring)

    for i in range(n):
        row_len_i = 0
        for j in range(n):
            if topology_ring[i][j] == 1:
                row_len_i += 1
        topology_ring[i] = topology_ring[i] / row_len_i

    # print("weighted asymmetric confusion matrix:")
    # print(topology_ring)
    return topology_ring




def generate_symmetric_ring_topology(num_clients):
    """Generate a doubly-stochastic adjacency matrix for a symmetric ring graph.

    Uses NetworkX's Watts-Strogatz graph with k=2 (each node connected to its
    two nearest neighbours) and p=0 (no random rewiring), which produces a pure
    ring.  The resulting binary adjacency matrix is then row-normalised so each
    row sums to 1 (equal weights for all neighbours, making it doubly stochastic
    for a regular ring).

    Args:
        num_clients: Number of federated nodes (ring size).

    Returns:
        np.ndarray: [N × N] symmetric, doubly-stochastic mixing matrix.
    """
    n = num_clients
    # first generate a ring topology
    # ring by connecting only to 2 neighbors
    topology_ring = nx.to_numpy_array(nx.watts_strogatz_graph(n, 2, 0))
    #print(topology_ring)


    # # generate symmetric topology
    topology_symmetric = topology_ring.copy()

    #make it doubly stochastic
    for i in range(n):
        row_len_i = 0
        for j in range(n):
            if topology_symmetric[i][j] == 1:
                row_len_i += 1
        topology_symmetric[i] = topology_symmetric[i] / row_len_i
    # print("weighted symmetric confusion matrix:")

    #print(f'topology_symmetric_ring :\n {topology_symmetric}')

    return topology_symmetric



def generate_symmetric_topology(neighbor_num, num_clients):
    """Generate a symmetric (but not necessarily doubly-stochastic) mixing matrix.

    Construction:
      1. Start from a pure ring (Watts-Strogatz, k=2, p=0).
      2. Overlay additional links from a denser Watts-Strogatz graph
         (k=``neighbor_num``).  Diagonal is set to 1 (self-loops).
      3. Row-normalise uniformly so each row sums to 1.

    When ``neighbor_num == num_clients - 1`` every pair of clients is connected
    (fully-connected topology).

    Args:
        neighbor_num: Target degree for the extra Watts-Strogatz links.
            Set to ``num_clients`` for a fully-connected graph.
        num_clients: Number of federated nodes.

    Returns:
        np.ndarray: [N × N] row-normalised symmetric mixing matrix.
    """
    n = num_clients
    # first generate a ring topology
    # ring by connecting only to 2 neighbors
    topology_ring = nx.to_numpy_array(nx.watts_strogatz_graph(n, 2, 0))
    # print(topology_ring)

    # randomly add some links for each node (symmetric)
    k = int(neighbor_num)
    topology_random_link = nx.to_numpy_array(nx.watts_strogatz_graph(n, k, 0))
    # print(f"Random link (symmetric): {topology_random_link}")

    # generate symmetric topology
    topology_symmetric = topology_ring.copy()
    for i in range(n):
        for j in range(n):
            if topology_symmetric[i][j] == 0 and topology_random_link[i][j] == 1:
                topology_symmetric[i][j] = topology_random_link[i][j]
    np.fill_diagonal(topology_symmetric, 1)
    # print(f"Symmetric topology: {topology_symmetric}")

    for i in range(n):
        row_len_i = 0
        for j in range(n):
            if topology_symmetric[i][j] == 1:
                row_len_i += 1
        topology_symmetric[i] = topology_symmetric[i] / row_len_i
    # print(f"Weighted symmetric confusion matrix: {topology_symmetric}")

    return topology_symmetric




def topology_manager(args):
    """Load or generate and cache the mixing matrix for the requested topology.

    Supported topologies (``args.topo``):
      - ``'fc'``     : Fully-connected — every client communicates with all others.
      - ``'ring'``   : Ring — each client connected to its two nearest neighbours.
      - ``'sparse'`` : Sparse Watts-Strogatz — ``args.sparse_neighbors`` per node.

    The mixing matrix and its MST are pickled to
    ``data/<topo>_adj_<num_clients>.pkl`` on first run and loaded from there
    on subsequent runs (avoids re-generating the same graph topology).

    Args:
        args: Namespace with fields:
            ``topo`` (str), ``num_clients`` (int),
            ``sparse_neighbors`` (int, used only for 'sparse' topo).

    Returns:
        np.ndarray: [N × N] mixing matrix for the requested topology.

    Side effects:
        Writes ``data/<topo>_adj_<num_clients>.pkl`` on first call.
    """
    if args.topo == 'fc':
        filename = 'fc_adj_'
    elif args.topo == 'ring':
        filename = 'ring_adj_'
    elif args.topo == 'sparse':
        filename = 'sparse_adj_'


    #load saved adj mat
    if os.path.isfile('data/'+filename+str(args.num_clients)+'.pkl'):
        with open('data/'+filename+str(args.num_clients)+'.pkl', 'rb') as f:
            mixing_mat, mst = pickle.load(f)
            f.close()
            LOGGER.info("Adjacency and MST Loaded from file")
            if(args.topo=='sparse'):
                LOGGER.info("Using nx.watts_strogatz_graph for sparse topology.")
    else:
        if args.topo == 'fc':
            LOGGER.info("Using fc topology")
            mixing_mat = generate_symmetric_topology(args.num_clients, args.num_clients)
        elif args.topo == 'ring':
            LOGGER.info("Using ring topology")
            mixing_mat = generate_symmetric_ring_topology(args.num_clients)
        elif args.topo == 'sparse':
            LOGGER.info("Using nx.watts_strogatz_graph for sparse topology.")
            mixing_mat = generate_symmetric_topology(args.sparse_neighbors, args.num_clients)
        

        #get mst
        mst = get_mst(mixing_mat)
        with open('data/'+filename+str(args.num_clients)+'.pkl', 'wb') as f: 
            pickle.dump([mixing_mat,mst], f)
            f.close()
            LOGGER.info("Adjacency and MST Saved in %s", 'data/'+filename+str(args.num_clients)+'.pkl')
    
    return mixing_mat



def sinkhorn_normalization(matrix, max_iter=100, tol=1e-6):
    """
    Perform Sinkhorn-Knopp normalization to ensure the matrix is doubly stochastic.
    Args:
        matrix: Symmetric matrix to normalize.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
    Returns:
        Doubly stochastic matrix.
    """
    for _ in range(max_iter):
        # Normalize rows
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)
        # Normalize columns
        matrix = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1e-12)
        
        # Check for convergence
        if np.allclose(matrix.sum(axis=1), 1, atol=tol) and np.allclose(matrix.sum(axis=0), 1, atol=tol):
            break
    return matrix

    

def update_adjacency_matrix(adj, clients, num_classes, alpha=None):
    """
    Update the adjacency matrix with weights based on local data.
    Args:
        adj: Original adjacency matrix (0 for no connection, 1 for connection).
        clients: List of clients, each with a sample_per_class attribute.
        num_classes: Total number of classes.
        alpha: Class importance weights (optional).
    Returns:
        Updated doubly stochastic adjacency matrix.
    """
    num_clients = len(adj)
    LOGGER.info("Total clients: %s", num_clients)
    
    if alpha is None:
        alpha = np.ones(num_classes)  # Equal weight for all classes
    
    # Compute pairwise weights
    for i in range(num_clients):
        for j in range(num_clients):
            if adj[i][j] == 0 or i == j:
                continue  # Skip if no connection or self-loop
            
            weight_ij = 0
            for c in range(num_classes):
                N_i_c = clients[i].sample_per_class[c]
                N_j_c = clients[j].sample_per_class[c]
                if N_i_c + N_j_c > 0:
                    weight_ij += alpha[c] * (N_j_c / (N_i_c + N_j_c))
            
            adj[i][j] = weight_ij
            adj[j][i] = weight_ij  # Ensure symmetry

    # Normalize rows and columns to make the matrix doubly stochastic
    adj = sinkhorn_normalization(adj)

    LOGGER.debug("Adjacency matrix after normalization:\n%s", adj)
    return adj
