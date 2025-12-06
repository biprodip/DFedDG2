import os 
import pickle 
import networkx as nx
import numpy as np
from comm_utils.mst import *
import pickle


def generate_asymmetric_topology(undirected_neighbor_num, num_clients):
        
        n = num_clients
        # randomly add some links for each node (symmetric)
        k = undirected_neighbor_num
        # print("neighbors = " + str(k))
        #topology_random_link = np.array(nx.to_numpy_matrix(nx.watts_strogatz_graph(n, k, 0)), dtype=np.float32)
        topology_random_link = nx.to_numpy_array(nx.watts_strogatz_graph(n, k, 0))

        # print("randomly add some links for each node (symmetric): ")
        print(f'Random topology:{topology_random_link}')

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
        '''
        neighbor_num : adjacent node count of every node in the graph
        num_clients : total num of clients
        '''
        
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
        '''
        neighbor_num : adjacent node count of every node in the graph 
                       (if neighbor_num == (num_clients - 1) then fc)
        num_clients : total num of clients 
        '''
        
        
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
            print('Adjacency and MST Loaded from file')
            if(args.topo=='sparse'):
                print('Using nx.watts_strogatz_graph for sparse topology.')
    else:
        if args.topo == 'fc':
            print('Using fc topology')
            mixing_mat = generate_symmetric_topology(args.num_clients, args.num_clients)    
        elif args.topo == 'ring':
            print('Using ring topology')
            mixing_mat = generate_symmetric_ring_topology(args.num_clients)
        elif args.topo == 'sparse':
            print('Using nx.watts_strogatz_graph for sparse topology.')
            mixing_mat = generate_symmetric_topology(args.sparse_neighbors, args.num_clients)
        

        #get mst
        mst = get_mst(mixing_mat)
        with open('data/'+filename+str(args.num_clients)+'.pkl', 'wb') as f: 
            pickle.dump([mixing_mat,mst], f)
            f.close()
            print('Adjacency and MST Saved in'+'data/'+filename+str(args.num_clients)+'.pkl')

        # print(adj_mat)
    
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
    print(f'Total clients: {num_clients}')
    
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

    print(f'Adjacency matrix after normalization:\n{adj}')
    return adj
