import cvxpy as cp
import networkx as nx
import numpy as np
import os
from comm_utils.mst import *
import pickle




def get_communication_graph(n, p, seed):
    return nx.generators.random_graphs.binomial_graph(n=n, p=p, seed=seed)


def compute_mixing_matrix(adjacency_matrix):
    """
    computes the mixing matrix associated to a graph defined by its `adjacency_matrix` using
    FMMC (Fast Mixin Markov Chain), see https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf

    :param adjacency_matrix: np.array()
    :return: optimal mixing matrix as np.array()
    """
    network_mask = 1 - adjacency_matrix
    N = adjacency_matrix.shape[0]

    s = cp.Variable()
    W = cp.Variable((N, N))
    objective = cp.Minimize(s)

    constraints = [
        W == W.T,
        W @ np.ones((N, 1)) == np.ones((N, 1)),
        cp.multiply(W, network_mask) == np.zeros((N, N)),
        -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
        W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N),
        np.zeros((N, N)) <= W
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if W.value is None:
        raise ValueError("Optimization problem did not converge.")


    mixing_matrix = W.value

    mixing_matrix *= adjacency_matrix
    mixing_matrix = np.multiply(mixing_matrix, mixing_matrix >= 0)


    # Force symmetry (for numerical stability)
    for i in range(N):
        if np.abs(np.sum(mixing_matrix[i, i:])) >= 1e-20:
            mixing_matrix[i, i:] *= (1 - np.sum(mixing_matrix[i, :i])) / np.sum(mixing_matrix[i, i:])
            mixing_matrix[i:, i] = mixing_matrix[i, i:]

    return mixing_matrix



def get_mixing_matrix(args, n, p, seed):
    
    if n==4:
        p = 0.6 #for FNLI setup only; Default: p is proportional to sampling rate in server based FL
        seed = 7
    else:
        seed = 3 #converges for cvx problem with this value, for more than 4 clients
    
    filename = 'sparse_adj_'

    # Construct the relative file path
    relative_path = 'data/' + filename + str(args.num_clients) + '.pkl'

    # Convert to absolute path
    absolute_path = os.path.abspath(relative_path) 
    
    #load saved adj mat
    if os.path.isfile(relative_path):
        print(f'Found {absolute_path} adjacency matrix file.')
        with open(relative_path, 'rb') as f:
            mixing_mat, mst = pickle.load(f)
            f.close()
            print('Adjacency and MST Loaded from file.')
            print('Loaded Erdos Renoyi graph for sparse topology.')
    else:
        print(f'Adjacency matrix {absolute_path} file not found.')
        #create mixing matrix
        graph = get_communication_graph(n, p, seed)
        adj_mat = nx.adjacency_matrix(graph, weight=None).todense()
        mixing_mat = compute_mixing_matrix(adj_mat)
        print('Created Erdos Renoyi graph for sparse topology.')
        print(f'Adjacency_matrix : {adj_mat}')
        
        #get mst
        mst = get_mst(adj_mat)
        with open(relative_path, 'wb') as f: 
                pickle.dump([mixing_mat, mst], f)
                f.close()
                print('Adjacency, MST and Mixing Matrix saved in'+'data/'+filename+str(args.num_clients)+'.pkl')
    
    return mixing_mat