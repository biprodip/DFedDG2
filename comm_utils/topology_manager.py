import os 
import pickle 
import networkx as nx
import numpy as np



def generate_asymmetric_topology(undirected_neighbor_num, num_clients):
        
        n = num_clients
        # randomly add some links for each node (symmetric)
        k = undirected_neighbor_num
        # print("neighbors = " + str(k))
        #topology_random_link = np.array(nx.to_numpy_matrix(nx.watts_strogatz_graph(n, k, 0)), dtype=np.float32)
        topology_random_link = nx.to_numpy_array(nx.watts_strogatz_graph(n, k, 0))

        # print("randomly add some links for each node (symmetric): ")
        print(topology_random_link)

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
        print(topology_ring)
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
    if os.path.isfile(os.path.join(args.save_folder_name,'../data/'+filename+str(args.num_clients)+'.pkl')):
        with open(os.path.join(args.save_folder_name,'../data/'+filename+str(args.num_clients)+'.pkl'), 'rb') as f:
            ADJ, MST = pickle.load(f)
            f.close()
            print('Adjacency and MST Loaded from file')
    else:
        if args.topo == 'fc':
            ADJ = generate_symmetric_topology(args.num_clients, args.num_clients)    
        elif args.topo == 'ring':
            ADJ = generate_symmetric_ring_topology(args.num_clients)
        elif args.topo == 'sparse':
            ADJ = generate_symmetric_topology(args.sparse_neighbors, args.num_clients)
        

        #get mst
        if args.mst_comm :
            MST = get_mst(ADJ)
            with open('../data/'+filename+str(args.num_clients)+'.pkl', 'wb') as f: 
                pickle.dump([ADJ,MST], f)
                f.close()
                print('Adjacency and MST Saved in'+'../data/'+filename+str(args.num_clients)+'.pkl')

        # print(ADJ)
    
    return MST if args.mst_comm else ADJ