from utils.utils_proto import *
import pickle



# Function to count total parameters in a model
def count_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# Function to count non-zero parameters in a model given the mask
def count_nonzero_params(model, mask):
    non_zero_params = 0
    for name, param in model.named_parameters():
        if name in mask:
            non_zero_params += torch.sum(mask[name] > 0).item()
    return non_zero_params



def comm_dis_pfl(args, adj, clients, debug=False, test_loader=None):
    '''
    For a client i, Aggregates(FedAvg) all clients j if j is adjacent to i and (i!=j)
    based on the adjacency matrics adj. 

    Not random gossip (where adjacents are randomly selected).
    '''
        
    print('DisPFL training.')
    # torch.manual_seed(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)

    
    reached_consensus = False
    avg_l_acc = []
    avg_g_acc = []
    avg_l_auc = []
    avg_g_auc = []
    avg_l_unc = []
    avg_g_unc = []

    #cluster_heat = np.zeros([args.num_clusters,args.num_clusters])
    client_heat = np.zeros([args.num_clients,args.num_clients])
    
    def update_clients(c):
       print(f"Updating client {c.id}")
       c.update()
    

    for e in range(args.num_rounds):
        
        print(f'Round : {e}')

        # Update each client
        for m in clients:
            print(f'Updating {m.id}')
            m.update()
            print(f'Performance: {m.performance_test()}\n')


        req_aggregation = [True for i in range(len(clients))]
        # Set up variables where we do the averaging calculation without disturbing the previous weights
        for i in range(len(clients)):
            client_updates = []
            client_masks = []
                   
            # Select clients to train and participate in averaging
            adj_clients = [clients[c] for c in range(len(clients)) if (adj[c][i] and c!=i)]
          
            #sel_neighbors = [adj_clients[j] for j in sel_neighbor_indx] 
            sel_neighbors = adj_clients

            for sc in sel_neighbors:
                model_update, mask = sc.get_model_and_mask()
                client_updates.append(model_update)
                client_masks.append(mask)
            
            #append self model and mask
            model_update, mask = clients[i].get_model_and_mask()
            client_updates.append(model_update)
            client_masks.append(mask)

            clients[i].aggregate_models(client_updates, client_masks) #compute aggregated model but load later(synchronous update)
            aggregated_mask = client_masks[0]  # Assume all clients have the same mask initially


        # average and update self
        for i in range(len(clients)):
                clients[i].set_model_and_mask() #load aggregated model and mask (saved locally)
                print(f'Aggregated client {i}')
                
       
       #list every round avg performance of all clients (local test data and global test data)
       #lacc, gacc, lauc, gauc, lunc, gunc = evaluate_clients(clients, test_loader)
        lacc = evaluate_clients(clients, test_loader)
        print(f'Avg ACC:{lacc}')
        avg_l_acc.append(lacc)
        # avg_g_acc.append(gacc)
        # avg_l_auc.append(lauc)
        # avg_g_auc.append(gauc)
        # avg_l_unc.append(lunc)
        # avg_g_unc.append(gunc)


        
        # Normalize the matrix
        #client_heat = (client_heat - np.min(client_heat)) / (np.max(client_heat) - np.min(client_heat))
        # Set diagonal elements to zero
        np.fill_diagonal(client_heat, 0)
  
    return avg_l_acc #, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc #client_heat 