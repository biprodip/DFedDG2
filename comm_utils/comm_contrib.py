import math
from utils import *
from eval_submod import *

def calculate_threshold_exponential(current_iteration, initial_threshold = 0.5, decay_factor=0.80, final_threshold = .3):
    current_threshold = initial_threshold * math.pow(decay_factor, current_iteration)
    current_threshold = max(current_threshold, final_threshold)
    current_threshold = 1- current_threshold
    return current_threshold


def comm_contrib(args, adj, clients, debug=False, test_loader=None):
    '''
    For a client i, Aggregates(FedAvg) all clients j if j is adjacent to i and (i!=j)
    based on the adjacency matrics adj. 

    Not random gossip (where adjacents are randomly selected).
    '''
        
    print('Pens training.')
    # torch.manual_seed(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    # torch.autograd.set_detect_anomaly(True)
    
    
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
       c.performance_train(c.val_loader)
    

    for e in range(args.num_rounds):
        
        print(f'Round : {e}')
        # Update each client
    
        for m in clients:
            print(f'Updating {m.id}')
            m.update()
            m.performance_train(m.val_loader) #store unc in self.unc

        if reached_consensus:
          break
        

        req_aggregation = [True for i in range(len(clients))]
        # Set up variables where we do the averaging calculation without disturbing the previous weights
        tot_train_data = 0 #############
        new_models = []
        Ni = [1 for i in range(len(clients))] #selected true neighbor length of client i every iteration(we devide by this ncorresponding umber during averaging)
        #copy self models
        for i in range(len(clients)):
         new_models.append(copy.deepcopy(clients[i].model.state_dict()))  ##copying all client models (adjacent?) 
          
        
        #args.grad_thres = calculate_threshold_exponential(e, args.grad_thres)
        print(f'Current gradient similarity threshold:{args.grad_thres}')

        # Start averaging towards the goal. Here we use equal neighbor averaging method.
        for i in range(len(clients)):
          
          # Select clients to train and participate in averaging
          adj_clients = [clients[c] for c in range(len(clients)) if (adj[c][i] and c!=i)]
          neighbors, _ = clients_to_communicate_with(args, clients[i], adj_clients)
          
          print(f'{len(neighbors)} neighbors selected from {len(adj_clients)} peers.')
          if len(neighbors)<=0:
              print(f'No neighbor for client {i}.')
              req_aggregation[i] = False
              continue  #process for next client (this client should continue local learning)
          
          
          weights = cont_weights(args, clients[i] ,neighbors)
              
          #id -id communication count
          if(len(neighbors)>0):
            s_c = [c.id for c in neighbors]
            for peer in s_c:
              client_heat[clients[i].id][peer]+=1  #count communication
              client_heat[peer][clients[i].id]+=1
            
          #Ni[i]=len(neighbors)+1 #including self req if equal avg
          
          
          if req_aggregation[i]:
            for key in clients[i].model.state_dict().keys():
              new_models[i][key] = weights[-1] * clients[i].model.state_dict()[key] #self portion
            
            for ind, sc in enumerate(neighbors):
              for key in sc.model.state_dict().keys():              
              #if args.submod:
                    #weighted sum
                    new_models[i][key] += weights[ind] * sc.model.state_dict()[key]
              
              #else:
              #  new_models[i][key] += sc.model.state_dict()[key]

        # for ind, sc in enumerate(neighbors):
        #     for key in sc.model.state_dict().keys():              
        #       #weighted sum
        #       #new_models[i][key] += clients[j].model.state_dict()[key]*len(clients[j].train_loader.dataset)  #############nnnnnnnnn
        #       #if args.submod:
        #         if req_aggregation[i]:
        #             new_models[i][key] += weights[ind] * sc.model.state_dict()[key]
        #       #else:
        #       #  new_models[i][key] += sc.model.state_dict()[key]



        # for i in range(len(clients)):
        #     if req_aggregation[i]:
        #         # if not args.submod:
        #         #     clients[i].avg_model(new_models[i], Ni[i]) #N[i]: tot neighbors of client i each round
        #         #     print(f'Aggregated client FedAvg {i}')
        #         # else:
        #             clients[i].avg_model(new_models[i], 2)  #self model and another model that is weighted sum of all weights
        #             print(f'Aggregated client on contribution {i}.')
        #     else:
        #         print(f'Skipped aggregation for client {i}')

        
        for i in range(len(clients)):
            if req_aggregation[i]:
                clients[i].load_model(new_models[i]) #N[i]: tot neighbors of client i each round
                print(f'Loaded aggregated model {i}')
                

       #list every round avg performance of all clients (local test data and global test data)
        lacc, gacc, lauc, gauc, lunc, gunc = evaluate_clients(clients, test_loader)
        avg_l_acc.append(lacc)
        avg_g_acc.append(gacc)
        avg_l_auc.append(lauc)
        avg_g_auc.append(gauc)
        avg_l_unc.append(lunc)
        avg_g_unc.append(gunc)


        
        # Normalize the matrix
        client_heat = (client_heat - np.min(client_heat)) / (np.max(client_heat) - np.min(client_heat))
        # Set diagonal elements to zero
        np.fill_diagonal(client_heat, 0)

  
    return avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc, client_heat 