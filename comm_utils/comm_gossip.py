from utils.utils_proto import *
import pickle

def comm_gossip(args, adj, clients, debug=False, test_loader=None):
    '''
    For a client i, Aggregates(FedAvg) all clients j if j is adjacent to i and (i!=j)
    based on the adjacency matrics adj. 

    Not random gossip (where adjacents are randomly selected).
    '''
        
    print('Gossip training.')
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
    

    for e in range(args.num_rounds):
        
        if args.check_running:
            if e%5 == 0:
                user_in = input(f"0/1 break/cont: ")
                if user_in == '0':
                    print('Exit learning.')
                    break #model will be saved


        print(f'Round : {e}')
        # Update each client
    
        for m in clients:
            print(f'Updating {m.id}')
            if args.algorithm == 'Ditto':
              m.ptrain()
            m.update()
            print(f'Performance: {m.performance_test()}\n')



        req_aggregation = [True for i in range(len(clients))]
        # Set up variables where we do the averaging calculation without disturbing the previous weights
        tot_train_data = 0 #############
        new_models = []
        Ni = [1 for i in range(len(clients))] #selected neighbor length of client i every iteration(we devide by this ncorresponding umber during averaging)
        
        #copy self models
        for i in range(len(clients)):
          new_models.append(copy.deepcopy(clients[i].model.state_dict()))   
          
         
        # Start averaging towards the goal. Here we use equal neighbor averaging method.
        for i in range(len(clients)):
          
          # Select clients to train and participate in averaging
          adj_clients = [clients[c] for c in range(len(clients)) if (adj[c][i] and c!=i)]
          
          # Random selection
          # print('FAvg aggregation')
          # V_set = set(range(len(adj_clients)))
          # m = int(args.sub_mod_sel_ratio * len(adj_clients)) #select .7 of neighbors
          # sel_neighbor_indx = np.random.choice(list(V_set), m, replace=False)
          # Ni[i]=len(sel_neighbor_indx)+1  #including self
                  
          
          #sel_neighbors = [adj_clients[j] for j in sel_neighbor_indx] 
          sel_neighbors = adj_clients

              
          #id -id communication count
          s_c = [c.id for c in sel_neighbors]
          # for peer in s_c:
          #     client_heat[clients[i].id][peer]+=1  #count communication
          #     client_heat[peer][clients[i].id]+=1W0 = [tens.detach() for tens in list(self.model0.parameters())] #weight now
        # Wt = [tens.detach() for tens in list(self.model.parameters())] #previous weight
        
            
          N = 0
          for sc in sel_neighbors:
            N = N + len(sc.train_loader.dataset) 
          #self
          N = N + len(clients[i].train_loader.dataset)

          #self weight
          W = len(clients[i].train_loader.dataset)/N
          for key in clients[i].model.state_dict().keys():              
            new_models[i][key] = W * clients[i].model.state_dict()[key]

          
          for sc in sel_neighbors:
            # Record each key's value, while also keeping track of distance
            
            W = len(sc.train_loader.dataset)/N
            for key in sc.model.state_dict().keys():  
              new_models[i][key] += W * sc.model.state_dict()[key]

            # for key in sc.model.state_dict().keys():              
            #   new_models[i][key] += sc.model.state_dict()[key]    
            
        
        
            
        # average and update self
        for i in range(len(clients)):
                #clients[i].avg_model(new_models[i], Ni[i]) #N[i]: tot neighbors of client i each round
                clients[i].avg_model(new_models[i], 1) #N[i]: tot neighbors of client i each round
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

    
    # for c in clients:
    #     #save client_models
    #     filename = args.params_dir + 'checkpoint_{}_{}_client_{}.pth.tar'.format(args.algorithm, args.dataset, c.id)
    #     torch.save({
    #         'model_state_dict': c.model.state_dict(),
    #         'optimizer_state_dict': c.optimizer.state_dict(),
    #         }, filename)
    #     print('Checkpoint file: ',filename)

        
    #     filename = args.params_dir + 'id_loaders_{}_{}_client_{}.pkl'.format(args.algorithm, args.dataset, c.id)
    #     with open(filename, 'wb') as f:
    #         if args.algorithm in ['FedAvg','Ditto']:
    #             pickle.dump([c.train_loader, c.test_loader, c.id_labels], f)
    #         else:
    #             pickle.dump([c.train_loader, c.test_loader, c.id_labels, c.local_protos, c.global_protos], f)
    #         print('Loaders file: ',filename)
    #         f.close()

    # print('Clients saved in file: ',filename)


  
    return avg_l_acc #, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc #client_heat 