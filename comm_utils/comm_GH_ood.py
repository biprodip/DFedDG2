from utils.utils_proto import *
#from eval_submod import *
from collections import defaultdict
import torch.nn.functional as F


def receive_protos(client, selected_clients):
        #append labels to protos
        assert (len(selected_clients) > 0)
        selected_clients.append(client)

        uploaded_ids = []
        uploaded_protos_id = []
        uploaded_protos_ood = []

        for sc in selected_clients:
            uploaded_ids.append(sc.id)
            for cc in sc.local_protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=client.device)
                if cc in client.id_labels:
                    #print('ID protos received')
                    uploaded_protos_id.append((sc.local_protos[cc], y))
                else:
                    #print('OOD protos received')
                    uploaded_protos_ood.append((sc.local_protos[cc], y))  

        print(f'Tot ID protos {len(uploaded_protos_id)} Tot OOD protos {len(uploaded_protos_ood)}')  
        return uploaded_protos_id, uploaded_protos_ood 




def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            





def comm_GH_ood(args, adj, clients, debug=False, test_loader=None):
    '''
    For a client i, Aggregates(FedAvg) all clients j if j is adjacent to i and (i!=j)
    based on the adjacency matrics adj. 

    Not random gossip (where adjacents are randomly selected).
    '''
        
    print('Comm GH OOD training.')
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
    #client_heat = np.zeros([args.num_clients,args.num_clients])
    

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
            m.update()
            m.collect_protos()

        

        req_aggregation = [True for i in range(len(clients))]
        # Set up variables where we do the averaging calculation without disturbing the previous weights
        tot_train_data = 0 #############
        
        # Start averaging towards the goal. Here we use equal neighbor averaging method.
        
        for i in range(len(clients)):
          
        #receive protos
            if req_aggregation[i]:
                adj_clients = [clients[c] for c in range(len(clients)) if (adj[c][i] and c!=i)]
                
                # neighbors, _ = clients_to_communicate_with(args, clients[i], adj_clients)
                
                # print(type(neighbors))
                neighbors = adj_clients 
                

                print(f'{len(neighbors)} neighbors selected from {len(adj_clients)} peers.')
                if len(neighbors)<=0:
                    print(f'No neighbor for client {i}.')
                    req_aggregation[i] = False
                    continue  #process for next client (this client should continue local learning)


                args.clustered_agg = False
                if args.clustered_agg:
                    rec_neighbors_protos, rec_ood_protos = receive_grouped_protos(clients[i], neighbors, adj_clients) 
                    global_protos_label =  proto_aggregation_clustered(clients[i], args.num_classes, rec_neighbors_protos, rec_ood_protos)  #####************    ID from neighbors, ood from othersclient
                    #print(f'Global protos : {global_protos_label}') # We have aonly [4 2 1 5 9] protos
                else:
                    received_protos_id, received_protos_ood = receive_protos(clients[i], adj_clients) #neighbors adj_clients
                    #clients[i].uploaded_protos = copy.deepcopy(received_protos)
                    if len(received_protos_ood) >0:
                        clients[i].train_head_ood(received_protos_id, received_protos_ood, args.ood_train_method) 
                        print('OOD head trained')
                        clients[i].set_parameters(copy.deepcopy(clients[i].gh_head))
                        print('Client head updated with ood aware head') 
                    else:
                        clients[i].train_head_ood(received_protos_id)
                        print('No ood protos, so only id head trained on received id protos!!!!')
                
                #print(clients[i].model.head.parameters())
                #print(clients[i].gh_head.parameters())

                
                print(f'Aggregated for client {i}')


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
        # client_heat = (client_heat - np.min(client_heat)) / (np.max(client_heat) - np.min(client_heat))
        # # Set diagonal elements to zero
        # np.fill_diagonal(client_heat, 0)

  
    #return avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc, client_heat 
    return avg_l_acc