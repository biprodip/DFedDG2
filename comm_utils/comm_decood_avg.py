from utils.utils_proto import *
#from eval_submod import *
from collections import defaultdict
import torch.nn.functional as F



def receive_protos(client,sel_clients):
    assert (len(sel_clients) > 0)

    uploaded_ids = []
    uploaded_protos = []
    for cl in sel_clients:
        #uploaded_ids.append(cl.id)
        uploaded_protos.append(cl.local_protos)
    
    uploaded_protos.append(client.local_protos) #self protos for aggregation
    return uploaded_protos



def receive_grouped_protos(client, neighbors, adjacents):
    assert (len(adjacents) > 0)
    # if all are neighhbors than only neighbor labels are available, ood protos(labels) are not available
    # own proto
    
    n_ids = [c.id for c in neighbors]

    rec_neighbors_protos = []
    rec_ood_protos = []
    
    for cl in adjacents:
        if cl.id in n_ids:
            #print(f"{cl.id} is neighbor")
            rec_neighbors_protos.append(cl.local_protos)  #all id  and ood from neighbors
        else:    
            #print(f"{cl.id} is not neighbor")
            rec_ood_protos.append(cl.local_protos)        #all id  and ood from others
    
    rec_neighbors_protos.append(client.local_protos) #self protos for aggregation
    return rec_neighbors_protos, rec_ood_protos




def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)

    for local_protos in local_protos_list:   #every client protos set 
        for label in local_protos.keys():    #one by one prototype based on labels
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data

            tmp_proto = F.normalize((proto / len(proto_list)),dim=0)  ######*****
            #tmp_proto = proto / len(proto_list)  ######*****
        else:
            tmp_proto = F.normalize(proto_list[0] , dim=0) #proto_list[0].data
            #tmp_proto = proto_list[0] #proto_list[0].data
        agg_protos_label[label] = tmp_proto.detach()

    return agg_protos_label





def proto_aggregation_clustered(client, labels, rec_neighbors_protos, rec_ood_protos):
    agg_protos_label = defaultdict(list)
    protos_received = [False for _ in range(labels)]
            
    #receive from ood clients
    for rec_protos in rec_ood_protos:
        for label in rec_protos.keys():
            if label not in client.id_labels:
                agg_protos_label[label].append(rec_protos[label])
                protos_received[label] = True


    for rec_protos in rec_neighbors_protos:
        for label in rec_protos.keys():
            if label in client.id_labels:  #take id protos from protos_label
                agg_protos_label[label].append(rec_protos[label])
            elif not protos_received[label]: #not received from ood clients
                agg_protos_label[label].append(rec_protos[label])
                #print('OOD protos should be received from ID neighbors, but using self protos(1e-08)!!!')
 

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            # tmp_proto = F.normalize((proto / len(proto_list)),dim=0)  ######*****
            tmp_proto = proto / len(proto_list)  ######*****

            agg_protos_label[label] = tmp_proto.detach()
        else:
            # tmp_proto = F.normalize(proto_list[0] , dim=0) #proto_list[0].data
            tmp_proto = proto_list[0] #proto_list[0].data

            agg_protos_label[label] = tmp_proto.detach()

    return agg_protos_label





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
            



def comm_decood_avg(args, adj, clients, debug=False, test_loader=None):
    '''
    For a client i, Aggregates(FedAvg) all clients j if j is adjacent to i and (i!=j)
    based on the adjacency matrics adj. 

    Not random gossip (where adjacents are randomly selected).
    '''
        
    print('Comm proto grad training.')
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
    
    # def update_clients(c):
    #    print(f"Updating client {c.id}")
    #    c.update()
    #    c.performance_train(c.val_loader)
    

    for e in range(args.num_rounds):

        # if args.check_running:
        #     if e%5 == 0:
        #         user_in = input(f"0/1 break/cont: ")
        #         if user_in == '0':
        #             print('Exit learning.')
        #             break #model will be saved
        
        print(f'Round : {e}')
        # Update each client
    
        for m in clients:
            print(f'Updating {m.id}')
            # if e == 0:
            #     m.init_local_proto()            
            m.update()
            #print('Implement prototype uncertainty approach')
            #m.performance_train(m.val_loader) #compute uncertainty unc in self.unc
        

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
          
        #receive protos
            if req_aggregation[i]:
                adj_clients = [clients[c] for c in range(len(clients)) if (adj[c][i] and c!=i)]
                
                #neighbors, _ = clients_to_communicate_with(args, clients[i], adj_clients)
                
                #print(type(neighbors))
                neighbors = adj_clients 



                print(f'{len(neighbors)} neighbors selected from {len(adj_clients)} peers.')
                if len(neighbors)<=0:
                    print(f'No neighbor for client {i}.')
                    req_aggregation[i] = False
                    continue  #process for next client (this client should continue local learning)

                

                s_c = [c.id for c in neighbors]
                # for peer in s_c:
                #     client_heat[clients[i].id][peer]+=1  #count communication
                #     client_heat[peer][clients[i].id]+=1W0 = [tens.detach() for tens in list(self.model0.parameters())] #weight now
                # Wt = [tens.detach() for tens in list(self.model.parameters())] #previous weight
                
                    
                N = 0
                for sc in neighbors:
                    N = N + len(sc.train_loader.dataset) 
                #self
                N = N + len(clients[i].train_loader.dataset)

                #self weight
                W = len(clients[i].train_loader.dataset)/N
                for key in clients[i].model.state_dict().keys():              
                    new_models[i][key] = W * clients[i].model.state_dict()[key]

                
                for sc in neighbors:
                    # Record each key's value, while also keeping track of distance
                    
                    W = len(sc.train_loader.dataset)/N
                    for key in sc.model.state_dict().keys():  
                        new_models[i][key] += W * sc.model.state_dict()[key]



                #prototype aggregation
                args.clustered_agg = False
                if args.clustered_agg:
                    rec_neighbors_protos, rec_ood_protos = receive_grouped_protos(clients[i], neighbors, adj_clients) 
                    global_protos_label =  proto_aggregation_clustered(clients[i], args.num_classes, rec_neighbors_protos, rec_ood_protos)  #####************    ID from neighbors, ood from othersclient
                    #print(f'Global protos : {global_protos_label}') # We have aonly [4 2 1 5 9] protos
                else:
                    rec_protos = receive_protos(clients[i], adj_clients) #neighbors adj_clients
                    global_protos_label =  proto_aggregation(rec_protos)  #####************    #uiform average  Normalized protos
                    #print(f'Global protos : {global_protos_label}')



                clients[i].set_global_protos(global_protos_label)   #*****************  set global protos for clients  (normalized here)
                clients[i].set_dis_loss_protos(global_protos_label)   #*****************  set global protos for clients (normalized here)
                

        # average and update self
        for i in range(len(clients)):
                #clients[i].avg_model(new_models[i], Ni[i]) #N[i]: tot neighbors of client i each round
                clients[i].avg_model(new_models[i], 1) #New models already contains weighted sum
                print(f'Aggregated client {i}')                
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