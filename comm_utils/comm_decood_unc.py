from utils.utils_proto import *
#from eval_submod import *
from collections import defaultdict
import torch.nn.functional as F



# def receive_protos(client,sel_clients):
#     assert (len(sel_clients) > 0)

#     uploaded_ids = []
#     uploaded_protos = []
    
#     for cl in sel_clients:
#         uploaded_ids.append(cl.id)
#         uploaded_protos.append(cl.local_protos)

    
#     uploaded_protos.append(client.local_protos) #self protos for aggregation
#     uploaded_ids.append(client.id)
#     return uploaded_protos,uploaded_ids



def receive_protos_kappa(client,sel_clients):
    assert (len(sel_clients) > 0)

    uploaded_ids = []
    uploaded_protos = []
    uploaded_kappas = []


    for cl in sel_clients:
        uploaded_ids.append(cl.id)
        uploaded_protos.append(cl.local_protos)
        uploaded_kappas.append(cl.kappa_hats)

    
    uploaded_protos.append(client.local_protos) #self protos for aggregation
    uploaded_ids.append(client.id)
    uploaded_kappas.append(client.kappa_hats)

    return uploaded_protos,uploaded_ids,uploaded_kappas



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




def hoeffding_sel(scores, indices):
    # Compute the average similarity score

    average_score = sum(scores) / len(scores)
    print(f'Avg_loss: {average_score}')

    # Set the confidence level
    confidence_level = 0.95
    delta = 1 - confidence_level

    # Calculate epsilon using Hoeffding's Inequality
    epsilon = torch.sqrt((torch.log(torch.tensor(2.0 / delta))) / (2 * len(scores)))
    print(f'Epsilon: {epsilon} Avg-epsilon: {average_score - epsilon}')

    # Aggregate clients based on similarity
    if sel_on_kappa:
        selected_ind = [i for i in range(len(scores)) if scores[i] >= (average_score - epsilon)] #> # higher kappa less unc(compact)
    else:
        selected_ind = [i for i in range(len(scores)) if scores[i] <= (average_score - epsilon)] #>
    #pritn(f'Epsilon: {selected_ind}')

    return selected_ind

     

def avg_sel(scores, indices, sel_on_kappa):
    # Compute the average similarity score
    # average_score = sum(scores) / len(scores)
    # scores = scores.numpy()
    scores = torch.tensor(scores)

    average_score = torch.mean(scores)
    std_score = torch.std(scores)

    # average_score = np.mean(np.array(scores))
    # std_score = np.std(np.array(scores))
    # print(f'Avg_loss: {average_score}')
    # print(f'Std: {std_score}')
    
    print('Selected based on criteria: scores[i] >= average_score-std_score') 
    # Aggregate clients based on similarity
    if sel_on_kappa:
        selected_ind = [i for i in range(len(scores)) if scores[i] >= average_score] # higher kappa less unc(compact)   average_score-std_score
    else:
        selected_ind = [i for i in range(len(scores)) if scores[i] <= average_score+std_score] # >


    # pritn(f'Epsilon: {selected_ind}')

    return selected_ind




def eval_agg_protos_tar_unc(client, rec_protos_list, is_avg_sel = True, verbose = False):
    '''
    client: arbitrary clinet (receiver clinet)
    rec_proto_list : protos recived at client i
    is_avg_sel : uniform average
    '''

    assert (len(rec_protos_list) > 0)

    proto_uncs = []
    proto_loss = []

    agg_protos_label = defaultdict(list)

    if verbose:
        print(f'\nHost client {client.id} id label: {client.id_labels} cluster_id: {client.cluster}')
        #print(client.get_mis_classification())

    #from received protos, group according to labels
    for rec_protos in rec_protos_list:   #every client protos set 
        for label in rec_protos.keys():    #one by one prototype based on labels
            agg_protos_label[label].append(rec_protos[label])


    # for [label, proto_list] in agg_protos_label.items():
    #     print(f'Received {len(proto_list)} protos of class {label}')
    # print('\n')

    #evaluate single protos 
    for [label, proto_list] in agg_protos_label.items():
        if verbose:
            print(f'Received {len(proto_list)} protos of class {label} for evaluation.')

        loss_at_target = []
        proto_ind = []
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data

            for ind, proto_i in enumerate(proto_list):
                #proto += proto_i.data

                #collect loss on local data and select proto based on performance
                if label in client.id_labels:
                    #evaluate uncertainty
                    #print(client.get_local_unc(proto_i))

                    #print(client.get_local_unc())

                    #evaluate loss/misclassification
                    loss_at_target.append(client.get_mis_classification(label,proto_i))                    
                    proto_ind.append(ind)
                    # print(client.get_mis_classification(label,proto_i))
                else: #not id label, no performance evaluation, select all and aggregate
                    proto += proto_i.data


            #selection if id labels and aggregation                 
            if label in client.id_labels:
                if is_avg_sel:
                    sel_ind = avg_sel(loss_at_target, proto_ind) #select above/below average
                else:
                    sel_ind = hoeffding_sel(loss_at_target, proto_ind)

                print(f'Loss at target: {loss_at_target}')
                print(f'Selected protos: {sel_ind}')

                #agg from selected proto
                for ind, proto_i in enumerate(proto_list):
                    if sel_ind is not None:
                        if ind in sel_ind:
                            proto += proto_i.data               #weighted aggregation implementation **********************************
                    # else:
                    #       proto += proto_i.data
       
                if(len(sel_ind)>1):
                    tmp_proto = F.normalize((proto / len(sel_ind)),dim=0)  ######*****
                else:
                    tmp_proto = F.normalize(proto_list[0] , dim=0) #proto_list[0].data
            else: #non id proto
                tmp_proto = F.normalize((proto / len(proto_list)),dim=0)  ######*****
                

        else: #if only one proto
             tmp_proto = F.normalize(proto_list[0] , dim=0) #proto_list[0].data

        agg_protos_label[label] = tmp_proto.detach()

    return agg_protos_label







def eval_agg_protos_src_unc(client, rec_protos_list,  rec_client_ids, is_avg_sel = True, verbose = True):
    #receive proto from client i
    assert (len(rec_protos_list) > 0)

    proto_uncs = []
    proto_loss = []

    agg_protos_label = defaultdict(list)

    if verbose:
        print(f'\nHost client {client.id} id label: {client.id_labels} Cluster id:{client.cluster}')
        #print(client.get_mis_classification())

    #from received proto set from clients, group according to label
    for rec_protos in rec_protos_list:   #protos set from client
        for label in rec_protos.keys():    #one prototype based on labels
            agg_protos_label[label].append(rec_protos[label])


    # for [label, proto_list] in agg_protos_label.items():
    #     print(f'Received {len(proto_list)} protos of class {label}')
    # print('\n')

    #evaluate every prototype on source/target 
    for [label, proto_list] in agg_protos_label.items(): #select one proto based on label
        if verbose:
            print(f'Received {len(proto_list)} protos of class {label}')

        loss_at_src = []
        kappa_hat_at_src = []
        proto_ind = []
        
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data

            for ind, proto_i in enumerate(proto_list):
                #proto += proto_i.data

                #collect loss on local data and select proto based on performance
                if label in client.id_labels:
                    #evaluate uncertainty
                    #print(client.get_local_unc(proto_i))

                    tmp_kappa_hat = client.get_local_unc()  #receive kappa_hat of all protos(dictionary)


                    #evaluate loss/misclassification
                    kappa_hat_at_src.append(tmp_kappa_hat[label])
                    loss_at_src.append(client.get_mis_classification(label,proto_i))                    
                    proto_ind.append(ind)
                    # print(client.get_mis_classification(label,proto_i))
                else: #not id label, no performance evaluation, select all and aggregate
                    proto += proto_i.data


            #selection if id labels and aggregation                 
            if label in client.id_labels:
                if sel_on_kappa:  #unc from kappa_hat of every proto
                    if is_avg_sel:
                        sel_ind = avg_sel(kappa_hat_at_scr, proto_ind) #select above/below average
                    else:
                        sel_ind = hoeffding_sel(kappa_hat_at_scr, proto_ind)
                else:
                    if is_avg_sel:
                        sel_ind = avg_sel(loss_at_src, proto_ind) #select above/below average
                    else:
                        sel_ind = hoeffding_sel(loss_at_src, proto_ind)

                if sel_on_kappa:
                    print(f'Kappa_hat at src : {kappa_hat_at_src}')
                else:
                    print(f'Loss at src: {loss_at_src}')
                print(f'Selected protos: {sel_ind}')


                #agg from selected proto
                for ind, proto_i in enumerate(proto_list):
                    if sel_ind is not None:
                        if ind in sel_ind:
                            proto += proto_i.data
                    # else:
                    #       proto += proto_i.data
       
                if(len(sel_ind)>1):
                    tmp_proto = F.normalize((proto / len(sel_ind)),dim=0)  ######*****
                else:
                    tmp_proto = F.normalize(proto_list[0] , dim=0) #proto_list[0].data
            else: #non id proto
                tmp_proto = F.normalize((proto / len(proto_list)),dim=0)  ######*****
                

        else: #if only one proto
             tmp_proto = F.normalize(proto_list[0] , dim=0) #proto_list[0].data

        agg_protos_label[label] = tmp_proto.detach()

    return agg_protos_label







def eval_agg_protos_src_unc_kappa(client, rec_protos_list, rec_client_ids, rec_kappa_hat_lists, is_avg_sel = True, sel_on_kappa = True, verbose = True):
    #receive proto from client i
    assert (len(rec_protos_list) > 0)

    proto_uncs = []
    proto_loss = []

    agg_protos_label = defaultdict(list)
    kappa_hat_label = defaultdict(list)
    
    if verbose:
        print(f'\nHost client {client.id} id label: {client.id_labels} Cluster id:{client.cluster}')
        #print(client.get_mis_classification())

    #from received proto set from clients, group according to label
    # for i,rec_protos in enumerate(rec_protos_list):     #protos set from client
    #     for label in rec_protos.keys():    #one prototype based on labels
    #         agg_protos_label[label].append(rec_protos[label])
    #         kappa_label[label].append()
        
    #     proto_owner_id = rec_client_ids[i]


    for rec_protos,rec_kappa_hats in zip(rec_protos_list,rec_kappa_hat_lists):     #protos set from client
        # print(f'rec_protos: {rec_protos}')
        # print(f'rec_kappa_hats: {rec_kappa_hats}')
        for label in rec_protos.keys():    #one prototype based on labels
            agg_protos_label[label].append(rec_protos[label])
            kappa_hat_label[label].append(rec_kappa_hats[label])
        
        # print(f'Label wise agg_protos_label: {agg_protos_label}')
        # print(f'Label wise kappa_hat_label:{kappa_hat_label}')

        # proto_owner_id = rec_client_ids[i]
        
    #Now if there is 5 prototypes of 7(from 5 clients), we have 5 kappa values of 7    



    # for [label, proto_list] in agg_protos_label.items():
    #     print(f'Received {len(proto_list)} protos of class {label}')
    # print('\n')

    #evaluate every prototype 
    #get corresponding kappa_hat
    #aggregate 
    for [label, proto_list] in agg_protos_label.items(): #select one proto based on label
        if verbose:
            print(f'Received {len(proto_list)} protos of class {label}')

        loss_at_src = []
        kappa_hat_at_src = []
        proto_ind = []
        
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data

            for ind, proto_i in enumerate(proto_list):
                #proto += proto_i.data

                #collect loss on local data and select proto based on performance
                if label in client.id_labels:
                    #evaluate uncertainty
                    #print(client.get_local_unc(proto_i))

                    # print(f'Kappa of label:{label}: {kappa_hat_label[label]}')
                    # print(f'\nKappa of current proto:{proto_i} Kappa {kappa_hat_label[label][ind].detach()}')

                    ###################### tmp_kappa_hat = client.get_local_unc()  #receive kappa_hat of all protos(dictionary)
                    
                    #evaluate loss/misclassification
                    if sel_on_kappa:
                        kappa_hat_at_src.append(kappa_hat_label[label][ind].detach())
                    else:
                        loss_at_src.append(client.get_mis_classification(label,proto_i))                    
                    proto_ind.append(ind)
                    # print(client.get_mis_classification(label,proto_i))
                else: #not id label, no performance evaluation, select all and aggregate
                    proto += proto_i.data


            #selection based on loss/kappa/unc and get selected indices                 
            if label in client.id_labels:
                if sel_on_kappa:  #unc from kappa_hat of every proto
                    if is_avg_sel:
                        sel_ind = avg_sel(kappa_hat_at_src, proto_ind, sel_on_kappa) #select above/below average
                    else:
                        sel_ind = hoeffding_sel(kappa_hat_at_src, proto_ind, sel_on_kappa)
                else:
                    if is_avg_sel:
                        sel_ind = avg_sel(loss_at_src, proto_ind, sel_on_kappa) #select above/below average
                    else:
                        sel_ind = hoeffding_sel(loss_at_src, proto_ind, sel_on_kappa)

                if sel_on_kappa:
                    print(f'Kappa_hat at src : {kappa_hat_at_src}')
                else:
                    print(f'Loss at src: {loss_at_src}')
                
                print(f'Selected protos: {sel_ind}')


                #agg from selected proto
                for ind, proto_i in enumerate(proto_list):
                    if sel_ind is not None:
                        if ind in sel_ind:
                            proto += proto_i.data
                    # else:
                    #       proto += proto_i.data
       
                if(len(sel_ind)>1):
                    tmp_proto = F.normalize((proto / len(sel_ind)),dim=0)  ######*****
                else:
                    tmp_proto = F.normalize(proto_list[0] , dim=0) #proto_list[0].data
            else: #non id proto
                tmp_proto = F.normalize((proto / len(proto_list)),dim=0)  ######*****
                

        else: #if only one proto
             tmp_proto = F.normalize(proto_list[0] , dim=0) #proto_list[0].data

        agg_protos_label[label] = tmp_proto.detach()

    return agg_protos_label









def proto_aggregation(local_protos_list,verbose = False):
    agg_protos_label = defaultdict(list)

    for local_protos in local_protos_list:   #every client protos set 
        for label in local_protos.keys():    #one by one prototype based on labels
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if verbose:
            print(f'Received {len(proto_list)} proto of class {label}')

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
            



def comm_decood_unc(args, adj, clients, debug=False, test_loader=None):
    '''
    For a client i, Aggregates(FedAvg) all clients j if j is adjacent to i and (i!=j)
    based on the adjacency matrics adj. 
    Do only prototype aggregation no model aggregation

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
            # if e == 0:
            #     m.init_local_proto()            
            m.update()
            #print('Implement prototype uncertainty approach')
            #m.performance_train(m.val_loader) #compute uncertainty unc in self.unc
        

        req_aggregation = [True for i in range(len(clients))]
        # Set up variables where we do the averaging calculation without disturbing the previous weights
        tot_train_data = 0 #############
        
        # Start averaging towards the goal. Here we use equal neighbor averaging method.
        
        for i in range(len(clients)):
          
        #receive protos
            if req_aggregation[i]:
                adj_clients = [clients[c] for c in range(len(clients)) if (adj[c][i] and c!=i)]
                
                #neighbors, _ = clients_to_communicate_with(args, clients[i], adj_clients)
                
                #print(type(neighbors))

                # sel_neighbors = adj_clients

                    
                # #id -id communication count
                # s_c = [c.id for c in sel_neighbors]
                # # for peer in s_c:
                # #     client_heat[clients[i].id][peer]+=1  #count communication
                # #     client_heat[peer][clients[i].id]+=1W0 = [tens.detach() for tens in list(self.model0.parameters())] #weight now
                # # Wt = [tens.detach() for tens in list(self.model.parameters())] #previous weight
        

                neighbors = adj_clients 



                print(f'{len(neighbors)} neighbors selected from {len(adj_clients)} peers.')
                if len(neighbors)<=0:
                    print(f'No neighbor for client {i}.')
                    req_aggregation[i] = False
                    continue  #process for next client (this client should continue local learning)

                
                # #############################################
                # if req_aggregation[i]:
                #     for key in clients[i].model.state_dict().keys():
                #         new_models[i][key] = weights[-1] * clients[i].model.state_dict()[key] #self portion
                        
                # for ind, sc in enumerate(neighbors):
                #     for key in sc.model.state_dict().keys():              
                #         new_models[i][key] += weights[ind] * sc.model.state_dict()[key]
                
                # for i in range(len(clients)):
                #     if req_aggregation[i]:
                #         clients[i].load_model(new_models[i]) #N[i]: tot neighbors of client i each round
                #         print(f'Loaded aggregated model {i}')
                # ###############################################  


                args.clustered_agg = False
                if args.clustered_agg:
                    rec_neighbors_protos, rec_ood_protos = receive_grouped_protos(clients[i], neighbors, adj_clients) 
                    global_protos_label =  proto_aggregation_clustered(clients[i], args.num_classes, rec_neighbors_protos, rec_ood_protos)  #####************    ID from neighbors, ood from othersclient
                    #print(f'Global protos : {global_protos_label}') # We have aonly [4 2 1 5 9] protos
                else:
                    # rec_protos, proto_client_ids = receive_protos(clients[i], adj_clients) #neighbors adj_clients
                    rec_protos, rec_client_ids, kappa_hats = receive_protos_kappa(clients[i], adj_clients) #neighbors adj_clients
                    
                    if e>=0: #e>0:
                        #evaluate_protos(clients[i],rec_protos)
                        # print('Evaluating protos in receiver client data.')
                        # global_protos_label = eval_agg_protos_tar_unc(clients[i], rec_protos, True) #select based on loss or uncertainty and aggregate #True for average based selection
                        print('Evaluating protos in sending client data.')
                        # global_protos_label = eval_agg_protos_src_unc(clients[i], rec_protos, True) #select based on loss or uncertainty and aggregate #True for average based selection
                        global_protos_label = eval_agg_protos_src_unc_kappa(clients[i], rec_protos, rec_client_ids, kappa_hats, True, args.sel_on_kappa, True)

                    else:
                        global_protos_label = proto_aggregation(rec_protos)  #####************    #uiform average  Normalized protos

                   
                    # global_protos_label =  proto_aggregation(rec_protos)  #####************    #uiform average  Normalized protos

                    # print(f'Global protos : {global_protos_label}')


                clients[i].set_global_protos(global_protos_label)      #*****************  set global protos for clients  (normalized here)
                #clients[i].set_dis_loss_protos(global_protos_label)   #*****************  set global protos for clients (normalized here) (if commented, disloss proto is local)
                
                print(f'Aggregated for client {i}')


                


        #list every round avg performance of all clients (local test data and global test data)
        #lacc, gacc, lauc, gauc, lunc, gunc = evaluate_clients(clients, test_loader)
        lacc = evaluate_clients(clients, test_loader)
        print(f'Avg ACC:{lacc}\n')
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