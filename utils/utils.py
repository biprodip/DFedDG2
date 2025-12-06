import copy
import torch
import numpy as np
import torch.nn as nn
from numpy import random 
from multiprocessing import pool
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity



def count_params(client, comm_algo):
        tot_param = 0
        if comm_algo=='comm_dis_pfl':
            total_params_to_send, non_zero_params, mask_params = client.count_params()  #from model and masks
            print(f'Parameters of client: total: {total_params_to_send}, non_zero_params: {non_zero_params} mask: {mask_params}')

        elif comm_algo=='comm_gossip' or comm_algo=='comm_penz':
            model_params = client.count_params()  #from model and masks
            print(f'Total parameters of client: {model_params}')

        elif comm_algo=='comm_decood_w':
            proto_params, model_params = client.count_params()  #from model and masks  
            print(f'Parameters of client: all_class_proto_params : {proto_params} model_params : {model_params}')
        
        else:
            print('utils.py: Prototype count function undefined for specified algorithm!')




def get_matrix_cosine_similarity_from_grads(local_model_grads):
    """
    return the similarity matrix where the distance chosen to
    compare two clients is set with `distance_type`
    """
    n_clients = len(local_model_grads)
    
    grad_1, grad_2 = local_model_grads[0], local_model_grads[1]

    # Convert PyTorch gradients to NumPy arrays
    grad_1_np = [g.detach().cpu().numpy() for g in grad_1]
    grad_2_np = [g.detach().cpu().numpy() for g in grad_2]

    # Reshape the gradients to 1D vectors
    grad_1_flat = np.concatenate([g.flatten() for g in grad_1_np])
    grad_2_flat = np.concatenate([g.flatten() for g in grad_2_np])

    # Calculate cosine similarity
    similarity = cosine_similarity([grad_1_flat], [grad_2_flat])

    return similarity


# def evaluate_clients(clients, test_loader=None):
#    round_avg_lacc = 0
   
#    K = len(clients)
#    # Load averaged weights in client(global models)
#    for i in range(K):
       
#        #local test performance
#        test_acc, test_num,_ = clients[i].performance_test()
#        #clients[i].l_test_loss_hist.append(rl_loss)
#        clients[i].l_test_acc_hist.append(test_acc/test_num)
#        print(f'Clinet id: {clients[i].id},\nLacc: {test_acc/test_num:.2f}')

#        round_avg_lacc += test_acc/test_num   
       
   
#    round_avg_lacc /= K
   
#    return round_avg_lacc



def evaluate_clients(clients, test_loader=None):
   round_avg_lacc = 0
   
   K = len(clients)

   # Load averaged weights in client(global models)
   for i in range(K):
       
       #local test performance
       test_acc, test_auc, test_unc = clients[i].performance_test()
       #clients[i].l_test_loss_hist.append(rl_loss)
       clients[i].l_test_acc_hist.append(test_acc)
       print(f'Client id: {clients[i].id}, Test Lacc: {test_acc:.2f}')

       round_avg_lacc += test_acc   
   
   round_avg_lacc /= K
   
   return round_avg_lacc



def evaluate_clients_tmp(clients, test_loader):
   round_avg_lacc = 0
   round_avg_gacc = 0
   round_avg_lauc = 0
   round_avg_gauc = 0
   round_avg_lunc = 0
   round_avg_gunc = 0
   
   K = len(clients)
   # Load averaged weights in client(global models)
   for i in range(K):
       
       #local test performance
       rl_acc, rl_auc, rl_unc = clients[i].performance_test(clients[i].test_loader)
       #clients[i].l_test_loss_hist.append(rl_loss)
       clients[i].l_test_acc_hist.append(rl_acc)
       clients[i].l_test_auc_hist.append(rl_auc)
       clients[i].l_test_unc_hist.append(rl_unc)
       print(f'Clinet id: {clients[i].id},\nLacc: {rl_acc:.2f},  Lunc: {rl_unc:.2f}')

       round_avg_lacc += rl_acc  
       round_avg_lauc += rl_auc  
       round_avg_lunc += rl_unc  
       
       #clients[i].model.load_state_dict(new_models[i])
       #global test_set
       if test_loader is not None:
           rg_acc, rg_auc, rg_unc = clients[i].performance_test(test_loader)
           #clients[i].g_test_loss_hist.append(rg_loss)
           clients[i].g_test_acc_hist.append(rg_acc)
           clients[i].g_test_auc_hist.append(rg_auc)
           clients[i].g_test_unc_hist.append(rg_unc)
           #print(f'Clinet id: {clients[i].id}, Lloss: {rl_loss:.2f}  Lacc: {rl_acc:.2f}, Lunc: {rl_unc:.2f}, Gloss: {rg_loss:.2f}  Gacc: {rg_acc:.2f} Gunc: {rg_unc:.2f}')
           print(f'Gacc: {rg_acc:.2f} Gunc: {rg_unc:.2f} ')
                  
           round_avg_gacc += rg_acc
           round_avg_gauc += rg_auc
           round_avg_gunc += rg_unc
   
   round_avg_lacc /= K
   round_avg_lauc /= K
   round_avg_lunc /= K
   if test_loader is not None:  #if global test set exists
       round_avg_gacc /= K
       round_avg_gauc /= K
       round_avg_gunc /= K
   else:
       print('No global test set, so no global result.')
   
   return round_avg_lacc, round_avg_gacc, round_avg_lauc, round_avg_gauc,round_avg_lunc, round_avg_gunc



##neighbor selection
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def eval_protos(protos1, protos2):
    loss_mse = nn.MSELoss()
    #print(f'Eval protos diversity loss: {nn.loss_mse(protos1, protos2)}')
    return loss_mse(protos1, protos2)

# def cosine_similarity(v1, v2):
#     return nn.functional.cosine_similarity(v1, v2, dim=0)


# def get_gradients(model):
#     gradients = []
#     for param in model.parameters():
#         if param.requires_grad and param.grad is not None:
#             gradients.append(param.grad.view(-1))
#     return torch.cat(gradients)    



def clients_to_communicate_with(args, client, clients):
    '''
    param: args: arguments
           client: current client (c)
           clients : adjacent clients of c     

    returns: neighbors: selected adjacent clients
             weights    
              
    '''
    weights = None
    
    adj = [c.id for c in clients]
    ############################print(f'{client.id} and adjacents are {adj}')
        
    if args.neighbour_selection == "random":
        print('\nRandom neighbour selection')
        #critical case
        n_neighbours = (args.num_sel_clients if len(clients) >= args.num_sel_clients else len(clients))
        #select neighbours
        neighbours = np.random.choice(clients, n_neighbours, replace=False)
        
    elif args.neighbour_selection == "ideal":
        print('\nIdeal same cluster random neighbours selection')
        #same cluster random neighbours selection

        clients_to_choose_from = [client_ for client_ in clients if client_.cluster == client.cluster ]
        if len(clients_to_choose_from) >= args.num_sel_clients:
            neighbours = np.random.choice(clients_to_choose_from, args.num_sel_clients, replace=False )
        else:
            print('Total clients to select is less than clients pool')

    
    elif args.neighbour_selection == "loss_based":
        #selected if greater than avg 1/loss of all peers
        print('\nLoss based neighbor selection in penz')
        # Sample 10 times
        
        for _ in range(args.n_samplings):
            #clients_to_consider = np.random.choice(clients, args.n_clients_to_consider, replace=False)
            clients_to_consider = clients #adjacents

            other_clients_metric = OrderedDict()
            for other_client in clients_to_consider:
                #get the loss
                train_loss,acc, auc, unc = client.performance_train(other_client.train_loader)
                #train_loss, train_acc = evaluate(client.model, other_client.train_loader)
                other_clients_metric[other_client] = 1 / (train_loss + 1e-5) #zero div protection
            
            
            # other_clients_sorted_by_metric = sorted(
            #     other_clients_metric, key=other_clients_metric.get, reverse=True
            # )
            
            
            # if args.neighbour_exploration == "greedy":
            #     neighbours = other_clients_sorted_by_metric[: args.num_sel_clients]
            
            # elif args.neighbour_exploration == "sampling":
            #     print('Sampling based neighbours')
            #     probs = np.array(list(other_clients_metric.values()))
            #     probs = probs / sum(probs)
            #     neighbours = np.random.choice(
            #         list(other_clients_metric.keys()),
            #         size=args.num_sel_clients,
            #         replace=False,
            #         p=probs,
            #     )
                
            # elif args.neighbour_exploration == "weights":
            #     print('weighted performance based neighbours')
            #     val_loss, val_acc = (
            #         client.history["val_losses"][-1],
            #         client.history["val_accs"][-1],
            #     )
            #     own_client_metric = (
            #         1 / (val_loss + 1e-5)
            #         #val_acc
            #     )
            #     weights = np.array(
            #         list(other_clients_metric.values()) + [own_client_metric]
            #     )
            #    neighbours = clients_to_consider

            # elif args.neighbour_exploration == "topk":
            #     print('topK performance based neighbours')
            #     candidates = other_clients_sorted_by_metric[: args.topk]
            #     neighbours = np.random.choice(candidates, size=args.num_sel_clients)
            
            avg_metric = sum(other_clients_metric.values()) / len(other_clients_metric)
            neighbours = [k for k, v in zip(other_clients_metric.keys(),other_clients_metric.values()) if v > avg_metric]
            

    elif args.neighbour_selection == "grad_based":
        #selected if greater than average similarity or args.threshold(0.75) 
        print('\nGrad based neighbor selection.')
        # Sample 10 times
        
        for _ in range(args.n_samplings):
            #clients_to_consider = np.random.choice(clients, args.n_clients_to_consider, replace=False)
            clients_to_consider = clients #adjacents

            other_clients_metric = OrderedDict()
            for other_client in clients_to_consider:
                #get the loss

                all_grads = []
                all_grads += [client.grad]
                all_grads += [other_client.grad]
                
    
                ############
                # model1_gradients = get_gradients(client.model)
                # model2_gradients = get_gradients(other_client.model)

                # model1_gradients = get_gradients(client.head)
                # model2_gradients = get_gradients(other_client.head)

                #sim_score = cosine_similarity(model1_gradients, model2_gradients)
                sim_score = get_matrix_cosine_similarity_from_grads(all_grads)
                print(f'Grad similarity with id: {other_client.id} is {sim_score}')

                
                if sim_score>args.grad_thres: #keep only positive cosine similarities
                    other_clients_metric[other_client] = sim_score    
                
                #other_clients_metric[other_client] = sim_score

            other_clients_sorted_by_metric = sorted(   #returned keys
                other_clients_metric, key=other_clients_metric.get, reverse=True
            )
            
            neighbours = other_clients_sorted_by_metric
            # if args.neighbour_exploration == "greedy":
            #     neighbours = other_clients_sorted_by_metric[: args.num_sel_clients]
            
            # elif args.neighbour_exploration == "sampling":
            #     print('Sampling based neighbours')
            #     print(other_clients_metric.values())
            #     probs = nn.functional.softmax(other_clients_metric.values(), dim=0)  ####negative values      #######################################
            #     neighbours = np.random.choice(
            #         list(other_clients_metric.keys()),
            #         size=args.num_sel_clients,
            #         replace=False,
            #         p=probs,
            #     )

    elif args.neighbour_selection == "hybrid":  #loss based selection of n_neighbor and diversity based weight
        print('performance based neighbor selection')
        # Sample 10 times
        
        for _ in range(args.n_samplings):
            #clients_to_consider = np.random.choice(clients, args.n_clients_to_consider, replace=False)
            clients_to_consider = clients #adjacents

            #select based on loss
            other_clients_metric = OrderedDict()
            for other_client in clients_to_consider:
                train_loss,acc, auc, unc = client.performance_train(other_client.train_loader)
                #train_loss, train_acc = evaluate(client.model, other_client.train_loader)
                other_clients_metric[other_client] = (
                    1 / (train_loss + 1e-5) #zero div protection
                    #train_acc
                )
            other_clients_sorted_by_metric = sorted(
                other_clients_metric, key=other_clients_metric.get, reverse=True
            )
            
            #update clients to consider based on loss
            if args.neighbour_exploration == "greedy":
                clients_to_consider = other_clients_sorted_by_metric[: args.num_sel_clients]
            
            elif args.neighbour_exploration == "sampling":
                print('Sampling based neighbours')
                probs = np.array(list(other_clients_metric.values()))
                probs = probs / sum(probs)
                clients_to_consider = np.random.choice(
                    list(other_clients_metric.keys()),
                    size=args.num_sel_clients,
                    replace=False,
                    p=probs,
                )
            
            #diversoty based weighting of selected clients
            diversity_client_metric = OrderedDict()
            
            client.collect_protos()  #update local protos on local train data
            for other_client in clients_to_consider:
                other_client.collect_protos(client.train_loader) #gt protos on provided dataset
                diversity_Loss  = eval_protos(client.protos, other_client.protos)

                #train_loss, train_acc = evaluate(client.model, other_client.train_loader)
                diversity_client_metric[other_client] = diversity_Loss  #+ other loss

            # other_clients_sorted_by_diversity = sorted(
            #     diversity_client_metric, key=diversity_client_metric.get, reverse=True
            # )

            if args.neighbour_exploration == "weights":
                print('weighted performance based neighbours')
                
                weight = np.array(list(diversity_client_metric.values()))
                weight = weight / sum(weight)
                
                neighbours = clients_to_consider
            

    elif args.neighbour_selection == "mixed_loss":       #select and weight everyone based on mixed loss
        print('performance based neighbor selection')
        # Sample 10 timesselected
        
        for _ in range(args.n_samplings):
            #clients_to_consider = np.random.choice(clients, args.n_clients_to_consider, replace=False)
            clients_to_consider = clients #adjacents

            #select based on loss
            other_clients_metric = OrderedDict()
            
            client.collect_protos()  #update local protos on local train data
            for other_client in clients_to_consider:
                target_loss = client.performance_train(other_client.train_loader)
                #train_loss, train_acc = evaluate(client.model, other_client.train_loader)
                
                other_client.collect_protos(client.train_loader) #gt protos on provided dataset
                diversity_Loss  = eval_protos(client.protos, other_client.protos)
                
                other_clients_metric[other_client] =  alpha * (1 / (target_loss + 1e-5)) + (1-alpha) * diversity_Loss  #+ other loss

            
            other_clients_sorted_by_metric = sorted(
                other_clients_metric, key=other_clients_metric.get, reverse=True
            )
            
            
            #update clients to consider based on loss
            if args.neighbour_exploration == "greedy":
                clients_to_consider = other_clients_sorted_by_metric[: args.num_sel_clients]
            elif args.neighbour_exploration == "sampling":
                print('Sampling based neighbours')
                probs = np.array(list(other_clients_metric.values()))
                probs = probs / sum(probs)
                neighbours = np.random.choice(
                    list(other_clients_metric.keys()),
                    size=args.num_sel_clients,
                    replace=False,
                    p=probs,
                )
            elif args.neighbour_exploration == "weights":
                print('weighted performance based neighbours')
                val_loss, val_acc = (
                    client.history["val_losses"][-1],
                    client.history["val_accs"][-1],
                )
                own_client_metric = (
                    1 / (val_loss + 1e-5)
                    #val_acc
                )
                weights = np.array(
                    list(other_clients_metric.values()) + [own_client_metric]
                )
                neighbours = clients_to_consider
            
            elif args.neighbour_exploration == "topk":
                print('topK performance based neighbours')
                candidates = other_clients_sorted_by_metric[: args.topk]
                neighbours = np.random.choice(candidates, size=args.num_sel_clients)



    return neighbours, weights