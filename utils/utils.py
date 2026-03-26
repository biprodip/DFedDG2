"""utils.py — General utility functions for DFedDG2 federated learning experiments.

Contains helpers for parameter counting, gradient similarity, per-round client
evaluation, and neighbour-selection strategies used by decentralised algorithms.

Neighbour selection strategies (``clients_to_communicate_with``)
----------------------------------------------------------------
'random'     : Uniformly sample ``args.num_sel_clients`` from all adjacents.
'ideal'      : Randomly pick same-cluster clients only (oracle baseline).
'loss_based' : Select clients whose inverse loss exceeds the peer average.
'grad_based' : Select by cosine similarity of local gradients (threshold-gated).
'hybrid'     : Loss-based selection + prototype-diversity weighting.
'mixed_loss' : Combined loss + diversity metric for selection and weighting.
"""

import copy
import logging
import torch
import numpy as np
import torch.nn as nn
from numpy import random
from multiprocessing import pool
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)



def count_params(client, comm_algo):
    """Print the parameter count for a client, dispatching by communication algorithm.

    Different algorithms expose different ``count_params()`` return signatures on
    the client object:
      - ``'comm_dis_pfl'``    → returns (total, non_zero, mask) — pruned models.
      - ``'comm_gossip'`` / ``'comm_penz'`` → returns a single integer count.
      - ``'comm_decood_w'``   → returns (proto_params, model_params).

    Args:
        client: A client object with a ``count_params()`` method.
        comm_algo: String identifier of the communication algorithm in use.

    Returns:
        None.  Parameter counts are printed to stdout only.
    """
    tot_param = 0
    if comm_algo == 'comm_dis_pfl':
        total_params_to_send, non_zero_params, mask_params = client.count_params()  #from model and masks
        LOGGER.info("Parameters of client: total: %s, non_zero_params: %s mask: %s", total_params_to_send, non_zero_params, mask_params)

    elif comm_algo=='comm_gossip' or comm_algo=='comm_penz':
        model_params = client.count_params()  #from model and masks
        LOGGER.info("Total parameters of client: %s", model_params)

    elif comm_algo=='comm_decood_w':
        proto_params, model_params = client.count_params()  #from model and masks
        LOGGER.info("Parameters of client: all_class_proto_params : %s model_params : %s", proto_params, model_params)

    else:
        LOGGER.warning("utils.py: Prototype count function undefined for specified algorithm!")




def get_matrix_cosine_similarity_from_grads(local_model_grads):
    """Compute cosine similarity between the gradients of two client models.

    Flattens all parameter gradients into a single 1-D vector per client, then
    returns the cosine similarity between the first two entries of
    ``local_model_grads``.  Only the first two clients are compared regardless
    of list length.

    Args:
        local_model_grads: List of gradient lists, where each inner list
            contains per-parameter gradient tensors (PyTorch).  At least two
            entries are required.

    Returns:
        np.ndarray: 1×1 cosine similarity matrix (scalar wrapped in array).
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




def evaluate_clients(clients, test_loader=None):
    """Evaluate all clients on their local test sets and return mean accuracy.

    Calls ``client.performance_test()`` for each client, appends the result to
    ``client.l_test_acc_hist``, and accumulates the mean across all K clients.

    Note: ``test_loader`` is accepted for interface compatibility but is not
    used; each client evaluates on its own internal test loader.

    Args:
        clients: List of client objects with ``performance_test()`` and
            ``l_test_acc_hist`` attributes.
        test_loader: Unused.  Present for API compatibility with the global-set
            variant (``evaluate_clients_tmp``).

    Returns:
        float: Mean local test accuracy across all K clients for this round.
    """
    round_avg_lacc = 0

    K = len(clients)

    # Evaluate each client on its local test split
    for i in range(K):
       
       #local test performance
       test_acc, test_auc, test_unc = clients[i].performance_test()
       #clients[i].l_test_loss_hist.append(rl_loss)
       clients[i].l_test_acc_hist.append(test_acc)
       LOGGER.info("Client id: %s, Test Lacc: %.2f", clients[i].id, test_acc)

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
       LOGGER.info("Clinet id: %s,\nLacc: %.2f,  Lunc: %.2f", clients[i].id, rl_acc, rl_unc)

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
           LOGGER.info("Gacc: %.2f Gunc: %.2f", rg_acc, rg_unc)
                  
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
       LOGGER.warning("No global test set, so no global result.")
   
   return round_avg_lacc, round_avg_gacc, round_avg_lauc, round_avg_gauc,round_avg_lunc, round_avg_gunc



##neighbor selection
def accuracy(outputs, labels):
    """Fraction of correct top-1 predictions (normalised accuracy).

    Args:
        outputs: Class logits, shape [N, C].
        labels: Ground-truth class indices, shape [N].

    Returns:
        torch.Tensor: Scalar accuracy in [0, 1].
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def eval_protos(protos1, protos2):
    """Compute MSE dissimilarity between two prototype tensors.

    Used in diversity-based neighbour selection to quantify how different two
    clients' per-class feature means are.  Higher MSE → more complementary
    prototypes → potentially better aggregation partner.

    Args:
        protos1: Prototype tensor from the local client.
        protos2: Prototype tensor from a candidate neighbour.

    Returns:
        torch.Tensor: Scalar MSE loss between the two prototype tensors.
    """
    loss_mse = nn.MSELoss()
    return loss_mse(protos1, protos2)



def clients_to_communicate_with(args, client, clients):
    """Select a subset of adjacent clients to communicate with this round.

    Implements multiple neighbour-selection policies controlled by
    ``args.neighbour_selection``:

    - ``'random'``     : Uniformly sample ``args.num_sel_clients`` from all
                         adjacents.
    - ``'ideal'``      : Oracle — pick same-cluster clients only.
    - ``'loss_based'`` : Keep clients whose inverse-loss exceeds peer mean.
    - ``'grad_based'`` : Keep clients with gradient cosine similarity above
                         ``args.grad_thres``; sort descending.
    - ``'hybrid'``     : Loss-based shortlist then diversity-based weighting.
    - ``'mixed_loss'`` : Combined loss + diversity metric for joint selection.

    Args:
        args: Namespace with fields: ``neighbour_selection``, ``num_sel_clients``,
            ``n_samplings``, ``neighbour_exploration``, ``grad_thres``, etc.
        client: The querying client whose local data / gradients are used to
            score candidates.
        clients: List of adjacent client objects to consider.

    Returns:
        tuple[list, np.ndarray | None]:
            - neighbours: Selected client objects for this round.
            - weights: Optional aggregation weights (None for most strategies).
    """
    weights = None
    
    adj = [c.id for c in clients]
        
    if args.neighbour_selection == "random":
        LOGGER.info("Random neighbour selection")
        #critical case
        n_neighbours = (args.num_sel_clients if len(clients) >= args.num_sel_clients else len(clients))
        #select neighbours
        neighbours = np.random.choice(clients, n_neighbours, replace=False)
        
    elif args.neighbour_selection == "ideal":
        LOGGER.info("Ideal same cluster random neighbours selection")
        #same cluster random neighbours selection

        clients_to_choose_from = [client_ for client_ in clients if client_.cluster == client.cluster ]
        if len(clients_to_choose_from) >= args.num_sel_clients:
            neighbours = np.random.choice(clients_to_choose_from, args.num_sel_clients, replace=False)
        else:
            LOGGER.warning("Total clients to select is less than clients pool")

    
    elif args.neighbour_selection == "loss_based":
        #selected if greater than avg 1/loss of all peers
        LOGGER.info("Loss based neighbor selection in penz")
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
            
            avg_metric = sum(other_clients_metric.values()) / len(other_clients_metric)
            neighbours = [k for k, v in zip(other_clients_metric.keys(),other_clients_metric.values()) if v > avg_metric]
            

    elif args.neighbour_selection == "grad_based":
        #selected if greater than average similarity or args.threshold(0.75)
        LOGGER.info("Grad based neighbor selection.")
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
                
                #sim_score = cosine_similarity(model1_gradients, model2_gradients)
                sim_score = get_matrix_cosine_similarity_from_grads(all_grads)
                LOGGER.debug("Grad similarity with id: %s is %s", other_client.id, sim_score)

                
                if sim_score>args.grad_thres: #keep only positive cosine similarities
                    other_clients_metric[other_client] = sim_score    
                
                #other_clients_metric[other_client] = sim_score

            other_clients_sorted_by_metric = sorted(   #returned keys
                other_clients_metric, key=other_clients_metric.get, reverse=True
            )
            
            neighbours = other_clients_sorted_by_metric
       

    elif args.neighbour_selection == "hybrid":  #loss based selection of n_neighbor and diversity based weight
        LOGGER.info("performance based neighbor selection")
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
                LOGGER.info("Sampling based neighbours")
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


            if args.neighbour_exploration == "weights":
                LOGGER.info("weighted performance based neighbours")
                
                weight = np.array(list(diversity_client_metric.values()))
                weight = weight / sum(weight)
                
                neighbours = clients_to_consider
            

    elif args.neighbour_selection == "mixed_loss":       #select and weight everyone based on mixed loss
        LOGGER.info("performance based neighbor selection")
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
                
                alpha = 0.5  # weight balancing loss vs diversity
                other_clients_metric[other_client] =  alpha * (1 / (target_loss + 1e-5)) + (1-alpha) * diversity_Loss  #+ other loss

            
            other_clients_sorted_by_metric = sorted(
                other_clients_metric, key=other_clients_metric.get, reverse=True
            )
            
            
            #update clients to consider based on loss
            if args.neighbour_exploration == "greedy":
                clients_to_consider = other_clients_sorted_by_metric[: args.num_sel_clients]
            elif args.neighbour_exploration == "sampling":
                LOGGER.info("Sampling based neighbours")
                probs = np.array(list(other_clients_metric.values()))
                probs = probs / sum(probs)
                neighbours = np.random.choice(
                    list(other_clients_metric.keys()),
                    size=args.num_sel_clients,
                    replace=False,
                    p=probs,
                )
            elif args.neighbour_exploration == "weights":
                LOGGER.info("weighted performance based neighbours")
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
                LOGGER.info("topK performance based neighbours")
                candidates = other_clients_sorted_by_metric[: args.topk]
                neighbours = np.random.choice(candidates, size=args.num_sel_clients)

    else:
        raise ValueError(
            f"Unknown neighbour_selection '{args.neighbour_selection}'. "
            "Choose from: random, ideal, loss_based, grad_based, hybrid, mixed_loss"
        )

    return neighbours, weights