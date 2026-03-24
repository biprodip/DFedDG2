"""comm_decood.py — Prototype-only gossip communication for DECOOD.

This module implements the non-gossip decentralized communication round used by
the 'decood' algorithm variant.  Unlike comm_vmf_gossip, there is **no** model
weight exchange — only per-class prototype vectors are aggregated across graph
neighbours.

Key functions
-------------
receive_protos              : flat collection of neighbour prototypes (+ self).
receive_grouped_protos      : split collection into neighbour vs. OOD-client protos.
proto_aggregation           : uniform mean of prototype lists → normalised result.
proto_aggregation_clustered : ID labels from neighbours; OOD labels from others.
evaluate                    : thin wrapper around server-style test/train metrics.
comm_decood                 : main training loop — local update then proto agg.
"""

from utils.utils_proto import *
#from eval_submod import *
from collections import defaultdict
import torch.nn.functional as F



def receive_protos(client, sel_clients):
    """Collect local prototypes from selected neighbours and the client itself.

    Args:
        client: The aggregating client (self).  Its own prototypes are always
            appended last so that proto_aggregation includes the self prototype.
        sel_clients: List of neighbour client objects whose ``local_protos``
            dicts will be included.

    Returns:
        tuple[list[dict], list[int]]:
            - uploaded_protos: List of per-class prototype dicts, one per
              contributing client (neighbours first, self last).
            - uploaded_ids: Corresponding client IDs in the same order.
    """
    assert (len(sel_clients) > 0)

    uploaded_ids = []
    uploaded_protos = []

    for cl in sel_clients:
        uploaded_ids.append(cl.id)
        uploaded_protos.append(cl.local_protos)

    uploaded_protos.append(client.local_protos)  # self protos for aggregation
    uploaded_ids.append(client.id)
    return uploaded_protos, uploaded_ids



def receive_grouped_protos(client, neighbors, adjacents):
    """Separate received prototypes into two groups: graph-neighbours vs. others.

    In the clustered aggregation strategy, ID-class prototypes are taken from
    actual graph neighbours (who share the same in-distribution classes), while
    OOD-class prototypes are sourced from adjacent clients outside the neighbour
    set.  If every adjacent client is a neighbour, ``rec_ood_protos`` will be
    empty.

    Args:
        client: The aggregating client.  Its own ``local_protos`` are appended
            to ``rec_neighbors_protos`` (treated as a self-neighbour).
        neighbors: Subset of ``adjacents`` that are selected graph-neighbours
            (used for ID-label protos).
        adjacents: Full adjacency list — all graph-connected clients.

    Returns:
        tuple[list[dict], list[dict]]:
            - rec_neighbors_protos: Prototype dicts from neighbours + self.
            - rec_ood_protos: Prototype dicts from adjacent non-neighbours
              (used to supply OOD-class prototypes).
    """
    assert (len(adjacents) > 0)
    # If all adjacents are neighbours, ood protos (labels) are not available
    # via the non-neighbour path.
    n_ids = [c.id for c in neighbors]

    rec_neighbors_protos = []
    rec_ood_protos = []

    for cl in adjacents:
        if cl.id in n_ids:
            rec_neighbors_protos.append(cl.local_protos)  # ID and OOD protos from neighbours
        else:
            rec_ood_protos.append(cl.local_protos)        # ID and OOD protos from others

    rec_neighbors_protos.append(client.local_protos)  # self protos for aggregation
    return rec_neighbors_protos, rec_ood_protos




def proto_aggregation(local_protos_list, verbose=False):
    """Uniformly average per-class prototypes across a list of client prototype dicts.

    For each class label, the contributing prototype tensors are summed and
    divided by their count, then L2-normalised (F.normalize, dim=0) so that the
    result lives on the unit hypersphere — consistent with the cosine-similarity
    objective used during training.

    Args:
        local_protos_list: List of dicts mapping class label → prototype tensor
            (one dict per participating client including self).
        verbose: If True, print the number of prototypes received per class.

    Returns:
        defaultdict[int, Tensor]: Aggregated normalised prototype per class
            (detached from the computation graph).
    """
    agg_protos_label = defaultdict(list)

    for local_protos in local_protos_list:   #every client protos set 
        for label in local_protos.keys():    #one by one prototype based on labels
            agg_protos_label[label].append(local_protos[label])
    # print(f'agg_protos_label: {agg_protos_label}')

    for [label, proto_list] in agg_protos_label.items():
        if verbose:
            print(f'Received {len(proto_list)} proto of class {label}')
            # print(proto_list)
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
    # print(f'Agg list:{agg_protos_label}')

    return agg_protos_label




def proto_aggregation_clustered(client, labels, rec_neighbors_protos, rec_ood_protos):
    """Aggregate prototypes with different sources for ID vs. OOD labels.

    Strategy:
      - **OOD labels** (classes absent locally): use prototypes from adjacent
        non-neighbour clients (``rec_ood_protos``).  If unavailable there,
        fall back to neighbour protos.
      - **ID labels** (locally present classes): always sourced from neighbours
        (``rec_neighbors_protos``), including self.

    Unlike ``proto_aggregation``, the result is **not** L2-normalised — a
    simple mean is used instead (see commented-out normalize line).

    Args:
        client: The aggregating client; its ``id_labels`` set defines which
            classes are in-distribution for this client.
        labels: Total number of classes (used to initialise ``protos_received``
            boolean array).
        rec_neighbors_protos: Prototype dicts from graph-neighbours + self.
        rec_ood_protos: Prototype dicts from adjacent non-neighbour clients.

    Returns:
        defaultdict[int, Tensor]: Aggregated prototype per class (detached).
    """
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
    """Compute and record test accuracy and training loss for a server-style client.

    Calls ``self.test_metrics()`` and ``self.train_metrics()`` (server-side
    methods), appends results to running history lists, and prints a summary.

    Args:
        self: Server or client object exposing ``test_metrics()``,
            ``train_metrics()``, ``rs_test_acc``, and ``rs_train_loss``.
        acc: External list to append test accuracy to.  If None, appends to
            ``self.rs_test_acc`` instead.
        loss: External list to append train loss to.  If None, appends to
            ``self.rs_train_loss`` instead.

    Note:
        This function uses ``self`` as a positional argument, which means it
        is *not* a method bound to a class — it must be called as
        ``evaluate(server_obj)``.
    """
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
            



def comm_decood(args, adj, clients, debug=False, test_loader=None):
    """Main training loop for the DECOOD (prototype-only gossip) algorithm.

    Each round proceeds in two phases:

    1. **Local update** — every client runs one epoch of local training
       (``client.update()``), computing the DECOOD loss on its own data.

    2. **Prototype aggregation** — for each client *i*, adjacent clients
       (``adj[c][i] == 1, c != i``) supply their ``local_protos``.  The mean
       prototype per class is computed via ``proto_aggregation`` (or the
       clustered variant if ``args.clustered_agg`` is True) and then pushed
       back into the client via ``set_global_protos`` and
       ``set_dis_loss_protos``.

    Note: ``args.clustered_agg`` is hard-coded to ``False`` inside this
    function; the clustered branch is present for experimental use only.

    Args:
        args: Experiment configuration namespace.  Relevant fields:
            ``num_rounds``, ``global_seed``, ``num_clients``,
            ``clustered_agg``, ``num_classes``.
        adj: [N × N] adjacency tensor (float).  ``adj[c][i] == 1`` means
            client *c* is a graph-neighbour of client *i*.
        clients: List of client objects (one per federated node).
        debug: Unused; reserved for future verbose logging.
        test_loader: Unused in this function; evaluation is performed via
            each client's own test loader through ``evaluate_clients``.

    Returns:
        list[float]: Per-round average local test accuracy across all clients.

    Side effects:
        Updates ``client.global_protos`` and ``DisLoss`` prototype buffers
        in-place for every client each round.
    """
        
    print('Comm Decood training.')
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



                print(f'Agg of client {i}: {len(neighbors)} neighbors selected from {len(adj_clients)} peers.')
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
                    rec_protos, proto_client_ids = receive_protos(clients[i], adj_clients) #neighbors adj_clients
                    global_protos_label =  proto_aggregation(rec_protos)  #####************    #uiform average  Normalized protos

                    # print(f'Global protos : {global_protos_label}')


                clients[i].set_global_protos(global_protos_label)      #*****************  set global protos for clients  (normalized here)
                clients[i].set_dis_loss_protos(global_protos_label)   #*****************  set global protos for clients (normalized here) (if commented, disloss proto is local)
                
                # print(f'Aggregated for client {i}')


                


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