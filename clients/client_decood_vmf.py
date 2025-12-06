import os
import copy
import torch
import random
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from utils.losses_decood import *
from utils.imb_losses import *
from sklearn import metrics
# from data.data_utils import DatasetSplit
from lib.data_utils import DatasetSplit
from sklearn.preprocessing import label_binarize



class ClientDecoodVMF():
    def __init__(self, id, args, adj, dataset_train=None, user_data_idx_train=None, dataset_test=None, user_data_idx_test=None, cluster=None):
        '''
        Computes local class  prototypes form local embedding. 
        Receives the aggregated global prototypes from communcation script.
        In next iteration, moves local embeddings closer to global prototypes. Again computes local protos.   
        L = L_CE(or L_bs) + CIDER losses(L_pl)
        No distingushing between ID and OOD protos. Just pushing apaprt dissimilar protos protos.
        '''
        
        self.id = id
        self.comm = args.comm
        self.device = args.device
        self.is_bayes = args.is_bayes
        self.adj = adj
        self.local_bs = args.local_bs
        self.local_epochs = args.local_epochs
        self.save_folder_name = args.save_folder_name
        self.global_seed = args.global_seed
        # self.w = args.w
        self.grad = []                            #model gradients after localupdate
        self.test_on_cosine = args.test_on_cosine #local test data
        self.feat_dim = args.feat_dim
        self.neighbors_id = None                  #fixed neighbors like Penz
        self.data_type = args.data_type           # if point cloud/image etc
        self.use_imb_loss = args.use_imb_loss
        self.cluster = cluster  #cluster_id (not used)
        self.clients_in_same_cluster = []

        #PENS params
        # self.neighbours_history = np.empty((0, args.num_clients-1), int) #max possible neighbour

        #dataset
        if dataset_test is not None:
            self.train_loader, self.val_loader, self.test_loader = self.train_val_test(
                dataset_train, list(user_data_idx_train), dataset_test, list(user_data_idx_test))
        else:
            self.train_loader, self.val_loader, self.test_loader = self.train_val_test(
                dataset_train, list(user_data_idx_train), None, None)

        self.train_size = len(self.train_loader.dataset)
        self.unc = -1 
        
        self.num_classes = args.num_classes 
        self.sample_per_class = torch.zeros(self.num_classes)
        self.test_sample_per_class = torch.zeros(self.num_classes)
        
        print(f'Cluster:{self.cluster}')
        for x, y, _ in self.train_loader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        print(f'Client id: {self.id}, Train dist: {self.sample_per_class}')
        self.id_labels = [i for i in range(self.num_classes) if self.sample_per_class[i]>0]
        # #print(f'ID Labels: {self.id_labels}')
        
        for x, y,_ in self.test_loader:
            for yy in y:
                self.test_sample_per_class[yy.item()] += 1
        print(f'Client id: {self.id}, Test dist: {self.test_sample_per_class}')

        print(f'Client data count: Train: {len(self.train_loader.dataset)} Test: {len(self.test_loader.dataset)}\n')

        self.ood_labels = [l for l in range(args.num_classes) if l not in self.id_labels]
        # print(f'OOD labels: {self.ood_labels}')

        
        #model
        self.model = copy.deepcopy(args.model)     #its a base head split model /updated here and loaded by average model from outside
        self.model0 = copy.deepcopy(args.model)    #gradient computation origin model
        
        #proto
        self.local_protos = None      #[1e-08 for _ in range(args.num_classes)]
        self.global_protos = None     #[1e-08 for _ in range(args.num_classes)]
        self.aggregated_protos = None #store temp proto for sync
        self.aggregated_kappas = None
        self.LAMBDA = args.LAMBDA
        
        #self.client_protos_set = None # [None for _ in range(self.num_clients)]
        self.normalize = args.normalize
        self.loss_mse = nn.MSELoss(reduction='sum')
        self.loss_cs = nn.CosineSimilarity(dim=1)
        self.loss_CE = nn.CrossEntropyLoss()
        
        #self.lamda = args.lamda
        self.tau = args.tau  #PCL

        #loss lr optimizer
        self.decood_loss_code = args.decood_loss_code
        self.loss_dis = DisLoss(args, self.model, self.train_loader, temperature = args.tau).cuda()
        self.loss_comp = CompLoss(args,temperature = args.tau).cuda()
        self.kappa_hats = self.loss_dis.kappa_hat

        if self.ood_labels:
            self.ood_init_proto = copy.deepcopy(self.loss_dis.prototypes[self.ood_labels[0]])
            # print(f'OOD Template: {self.ood_init_proto}')   
        
        
        self.learning_rate_decay = args.learning_rate_decay
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum = args.momentum, nesterov = True, weight_decay = args.weight_decay)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay
        )


        #aggregation params
        self.req_avg = False
        self.n_k = 1
        self.agg_count = 1 #total nodes below this from where it was updated

        #performance

        self.unc = 0
        self.l_test_acc_hist = []
        self.l_test_auc_hist = []
        self.l_test_unc_hist = []
        self.g_test_acc_hist = []
        self.g_test_auc_hist = []
        self.g_test_unc_hist = []

        

    def train_val_test(self, dataset_train, train_idxs, dataset_test, test_idxs):
        '''
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        '''
        random.shuffle(train_idxs)   #results are averaged
        
        if dataset_test is not None:
            idxs_train = train_idxs[:int(len(train_idxs))]
            # idxs_val = train_idxs[int(0.9*len(train_idxs)):] #10%
            idxs_test = test_idxs #idxs[int(0.9*len(idxs)):]  
        else:
            idxs_train = train_idxs[:int(0.8*len(train_idxs))]
            idxs_val = train_idxs[int(0.8*len(train_idxs)):int(0.9*len(train_idxs))] #10%
            idxs_test = train_idxs[int(0.9*len(train_idxs)):]   #10%


        trainloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_train),
                                 batch_size=self.local_bs, shuffle=True, drop_last=False)
        
        if (dataset_test is not None) and (test_idxs is not None):
            validloader = None
            print('No validation set.')
            testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, idxs_test),
                                batch_size=self.local_bs, shuffle=False, drop_last=False)
        else:
            validloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_val),
                                 batch_size=self.local_bs, shuffle=False, drop_last=True)
        
            testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_test),
                                batch_size=self.local_bs, shuffle=False, drop_last=True)

        
        return trainloader, validloader, testloader
    



    def update(self):
        trainloader = self.train_loader

        self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        

        # if self.comm == 'comm_decood_w':
        #     max_local_epochs = max(1, int(self.id%self.local_epochs))        
        #     # max_local_epochs = np.random.randint(1, max_local_epochs)        
        #     print(f'Local epoch: {max_local_epochs}***')
        # else:
        #     print(f'Local epoch: {max_local_epochs}')

        print(f'Local epoch: {max_local_epochs}')


        # if self.train_slow:
        #     max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        avg_comp_loss = 0
        avg_dis_loss = 0
        avg_CE_loss = 0
        avg_lp_loss = 0

        for epoch in range(max_local_epochs):
            
            # for i, (x, y) in enumerate(trainloader):
            for (x, y, indices) in trainloader:

                x = x[0]
                # print(x.shape,y.shape)
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                if torch.isnan(x).any():
                    print(f"NaN in input batch {i}")
                

                rep = self.model.base(x)

                if torch.isnan(rep).any() or torch.isinf(rep).any():
                    print("NaN or Inf detected in base model output!")

                if self.normalize:
                    rep = F.normalize(rep+1e-8, dim=1)    
                
                    
                  
                output = self.model.head(rep)
                if self.use_imb_loss:
                    loss_CE = balanced_softmax_loss(y, output, self.sample_per_class)
                else:
                    loss_CE = self.loss_CE(output, y)  #CE loss

                #ld = self.loss_dis(rep,y,self.id_labels)

                ld = self.loss_dis(rep,y, self.loss_dis.prototypes)
                lc = self.loss_comp(rep,self.loss_dis.prototypes,y) #compute prototypes and normalize
                #lc = self.loss_comp(rep,self.local_protos,y)
                
                
                # #fedproto                
                # lp = 0
                # if self.global_protos is not None:
                #     #make space for new protos
                #     proto_new = copy.deepcopy(rep.detach())
                #     #y is batch, for every sample(i), get global proto(representative) of that class(as new proto) 
                #     for i, yy in enumerate(y):
                #         y_c = yy.item()
                #         if type(self.global_protos[y_c]) != type([]):
                #             proto_new[i, :] = self.global_protos[y_c].data 
                #     lp += self.loss_mse(proto_new, rep)   #diff bet global rep of every(ith) sample and current rep(proto)
                
                

                if self.decood_loss_code =='CD':
                    loss =   lc + ld     #default CD
                elif self.decood_loss_code =='ECD':
                    loss =  loss_CE + self.LAMBDA *(lc + ld)  #0,0.2,0.4
                elif self.decood_loss_code =='ECP':
                    loss =  loss_CE + .2 * lc + .1* lp
                elif self.decood_loss_code =='EC':
                    loss = loss_CE + self.LAMBDA * lc
                    

                avg_CE_loss += loss_CE.data
                avg_comp_loss += lc.data
                avg_dis_loss += ld.data                
                # avg_lp_loss +=lp

                # lce_hist.append(avg_CE_loss.item())
                # lc_hist.append(avg_comp_loss.item())
                # lc_hist.append(avg_dis_loss.item())
                
                if epoch == max_local_epochs-1:
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                
                # print(f'Batch loss : LCE:{loss_CE.data}  Lc: {lc.data}   Ld: {ld.data}   Lp: {lp}')

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()


        avg_CE_loss = avg_CE_loss/ (len(self.train_loader.dataset) * max_local_epochs)
        avg_comp_loss = avg_comp_loss / (len(self.train_loader.dataset) * max_local_epochs)
        avg_dis_loss = avg_dis_loss / (len(self.train_loader.dataset) * max_local_epochs)
        avg_lp_loss = avg_lp_loss / (len(self.train_loader.dataset) * max_local_epochs)

        #print(f'Epoch loss : LCE:{avg_CE_loss/len(self.train_loader.dataset)} Ll: {avg_comp_loss/len(self.train_loader.dataset)}   Lg: {avg_dis_loss/len(self.train_loader.dataset)}')
        print(f'Client {self.id} Avg Training : LCE:{avg_CE_loss}  LC: {avg_comp_loss}   LD: {avg_dis_loss}   LP: {avg_lp_loss}')
        
        
        #compute representative protos from collected rep of train samples
        self.local_protos = agg_func(self.normalize, protos, self.num_classes, self.device) #get normalized aggregated proto  EDOnt normalize in agg if rep is already norm here
        
        # Update OOD class prototypes from DisLoss prototypes if available
        for ood_class in self.ood_labels:
            ood_proto = self.loss_dis.prototypes[ood_class]
            if not torch.equal(ood_proto, self.ood_init_proto):  # it has been updated from default value by neighbor client
                # print(f'OOD proto updated for class {ood_class} from neighbors.')
                self.local_protos[ood_class] = ood_proto.clone().detach()
        
        self.kappa_hats = self.loss_dis.kappa_hat #classwise uncertainty of prototypes

        #compute gradient
        # W0 = [tens.detach() for tens in list(self.model0.head.parameters())] #weight now
        # Wt = [tens.detach() for tens in list(self.model.head.parameters())] #previous weight
        # self.grad = [wc - wp for wc,wp in zip(W0,Wt)] #grad respect to origin


    

    def count_params(self, prototypes=None):
        """
        Counts the number of parameters to be sent if only prototypes are shared.

        Parameters:
            prototypes (dict): A dictionary of class-wise prototypes, where each entry 
                            prototypes[class_id] is a tensor representing the prototype 
                            for that class.

        Returns:
            int: Total number of prototype parameters to be sent.
        """
        model_params = sum(p.numel() for p in self.model.parameters())
        prototypes = self.local_protos
        
        total_proto_params = 0
        
        for proto in prototypes.values():
            # Count the number of parameters in each prototype (embedding size)
            total_proto_params += proto.numel()
            
        return total_proto_params, model_params



    
    def init_local_proto(self):
        trainloader = self.train_loader

        # self.model.to(self.device)
        self.model.eval()

        max_local_epochs = self.local_epochs

        protos = defaultdict(list)

        for epoch in range(max_local_epochs):

            # for i, (x, y) in enumerate(trainloader):
            for (x, y, indices) in trainloader:
                x = x[0]
                
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                
                if self.normalize:
                    rep = F.normalize(rep, dim=1)                                                          #Close it if normalize in the loss ###################************
                    
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        #compute representative protos from collected rep of train samples
        self.local_protos = agg_func(self.normalize, protos,self.num_classes, self.device)                 #Normalized

    

    def save_aggregated_protos(self, global_protos):
        self.aggregated_protos = global_protos

    def save_aggregated_kappas(self, global_kappas):
        self.aggregated_kappas = global_kappas



    # def set_global_protos(self, global_protos):
    #     self.global_protos = global_protos
    

    # def set_dis_loss_protos(self,protos):
    #     for c in range(self.num_classes):
    #         self.loss_dis.prototypes[c] = protos[c].data



    def set_global_protos(self):
        self.global_protos = self.aggregated_protos
    

    def set_dis_loss_protos(self):
        for c in range(self.num_classes):
            # print(c)
            # if self.id==2 or self.id==6:
            #     print(f'Class: {c} __prototype: {self.loss_dis.prototypes[c]}')

            self.loss_dis.prototypes[c] = self.aggregated_protos[c].data


    def print_protos(self):
        # print('Global aggregated proto:\n')
        # for c in range(self.num_classes):
        #     print(f'Class: {c} , prototype: {self.global_protos[c]}')

        print('Dis loss proto:\n')
        for c in range(self.num_classes):
            print(f'Class: {c} , prototype: {self.loss_dis.prototypes[c]}')

    
 
    
    def avg_model(self,run_agg_model_sd,run_agg_n_k):
        #Ni = np.sum(adj[self.id,:]>0)+1 #since (i,i)=0, to count self(+1)
        for key in run_agg_model_sd.keys():
            run_agg_model_sd[key] /= run_agg_n_k
        #self.agg_count = 1

        self.model.load_state_dict(run_agg_model_sd)
        #print(f'Client {self.id} averaged.')
        self.req_avg = False




    #can we you do a copy
    def agg(self, run_agg_model_sd, run_agg_n_k, rec_params):
       
        # approach 1
        rec_model_sd = rec_params[0]
        run_agg_n_k += rec_params[1] #rec n_k

        for key in run_agg_model_sd.keys():
            # if (consensus_all_pair_err_min==True and j>i): #dist of i<=j has been computed
            #      client_wise_dist += torch.norm(clients[j].model.state_dict()[key] - clients[i].model.state_dict()[key]) #diversity
            
            #run_agg_model_sd[key] *= len(self.train_loader.dataset)/m_t
            #run_agg_model_sd[key] += rec_model[key]* (rec_dataset_len/m_t)
            run_agg_model_sd[key] += rec_model_sd[key]
            #run_agg_model_sd[key] /= 2

        #self.model.load_state_dict(run_agg_model_sd)
        #self.agg_count += 1 #aggregated from one more node
        #self.req_avg = True


        #print(f'Aggregation done at client {self.id}')
        return run_agg_model_sd, run_agg_n_k
    
    


    # Function to perform DFS on the graph
    def get_mst_agg_model(self, visited, clients, adj):                         
        # Set current node as visited
        visited[self.id] = True

        run_agg_model_sd = copy.deepcopy(self.model.state_dict()) 
        run_agg_n_k = copy.deepcopy(len((self.train_loader)))

        # For every node of the graph
        for i in range(len(visited)):
            if (adj[self.id][i] == 1 and (not visited[i])):
                 run_agg_model_sd, run_agg_n_k = self.agg(run_agg_model_sd, run_agg_n_k, clients[i].get_mst_agg_model(visited,clients,adj))

    
        #if not aggregated than selfcopy in runn_agg is returned         
        return run_agg_model_sd, run_agg_n_k




    def performance_test(self, data_loader=None, saveFlag=0):
        if data_loader is None:
            data_loader = self.test_loader

        self.model.eval()

        test_acc = 0
        test_num = 0
        results = []  # To store predictions and uncertainties

        if self.local_protos is not None:
            with torch.no_grad():
                for (x, y, indices) in data_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)

                    if self.normalize:
                        rep = F.normalize(rep, dim=1)  # Normalize if required

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in self.local_protos.items():
                            output[i, j] = F.cosine_similarity(r.unsqueeze(0), pro.unsqueeze(0))

                    predictions = torch.argmax(output, dim=1)  # Get predicted class indices
                    test_acc += (torch.sum(predictions == y)).item()
                    test_num += y.shape[0]

                    if saveFlag == 1:
                        # Compute probabilities (softmax)
                        probabilities = F.softmax(output / self.tau, dim=1)  # Adjust with temperature if needed

                        # Compute entropy-based uncertainty for each prediction
                        for i in range(y.shape[0]):
                            predicted_class = predictions[i].item()
                            
                            # Get the predicted class probability
                            predicted_probability = probabilities[i, predicted_class].item()

                            # Calculate entropy (Uncertainty)
                            entropy = -torch.sum(probabilities[i] * torch.log(probabilities[i] + 1e-8)).item()  # Add epsilon for numerical stability

                            # Store results
                            results.append({
                                "sample_index": indices[i].item(),
                                "true_label": y[i].item(),
                                "predicted_label": predicted_class,
                                "predicted_probability": predicted_probability,
                                "uncertainty": entropy
                            })

            # Save predictions and uncertainties to a file if saveFlag is set
            if saveFlag == 1:
                client_id = self.id if hasattr(self, 'id') else 'unknown_client'
                save_path = f"predictions_uncertainties_client_{client_id}.json"
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Predictions and uncertainties saved to {save_path}")
                print(f'Saved uncertainties of client {client_id}')    

            return test_acc / test_num, 0, 0
        else:
            return 0, 1e-5, 0



    def get_local_unc(self):
        trainloader = self.train_loader

        # self.model.to(self.device)
        self.model.eval()

        max_local_epochs = self.local_epochs

        protos = defaultdict(list)
        kappa_hat = defaultdict(list)
        R_hat = defaultdict(list)


        for epoch in range(max_local_epochs):

            # for i, (x, y) in enumerate(trainloader):
            for (x, y, indices) in trainloader:
                x = x[0]

                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                
                if self.normalize:
                    rep = F.normalize(rep, dim=1)                                           #Close it if normalize in the loss ###################
                    
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        #compute representative protos from collected rep of train samples
        new_protos = agg_func( False, protos,self.num_classes, self.device)                 #Compute kappa from unnormalized protos
        # print(f'mus: {new_protos[0]}')
        
        for [label, proto] in protos.items():
            # print(proto)
            R_hat[label] = torch.norm(proto).detach()
            R_hat_sqr = R_hat[label]*R_hat[label]
            kappa_hat[label] = (R_hat[label] * (self.feat_dim - R_hat_sqr)) / (1 - R_hat_sqr) 
            print(f'Label: {label}, R_hat[label]: {R_hat[label].detach()}, kappa_hat[label]: {kappa_hat[label].detach()}')
        
        return kappa_hat 
        

        
    
    def get_mis_classification(self, rec_label=None, rec_proto=None):
        '''
        Evaluate the received prototype on local training dataset and compute the uncertainty
        We replace local prototype of received label with received prototype and evaluate  
        '''
        data_loader = self.train_loader

        self.model.eval()

        test_loss = 0
        test_num = 0
        
        if self.local_protos is not None:
            #create space
            tmp_local_protos = copy.deepcopy(self.local_protos)
            if rec_label is not None:
                tmp_local_protos[rec_label] = rec_proto
                print(f'Testing received proto of label :{rec_label} @ client : {self.id} ')
            else:
                print('Evaluating misclassification on local protos')

            # print('Normalize before testing ID')
            with torch.no_grad():
                # for x, y in data_loader:
                for (x, y, indices) in data_loader:
                    x = x[0]
                    
                    x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)

                    if self.normalize:
                        rep = F.normalize(rep, dim=1)                                                #Close it if normalize in the loss ###################*************


                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device) #all one
                    for i, r in enumerate(rep):
                        for j, pro in tmp_local_protos.items():  #global_protos
                            if type(pro) != type([]):
                                if self.test_on_cosine == False:
                                    output[i, j] = self.loss_mse(r, pro)
                                else: 
                                    output[i, j] = F.cosine_similarity(r.unsqueeze(0), pro.unsqueeze(0))
                                    # ed = self.loss_mse(r, pro)

                    test_loss += (torch.sum(torch.argmin(output, dim=1) != y)).item()  #argmin returns index
                    test_num += y.shape[0]

            return test_loss/test_num




    def performacne_train(self):
        trainloader = self.load_train_data()
    
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            # for x, y in trainloader:
            for (x, y, indices) in trainloader:
                x = x[0]
                
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                if self.normalize:
                    rep = F.normalize(rep, dim=1)        #Close it if normalize in the loss ###################


                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

     
        return losses, train_num




def agg_func(normalize, protos, num_classes = 10, device = 'cpu'):
    """
    Returns the average of the weights.
    """
    #print(len(protos[0][0]))
    # unnorm_agg_proto = torch.zeros([num_classes, len(protos[0][0])],dtype = torch.float).to(device)
    # print(f'Shape Unnorm Protos: {unnorm_agg_proto.shape}')
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            
            if normalize:
                protos[label] = F.normalize((proto / len(proto_list)) , dim=0) #proto / len(proto_list)
            else:
                protos[label] = proto / len(proto_list)                    #################****
        
        else:
            if normalize:
                protos[label] = F.normalize(proto_list[0] , dim=0) #proto_list[0]   #################***
            else:
                protos[label] = proto_list[0]
    
    return protos