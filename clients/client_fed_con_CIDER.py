import os
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from CIDER_losses import *
#import matplotlib.pyplot as plt
from sklearn import metrics
from data_utils import DatasetSplit
from sklearn.preprocessing import label_binarize


class ClientFedConCIDER():
    def __init__(self, id, args, adj, dataset_train, user_data_idx_train, dataset_test, user_data_idx_test, cluster=None):

        self.id = id
        self.device = args.device
        self.is_bayes = args.is_bayes
        self.adj = adj
        self.local_bs = args.local_bs
        self.local_epochs = args.local_epochs
        self.save_folder_name = args.save_folder_name
        self.global_seed = args.global_seed
        self.w = args.w
        self.grad = []
        
        #PENS params
        self.cluster = cluster  #cluster_id (not used)
        self.neighbours_history = np.empty((0, args.num_clients-1), int) #max possible neighbour
        self.clients_in_same_cluster = []

        #dataset
        self.train_loader, self.val_loader, self.test_loader = self.train_val_test(
            dataset_train, list(user_data_idx_train), dataset_test, list(user_data_idx_test))
        self.train_size = len(self.train_loader.dataset) 
        
        self.num_classes = args.num_classes 
        self.sample_per_class = torch.zeros(self.num_classes)
        
        print(f'Cluster:{self.cluster}')
        for x, y in self.train_loader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        print(f'Client id: {self.id}, Sample per class: {self.sample_per_class}, Shape:{self.sample_per_class.shape}')

        
        #model
        self.model = copy.deepcopy(args.model)     #its a base head split model /updated here and loaded by average model from outside
        self.model0 = copy.deepcopy(args.model)    #gradient computation origin model
        
        #proto
        self.local_protos = [None for _ in range(args.num_classes)]
        self.global_protos = [None for _ in range(args.num_classes)]
        #self.client_protos_set = None # [None for _ in range(self.num_clients)]
        self.loss_mse = nn.MSELoss()
        
        self.lamda = args.lamda
        self.tau = args.tau  #PCL

        #loss lr optimizer
        self.loss_dis = DisLoss(args, self.model, self.train_loader, temperature = args.temp).cuda()
        self.loss_comp = CompLoss(args,temperature = args.temp).cuda()
        self.loss_CE = nn.CrossEntropyLoss()
        
        self.learning_rate_decay = args.learning_rate_decay
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum = args.momentum, nesterov = True, weight_decay = args.weight_decay)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
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
        random.shuffle(train_idxs)
        idxs_train = train_idxs[:int(0.8*len(train_idxs))]
        idxs_val = train_idxs[int(0.8*len(train_idxs)):int(0.9*len(train_idxs))] #10%
        idxs_test = test_idxs #idxs[int(0.9*len(idxs)):]   #10%

        trainloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_train),
                                 batch_size=self.local_bs, shuffle=True)
        validloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, idxs_test),
                                batch_size=self.local_bs, shuffle=False)
        return trainloader, validloader, testloader

    
    # def show(self):
        
    #     images,labels = next(iter(self.train_loader))
    #     fig, axes = plt.subplots(figsize=(10,4),ncols=4)
    #     for ii in range(4):
    #         ax = axes[ii]
    #         imshow(images[ii],ax=ax,normalize=False)





    def update(self):
        trainloader = self.train_loader

        self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        

        # if self.train_slow:
        #     max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
        for epoch in range(max_local_epochs):
            
            avg_comp_loss = 0
            avg_dis_loss = 0
            avg_CE_loss = 0

            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
              
                #CE loss computation
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss_CE = self.loss_CE(output, y)  #CE loss
                
                #rep = F.normalize(rep, dim=1)               ####################******************************
                lg = self.loss_dis(rep,y)
                ll = self.loss_comp(rep,self.loss_dis.prototypes,y)
                #loss = ll + lg
                #loss = loss_CE + 0.5*(ll + lg)   #self.w

                loss = loss_CE


                # print(f'Local protos shape: {local_protos_emb.shape}')
                # print(f'Rep shape: {rep.shape}')
                                
                avg_CE_loss += loss_CE.data
                avg_comp_loss += ll.data
                avg_dis_loss += lg.data
                
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            print(f'Epoch loss : LCE:{avg_CE_loss/len(self.train_loader.dataset)} Ll: {avg_comp_loss/len(self.train_loader.dataset)}   Lg: {avg_dis_loss/len(self.train_loader.dataset)}')
        
        
        #compute representative protos from collected rep of train samples
        self.local_protos = agg_func(protos, self.num_classes, self.device)
        #self.local_protos = F.normalize(protos, dim=1)###############******************


        #compute gradient
        W0 = [tens.detach() for tens in list(self.model0.head.parameters())] #weight now
        Wt = [tens.detach() for tens in list(self.model.head.parameters())] #previous weight
        self.grad = [wc - wp for wc,wp in zip(W0,Wt)] #grad respect to origin



    
    def init_local_proto(self):
        trainloader = self.train_loader

        # self.model.to(self.device)
        self.model.eval()

        max_local_epochs = self.local_epochs

        protos = defaultdict(list)

        for epoch in range(max_local_epochs):

            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                rep = F.normalize(rep, dim=1)

                    
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        #compute representative protos from collected rep of train samples
        self.local_protos = agg_func( protos,self.num_classes, self.device)                 #Normalize too
        #self.local_protos = F.normalize(protos, dim=1)###############******************




    def set_protos(self, global_protos):
        self.global_protos = global_protos

    
    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        protos = agg_func(protos, self.num_classes, self.device)
        #protos = F.normalize(protos, dim=1)###############******************

    
    def performance_test(self):
        testloaderfull = self.test_loader

        self.model.eval()

        test_acc = 0
        test_num = 0
        
        if self.global_protos is not None:
            with torch.no_grad():
                for x, y in testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device) #all one
                    for i, r in enumerate(rep):
                        for j, pro in self.local_protos.items():  #global_protos
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()  #argmin returns index
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0



    def performacne_train(self):
        trainloader = self.load_train_data()
    
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
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


def agg_func(protos, num_classes = 10, device = 'cpu'):
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
    
            #unnorm_agg_proto[label,:] = proto / len(proto_list)      #################*****************************************
            #protos[label] = F.normalize(unnorm_agg_proto, dim = 1)   #################*****************************************
            

            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]                            #################*****************************************
        
        #protos[label] = F.normalize(protos[label], dim = 1)

    return protos