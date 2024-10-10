import os
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader

from sklearn import metrics
from utils.losses_decood import *
from lib.data_utils import DatasetSplit
from sklearn.preprocessing import label_binarize


class ClientFedGHProto():
    def __init__(self, id, args, adj, dataset_train, user_data_idx_train, dataset_test=None, user_data_idx_test=None, cluster=None):

        '''
        Lce+Lcomp
        Head: LCE
        '''
        self.id = id
        self.device = args.device
        self.is_bayes = args.is_bayes
        self.adj = adj
        self.local_bs = args.local_bs
        self.proto_bs = 2
        self.local_epochs = args.local_epochs
        self.save_folder_name = args.save_folder_name
        self.global_seed = args.global_seed
        self.grad = []


        self.ood_header_epoch = args.ood_header_epoch
        self.id_header_epoch = args.id_header_epoch
        self.proto_bs_id = args.proto_bs_id
        self.proto_bs_ood = args.proto_bs_ood
        self.normalize = args.normalize




        #PENS params
        self.cluster = cluster  #cluster_id (not used)
        self.neighbours_history = np.empty((0, args.num_clients-1), int) #max possible neighbour
        self.clients_in_same_cluster = []

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
        for x, y in self.train_loader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        print(f'Client id: {self.id}, Train dist: {self.sample_per_class}')
        self.id_labels = [i for i in range(self.num_classes) if self.sample_per_class[i]>0]
        #print(f'ID Labels: {self.id_labels}')
        
        for x, y in self.test_loader:
            for yy in y:
                self.test_sample_per_class[yy.item()] += 1
        print(f'Client id: {self.id}, Test dist: {self.test_sample_per_class}\n')
        
        
        #model
        self.model = copy.deepcopy(args.model)     #its a base head split model /updated here and loaded by average model from outside
        self.model0 = copy.deepcopy(args.model)    #gradient computation origin model
        self.head = self.model.head
        
        self.gh_head = copy.deepcopy(self.model.head)  #GH
        
        #proto
        self.local_protos = None #[None for _ in range(args.num_classes)]
        self.global_protos = None #[None for _ in range(args.num_classes)]

        
        
        #for integrating proto
        #self.loss_mse = nn.MSELoss(reduction='sum')
        
        # self.lamda = args.lamda
        self.tau = args.tau  #PCL

        #loss lr optimizer
        self.loss_dis = DisLoss(args, self.model, self.train_loader, temperature = args.tau).cuda()
        self.loss_comp = CompLoss(args,temperature = args.tau).cuda()
        #for integrating proto


        #loss lr optimizer
        self.loss = nn.CrossEntropyLoss()
        
        self.learning_rate = args.lr
        self.learning_rate_decay = args.learning_rate_decay
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay
        )
        
        self.gh_loss = nn.CrossEntropyLoss()
        self.gh_lr = 0.01
        self.opt_gh = torch.optim.SGD(self.gh_head.parameters(), lr=self.gh_lr) #according to paper

        
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
        if dataset_test is not None:
            idxs_train = train_idxs[:int(0.9*len(train_idxs))]
            idxs_val = train_idxs[int(0.9*len(train_idxs)):] #10%
            idxs_test = test_idxs #idxs[int(0.9*len(idxs)):]  
        else:
            idxs_train = train_idxs[:int(0.8*len(train_idxs))]
            idxs_val = train_idxs[int(0.8*len(train_idxs)):int(0.9*len(train_idxs))] #10%
            idxs_test = train_idxs[int(0.9*len(train_idxs)):]   #10%


        trainloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_train),
                                 batch_size=self.local_bs, shuffle=True)
        validloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        
        if dataset_test is not None:
            testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, idxs_test),
                                batch_size=self.local_bs, shuffle=False)
        else:
            testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_test),
                                batch_size=self.local_bs, shuffle=False)

        
        return trainloader, validloader, testloader




    def update(self):
        trainloader = self.train_loader
        #print(len(trainloader.dataset))

        self.model.to(self.device)
        self.model.train()
        
        max_local_epochs = self.local_epochs
        # if self.train_slow:
        #     max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        avg_comp_loss = 0
        avg_dis_loss = 0
        avg_CE_loss = 0

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # if self.train_slow:
                #     time.sleep(0.1 * np.abs(np.random.rand()))
                # print(x.shape)
                # output = self.model(x)
                # print(type(self.model))
                # print(output.shape)

                # rep = self.model.base(x)
                # output = self.model.head(rep)

                rep = self.model.base(x)

                if self.normalize:
                    rep = F.normalize(rep, dim=1)               #Close it if normalize in the loss ###################******************

                output = self.model.head(rep)
                loss_CE = self.loss(output, y)  #CE loss
                
                ld = self.loss_dis(rep,y)
                lc = self.loss_comp(rep,self.loss_dis.prototypes,y) #compute prototypes and normalize
                
                # loss =  loss_CE + .2 * lc    #best till now!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!********!!!!!!!!!!!!
                loss =  .4 * loss_CE + .6 * lc    #best till now!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!********!!!!!!!!!!!!


                avg_CE_loss += loss_CE.data
                avg_comp_loss += lc.data
                avg_dis_loss += ld.data

                # loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        avg_CE_loss = avg_CE_loss/ (len(self.train_loader.dataset) * max_local_epochs)
        avg_comp_loss = avg_comp_loss / (len(self.train_loader.dataset) * max_local_epochs)
        avg_dis_loss = avg_dis_loss / (len(self.train_loader.dataset) * max_local_epochs)
        #print(f'Epoch loss : LCE:{avg_CE_loss/len(self.train_loader.dataset)} Ll: {avg_comp_loss/len(self.train_loader.dataset)}   Lg: {avg_dis_loss/len(self.train_loader.dataset)}')
        print(f'Avg loss : LCE:{avg_CE_loss} Lc: {avg_comp_loss}   Ld: {avg_dis_loss}')


        #self.train_time_cost['num_rounds'] += 1
        #self.train_time_cost['total_cost'] += time.time() - start_time

        
    
    def train_head(self,uploaded_protos):
        proto_loader = DataLoader(uploaded_protos, self.proto_bs, drop_last=False, shuffle=True)

        for epoch in range(self.id_header_epoch):
            epoch_loss = 0
            for p, y in proto_loader:
                out = self.gh_head(p)
                loss = self.gh_loss(out, y)
                self.opt_gh.zero_grad()
                loss.backward()
                self.opt_gh.step()
                epoch_loss = 0.8 * epoch_loss + .2 * loss.data
            print(f'Head training epoch loss : {epoch_loss}')
    
    
    def set_parameters(self, global_head):
        for new_param, old_param in zip(global_head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()



    def set_global_protos(self, global_protos):
        self.global_protos = global_protos
    

    def set_dis_loss_protos(self,protos):
        for c in range(self.num_classes):
            self.loss_dis.prototypes[c] = protos[c].data



    def train_head_ood(self,uploaded_protos_id, uploaded_protos_ood, ood_train_method):
        proto_loader_id = DataLoader(uploaded_protos_id, self.proto_bs_id, drop_last=False, shuffle=True)
        proto_loader_ood = DataLoader(uploaded_protos_ood, self.proto_bs_ood, drop_last=False, shuffle=True)
        
        self.gh_head.train()
        for epoch in range(self.ood_header_epoch):
            loss_avg = 0.0 
            #for in_set, out_set in zip(proto_loader_id, proto_loader_ood):
            for in_set in proto_loader_id:           #just comment our this and following line
                for out_set in  proto_loader_ood:

                    data = torch.cat((in_set[0], out_set[0]), 0) #we ignore ood_proto labels
                    target = in_set[1]
                    #print(f'ID-OOD data shape: {len(in_set[0])} and {len(out_set[0])}')  #id batch 128, ood_batch 256 total 354 

                    data, target = data.cuda(), target.cuda()
                    #print('data batch loaded')

                    # forward
                    x = self.gh_head(data)      #output 10 classes
                    #print(x.shape)     #384*10

                    # backward
                    #scheduler.step() #original code
                    self.opt_gh.zero_grad()


                    #loss = F.cross_entropy(x[:len(in_set[0])], target)
                    loss = self.gh_loss(x[:len(in_set[0])], target)
                    # cross-entropy from softmax distribution to uniform distribution
                    if ood_train_method == 'energy':
                        if(len(out_set[0])>1):
                            Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1) #along x axis
                        else:
                            Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=0) #along x axis
                        if(len(in_set[0])>1):
                            Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
                        else:
                            Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=0)
                        
                        loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())
                    elif ood_train_method == 'OE':
                        loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

                    loss.backward()
                    self.opt_gh.step()
                    #scheduler.step() #modified code


                    # exponential moving average
                    loss_avg = loss_avg * 0.8 + float(loss) * 0.2
                    # print(f'Avg loss : {loss_avg}')
            print(f'Epoch {epoch} : Loss of ood_head training:{loss_avg}')
    


    def collect_protos(self):
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
                
                if self.normalize:
                    rep = F.normalize(rep, dim=1) #######################

                    
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        #compute representative protos from collected rep of train samples
        self.local_protos = agg_func( self.normalize, protos,self.num_classes, self.device)                 #Normalized



    def performance_test(self, data_loader=None):
        'performance on local test data'
        
        if data_loader is None:
            data_loader = self.test_loader
        
        # if self.req_avg:
        #     self.avg_model(self.adj,self.agg_count)            

        self.model.eval()


        eps = torch.tensor([1e-16]).to(self.device)
        acc = 0
        #num = 0
        losses = 0
        y_prob = []
        y_true = []
        unc = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                
                rep = self.model.base(x)
                output = self.model.head(rep)

                if self.normalize:
                    rep = F.normalize(rep, dim=1)  ############################# since protos are normalized (along x axis)


                tmp_out = F.softmax(output,dim=1)
                entr = (-(tmp_out.mul(tmp_out.add(eps).log())).sum(dim=1)) ############### global
                unc += sum(entr) 
                
                # loss = self.loss(output, y)
                # losses += loss.item() * y.shape[0]
                
                acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                y_prob.append(F.softmax(output,dim=1).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro') #avg
        
        # loss = losses/len(data_loader.dataset) #avg
        acc = acc/len(data_loader.dataset)     #avg
        unc = unc.item()/len(data_loader.dataset)
        #print(type(unc))
        
        return acc, auc, unc



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
                protos[label] = proto / len(proto_list)                    #################*******************************
        
        else:
            if normalize:
                protos[label] = F.normalize(proto_list[0] , dim=0) #proto_list[0]   #################*******************
            else:
                protos[label] = proto_list[0]
    
    return protos