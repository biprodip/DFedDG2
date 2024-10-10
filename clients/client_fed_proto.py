import os
import copy
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


from sklearn import metrics
from lib.data_utils import DatasetSplit
from sklearn.preprocessing import label_binarize


class ClientFedProto():
    def __init__(self, id, args, adj, dataset_train, user_data_idx_train, dataset_test=None, user_data_idx_test=None, cluster=None):

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
        #self.neighbours_history = np.empty((0, args.num_clients-1), int) #max possible neighbour
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
        
        #proto
        self.local_protos = None #[1e-08 for _ in range(args.num_classes)]
        self.global_protos = None #[1e-08 for _ in range(args.num_classes)]
        self.loss_mse = nn.MSELoss()        
        self.lamda_fed_proto = args.lamda_fed_proto

        #loss lr optimizer
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = args.lr
        self.learning_rate_decay = args.learning_rate_decay
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
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
        
        self.model.train()
        max_local_steps = self.local_epochs
        
        tmp_protos = defaultdict(list)
        for step in range(max_local_steps): #local epochs

            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                #if global proto available
                if self.global_protos is not None:
                    #make space for new protos
                    proto_new = copy.deepcopy(rep.detach())
                    #y is batch, for every sample(i), get global proto(representative) of that class(as new proto) 
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data 
                    loss += self.loss_mse(proto_new, rep) * self.lamda_fed_proto   #diff bet global rep of every(ith) sample and current rep(proto)

                
                #store all rep of every class y_c(c_j) to create new avg(class wise global rep) 
                if step == max_local_steps -1:
                    for i, yy in enumerate(y): 
                        y_c = yy.item()
                        tmp_protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                #print(f'Loss: {loss.data}')
                loss.backward()
                self.optimizer.step()

        self.local_protos = agg_protos(tmp_protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()




    def update_ood(self):
        
        self.model.train()
        max_local_steps = self.local_epochs
        
        tmp_protos = defaultdict(list)
        
        # protos_array = numpy.zeros([1,args.self.num_classes]) #dist frm a emb to all ood protos
        # for y in range(self.num_classes):


        for step in range(max_local_steps): #local epochs

            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                #if global proto available
                #proto_dist = torch.zeros([1,self.num_classes],dtype=float).cuda() #dist sum of a proto to others
                batch_ood_dist = 0

                if self.global_protos is not None:
                    #make space for new protos
                    proto_new = copy.deepcopy(rep.detach())
                    #y is batch, for every sample(i), get global proto(representative) of that class(as new proto) 
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        d = 0
                        for cl in range(self.num_classes):
                            if y_c != cl and type(self.global_protos[cl]) != type([]):
                                d = d + self.loss_mse(proto_new[i, :], self.global_protos[cl].data)
                        batch_ood_dist = batch_ood_dist + (d/(self.num_classes-1))                               
                    
                    
                    #print(f'Len of rep: {len(rep)}')
                    loss -= (batch_ood_dist/len(rep)) * self.lamda_fed_proto   #diff bet global rep of every(ith) sample and current rep(proto)

                
                #store all rep of every class y_c(c_j) to create new avg(class wise global rep) 
                if step == max_local_steps -1:
                    for i, yy in enumerate(y): 
                        y_c = yy.item()
                        tmp_protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                #print(f'Loss: {loss.data}')

                loss.backward()
                self.optimizer.step()

        self.local_protos = agg_protos(tmp_protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()





        
    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)



    def collect_protos(self):
        self.model.eval()

        tmp_protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    tmp_protos[y_c].append(rep[i, :].detach().data)

        self.local_protos = agg_protos(tmp_protos)


        

    def performance_train(self, data_loader=None):
        "performance on local test data"

        if data_loader is None:
            data_loader = self.train_loader

        if self.req_avg:
            self.avg_model(self.adj,self.agg_count)            

        self.model.eval()


        eps = torch.tensor([1e-16]).to(self.device)
        acc = 0
        #num = 0
        losses = 0
        y_prob = []
        y_true = []
        unc = 0
        
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in data_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                tmp_output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device) #NxC

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            tmp = proto_new[i, :]
                            proto_new[i, :] = self.global_protos[y_c].data
                            tmp_output[i,y_c] = self.loss_mse(self.global_protos[y_c].data, tmp) # acc calculation
                    

                    loss += self.loss_mse(proto_new, rep) * self.lamda
                    acc += (torch.sum(torch.argmin(tmp_output, dim=1) == y)).item()
                
                else:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.local_protos[y_c]) != type([]):
                            tmp = proto_new[i, :]
                            proto_new[i, :] = self.local_protos[y_c].data
                            tmp_output[i,y_c] = self.loss_mse(self.local_protos[y_c].data, tmp) # acc calculation
                    

                    loss += self.loss_mse(proto_new, rep) * self.lamda
                    acc += (torch.sum(torch.argmin(tmp_output, dim=1) == y)).item()

                    
                #train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
            
            loss = losses/len(data_loader.dataset) #avg
            acc = acc/len(data_loader.dataset)     #avg
            auc = -1 #undefined
            unc = -1 #undefined
        
        return loss,acc, auc, unc




    def performance_test(self, data_loader=None):
        if data_loader is None:
            data_loader = self.test_loader

        self.model.eval()

        test_acc = 0
        test_num = 0
        
        if self.global_protos is not None:
            with torch.no_grad():
                for x, y in data_loader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device) #all one
                    if self.global_protos is not None:
                        for i, r in enumerate(rep):
                            for j, pro in self.global_protos.items():  #global_protos
                                if type(pro) != type([]):
                                    output[i, j] = self.loss_mse(r, pro)
                    else:
                        for i, r in enumerate(rep):
                            for j, pro in self.local_protos.items():  #global_protos
                                if type(pro) != type([]):
                                    output[i, j] = self.loss_mse(r, pro)                    

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()  #argmin returns index
                    test_num += y.shape[0]

            return test_acc/test_num, 0, 0
        else:
            return 0, 1e-5, 0




    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))



    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
        
    

def agg_protos(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            #print(f'len {len(proto_list)}')
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos