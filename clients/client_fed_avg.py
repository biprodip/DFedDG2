import os
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from lib.data_utils import DatasetSplit
from data.provider import * 
from sklearn.preprocessing import label_binarize



class ClientFedAvg():
    def __init__(self, id, args, adj, dataset_train, user_data_idx_train, dataset_test=None, user_data_idx_test=None, cluster=None):

        self.id = id
        self.device = args.device
        self.is_bayes = args.is_bayes
        self.adj = adj
        self.local_bs = args.local_bs
        self.local_epochs = args.local_epochs
        self.save_folder_name = args.save_folder_name
        self.global_seed = args.global_seed
        self.grad = []
        self.data_type = args.data_type

        #PENS params
        self.cluster = cluster  #cluster_id (not used)
        self.neighbours_history = np.empty((0, args.num_clients-1), int) #max possible neighbour
        self.clients_in_same_cluster = []

        #dataset
        if self.data_type == 'pc':
            if dataset_test is not None:
                self.train_loader, self.val_loader, self.test_loader = self.train_val_test_pc(
                    dataset_train, list(user_data_idx_train), dataset_test, list(user_data_idx_test))
            else:
                self.train_loader, self.val_loader, self.test_loader = self.train_val_test_pc(
                    dataset_train, list(user_data_idx_train), None, None)

        else:
            if dataset_test is not None:
                self.train_loader, self.val_loader, self.test_loader = self.train_val_test(
                    dataset_train, list(user_data_idx_train), dataset_test, list(user_data_idx_test))
            else:
                self.train_loader, self.val_loader, self.test_loader = self.train_val_test(
                    dataset_train, list(user_data_idx_train), None, None)

        self.train_size = len(self.train_loader.dataset)
        print(len(self.train_loader.dataset))
        print(len(self.val_loader.dataset))
        print(len(self.test_loader.dataset))
        
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


        
        # print(f'TestLoader :{len(self.test_loader.dataset)}')
        # print(f'Dataset_test length :{len(dataset_test)}')
        # print(f'user_data_idx_test len :{len(user_data_idx_test)}')


        
        #model
        self.model = copy.deepcopy(args.model)     #its a base head split model /updated here and loaded by average model from outside
        self.model0 = copy.deepcopy(args.model)    #gradient computation origin model
        
        #loss lr optimizer
        self.loss = nn.CrossEntropyLoss()
        
        self.learning_rate = args.lr
        self.learning_rate_decay = args.learning_rate_decay
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
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
                                 batch_size=self.local_bs, shuffle=False)
        validloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        
        if dataset_test is not None:
            testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, idxs_test),
                                batch_size=self.local_bs, shuffle=False)
        else:
            testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_test),
                                batch_size=self.local_bs, shuffle=False)

        
        return trainloader, validloader, testloader



    def train_val_test_pc(self, dataset_train, train_idxs, dataset_test, test_idxs):
        '''
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        # '''
        # random.shuffle(train_idxs)
        # if dataset_test is not None:
        #     idxs_train = train_idxs[:int(0.9*len(train_idxs))]
        #     idxs_val = train_idxs[int(0.9*len(train_idxs)):] #10%
        #     idxs_test = test_idxs #idxs[int(0.9*len(idxs)):]  
        # else:
        #     idxs_train = train_idxs[:int(0.8*len(train_idxs))]
        #     idxs_val = train_idxs[int(0.8*len(train_idxs)):int(0.9*len(train_idxs))] #10%
        #     idxs_test = train_idxs[int(0.9*len(train_idxs)):]   #10%


        # trainloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_train),
        #                          batch_size=self.local_bs, shuffle=True)
        # validloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        
        # if dataset_test is not None:
        #     testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, idxs_test),
        #                         batch_size=self.local_bs, shuffle=False)
        # else:
        #     testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_test),
        #                         batch_size=self.local_bs, shuffle=False)

        
        trainloader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=self.local_bs, shuffle=False)


        return trainloader, trainloader, trainloader



    def update(self):
        # if self.req_avg:
        #     self.avg_model(self.adj,self.agg_count)

        #trainloader = self.load_train_data()
        
        #start_time = time.time()

        self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_epochs
        #if self.train_slow:
        #    max_local_steps = np.random.randint(1, max_local_steps // 2)

        mean_correct_pc = []
        for step in range(max_local_steps):
            for i, (x, y) in enumerate(self.train_loader):
                
                #print(x.shape)
                if self.data_type == 'pc':
                    x, y = process_3d_pc(x, y)
                #print(x.shape)
                
                x = x.to(self.device)
                y = y.to(self.device)
                # output = self.model(x)

                if self.data_type == 'pc':
                #    x.requires_grad = True
                #    y.requires_grad = False
                    y = y.long()


                if self.data_type == 'pc':
                    rep,_ = self.model.base(x)
                else:
                    rep = self.model.base(x)

                output = self.model.head(rep)
                
 
                if self.data_type =='pc' or x.shape[-2] in [3, 6]:
                    loss = self.loss(output, y)
                else:
                    loss = self.loss(output, y)
                
                if self.data_type in ['pc','img']:
                    pred_choice = output.data.max(1)[1]
                    correct = pred_choice.eq(y.long().data).cpu().sum()
                    mean_correct_pc.append(correct.item() / float(x.size()[0]))

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        # self.model.cpu()
        if self.data_type in ['pc','img']:
            print(f'Training data perf: {np.mean(mean_correct_pc)}')

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()




    def update_backup(self):
        # if self.req_avg:
        #     self.avg_model(self.adj,self.agg_count)

        #trainloader = self.load_train_data()
        
        #start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_epochs
        #if self.train_slow:
        #    max_local_steps = np.random.randint(1, max_local_steps // 2)


        for step in range(max_local_steps):
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # output = self.model(x)

                rep = self.model.base(x)
                output = self.model.head(rep)
                
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        #self.train_time_cost['num_rounds'] += 1
        #self.train_time_cost['total_cost'] += time.time() - start_time

        #compute gradient
        # W0 = [tens.detach() for tens in list(self.model0.parameters())] #weight now
        # Wt = [tens.detach() for tens in list(self.model.parameters())] #previous weight
        # self.grad = [wc - wp for wc,wp in zip(W0,Wt)] #grad respect to origin
    
    
    
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
        


    def performance_train(self, data_loader):
        'performance on local test data'
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
        
        with torch.no_grad():
            for x, y in data_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                
                # output = self.model(x)
                rep = self.model.base(x)
                output = self.model.head(rep)


                tmp_out = F.softmax(output,dim=1)
                entr = (-(tmp_out.mul(tmp_out.add(eps).log())).sum(dim=1)) ############### global #sum accross every sample (batch size x 1)
                unc += sum(entr) #sum of batch (single value)
                
                loss = self.loss(output, y)
                losses += loss.item() * y.shape[0]
                
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
        
        loss = losses/len(data_loader.dataset) #avg
        acc = acc/len(data_loader.dataset)     #avg
        unc = unc.item()/len(data_loader.dataset)
        #print(type(unc))
        self.unc = unc
        
        return loss,acc, auc, unc




    def performance_test(self, data_loader=None):
        'performance on local test data'
        
        if data_loader is None:
            data_loader = self.test_loader
        
        # if self.req_avg:
        #     self.avg_model(self.adj,self.agg_count)            

        self.model.to(self.device)
        self.model.eval()


        eps = torch.tensor([1e-16]).to(self.device)
        acc = 0
        #num = 0
        losses = 0
        y_prob = []
        y_true = []
        unc = 0
        mean_correct_pc = []        
        with torch.no_grad():
            for x, y in data_loader:

                #print(x.shape)
                if self.data_type == 'pc':
                    x, y = process_3d_pc(x, y)
                #print(x.shape)
                
                x = x.to(self.device)
                y = y.to(self.device)

                if self.data_type == 'pc':
                    y = y.long()


                if self.data_type == 'pc':
                    rep,_ = self.model.base(x)
                else:
                    rep = self.model.base(x)

                output = self.model.head(rep)



                tmp_out = F.softmax(output,dim=1)
                entr = (-(tmp_out.mul(tmp_out.add(eps).log())).sum(dim=1)) ############### global
                unc += sum(entr) 
                
                # loss = self.loss(output, y)
                # losses += loss.item() * y.shape[0]
                
                
                
                if self.data_type in['pc','img']:
                    pred_choice = output.data.max(1)[1]
                    correct = pred_choice.eq(y.data).cpu().sum()
                    mean_correct_pc.append(correct.item() / float(x.size()[0]))                
                
                
                
                acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                y_prob.append(F.softmax(output,dim=1).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
                
        
        print(f'Mean (pc code style) Test performance: {np.mean(mean_correct_pc)}')
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro') #avg
        
        # loss = losses/len(data_loader.dataset) #avg
        acc = acc/len(data_loader.dataset)     #avg
        unc = unc.item()/len(data_loader.dataset)
        #print(type(unc))
        
        return acc, auc, unc


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, 'client_' + str(self.id) + '_' + item_name + '.pt'))


    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, 'client_' + str(self.id) + '_' + item_name + '.pt'))
