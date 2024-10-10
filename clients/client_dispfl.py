import os
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from lib.data_utils import DatasetSplit
from data.provider import * 
from sklearn.preprocessing import label_binarize



class ClientDisPFL():
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
        self.tmp_mask = None
        self.aggregated_model = None
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

        
        #model
        self.model = copy.deepcopy(args.model)     #its a base head split model /updated here and loaded by average model from outside
        self.model0 = copy.deepcopy(args.model)    #gradient computation origin model
        

        #PENS params
        self.anneal_factor = 0.5  # Example value for cosine annealing
        self.mask = initialize_mask(self.model, sparsity=0.5)  # 50% sparsity
        self.num_rounds = args.num_rounds



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
        #trainloader = self.load_train_data()        
        #start_time = time.time()
        
        
        #count initial params
        total_params = count_total_params(self.model)

        self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_epochs
        #if self.train_slow:
        #    max_local_steps = np.random.randint(1, max_local_steps // 2)

        mean_correct_pc = []
        for step in range(max_local_steps):
            for i, (x, y) in enumerate(self.train_loader):
                
                x = x.to(self.device)
                y = y.to(self.device)
                # output = self.model(x)

                output = self.model(x)                 
                loss = self.loss(output, y)
                                
                pred_choice = output.data.max(1)[1]
                correct = pred_choice.eq(y.long().data).cpu().sum()
                mean_correct_pc.append(correct.item() / float(x.size()[0]))

                self.optimizer.zero_grad()
                loss.backward()
                #apply mask to gradients
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in self.mask:
                            param.grad *= self.mask[name].to(self.device)

                self.optimizer.step()

        #update mask (updates mask and saves in tmp_mask) 
        #tmp_mask is loaded in mask synchronously at the end of each round
        self.update_mask(round = 0)


        # Count non-zero parameters in the aggregated model(tmp_mask)
        #next round this mask will be used for aggregation(sent to other clients)
        total_non_zero_params = count_nonzero_params(self.model, self.tmp_mask)
        # Track reduction in parameters
        params_reduction = total_params - total_non_zero_params
        print(f"Parameters reduced: {params_reduction} / {total_params} ({(params_reduction / total_params) * 100:.2f}% reduction)")


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        


    def update_mask(self, round):
        # Prune weights (e.g., based on weight magnitude and cosine annealing)
        drop_ratio = self.anneal_factor / 2 * (1 + math.cos((round * math.pi) / self.num_rounds))
        new_mask = copy.deepcopy(self.mask)
        num_remove = {}

        for name, param in self.model.named_parameters():
            if name in self.mask:
                mask_device = param.device
                self.mask[name] = self.mask[name].to(mask_device)  # Move mask to the same device as param
                num_non_zero = torch.sum(self.mask[name])
                num_remove[name] = math.ceil(drop_ratio * num_non_zero)

                temp_weights = torch.where(self.mask[name] > 0, torch.abs(param), torch.tensor(float('inf'), device=mask_device))
                _, idx = torch.sort(temp_weights.view(-1))
                new_mask[name].view(-1)[idx[:num_remove[name]]] = 0  # Set the smallest values to 0 (prune)

        # Regrow weights (e.g., based on gradient magnitude)
        self.regrow_mask(new_mask, num_remove)

        # Update mask with new pruning/regrowth
        self.tmp_mask = new_mask


    def regrow_mask(self, mask, num_remove):
        for name, param in self.model.named_parameters():
            if name in self.mask:
                gradient = param.grad
                # Ensure mask is on the same device as gradient
                mask[name] = mask[name].to(self.device)
                gradient = gradient.to(self.device)

                # Perform the regrowth operation
                temp = torch.where(mask[name] == 0, torch.abs(gradient), torch.tensor(float('-inf')).to(self.device))
                _, idx = torch.sort(temp.view(-1), descending=True)
                mask[name].view(-1)[idx[:num_remove[name]]] = 1  # Set the largest gradients back to 1 (regrow)


    def get_model_and_mask(self):
        return {name: param.cpu().clone() for name, param in self.model.named_parameters()}, self.mask


    def set_model_and_mask(self):
        self.model.load_state_dict(self.aggregated_model)
        self.mask = self.tmp_mask #updated mask stored in tmp_mask
    

    def aggregate_models(self, client_updates, client_masks):
        # Averaging the updates from clients (only the non-masked parameters)
        aggregated_model = copy.deepcopy(self.model.state_dict())

        for key in aggregated_model.keys():
            sum_weights = torch.zeros_like(aggregated_model[key], device=self.device)
            sum_masks = torch.zeros_like(aggregated_model[key], device=self.device)

            for i, client_update in enumerate(client_updates):
                # Ensure both the model updates and masks are on the correct device
                client_update[key] = client_update[key].to(self.device)
                client_masks[i][key] = client_masks[i][key].to(self.device)

                # Apply mask to model update
                sum_weights += client_update[key] * client_masks[i][key]
                sum_masks += client_masks[i][key]

            # Aggregate only the masked parts
            valid_mask = sum_masks > 0
            aggregated_model[key][valid_mask] = sum_weights[valid_mask] / sum_masks[valid_mask]

        self.aggregated_model = aggregated_model
        


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

                output = self.model(x)

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

                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                tmp_out = F.softmax(output,dim=1)
                entr = (-(tmp_out.mul(tmp_out.add(eps).log())).sum(dim=1)) ############### global
                unc += sum(entr) 
                
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
                
        
        print(f'Mean pc style test performance: {np.mean(mean_correct_pc)}')
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


def initialize_mask(model, sparsity=0.5):
    mask = {}
    for name, param in model.named_parameters():
        mask[name] = torch.zeros_like(param)
        dense_numel = int((1 - sparsity) * param.numel())
        if dense_numel > 0:
            temp = mask[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
    return mask


# Function to count total parameters in a model
def count_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# Function to count non-zero parameters in a model given the mask
def count_nonzero_params(model, mask):
    non_zero_params = 0
    for name, param in model.named_parameters():
        if name in mask:
            non_zero_params += torch.sum(mask[name] > 0).item()
    return non_zero_params
