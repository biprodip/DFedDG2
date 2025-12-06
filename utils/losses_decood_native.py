
"""
Aapted from SupCon: https://github.com/HobbitLong/SupContrast/
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import defaultdict


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).to(args.device)
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output






class CompLoss(nn.Module):
    '''
    Compactness Loss with class-conditional prototypes
    '''
    def __init__(self, args, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels):
        '''receiving normalized losses fro  disloss'''
        #prototypes = F.normalize(prototypes, dim=1) 
        
        proxy_labels = torch.arange(0, self.args.num_classes).to(self.args.device) #[0,...9]
        labels = labels.contiguous().view(-1, 1) #[2,5,3] to [[2],[5],[3]] 

        # print(proxy_labels.shape, (proxy_labels.T).shape, proxy_labels.permute(*torch.arange(proxy_labels.ndim - 1, -1, -1)).shape)
        mask = torch.eq(labels, proxy_labels.permute(*torch.arange(proxy_labels.ndim - 1, -1, -1))).float().to(self.args.device) #bz x cls    # [[0010000000],[0000010000],[0001000000]]

        # compute logits
        feat_dot_prototype = torch.div(
            torch.matmul(features, prototypes.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss




class DisLoss(nn.Module):
    '''
    Dispersion Loss with EMA prototypes
    '''
    def __init__(self, args, model, loader, temperature= 0.1, base_temperature=0.1):
        super(DisLoss, self).__init__()
        self.args = args
        self.epsilon = 1e-08                                        ##################***********
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer("prototypes", torch.zeros(self.args.num_classes,self.args.feat_dim))
        self.model = model
        self.loader = loader
        self.kappa_hat = defaultdict(list)
        self.R_hat = defaultdict(list)
        self.init_class_prototypes()



    def forward(self, features, labels, prototypes):    

        #prototypes = self.prototypes
        
        num_cls = self.args.num_classes
        for j in range(len(features)):
           
            
            tmp_proto = prototypes[labels[j].item()] * self.args.proto_m + features[j] * (1 - self.args.proto_m)
            self.R_hat[labels[j].item()] = torch.norm(tmp_proto).detach()

            # Compute full kappa formula
            R_hat = self.R_hat[labels[j].item()]
            R_hat_sqr = R_hat * R_hat
            self.kappa_hat[labels[j].item()] = (
                R_hat * (self.args.feat_dim - R_hat_sqr) / (1 - R_hat_sqr)
                if R_hat < 0.999 else 1e6  # Handle edge case
            )

            prototypes[labels[j].item()] = F.normalize(tmp_proto, dim=0)

            # #******** 
            # tmp_proto = prototypes[labels[j].item()] * self.args.proto_m + features[j]*(1-self.args.proto_m)
            # self.R_hat[labels[j].item()] =  torch.norm(tmp_proto).detach()
            # # R_hat_sqr = self.R_hat[labels[j].item()] * self.R_hat[labels[j].item()]
            # # kappa_hat[labels[j].item()] = (self.R_hat[labels[j].item()] * (self.feat_dim - R_hat_sqr)) / (1 - R_hat_sqr) 
            # self.kappa_hat[labels[j].item()] = self.R_hat[labels[j].item()] 

            # # print(label, R_hat[labels[j].item()], kappa_hat[labels[j].item()])

            # prototypes[labels[j].item()] = F.normalize(tmp_proto,dim=0)
            # #********

            # # prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] *self.args.proto_m + features[j]*(1-self.args.proto_m), dim=0)  #happens for local labels


        self.prototypes = prototypes.detach()

        labels = torch.arange(0, num_cls).to(self.args.device)     #all labels
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.T).float()).to(self.args.device)  ###############mT


        logits = torch.div(
            torch.matmul(prototypes, prototypes.T),
            self.temperature)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_cls).view(-1, 1).to(self.args.device),
            0
        )
        mask = mask * logits_mask
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss



    def init_class_prototypes(self):
        """Initialize class prototypes"""
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.args.num_classes
        with torch.no_grad():
            prototypes = torch.zeros(self.args.num_classes,self.args.feat_dim).to(self.args.device)       ####unknown protos will be zeros
            # for i, (input, target) in enumerate(self.loader):
            for i,(input, target) in enumerate(self.loader):
            
                # input = input[0]
                input, target = input.to(self.args.device), target.to(self.args.device)
                #print(f'{input.shape}')
                features = self.model.base(input)  #self.model(input)
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            
            
            for cls in range(self.args.num_classes):
                if prototype_counts[cls] > 0:
                    prototypes[cls] /= prototype_counts[cls]
                    self.R_hat[cls] = torch.norm(prototypes[cls]).detach()

                    # Compute full kappa formula
                    R_hat = self.R_hat[cls]
                    R_hat_sqr = R_hat * R_hat
                    self.kappa_hat[cls] = (
                        R_hat * (self.args.feat_dim - R_hat_sqr) / (1 - R_hat_sqr)
                        if R_hat < 0.999 else 1e6  # Handle edge case
                    )
                else:
                    prototypes[cls] += self.epsilon
                    self.R_hat[cls] = torch.norm(prototypes[cls]).detach()
                    self.kappa_hat[cls] = 0  # No reliable concentration

            
            
            # for cls in range(self.args.num_classes):
            #     if prototype_counts[cls] > 0:                  ###################***********
            #         prototypes[cls] /=  prototype_counts[cls]
                    
            #         #kappa computation
            #         self.R_hat[cls] =  torch.norm(prototypes[cls]).detach()
            #         # R_hat_sqr = self.R_hat[cls]*self.R_hat[cls]
            #         # self.kappa_hat[cls] = (self.R_hat[cls] * (self.feat_dim - R_hat_sqr)) / (1 - R_hat_sqr) 
            #         self.kappa_hat[cls] = self.R_hat[cls]
            #     else:    
            #         prototypes[cls] += self.epsilon                 ###################***********
                    
            #         #kappa computation
            #         self.R_hat[cls] =  torch.norm(prototypes[cls]).detach()
            #         # R_hat_sqr = self.R_hat[cls]*self.R_hat[cls]
            #         # self.kappa_hat[cls] = (self.R_hat[cls] * (self.feat_dim - R_hat_sqr)) / (1 - R_hat_sqr) 
            #         self.kappa_hat[cls] = self.R_hat[cls]


            # measure elapsed time
            duration = time.time() - start
            # print(f'Time to initialize prototypes: {duration:.3f}')
            
            #if self.args.normalize:
            prototypes = F.normalize(prototypes, dim=1)

            self.prototypes = prototypes





# class CompLoss(nn.Module):
#     '''
#     Compactness Loss with class-conditional prototypes
#     '''
#     def __init__(self, args, temperature=0.07, base_temperature=0.07):
#         super(CompLoss, self).__init__()
#         self.args = args
#         self.temperature = temperature
#         self.base_temperature = base_temperature

#     def forward(self, features, prototypes, labels):
#         '''receiving normalized losses fro  disloss'''
#         #prototypes = F.normalize(prototypes, dim=1) 
        
#         proxy_labels = torch.arange(0, self.args.num_classes).to(self.args.device) #[0,...9]
#         labels = labels.contiguous().view(-1, 1) #[2,5,3] to [[2],[5],[3]] 

#         # print(proxy_labels.shape, (proxy_labels.T).shape, proxy_labels.permute(*torch.arange(proxy_labels.ndim - 1, -1, -1)).shape)
#         mask = torch.eq(labels, proxy_labels.permute(*torch.arange(proxy_labels.ndim - 1, -1, -1))).float().to(self.args.device) #bz x cls    # [[0010000000],[0000010000],[0001000000]]

#         # compute logits
#         feat_dot_prototype = torch.div(
#             torch.matmul(features, prototypes.T),
#             self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
#         logits = feat_dot_prototype - logits_max.detach()

#         # compute log_prob
#         exp_logits = torch.exp(logits) 
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) 

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
#         return loss




# class DisLoss(nn.Module):
#     '''
#     Dispersion Loss with EMA prototypes
#     '''
#     def __init__(self, args, model, loader, temperature= 0.1, base_temperature=0.1):
#         super(DisLoss, self).__init__()
#         self.args = args
#         self.epsilon = 1e-08                                        ##################***********
#         self.temperature = temperature
#         self.base_temperature = base_temperature
#         self.register_buffer("prototypes", torch.zeros(self.args.num_classes,self.args.feat_dim))
#         self.model = model
#         self.loader = loader
#         self.kappa_hat = defaultdict(list)
#         self.R_hat = defaultdict(list)
#         self.init_class_prototypes()



#     def forward(self, features, labels):    

#         prototypes = self.prototypes
#         num_cls = self.args.num_classes
#         for j in range(len(features)):
           
#             #******** 
#             tmp_proto = prototypes[labels[j].item()] * self.args.proto_m + features[j]*(1-self.args.proto_m)
#             self.R_hat[labels[j].item()] =  torch.norm(tmp_proto).detach()
#             # R_hat_sqr = self.R_hat[labels[j].item()] * self.R_hat[labels[j].item()]
#             # kappa_hat[labels[j].item()] = (self.R_hat[labels[j].item()] * (self.feat_dim - R_hat_sqr)) / (1 - R_hat_sqr) 
#             self.kappa_hat[labels[j].item()] = self.R_hat[labels[j].item()] 

#             # print(label, R_hat[labels[j].item()], kappa_hat[labels[j].item()])

#             prototypes[labels[j].item()] = F.normalize(tmp_proto,dim=0)
#             #********

#             # prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] *self.args.proto_m + features[j]*(1-self.args.proto_m), dim=0)  #happens for local labels



#         self.prototypes = prototypes.detach()

#         labels = torch.arange(0, num_cls).to(self.args.device)     #all labels
#         labels = labels.contiguous().view(-1, 1)

#         mask = (1- torch.eq(labels, labels.T).float()).to(self.args.device)  ###############mT


#         logits = torch.div(
#             torch.matmul(prototypes, prototypes.T),
#             self.temperature)

#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(num_cls).view(-1, 1).to(self.args.device),
#             0
#         )
#         mask = mask * logits_mask
#         mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
#         mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
#         loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
#         return loss



#     def init_class_prototypes(self):
#         """Initialize class prototypes"""
#         self.model.eval()
#         start = time.time()
#         prototype_counts = [0]*self.args.num_classes
#         with torch.no_grad():
#             prototypes = torch.zeros(self.args.num_classes,self.args.feat_dim).to(self.args.device)       ####unknown protos will be zeros
#             for i, (input, target) in enumerate(self.loader):
#             # for (input, target, _) in self.loader:
            
#                 # input = input[0]
#                 input, target = input.to(self.args.device), target.to(self.args.device)
#                 # print(f'Input shape: {input.shape}')
#                 features = self.model.base(input)  #self.model(input)
#                 for j, feature in enumerate(features):
#                     prototypes[target[j].item()] += feature
#                     prototype_counts[target[j].item()] += 1
            
#             for cls in range(self.args.num_classes):
#                 if prototype_counts[cls] > 0:                  ###################***********
#                     prototypes[cls] /=  prototype_counts[cls]
                    
#                     #kappa computation
#                     self.R_hat[cls] =  torch.norm(prototypes[cls]).detach()
#                     # R_hat_sqr = self.R_hat[cls]*self.R_hat[cls]
#                     # self.kappa_hat[cls] = (self.R_hat[cls] * (self.feat_dim - R_hat_sqr)) / (1 - R_hat_sqr) 
#                     self.kappa_hat[cls] = self.R_hat[cls]
#                 else:    
#                     prototypes[cls] += self.epsilon                 ###################***********
                    
#                     #kappa computation
#                     self.R_hat[cls] =  torch.norm(prototypes[cls]).detach()
#                     # R_hat_sqr = self.R_hat[cls]*self.R_hat[cls]
#                     # self.kappa_hat[cls] = (self.R_hat[cls] * (self.feat_dim - R_hat_sqr)) / (1 - R_hat_sqr) 
#                     self.kappa_hat[cls] = self.R_hat[cls]


#             # measure elapsed time
#             duration = time.time() - start
#             # print(f'Time to initialize prototypes: {duration:.3f}')
            
#             #if self.args.normalize:
#             prototypes = F.normalize(prototypes, dim=1)

#             self.prototypes = prototypes