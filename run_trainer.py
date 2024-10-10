import copy
import torch
import pickle
import numpy as np
import torchvision
import argparse

from numpy import random 


from data.data_utils import *
from ood_data_utils import *
from models.model_manager import *

from comm_utils.topology_manager import *
from comm_utils.decentralized import * 
from comm_utils.comm_vanilla_proto import *
from comm_utils.comm_decood import * 
from comm_utils.comm_decood_w import * 
from comm_utils.comm_decood_avg import * 
from comm_utils.comm_decood_avg_w import * 
from comm_utils.comm_decood_avg_clust_w import * 
from comm_utils.comm_decood_unc import * 
from comm_utils.comm_dis_pfl import * 
from comm_utils.comm_gossip import *
from comm_utils.comm_GH import *
from comm_utils.comm_GH_ood import *
from comm_utils.comm_GH_proto import *
from comm_utils.comm_decood_avg_clust import * 
from comm_utils.comm_em_vmf import * 
from comm_utils.comm_local import *
from comm_utils.comm_penz import *

# from comm_clus import *
# from comm_contrib import *
from comm_utils.mst import *
from client_manager import *
# from comm_submod import *
#from comm_mst import *



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def parse_arguments():
    parser = argparse.ArgumentParser(description='arguments for DECOOD')

    parser.add_argument('--algorithm', default='FedProto', help='Algorithm name: FedConCIDER/FedGradProto/Decood')
    parser.add_argument('--data_type', default='img', help='Algorithm name: img/pc/text/tabular') 

    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    parser.add_argument('--mst_comm', default=False, action='store_true', help='Whether to use MST communication')
    parser.add_argument('--comm', default='vanilla_proto', help='Communication method : comm_GH/vanilla_proto/gossip/clus/mst/grad_proto_decood')
    parser.add_argument('--check_running', default=True, type=bool, help='Check running status')
    parser.add_argument('--decood_loss_code', default='CD', help='Decood loss type: ECD/ECP/EC')
    parser.add_argument('--num_clients', default=20, type=int, help='Total number of clients')
    parser.add_argument('--topo', default='fc', help='Topology type: ring/sparse/fc')
    parser.add_argument('--sparse_neighbors', default=5, type=int, help='Number of neighbors in sparse topology')
    parser.add_argument('--test_on_cosine', default=False, type=bool, help='mse/cosine based performance testing')
    
    
    parser.add_argument('--model', default=None, help='Model: assigned based on algorithm and dataset')
    parser.add_argument('--head', default=None, help='Head : assigned based on algorithm and dataset')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--ood_dataset', default='svhn', help='Out-of-distribution dataset name')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes')
    parser.add_argument('--dist', default='iid',  help='iid/path/dir')
    parser.add_argument('--w', default=1, type=int, help='Weight parameter')
    parser.add_argument('--feat_dim', default=512, type=int, help='Feature dimension')
    parser.add_argument('--normalize', default=True, type=bool, help='feature normalization')
    parser.add_argument('--sel_on_kappa', default=True, type=bool, help='kappa_based_selection_of_prototypes_for_aggregation')

    
    parser.add_argument('--temp', default=0.5, type=float, help='Temperature parameter')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum parameter')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay parameter')
    parser.add_argument('--training_mode', default='SupCon', help='Training mode')
    parser.add_argument('--posterior_type', default='soft', help='Training mode')


    parser.add_argument('--num_rounds', default=20, type=int, help='Number of rounds')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer type')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--local_bs', default=32, type=int, help='Local batch size')
    parser.add_argument('--test_bs', default=32, type=int, help='Test batch size')
    parser.add_argument('--learning_rate_decay', default=0.9, type=float, help='Learning rate decay')
    parser.add_argument('--local_epochs', default=3, type=int, help='Number of local epochs')

    parser.add_argument('--clustering', default='noise', help='Clustering type') #'rotation', 'noise', 'label', 'None'
    parser.add_argument('--noise_level', default=0.2, type = float,  help='Clustering type') #'rotation', 'noise'
    parser.add_argument('--num_clusters', default=1, type=int, help='Number of clusters')
    parser.add_argument('--num_sel_clients', default=4, type=int, help='Number of selected clients')
    parser.add_argument('--isCFL', default=True, type=bool, help='Whether clustered')
    parser.add_argument('--isDFL', default=True, type=bool, help='Whether DFL')
    parser.add_argument('--fc', default=True, type=bool, help='Whether FC')
    parser.add_argument('--clustered_agg', default=False, type=bool, help='Whether clustered aggregation')
    parser.add_argument('--proto_m', default=0.5, type=float, help='Proto M parameter')
    parser.add_argument('--lamda_fed_proto', default=0.1, type=float, help='Lambda fed proto parameter')
    parser.add_argument('--unequal', default=False, type=bool, help='Whether unequal')
    parser.add_argument('--shard_size', default=500, type=int, help='Shard size')
    parser.add_argument('--no_shard_per_client', default=4, type=int, help='Number of shards per client')
    parser.add_argument('--alpha', default=0.9, type=float, help='Alpha parameter')

    parser.add_argument('--neighbour_selection', default='grad_based', help='Neighbour selection type')
    parser.add_argument('--neighbour_exploration', default='greedy', help='Neighbour exploration type')
    parser.add_argument('--topk', default=7, type=int, help='Top k parameter')
    parser.add_argument('--tau', default=1, type=int, help='Tau parameter')
    parser.add_argument('--submod', default=False, type=bool, help='Whether submodular')
    parser.add_argument('--n_samplings', default=1, type=int, help='Number of samplings')
    parser.add_argument('--confidence_level', default=0.95, type=float, help='Confidence level')
    parser.add_argument('--grad_thres', default=0.0, type=float, help='Gradient threshold')
    parser.add_argument('--sub_mod_sel_ratio', default=0.7, type=float, help='Sub-modular selection ratio')
    parser.add_argument('--fed_avg_sel', default=0.7, type=float, help='Fed avg selection')
    parser.add_argument('--n_neighbor', default=4, type=int, help='Number of neighbors')
    parser.add_argument('--subset_ratio', default=0.5, type=float, help='Subset ratio')

    parser.add_argument('--th', default=0.1, type=float, help='Threshold')
    parser.add_argument('--is_bayes', default=False, type=bool, help='Whether Bayes')
    parser.add_argument('--p', default=0.1, type=float, help='P parameter')

    parser.add_argument('--device', default=None, help='Device type')
    parser.add_argument('--gpu', default=False, type=bool, help='Whether GPU')
    parser.add_argument('--no_cuda', default=False, type=bool, help='Whether to disable CUDA')
    parser.add_argument('--avg_error_thres', default=0.01, type=float, help='Average error threshold')
    parser.add_argument('--mu', default=0.75, type=float, help='Mu parameter')
    parser.add_argument('--global_seed', default=3, type=int, help='Global seed')

    parser.add_argument('--proto_bs_id', default=5, type=int, help='Proto bs id parameter')
    parser.add_argument('--proto_bs_ood', default=5, type=int, help='Proto bs ood parameter')
    parser.add_argument('--ood_header_epoch', default=100, type=int, help='OOD header epoch')
    parser.add_argument('--id_header_epoch', default=1, type=int, help='ID header epoch')

    parser.add_argument('--ood_train_method', default='OE', help='OOD training method: energy/OE')
    parser.add_argument('--m_in', default=-25, type=int, help='M in parameter')
    parser.add_argument('--m_out', default=-7, type=int, help='M out parameter')

    parser.add_argument('--save_folder_name', default='../results/', help='Save folder name')
    parser.add_argument('--params_dir', default='../params/', help='Parameters directory')


    args = parser.parse_args()
    return args




def main():
  args = parse_arguments()


  #set seeds
  torch.manual_seed(args.global_seed)
  random.seed(args.global_seed)
  np.random.seed(args.global_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.autograd.set_detect_anomaly(True)
  args.global_seed = [args.global_seed]
  
  #set cuda params
  use_cuda = not args.no_cuda and torch.cuda.is_available()
  args.device = torch.device('cuda' if use_cuda else 'cpu')

  ##
  g = torch.Generator()
  g.manual_seed(0)
  
  train_kwargs = {'batch_size': args.local_bs, 'num_workers':1, 'worker_init_fn':seed_worker, 'generator':g}
  test_kwargs = {'batch_size': args.test_bs, 'num_workers':1, 'worker_init_fn':seed_worker, 'generator':g}
  
  if use_cuda:
    cuda_kwargs = {'num_workers': 1,'pin_memory': True,'shuffle': False}  # 'shuffle': False
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)


  if args.comm == 'comm_decood_w' or args.comm == 'comm_decood_avg_w' or args.comm == 'comm_decood_avg_clust_w':
     adj_mat = get_mixing_matrix(n=args.num_clients, p=0.5, seed=3) #mixing is computed from adj_mat  
  else:
     adj_mat = topology_manager(args)

  print(adj_mat)

  args.model,args.feat_dim = get_model(args)
  print(f'Using feature dim: {args.feat_dim}')

  
  if args.algorithm != 'DisPFL':
    if args.dataset == 'cifar10':
     #    mobilenet default
     #    args.head = copy.deepcopy(args.model.classifier)
     #    args.model.classifier = nn.Identity() #freeze the head from original model
     #    args.model = BaseHeadSplit(args.model, args.head)

        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity() #freeze the head from original model
        args.model = BaseHeadSplit(args.model, args.head)
    elif args.dataset == 'cifar100':
     #    mobile_net
     #    args.head = copy.deepcopy(args.model.classifier)
     #    args.model.classifier = nn.Identity() #freeze the head from original model
     #    args.model = BaseHeadSplit(args.model, args.head)

        args.head = copy.deepcopy(args.model.fc[-1])
        args.model.fc[-1] = nn.Identity() #freeze the head from original model
        args.model = BaseHeadSplit(args.model, args.head)




  #for mobilenet torchvision only
  #  args.head = copy.deepcopy(args.model.classifier)
  #  args.model.classifier = nn.Identity() #freeze the head from original model
  #  args.model = BaseHeadSplit(args.model, args.head)
  
     #server = FedROD(args, i)


#   if args.algorithm in ['FedAvg','FedRod','FedPens','FedProto','FedGradProto','FedCon','FedConCIDER','Decood','Decood_avg','FedGH', 'FedGHOOD','FedGHProto']:
#             args.head = copy.deepcopy(args.model.fc)
#             args.model.fc = nn.Identity() #freeze the head from original model
#             args.model = BaseHeadSplit(args.model, args.head)
#             #server = FedROD(args, i)

  
  print(f'\nAlgorithm:  {args.algorithm}  \nDataset: {args.dataset}  \nDist: {args.dist}  \nFC: {args.fc} \nSubmod: {args.submod} \nCommunucation(MST): {args.comm} \nRound: {args.num_rounds}\n')
  
  
  #prepare train and test set (all clients will load data using their loader locally)\
  if args.clustering == 'label':
        #load datasets transformed(2D), only tensor transformed(3d)
        dataset_train, dataset_test = load_dataset(args) 
        
    
        # #global test set
        if args.data_type == 'pc':
          test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=test_bs, shuffle=False, drop_last=False)
        else:
          test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
        
        #label wise equally cluster train and test data 
        #then every cluster is iid or non-iid distributed among group of clients
        #->get train and test data(from train test set we loaded previously) clustered in 2 group of labels(randomly)
        #->split_data_label : multiple clusters of train data in train set, multiple clusters of test data in test set
        #(Selected unique labels from data.Then grouped unique labels. Then apply this to train data and test data.)
        train_set_clusters, test_set_clusters = cluster_train_test_set_label(dataset_train, dataset_test, args.num_clusters) #2 clusters
        print('Clustering done.')
        #distribute among clients
        # print(f'Training data: {train_set_clusters[0]}')
        client_id = 0
        clients = []

        # print(f'Train_set_cluster_keys: {len(train_set_clusters.keys())}')

        for cluster_idx in range(len(train_set_clusters.keys())):
          #suppose we have 3 clusters. So distribute(iid/non_iid) each cluster(set of samples according to labels) to a total of (tot_clients/3) clients
          #we take a cluster(set of samples)
          #for every client of that cluster, we select equal number of samples randomly
          #we get a total of (args.num_clients // args.num_clusters) splits of a single cluster and assign them to clients later
          no_cluster_clients = args.num_clients // args.num_clusters


          if args.dist == 'iid':
              user_data_idxs_train, user_data_idxs_test  = dist_iid_clust(train_set_clusters[cluster_idx], test_set_clusters[cluster_idx], no_cluster_clients, args.isCFL)             
          elif args.dist == 'path':
              shard_size_train = len(train_set_clusters[cluster_idx])/(no_cluster_clients*args.no_shard_per_client)
              shard_size_test = len(test_set_clusters[cluster_idx])/(no_cluster_clients*args.no_shard_per_client)
              user_data_idxs_train, user_data_idxs_test  = dist_pathological_clust(train_set_clusters[cluster_idx], test_set_clusters[cluster_idx], no_cluster_clients, shard_size_train,shard_size_test, args.isCFL)
          elif args.dist == 'dir':      
               user_data_idxs_train, user_data_idxs_test = dist_dirichlet_clust(train_set_clusters[cluster_idx], test_set_clusters[cluster_idx], no_cluster_clients, args.alpha, args.num_classes, args.global_seed, args.isCFL)              
          #     user_data_idxs_train = dist_dirichlet(train_set_clusters[cluster_idx], no_cluster_clients, args.alpha, args.num_classes, args.global_seed, args.isCFL)
          #     user_data_idxs_test = dist_dirichlet(test_set_clusters[cluster_idx], no_cluster_clients, args.alpha, args.num_classes, args.global_seed, args.isCFL)
              


              #print('Non-iid setup un defined for clustering')
          
          #for current cluster, create clients(equal to  total_clients/number_of_cluster(2)) in a loop and assign current cluster
          for i in range(args.num_clients // args.num_clusters):
               clients.append(get_clients(args.algorithm, client_id, args, adj_mat, train_set_clusters[cluster_idx], user_data_idxs_train[i], test_set_clusters[cluster_idx], user_data_idxs_test[i], cluster_idx))
               client_id = client_id + 1

        #print(f'Adjacent matrics: {adj_mat}')
        if args.comm=='comm_mst':
             avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc = comm_mst(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_penz':
             avg_l_acc = comm_penz(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_gossip':
             avg_l_acc = comm_gossip(args, adj_mat, clients, debug=False, test_loader=test_loader)
            #  avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc = comm_gossip(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_vanilla_proto':
             avg_l_acc = comm_vanilla_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood':
             #avg_l_acc = comm_grad_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
             avg_l_acc = comm_decood(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_w':
             #avg_l_acc = comm_grad_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
             avg_l_acc = comm_decood_w(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_avg':
             #avg_l_acc = comm_grad_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
             avg_l_acc = comm_decood_avg(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_avg_w':
             avg_l_acc = comm_decood_avg_w(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_unc':
             avg_l_acc = comm_decood_unc(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_avg_clust':
             #avg_l_acc = comm_grad_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
             avg_l_acc = comm_decood_avg_clust(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_dis_pfl':
             avg_l_acc = comm_dis_pfl(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_avg_clust_w':
             avg_l_acc = comm_decood_avg_clust_w(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_GH':
             avg_l_acc = comm_GH(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_GH_ood':
             avg_l_acc = comm_GH_ood(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_fed_GH_proto':
             avg_l_acc = comm_GH_proto(args,adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_local':
             avg_l_acc = comm_local(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_em_vmf':
             avg_l_acc = comm_em_vmf(args, adj_mat, clients, debug=False, test_loader=test_loader)

             


             #avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc, client_heat = comm_contrib(args, adj_mat, clients, debug=False, test_loader=test_loader)  #submodular maximization based gradient based node selection
             #avg_lacc, avg_gacc, avg_lunc, avg_gunc, client_heat = comm_clus(args, adj_mat, clients, debug=False, test_loader=test_loader)         


  elif args.clustering == 'noise':
        print('Noisy training')
        #load multiple rotated dataset  
        datasets_train, datasets_test = load_dataset(args)     
        # #global test set
        test_loader = torch.utils.data.DataLoader(datasets_test[0], **test_kwargs) #0 rotation testset
        
        train_set_clusters={} 
        test_set_clusters={}
        for cluster_idx in range(len(datasets_train.keys())):
            train_set_clusters[cluster_idx], test_set_clusters[cluster_idx] = cluster_train_test_set_rotation_noise(datasets_train[cluster_idx], datasets_test[cluster_idx], args.num_clusters) #2 clusters

        #distribute among clients
        client_id = 0
        clients = []
        
        for cluster_idx in range(len(train_set_clusters.keys())):
          no_cluster_clients = args.num_clients // args.num_clusters
          shard_size_train = len(train_set_clusters[cluster_idx])/(no_cluster_clients*args.no_shard_per_client)
          shard_size_test = len(test_set_clusters[cluster_idx])/(no_cluster_clients*args.no_shard_per_client)

          if args.dist == 'iid':
              user_data_idxs_train, user_data_idxs_test  = dist_iid_clust(train_set_clusters[cluster_idx], test_set_clusters[cluster_idx], no_cluster_clients, args.isCFL)             
          elif args.dist == 'path':
              shard_size_train = len(train_set_clusters[cluster_idx])/(no_cluster_clients*args.no_shard_per_client)
              shard_size_test = len(test_set_clusters[cluster_idx])/(no_cluster_clients*args.no_shard_per_client)
              user_data_idxs_train, user_data_idxs_test  = dist_pathological_clust(train_set_clusters[cluster_idx], test_set_clusters[cluster_idx], no_cluster_clients, shard_size_train,shard_size_test, args.isCFL)
          elif args.dist == 'dir':      
              user_data_idxs_train = dist_dirichlet(train_set_clusters[cluster_idx], no_cluster_clients, args.alpha, args.num_classes, args.global_seed, args.isCFL)
              user_data_idxs_test = dist_dirichlet(test_set_clusters[cluster_idx], no_cluster_clients, args.alpha, args.num_classes, args.global_seed, args.isCFL)
              

          #for current cluster, create clients(equal to  total_clients/number_of_cluster(2)) in a loop and assign current cluster
          for i in range(args.num_clients // args.num_clusters):
               clients.append(get_clients(args.algorithm, client_id, args, adj_mat, train_set_clusters[cluster_idx], user_data_idxs_train[i], test_set_clusters[cluster_idx], user_data_idxs_test[i], cluster_idx))
               client_id = client_id + 1
          
        #print(f'Adjacent matrics: {adj_mat}')
        if args.comm=='comm_mst':
             avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc = comm_mst(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_penz':
             avg_l_acc = comm_penz(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_gossip':
             avg_l_acc = comm_gossip(args, adj_mat, clients, debug=False, test_loader=test_loader)
            #  avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc = comm_gossip(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_vanilla_proto':
             avg_l_acc = comm_vanilla_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood':
             #avg_l_acc = comm_grad_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
             avg_l_acc = comm_decood(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_unc':
             avg_l_acc = comm_decood_unc(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_avg_clust':
             #avg_l_acc = comm_grad_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
             avg_l_acc = comm_decood_avg_clust(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_GH':
             avg_l_acc = comm_GH(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_GH_ood':
             avg_l_acc = comm_GH_ood(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_fed_GH_proto':
             avg_l_acc = comm_GH_proto(args,adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_local':
             avg_l_acc = comm_local(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_em_vmf':
             avg_l_acc = comm_em_vmf(args, adj_mat, clients, debug=False, test_loader=test_loader)


  elif args.clustering == 'rotation':
        #load multiple rotated dataset  
        datasets_train, datasets_test = load_dataset(args)     
        # #global test set
        test_loader = torch.utils.data.DataLoader(datasets_test[0], **test_kwargs) #0 rotation testset
        
        train_set_clusters={} 
        test_set_clusters={}
        for cluster_idx in range(len(datasets_train.keys())):
            train_set_clusters[cluster_idx], test_set_clusters[cluster_idx] = cluster_train_test_set_rotation_noise(datasets_train[cluster_idx], datasets_test[cluster_idx], args.num_clusters) #2 clusters

        #distribute among clients
        client_id = 0
        clients = []
        
        for cluster_idx in range(len(train_set_clusters.keys())):
          no_cluster_clients = args.num_clients // args.num_clusters

          if args.dist == 'iid':
              user_data_idxs_train, user_data_idxs_test  = dist_iid_clust(train_set_clusters[cluster_idx], test_set_clusters[cluster_idx], no_cluster_clients, args.isCFL)             
          elif args.dist == 'path':
              shard_size_train = len(train_set_clusters[cluster_idx])/(no_cluster_clients*args.no_shard_per_client)
              shard_size_test = len(test_set_clusters[cluster_idx])/(no_cluster_clients*args.no_shard_per_client)
              user_data_idxs_train, user_data_idxs_test  = dist_pathological_clust(train_set_clusters[cluster_idx], test_set_clusters[cluster_idx], no_cluster_clients, shard_size_train,shard_size_test, args.isCFL)
          elif args.dist == 'dir':      
              user_data_idxs_train = dist_dirichlet(train_set_clusters[cluster_idx], no_cluster_clients, args.alpha, args.num_classes, args.global_seed, args.isCFL)
              user_data_idxs_test = dist_dirichlet(test_set_clusters[cluster_idx], no_cluster_clients, args.alpha, args.num_classes, args.global_seed, args.isCFL)

          #for current cluster, create clients(equal to  total_clients/number_of_cluster(2)) in a loop and assign current cluster
          for i in range(args.num_clients // args.num_clusters):
               clients.append(get_clients(args.algorithm, client_id, args, adj_mat, train_set_clusters[cluster_idx], user_data_idxs_train[i], test_set_clusters[cluster_idx], user_data_idxs_test[i], cluster_idx))
               client_id = client_id + 1
          
        #print(f'Adjacent matrics: {adj_mat}')
        if args.comm=='mst':
             avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc = comm_mst(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='gossip':
             avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc = comm_gossip(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='penz':
             avg_l_acc = comm_penz(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='vanilla_proto':
             avg_l_acc = comm_vanilla_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)  #only proto aggregation and learn
        elif args.comm=='comm_grad_proto_decood':
             avg_l_acc = comm_grad_proto_decood(args, adj_mat, clients, debug=False, test_loader=test_loader)
             #avg_l_acc = comm_grad_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)     #similarity basedproto aggregation
             #avg_l_acc = comm_vanilla_proto(args, adj_mat, clients, debug=False, test_loader=test_loader) 
             #avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc, client_heat = comm_contrib(args, adj_mat, clients, debug=False, test_loader=test_loader)
             #avg_lacc, avg_gacc, avg_lunc, avg_gunc, client_heat = comm_clus(args, adj_mat, clients, debug=False, test_loader=test_loader) 
        elif args.comm=='comm_em_vmf':
             avg_l_acc = comm_em_vmf(args, adj_mat, clients, debug=False, test_loader=test_loader)



  elif args.clustering == 'None':  #No clustering   #non_iid/iid distribution based on raw train and test set data (No group of clients consideration, learn from all)
        
        dataset_train, dataset_test = load_dataset(args) 
        # global test set
        # #global test set
        if args.data_type == 'pc':
          test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.test_bs, shuffle=False, drop_last=False)
        else:
          test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
        

        
        if args.dist == 'iid':
              user_data_idxs_train  = dist_iid(dataset_train, args.num_clients, args.isCFL)             
        elif args.dist == 'path':
              shard_size_train = len(dataset_train)/(args.num_clients*args.no_shard_per_client)
              user_data_idxs_train  = dist_pathological(dataset_train,  args.num_clients, shard_size_train)
        elif args.dist == 'dir':      
              user_data_idxs_train = dist_dirichlet(dataset_train, args.num_clients, args.alpha, args.num_classes, args.global_seed, args.isCFL)

        
        #print('Non-iid setup un defined for clustering')
          

        #for current cluster, create clients(equal to  total_clients/number_of_cluster(2)) in a loop and assign current cluster
        client_id = 0
        clients = []        
        for i in range(args.num_clients):
               clients.append(get_clients(args.algorithm, client_id, args, adj_mat, dataset_train, user_data_idxs_train[i], None, None, 0))
               client_id = client_id + 1

        #print(f'Adjacent matrics: {adj_mat}')
        if args.comm=='comm_mst':
             avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc = comm_mst(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_gossip':
             avg_l_acc = comm_gossip(args, adj_mat, clients, debug=False, test_loader=test_loader)
            #  avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc = comm_gossip(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='penz':
             avg_l_acc = comm_penz(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_vanilla_proto':
             avg_l_acc = comm_vanilla_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood':
             #avg_l_acc = comm_grad_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
             avg_l_acc = comm_decood(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_unc':
             avg_l_acc = comm_decood_unc(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_decood_avg_clust':
             #avg_l_acc = comm_grad_proto(args, adj_mat, clients, debug=False, test_loader=test_loader)
             avg_l_acc = comm_decood_avg_clust(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_GH':
             avg_l_acc = comm_GH(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_GH_ood':
             avg_l_acc = comm_GH_ood(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_fed_GH_proto':
             avg_l_acc = comm_GH_proto(args,adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_local':
             avg_l_acc = comm_local(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm=='comm_em_vmf':
             avg_l_acc = comm_em_vmf(args, adj_mat, clients, debug=False, test_loader=test_loader)



  




  if args.isDFL:
    #save adjacency matrics
    filename = args.params_dir + 'adj_mat.pkl'
    with open(filename, 'wb') as f:
        pickle.dump([adj_mat], f)
        print('Adjacency saved in file: ',filename)
        f.close()

    for c in clients:
        #save client_models
        filename = args.params_dir + 'checkpoint_{}_{}_client_{}.pth.tar'.format(args.algorithm, args.dataset, c.id)
        torch.save({
            'model_state_dict': c.model.state_dict(),
            'optimizer_state_dict': c.optimizer.state_dict(),
            }, filename)
        print('Checkpoint file: ',filename)

        
        filename = args.params_dir + 'id_loaders_{}_{}_client_{}.pkl'.format(args.algorithm, args.dataset, c.id)
        with open(filename, 'wb') as f:
            if args.algorithm in ['FedAvg','Ditto']:
                pickle.dump([c.train_loader, c.test_loader, c.id_labels], f)
            else:
                pickle.dump([c.train_loader, c.test_loader, c.id_labels, c.local_protos, c.global_protos], f)
                #pickle.dump([c.train_loader, c.test_loader, c.id_labels, c.local_protos, c.global_protos], f)

            print('Loaders file: ',filename)
            f.close()

    print('Clients saved in file: ',filename)

  else :
    #save server_models
    filename = args.params_dir + 'checkpoint_client_{}.pth.tar'.format(c.id)
    torch.save({
        'model_state_dict': server.model.state_dict(),
        'optimizer_state_dict': server.optimizer.state_dict(),
        }, filename)

    filename = args.params_dir + 'id_loaders_{}.pkl'.format(c.id)
    with open(filename, 'wb') as f:
        pickle.dump([server.train_loader, server.test_loader, server.id_labels, server.global_protos], f)
        print('Loaders file: ',filename)
        f.close()


    print('Clients saved in file: ',filename)

  #save results
  filename = os.path.join(args.save_folder_name, str(args.num_clients)+'_'+str(args.dataset)+'_'+ str(args.algorithm)+'_Graph_FC_'+str(args.fc) + '_iid_' + str(args.dist) +'_COMM_'+str(args.comm)+'.pkl')
  
  with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(avg_l_acc, f)
        
        # if args.comm in ['clus', 'hybrid', 'mixed']:
        #     pickle.dump([avg_l_acc, avg_g_acc, avg_l_auc, avg_g_auc, avg_l_unc, avg_g_unc, client_heat], f)
        # else:
        #     pickle.dump([avg_l_acc, avg_g_acc], f) #pickle.dump([avg_l_acc, avg_g_acc], f)
        # print('Results saved in file: ',filename)
    

  return 0
    

if __name__ == '__main__':
      ret = main()
      print('Execution finished.')
