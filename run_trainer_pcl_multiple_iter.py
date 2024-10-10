import copy
import torch
import pickle
import numpy as np
import torchvision
import argparse

from numpy import random 


# from ood_data_utils import *
from models.model_manager import *

from comm_utils.topology_manager import *
from comm_utils.decentralized import * 
# from comm_utils.comm_vanilla_proto import *
from comm_utils.comm_decood import * 
# from comm_utils.comm_decood_w import * 

# from comm_utils.comm_decood_avg import * 
# from comm_utils.comm_decood_avg_w import * 
# from comm_utils.comm_decood_avg_clust_w import * 
# from comm_utils.comm_decood_unc import * 
# from comm_utils.comm_dis_pfl import * 
# from comm_utils.comm_gossip import *
# from comm_utils.comm_GH import *
# from comm_utils.comm_GH_ood import *
# from comm_utils.comm_GH_proto import *
# from comm_utils.comm_decood_avg_clust import * 
# from comm_utils.comm_em_vmf import * 
# from comm_utils.comm_local import *
# from comm_utils.comm_penz import *

# from comm_clus import *
# from comm_contrib import *
from comm_utils.mst import *
from client_manager import *
# from comm_submod import *
#from comm_mst import *





import os, sys
from pathlib import Path
from datetime import datetime

lib_dir = (Path(__file__).parent /"lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))




# lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
# if str(lib_dir) not in sys.path:
#     sys.path.insert(0, str(lib_dir))


# from models.resnet import resnet18
#from models.vision_transformer import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
# from options import args_parser
# from update import LocalUpdate, LocalTest
# from models.models import ProjandDeci
# from models.multibackbone import alexnet, vgg11, mlp_m
from lib.pcl_utils import *
# from lib import data_utils






def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
    np.random.seed(1234)
    random.seed(1234)



def parse_arguments():
    parser = argparse.ArgumentParser(description='arguments for DECOOD')

    parser.add_argument('--algorithm', default='FedProto', help='Algorithm name: FedConCIDER/FedGradProto/Decood')
    parser.add_argument('--data_type', default='img', help='Algorithm name: img/pc/text/tabular') 
    parser.add_argument('--num_trials', type=int, default=3, help='Number of trials')


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
#     parser.add_argument('--alpha', default=0.9, type=float, help='Dirichlet dist parameter')

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
    parser.add_argument('--global_seed', default=1234, type=int, help='Global seed')

    parser.add_argument('--proto_bs_id', default=5, type=int, help='Proto bs id parameter')
    parser.add_argument('--proto_bs_ood', default=5, type=int, help='Proto bs ood parameter')
    parser.add_argument('--ood_header_epoch', default=100, type=int, help='OOD header epoch')
    parser.add_argument('--id_header_epoch', default=1, type=int, help='ID header epoch')

    parser.add_argument('--ood_train_method', default='OE', help='OOD training method: energy/OE')
    parser.add_argument('--m_in', default=-25, type=int, help='M in parameter')
    parser.add_argument('--m_out', default=-7, type=int, help='M out parameter')

    parser.add_argument('--save_folder_name', default='../results/', help='Save folder name')
    parser.add_argument('--params_dir', default='../params/', help='Parameters directory')






    # federated arguments
#     parser.add_argument('--rounds', type=int, default=60, help="number of rounds of training")
#     parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
#     parser.add_argument('--alg', type=str, default='fedpcl', help="algorithms")
#     parser.add_argument('--train_ep', type=int, default=1, help="the number of local episodes: E")
#     parser.add_argument('--local_bs', type=int, default=32, help="local batch size")
#     parser.add_argument('--test_bs', type=int, default=32, help="test batch size")
#     parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
#     parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
#     parser.add_argument('--weight_decay', type=float, default=0, help='Adam weight decay (default: 0)')
#     parser.add_argument('--device', default="cuda", type=str, help="cpu, cuda, or others")
#     parser.add_argument('--gpu', default=0, type=int, help="index of gpu")
#     parser.add_argument('--optimizer', type=str, default='adam', help="type of optimizer")
#     parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')

    # model arguments
    parser.add_argument('--model_type', type=str, default='cnn', help='model name') #cnn
    parser.add_argument('--num_bb', type=int, default=3, help='number of backbone')

    # data arguments
#     parser.add_argument('--dataset', type=str, default='digit', help="name of dataset, e.g. digit")
    parser.add_argument('--percent', type=float, default=1, help="percentage of dataset to train")
    parser.add_argument('--data_dir', type=str, default='../FedPCL/data/', help="name of dataset, default: './data/'")
    parser.add_argument('--train_size', type=int, default=10, help="number of training samples in total")
    parser.add_argument('--test_size', type=int, default=100, help="num of test samples per dataset")
#     parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--feature_iid', type=int, default=0, help='Default set to feature non-IID. Set to 1 for feature IID.')
    parser.add_argument('--label_iid', type=int, default=1, help='Default set to label non-IID. Set to 1 for label IID.')
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")
    parser.add_argument('--save_protos', type=int, default=1, help="whether to save protos or not")

    # Local arguments
    parser.add_argument('--n_per_class', type=int, default=10, help="num of samples per class")
    parser.add_argument('--ld', type=float, default=0.5, help="weight of proto loss")
    parser.add_argument('--t', type=float, default=2, help="coefficient of local loss")
    parser.add_argument('--alpha', type=float, default=1, help="diri distribution parameter")

    # noise
    parser.add_argument('--add_noise_img', type=int, default=0, help="whether to add noise to images")
    parser.add_argument('--add_noise_proto', type=int, default=0, help="whether to add noise to images")
    parser.add_argument('--noise_type', type=str, default='gaussian', help="laplacian, gaussian, exponential")
    parser.add_argument('--perturb_coe', type=float, default=0.1, help="perturbation coefficient")
    parser.add_argument('--scale', type=float, default=0.05, help="noise distribution std")


    args = parser.parse_args()
    return args



def generate_filename(args):
    """Generate a filename based on dataset, feature iid, and label iid settings."""
    feature_label_pattern = f"feature_{'iid' if args.feature_iid else 'non_iid'}_label_{'iid' if args.label_iid else 'non_iid'}_{args.dataset}.pkl"
    filepath = os.path.join('data/', feature_label_pattern)
    return filepath

def load_data(filename):
    """Load the dataset and user group data from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)



def main():
     args = parse_arguments()


     #set seeds
     # torch.manual_seed(args.global_seed)
     # random.seed(args.global_seed)
     # np.random.seed(args.global_seed)
     # torch.backends.cudnn.deterministic = True
     # torch.backends.cudnn.benchmark = False
     # torch.autograd.set_detect_anomaly(True)
     # args.global_seed = [args.global_seed]
     # np.random.seed(args.seed)
     # random.seed(args.seed)

     
     #set cuda params
     use_cuda = not args.no_cuda and torch.cuda.is_available()
     args.device = torch.device('cuda' if use_cuda else 'cpu')

     # ##
     g = torch.Generator()
     g.manual_seed(0)
     
     train_kwargs = {'batch_size': args.local_bs, 'num_workers':1, 'worker_init_fn':seed_worker, 'generator':g}
     test_kwargs = {'batch_size': args.test_bs, 'num_workers':1, 'worker_init_fn':seed_worker, 'generator':g}
     
     if use_cuda:
          cuda_kwargs = {'num_workers': 1,'pin_memory': True,'shuffle': False}  # 'shuffle': False
          train_kwargs.update(cuda_kwargs)
          test_kwargs.update(cuda_kwargs)


     args.device = args.device if torch.cuda.is_available() else 'cpu'
     print("Training on", args.device, '...')
     if args.device == 'cuda':
         torch.cuda.set_device(args.gpu)
         torch.cuda.manual_seed(args.seed)
         torch.manual_seed(args.seed)
     else:
         torch.manual_seed(args.seed)
     np.random.seed(args.seed)
     random.seed(args.seed)





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
          if args.dataset == 'digit':
               args.head = copy.deepcopy(args.model.fc)
               args.model.fc = nn.Identity() #freeze the head from original model
               args.model = BaseHeadSplit(args.model, args.head)
          if args.dataset == 'office':
               args.head = copy.deepcopy(args.model.fc)
               args.model.fc = nn.Identity() #freeze the head from original model
               args.model = BaseHeadSplit(args.model, args.head)
          if args.dataset == 'domainnnet':
               args.head = copy.deepcopy(args.model.fc)
               args.model.fc = nn.Identity() #freeze the head from original model
               args.model = BaseHeadSplit(args.model, args.head)



  
     print(f'\nAlgorithm:  {args.algorithm}  \nDataset: {args.dataset}  \nDist: {args.dist}  \nFC: {args.fc} \nSubmod: {args.submod} \nCommunucation(MST): {args.comm} \nRound: {args.num_rounds}\n')
  

     acc_mtx = torch.zeros([args.num_trials, args.num_clients])
     global_test_acc_mtx = torch.zeros([args.num_trials])


     for trial in range(args.num_trials):

          # # Generate filename for dataset
          # filename = generate_filename(args)

          # # Check if the file exists, if so, load the data; if not, prepare the data and save it
          # if os.path.exists(filename):
          #    print(f"Loading data from {filename}...")
          #    try:
          #        train_dataset_list, test_dataset_list, user_groups, user_groups_test = load_data(filename)
          #        print(f"Data loaded successfully from {filename}.")
          #    except (EOFError, pickle.UnpicklingError) as e:
          #        print(f"Error loading data from {filename}: {e}")
          # else:
          #    print("File not found...")

          # print(f'{args.dataset}: Data initialized .........................................................')
          # print(train_dataset_list, user_groups)
          # for i in range(args.num_clients):
          #    print(len(user_groups[i]))

          # test_loader = None




     # dataset initialization
          # feature iid, label non-iid
          if args.feature_iid and args.label_iid==0:
               if args.dataset == 'digit':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_mnistm_noniid(args.num_clients, args=args)
               elif args.dataset == 'office':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_caltech_noniid(args.num_clients, args=args)
               elif args.dataset == 'domainnet':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_real_noniid(args.num_clients, args=args)
          # feature non-iid, label iid
          elif args.feature_iid==0 and args.label_iid:
               if args.dataset == 'digit':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits(args.num_clients, args=args)
               elif args.dataset == 'office':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office(args.num_clients, args=args)
               elif args.dataset == 'domainnet':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domainnet(args.num_clients, args=args)
          # feature non-iid, label non-iid
          elif args.feature_iid==0 and args.label_iid==0:
               if args.dataset == 'digit':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits_noniid(args.num_clients, args=args)
               elif args.dataset == 'office':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office_noniid(args.num_clients, args=args)
               elif args.dataset == 'domainnet':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domainnet_noniid(args.num_clients, args=args)

          test_loader = None





          client_id = 0
          clients = []


          #for current cluster, create clients(equal to  total_clients/number_of_cluster(2)) in a loop and assign current cluster
          for i in range(args.num_clients):
               clients.append(get_clients(args.algorithm, client_id, args, adj_mat, train_dataset_list[i], user_groups[i], test_dataset_list[i], user_groups_test[i], 0))
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

               


          aggregated_trial_test_acc = 0
          total_test_samples = 0

          if args.isDFL:
          #save adjacency matrics
               filename = args.params_dir + 'adj_mat.pkl'
               with open(filename, 'wb') as f:
                    pickle.dump([adj_mat], f)
                    print('Adjacency saved in file: ',filename)
                    f.close()

               for c in clients:
                    #save clients performance of this trial  
                    acc_mtx[trial, c.id] = c.l_test_acc_hist[-1] #local_performance
                    aggregated_trial_test_acc += c.l_test_acc_hist[-1]*len(c.test_loader.dataset)
                    total_test_samples += len(c.test_loader.dataset)

                    #save client_models                 
                    filename = args.params_dir + 'id_loaders_{}_{}_{}_client_{}.pkl'.format(args.algorithm, args.dataset, args.comm, c.id)
                    with open(filename, 'wb') as f:
                         pickle.dump([c.local_protos, c.global_protos, c.l_test_acc_hist], f)
                         f.close()
               print('Clients saved in file: ',filename)
               aggregated_trial_test_acc /= total_test_samples
               global_test_acc_mtx[trial] = aggregated_trial_test_acc #weighted performance of all clients
          else :
               #save server_models
               filename = args.params_dir + 'id_loaders_{}.pkl'.format(c.id)
               with open(filename, 'wb') as f:
                    pickle.dump([server.global_protos, g_test_acc_hist], f)
                    print('Loaders file: ',filename)
                    f.close()



          print('Clients saved in file: ',filename)

          #save results
          filename = os.path.join(args.save_folder_name, 'avg_local_acc_'+str(args.num_clients)+'_'+str(args.dataset)+'_'+ str(args.algorithm)+'_feat_iid_'+str(args.feature_iid) + '_label_iid_' + str(args.label_iid) + '_iid_' + str(args.dist) +'_COMM_'+str(args.comm)+'.pkl')
          
          with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
               pickle.dump(avg_l_acc, f)
     
     
     #save acc_mtx
     filename = os.path.join(args.save_folder_name, 'acc_mtx_'+str(args.num_clients)+'_'+str(args.dataset)+'_'+ str(args.algorithm)+'_feat_iid_'+str(args.feature_iid) + '_label_iid_' + str(args.label_iid) + '_iid_' + str(args.dist) +'_COMM_'+str(args.comm)+'.pkl')
     with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
          pickle.dump(acc_mtx, f)
     
     #compute mean and std of acc_mtx
     # print("The avg test acc of all trials are:")
     # for j in range(args.num_clients):
     #      print('{:.2f}'.format(torch.mean(acc_mtx[:,j])*100))

     # print("The stdev of test acc of all trials are:")
     # for j in range(args.num_clients):
     #      print('{:.2f}'.format(torch.std(acc_mtx[:,j])*100))

     # acc_avg = torch.zeros([args.num_trials])
     # for i in range(args.num_trials):
     #      acc_avg[i] = torch.mean(acc_mtx[i,:]) * 100
     # print("The avg and stdev test acc of all clients in the trials:")
     # print('{:.2f}'.format(torch.mean(acc_avg)))
     # print('{:.2f}'.format(torch.std(acc_avg)))

     #
     mean_global_test_acc = torch.mean(global_test_acc_mtx)
     std_global_test_acc = torch.std(global_test_acc_mtx)

     print("The avg test acc of all trials are:")
     for j in range(args.num_clients):
          print('{:.2f}'.format(mean_global_test_acc))

     print("The stdev of test acc of all trials are:")
     for j in range(args.num_clients):
          print('{:.2f}'.format(std_global_test_acc))


     return 0




if __name__ == '__main__':
      ret = main()
      print('Execution finished.')
