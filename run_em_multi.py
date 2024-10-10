"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
from utils.em_utils import *
from utils.constants import *
from utils.args import *
import pickle


from pathlib import Path
from datetime import datetime
import os, sys


from torch.utils.tensorboard import SummaryWriter
lib_dir = (Path(__file__).parent /"lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from lib.pcl_utils import *
from lib.data_utils import DatasetSplit




def get_loader_from_dataset(dataset_train, train_idxs, dataset_test, test_idxs, local_bs):
    '''
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    '''
    random.shuffle(train_idxs)
    # idxs_train = train_idxs[:int(len(train_idxs))]
    # idxs_test = test_idxs #idxs[int(0.9*len(idxs)):]  

    trainloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, train_idxs),
                                batch_size=local_bs, shuffle=True, drop_last=True)
    validloader = trainloader
    testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, test_idxs),
                            batch_size=local_bs, shuffle=False, drop_last=True)
    
    return trainloader, validloader, testloader



def generate_filename(args):
    """Generate a filename based on dataset, feature iid, and label iid settings."""
    feature_label_pattern = f"feature_{'iid' if args.feature_iid else 'non_iid'}_label_{'iid' if args.label_iid else 'non_iid'}_{args.dataset}.pkl"
    filepath = os.path.join('data/', feature_label_pattern)
    return filepath


def load_data(filename):
    """Load the dataset and user group data from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)




def init_clients(args_, root_path, logs_dir):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")

    ############
    if args_.num_trials==1:
        filename = generate_filename(args)

        # Check if the file exists, if so, load the data; if not, prepare the data and save it
        if os.path.exists(filename):
            print(f"Loading data from {filename}...")
            try:
                train_dataset_list, test_dataset_list, user_groups, user_groups_test = load_data(filename)
                print(f"Data loaded successfully from {filename}.")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading data from {filename}: {e}")
        else:
            print("File not found...")
    else:
        # feature iid, label non-iid
        if args_.feature_iid and args_.label_iid==0:
               if args_.dataset == 'digit':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_mnistm_noniid(args_.num_clients, args=args_)
               elif args_.dataset == 'office':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_caltech_noniid(args_.num_clients, args=args_)
               elif args_.dataset == 'domainnet':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_real_noniid(args_.num_clients, args=args_)
        # feature non-iid, label iid
        elif args_.feature_iid==0 and args_.label_iid:
               if args_.dataset == 'digit':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits(args_.num_clients, args=args_)
               elif args_.dataset == 'office':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office(args_.num_clients, args=args_)
               elif args_.dataset == 'domainnet':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domainnet(args_.num_clients, args=args_)
        # feature non-iid, label non-iid
        elif args_.feature_iid==0 and args_.label_iid==0:
               if args_.dataset == 'digit':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits_noniid(args_.num_clients, args=args_)
               elif args_.dataset == 'office':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office_noniid(args_.num_clients, args=args_)
               elif args_.dataset == 'domainnet':
                    train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domainnet_noniid(args_.num_clients, args=args_)

        

    print(f'{args.dataset}: Data initialized .........................................................')
    print('Train dataset size:')
    for i in range(args.num_clients):
        print(len(user_groups[i]))


    #create list of train, validation and test iterators from train_dataset_list, test_dataset_list and user_groups_test    
    train_iterators = []
    val_iterators = []
    test_iterators = []
    for i in range(args.num_clients):
        train_iterator, val_iterator, test_iterator = get_loader_from_dataset(train_dataset_list[i], user_groups[i], test_dataset_list[i], user_groups_test[i], args.local_bs)
        
        ###########################################################
        num_classes = 10 #TODO: change this
        sample_per_class = torch.zeros(num_classes)
        test_sample_per_class = torch.zeros(num_classes)

        for x, y, _ in train_iterator:
            for yy in y:
                sample_per_class[yy.item()] += 1
        print(f'Train dist: {sample_per_class}')
        id_labels = [i for i in range(num_classes) if sample_per_class[i]>0]
        #print(f'ID Labels: {self.id_labels}')
        
        for x, y, _ in test_iterator:
            for yy in y:
                test_sample_per_class[yy.item()] += 1
        print(f'Test dist: {test_sample_per_class}')

        print(f'Client data count: Train: {len(train_iterator.dataset)} Test: {len(test_iterator.dataset)}\n')
        
        
        
        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    ############


    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        learners_ensemble =\
            get_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
            )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients
        )

        clients_.append(client)

    return clients_



def get_trial_perf(aggregator):
    clients_acc = []

    global_test_loss = 0.
    global_test_acc = 0.
    clients_loss = []

    total_n_test_samples = 0
    client_train_samples = []

    for client_id, client in enumerate(aggregator.clients):

        train_loss, train_acc, test_loss, test_acc = client.write_logs() #################  

        client_train_samples.append(len(client.train_iterator.dataset))

        clients_acc.append(test_acc)
        clients_loss.append(test_loss)
        
        global_test_loss += test_loss * client.n_test_samples
        global_test_acc += test_acc * client.n_test_samples

        total_n_test_samples += client.n_test_samples

    
    global_test_loss /= total_n_test_samples
    global_test_acc /= total_n_test_samples
    
   
    return clients_acc,clients_loss,client_train_samples,global_test_loss,global_test_acc  



def run_experiment(args_):
    torch.manual_seed(args_.seed)

    # data_dir = get_data_dir(args_.experiment)
    data_dir = '../FedPCL/data/'

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients = init_clients(args_, root_path=os.path.join(data_dir, "train"),
                           logs_dir=os.path.join(logs_dir, "train"))

    # print("==> Test Clients initialization..")
    # test_clients = init_clients(args_, root_path=os.path.join(data_dir, "test"),
    #                             logs_dir=os.path.join(logs_dir, "test"))
    test_clients = []

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)
    
    # print('Print 1')
    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            client_type=CLIENT_TYPE[args_.method],
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
        )
    
    print('Print 2')
    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed
        )
    
    print("Training..................")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    while current_round <= args_.n_rounds:

        aggregator.mix()
        aggregator.write_logs() ######

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round

    
    clients_acc,clients_loss,_,global_test_loss,global_test_acc =  get_trial_perf(aggregator)
    

    if "save_dir" in args_:
        save_dir = os.path.join(args_.save_dir)

        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)

    return clients_acc,clients_loss,global_test_loss,global_test_acc


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.cuda.manual_seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    args = parse_args()

    acc_mtx = torch.zeros([args.num_trials, args.num_clients])
    global_test_acc_mtx = torch.zeros([args.num_trials])

    for trial in range(args.num_trials):
        clients_acc, clients_loss, aggregated_trial_test_loss, aggregated_trial_test_acc = run_experiment(args)
        #for every trial store the clients weighted accuracy in the acc_mtx
        for i in range(args.num_clients):
            acc_mtx[trial, i] = clients_acc[i] #local accuracy (non weighted)
        
        #store the global test accuracy in the global_test_acc_mtx
        global_test_acc_mtx[trial] = aggregated_trial_test_acc #weighted avg 

    #calculate the mean accuracy and std deviation for each client and the global test accuracy
    mean_acc_mtx = torch.mean(acc_mtx, dim=0)
    std_acc_mtx = torch.std(acc_mtx, dim=0)
    mean_global_test_acc = torch.mean(global_test_acc_mtx)
    std_global_test_acc = torch.std(global_test_acc_mtx)

    print(f'Mean accuracy for each client: {mean_acc_mtx}')
    print(f'Standard deviation for each client: {std_acc_mtx}')
    print(f'Mean global test accuracy: {mean_global_test_acc}')
    print(f'Standard deviation for global test accuracy: {std_global_test_acc}')

