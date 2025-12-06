import os
import torch
import numpy as np
from torch import nn
from math import sqrt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

def dist_iid(dataset_train, dataset_test, num_clients, isCFL = False):
    """
    sample I.I.D. client data from MNIST dataset
    :param dataset: train dataset
    :param num_users: int
    :return: dictionary of client indices and its data sample indices (can be used as train and test) 
    """
    #1.shuffle data (indices)
    #2.create split for clients(datasize/client size)
    #3.create dictionary

    #train data processing
    # randomly shuffle label indices
    shuffled_indices = np.random.permutation(len(dataset_train))
    # split shuffled indices by the number of clients
    split_indices = np.array_split(shuffled_indices, num_clients)
    # construct a hashmap
    split_map_train = {i: split_indices[i] for i in range(num_clients)}


    if not isCFL: #not clustered
        return split_map_train #split_map_test
    else:
        shuffled_indices_test = np.random.permutation(len(dataset_test))
        split_indices_test = np.array_split(shuffled_indices_test, num_clients)
        split_map_test = {i: split_indices_test[i] for i in range(num_clients)} 
        return split_map_train, split_map_test     
    


def dist_non_iid_clust(dataset_train, dataset_test, num_clients, isCFL=False):
    """
    Sample non-I.I.D. client data from datasets.
    :param dataset_train: Training dataset
    :param dataset_test: Test dataset
    :param num_clients: Number of clients
    :param isCFL: Boolean flag indicating whether the data is clustered (default is False)
    :return: Dictionary of client indices and their data sample indices (for training and testing)
    """

    # Extract labels from the training dataset
    labels = np.array([target for _, target in dataset_train])

    # Shuffle the indices of the entire dataset
    shuffled_indices = np.random.permutation(len(labels))

    # Initialize variables to track assigned labels to clients
    labels_per_client = {i: [] for i in range(num_clients)}

    # Iterate over shuffled indices and distribute them among clients
    for i, index in enumerate(shuffled_indices):
        if i < (len(labels)/2):
            client_index = i % num_clients
        else:
            client_index = i % 2 #6 #first three imbalanced
        labels_per_client[client_index].append(index)

    # Construct a hashmap for training data
    split_map_train = {i: np.array(labels_per_client[i]) for i in range(num_clients)}

    if not isCFL:  # Not clustered
        return split_map_train
    else:
        # Extract labels from the test dataset
        labels_test = np.array([target for _, target in dataset_test])

        # Shuffle the indices of the entire test dataset
        shuffled_indices_test = np.random.permutation(len(labels_test))

        # Initialize variables to track assigned labels to clients for the test dataset
        labels_per_client_test = {i: [] for i in range(num_clients)}

        # Iterate over shuffled test indices and distribute them among clients
        for i, index_test in enumerate(shuffled_indices_test):
            client_index_test = i % num_clients
            labels_per_client_test[client_index_test].append(index_test)

        # Construct a hashmap for the test data
        split_map_test = {i: np.array(labels_per_client_test[i]) for i in range(num_clients)}

        return split_map_train, split_map_test



def dist_pathological(dataset, num_clients, shard_size):
    """
    Sample non-I.I.D client data from MNIST dataset
    param 
    dataset: nxd
    num_clients: int
    shard_size: int
    return: dictionary of client indices and its data sample indices (can be used as train and test)
    """
    #1.sort data according to labels
    #2.split sorted data in shards so that each shard contains specified number of samples
    # (note that some shard may have 2 type of labels(one label sample ends and another label sample starts)
    #3.assign every shard to a client in cyclic order
    #i.e. if there are 5 cleints and 10 shards, client 1 will get shard 1 and 11. clinet 2 will get shard 2 and 12 and son on..
    #we can do it randomply too
    #** smaller shard can assign one label to multiple clients,
    #large shards can make some deprive some clients from getting any shard

    #assert args.dataset in ['MNIST', 'CIFAR10'], '[ERROR] `pathological non-IID setting` is supported only for `MNIST` or `CIFAR10` dataset!'
    print(f'Data Length: {len(dataset)},Shard Size: {shard_size},Tot Clients: {num_clients}')
    #if not at least 2 classes per client, raise error
    assert len(dataset) / shard_size / num_clients == 2, '[ERROR] each client should have samples from class at least 2 different classes!'
        
    # sort data by labels
    sorted_indices = np.argsort(np.array(dataset.targets))
    
    #get (len(dataset) // shard_size) splitted group of data indices for every shard
    shard_indices = np.array_split(sorted_indices, len(dataset) // shard_size)

    # sort the list to conveniently assign samples to each clients from at least two~ classes
    split_indices = [[] for _ in range(num_clients)]
      
    # cyclically assign a shard to a client((idx % num_clients)th client) 
    for idx, shard in enumerate(shard_indices):
        split_indices[idx % num_clients].extend(shard)
        
    # construct a hashmap
    split_map = {i: split_indices[i] for i in range(num_clients)}
    return split_map
    


def dist_pathological_clust(train_dataset, test_dataset, num_clients, shard_size_train, shard_size_test, isCFL=False):
    """
    Sample non-I.I.D client data from train and test datasets.
    param 
    train_dataset: Training dataset (nxd)
    test_dataset: Testing dataset (mxd)
    num_clients: int
    shard_size: int
    return: Dictionary of client indices and their data sample indices for train and test sets
    """
    # If not at least 2 classes per client, raise an error
    print(f'Dataset Size: {len(train_dataset)}, Clients: {num_clients} ')
    #assert len(train_dataset) / shard_size / num_clients == 2, '[ERROR] each client should have samples from at least 2 different classes!'

    # Sort train and test data by labels
    if isCFL:
        sorted_train_indices = np.argsort(np.array(get_targets(train_dataset)))
        sorted_test_indices = np.argsort(np.array(get_targets(test_dataset)))
    else:
        sorted_train_indices = np.argsort(np.array(train_dataset.targets))
        sorted_test_indices = np.argsort(np.array(test_dataset.targets))

    # Get (len(train_dataset) // shard_size) and (len(test_dataset) // shard_size) splitted groups of data indices for every shard
    shard_indices_train = np.array_split(sorted_train_indices, len(train_dataset) // shard_size_train)
    shard_indices_test = np.array_split(sorted_test_indices, len(test_dataset) // shard_size_test)
    #print(f'Shard split size:{len(shard_indices_train)} Total Clients: {num_clients} in cluster.')

    # Sort the lists to conveniently assign samples to each client from at least two classes for train and test sets
    split_indices_train = [[] for _ in range(num_clients)]
    split_indices_test = [[] for _ in range(num_clients)]

    # Cyclically assign a shard to a client for train set
    for idx, shard in enumerate(shard_indices_train):
        split_indices_train[idx % num_clients].extend(shard)

    # Cyclically assign a shard to a client for test set
    for idx, shard in enumerate(shard_indices_test):
        split_indices_test[idx % num_clients].extend(shard)


    # Construct hashmaps for train and test sets
    split_map_train = {i: split_indices_train[i] for i in range(num_clients)}
    split_map_test = {i: split_indices_test[i] for i in range(num_clients)}

    return split_map_train, split_map_test

# # Example usage:
# # train_split_map, test_split_map = dist_pathological(train_dataset, test_dataset, num_clients, shard_size



def dist_dirichlet(dataset, num_clients, alpha, num_classes, global_seed, isCFL=False):
    # Non-IID split proposed in Hsu et al., 2019 (i.e., using Dirichlet distribution to simulate non-IID split)
    np.random.seed(global_seed)
    split_map = dict()

    # container
    client_indices_list = [[] for _ in range(int(num_clients))]

    # iterate through all classes
    # print(f'Classes in dataset....: {np.where(np.array(get_targets(dataset))==0)}')

    min_size = 0
    min_require_size = 10

    try_cnt = 1
    while min_size < min_require_size:
        for c in range(num_classes):
            # get corresponding class indices
            if isCFL:
                target_class_indices = np.where(np.array(get_targets(dataset))==c)[0] #wrong should be equal to c
            else:
                target_class_indices = np.where(np.array(dataset.targets) == c)[0]

            # shuffle class indices
            np.random.shuffle(target_class_indices)
            print(f'Target class {c} indices: {len(target_class_indices)}')

            # get label retrieval probability per each client based on a Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx) < len(dataset) / num_clients) for p, idx in zip(proportions, client_indices_list)])
            print(f'Proportions of class {c} : {proportions}')

            # normalize
            proportions = proportions / proportions.sum()
            #print(f'Prob propottions of class {c} : {proportions}')
            proportions = (np.cumsum(proportions) * len(target_class_indices)).astype(int)[:-1]
            print(f'Final proportions of class {c} : {proportions}')
            print(f'Sum of proportions {sum(proportions)}')
            

            # split class indices by proportions
            idx_split = np.array_split(target_class_indices, proportions)
            
            client_indices_list = [j + idx.tolist() for j, idx in zip(client_indices_list, idx_split)]
            min_size = min([len(idx_j) for idx_j in client_indices_list])

        print(f'Attempt: {try_cnt}')
        try_cnt += 1

    # shuffle finally and create a hashmap
    for j in range(num_clients):
        np.random.shuffle(client_indices_list[j])
        split_map[j] = client_indices_list[j]
    
    return split_map




def dist_dirichlet_clust(train_dataset, test_dataset, num_clients, alpha, num_classes, global_seed, isCFL=False):
    """
    Non-IID split using Dirichlet distribution for train and test datasets.
    param 
    train_dataset: Training dataset
    test_dataset: Testing dataset
    num_clients: Number of clients
    alpha: Dirichlet distribution parameter
    num_classes: Number of classes in the dataset
    global_seed: Global seed for reproducibility
    isCFL: Boolean parameter (default set to False)
    return: Dictionary of client indices and their data sample indices for train and test sets
    """
    # Container for train and test split maps
    split_map_train = dict()
    split_map_test = dict()

    # Containers for train and test client indices
    client_indices_list_train = [[] for _ in range(num_clients)]
    client_indices_list_test = [[] for _ in range(num_clients)]

    # Iterate through all classes
    for c in range(num_classes):
        # Get corresponding class indices for train and test datasets
        if isCFL:
            train_class_indices = np.where(np.array(get_targets(train_dataset))==c)[0]
            test_class_indices = np.where(np.array(get_targets(test_dataset))==c)[0]
            #print(train_class_indices)
            #print(test_class_indices)
        else:
            train_class_indices = np.where(np.array(train_dataset.targets) == c)[0]
            test_class_indices = np.where(np.array(test_dataset.targets) == c)[0]

        # Shuffle class indices for train and test datasets
        np.random.shuffle(train_class_indices)
        np.random.shuffle(test_class_indices)

        # Get label retrieval probability per each client based on a Dirichlet distribution
        proportions_train = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions_train = np.array([p * (len(idx) < len(train_dataset) / num_clients) for p, idx in zip(proportions_train, client_indices_list_train)])
        
        proportions_test = proportions_train #np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions_test = np.array([p * (len(idx) < len(test_dataset) / num_clients) for p, idx in zip(proportions_test, client_indices_list_test)])

        # Normalize
        proportions_train = proportions_train / proportions_train.sum()
        proportions_train = (np.cumsum(proportions_train) * len(train_class_indices)).astype(int)[:-1]
        proportions_test = proportions_test / proportions_test.sum()
        proportions_test = (np.cumsum(proportions_test) * len(test_class_indices)).astype(int)[:-1]

        # Split class indices by proportions for train and test sets
        idx_split_train = np.array_split(train_class_indices, proportions_train.cumsum() * len(train_class_indices))
        idx_split_test = np.array_split(test_class_indices, proportions_test.cumsum() * len(test_class_indices))

        # Update client indices lists for train and test sets
        client_indices_list_train = [j + idx.tolist() for j, idx in zip(client_indices_list_train, idx_split_train)]
        client_indices_list_test = [k + idx.tolist() for k, idx in zip(client_indices_list_test, idx_split_test)]

    # Shuffle finally and create hashmaps for train and test sets
    for j in range(num_clients):
        np.random.seed(global_seed)
        np.random.shuffle(client_indices_list_train[j])

        if len(client_indices_list_train[j]) > 10:
            split_map_train[j] = client_indices_list_train[j]

    for k in range(num_clients):
        np.random.seed(global_seed)
        np.random.shuffle(client_indices_list_test[k])

        if len(client_indices_list_test[k]) > 10:
            split_map_test[k] = client_indices_list_test[k]


    return split_map_train, split_map_test





class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:,row*size+i,col*size+j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)




class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]] #target was label
        
        return image.clone().detach(), label #target was label
        

def get_targets(data):
    return [data.__getitem__(i)[1] for i in range(data.__len__())]


def cluster_train_test_set_rotation_noise(dataset_train, dataset_test, n):
    '''
    Input : 
    train dataset,
    test dataset, 
    n: number of clusters
    
    return : just format identically to label clustering
    '''
    #train_set_clusters = {} #clusters
    #test_set_clusters = {}  

    train_idxs = np.array(np.arange(len(dataset_train), dtype=int), dtype="int64")
    test_idxs = np.array(np.arange(len(dataset_test), dtype=int), dtype="int64")
    
    #train_set_clusters[0] = DatasetSplit(dataset_train, train_idxs)  #contains ith cluster samples suppose all samples of class 1,3and 4
    #test_set_clusters[0] = DatasetSplit(dataset_test, test_idxs)

    
    return DatasetSplit(dataset_train, train_idxs), DatasetSplit(dataset_test, test_idxs)  #n group/cluster of samples







def cluster_train_test_set_label_fixed_train_size(dataset_train, dataset_test, n, num_samples):
    '''
    num_samples should be : num_client * desired_size_of_local_train_data
    considered n=1
    Input: 
    - train dataset
    - test dataset
    - n: number of clusters
    - num_samples: total number of training samples to consider

    From the train and test set, create train label-based clusters.
    First, all unique train labels are equally distributed in n clusters.
    Then, all corresponding data samples are assigned to those clusters 
    (no iid or non-iid distribution factor considered here). 

    Returns: 
    - train set and test set clusters
    '''

    # Step 1: Reduce the training dataset to `num_samples` samples
    idxs = np.arange(len(dataset_train), dtype=int)
    reduced_idxs = np.random.choice(idxs, num_samples, replace=False)
    labels = np.array(get_targets(dataset_train))
    labels_reduced = labels[reduced_idxs]

    # Get test labels (unchanged)
    idxs_test = np.arange(len(dataset_test), dtype=int)
    labels_test = np.array(get_targets(dataset_test))

    # Step 2: Cluster the unique train labels
    unique_labels = np.unique(labels_reduced)
    num_classes = len(unique_labels)

    labels_cluster = {}
    for i in range(n):
        labels_cluster[i] = np.random.choice(
            unique_labels, int(num_classes / n), replace=False
        )
        unique_labels = list(set(unique_labels) - set(labels_cluster[i]))

    # Step 3: Create train and test clusters
    train_set_clusters = {}  # clusters for train set
    test_set_clusters = {}   # clusters for test set

    for i in range(n):  # n clusters
        train_idxs = np.array([], dtype="int64")
        test_idxs = np.array([], dtype="int64")

        # Extract samples for each label and add them to the cluster
        for label in labels_cluster[i]:
            train_idxs_ = reduced_idxs[labels_reduced == label]
            train_idxs = np.concatenate((train_idxs, train_idxs_))
            test_idxs_ = idxs_test[labels_test == label]
            test_idxs = np.concatenate((test_idxs, test_idxs_))

        train_set_clusters[i] = DatasetSplit(dataset_train, train_idxs)
        test_set_clusters[i] = DatasetSplit(dataset_test, test_idxs)

    return train_set_clusters, test_set_clusters  # n groups/clusters of samples






def cluster_train_test_set_label(dataset_train, dataset_test, n):
    '''
    Input : 
    train dataset,
    test dataset, 
    n: number of clusters
    From train and test set, create train label based clusters  
    First All unique train labels are equally distributed in n clusters
    Then all corresponding data samples are assigned to those clusters (no iid or non-iid distribution factor considered here) 
    return : train set and test set clusters
    '''
    #Get unique train labels
    idxs = np.arange(len(dataset_train), dtype=int)
    labels = np.array(get_targets(dataset_train))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    # print(f'Unique labels: {unique_labels}')
    
    #Get test labels
    idxs_test = np.arange(len(dataset_test), dtype=int)
    labels_test = np.array(get_targets(dataset_test))

    #Group the labels into clusters(based on train labels: we care only train labels)
    #n clusters 
    labels_cluster = {}
    for i in range(n):
        labels_cluster[i] = np.random.choice(
            unique_labels, int(num_classes / n), replace=False
        )
        # print(f'Label cluster: {labels_cluster[i]}')
        unique_labels = list(set(unique_labels) - set(labels_cluster[i]))

    train_set_clusters = {} #clusters
    test_set_clusters = {}  


    #For ith cluster(some unique labels)  get all train(or test) samples of those classes and put in train_sets[i]
    for i in range(n): #n clusters
        train_idxs = np.array([], dtype="int64")
        test_idxs = np.array([], dtype="int64")
        #extract samples from according to label
        #and put them in train_sets[i] for ith cluster data similarly for testdata
        for label in labels_cluster[i]:
            train_idxs_ = idxs[label == labels[idxs]]
            train_idxs = np.concatenate((train_idxs, train_idxs_))
            test_idxs_ = idxs_test[label == labels_test[idxs_test]]
            test_idxs = np.concatenate((test_idxs, test_idxs_))

        train_set_clusters[i] = DatasetSplit(dataset_train, train_idxs)  #contains ith cluster samples suppose all samples of class 1,3and 4
        test_set_clusters[i] = DatasetSplit(dataset_test, test_idxs)

        # print(f'Train idx in cluster: {np.unique(labels[train_idxs], return_counts=True)}')
        # print(np.unique(labels_test[test_idxs], return_counts=True))

    return train_set_clusters, test_set_clusters  #n group/cluster of samples



def load_dataset(args):
    """ Returns train and test datasets 
    #a user group : which is a dict where the keys are the user index and the values are the 
    #corresponding data index in dataset for each of those users.
    """
     #assert args.dataset in ['MNIST', 'CIFAR10'], '[ERROR] `pathological non-IID setting` is supported only for `MNIST` or `CIFAR10` dataset!'

    if args.dataset == 'mnist':
        data_dir = '../data/mnist/'
    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist/'
    else:
        data_dir = '../data/cifar/'
    print(f'\nDownloading in :{os.getcwd()+data_dir}\n')


    
    #download/load dataset dis_PFL
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] #CIDER and dispfl
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    
    if args.dataset in ['cifar10','cifar100']:
        normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    elif args.dataset in ['mnist','fmnist']:
        normalize = transforms.Normalize((0.5), (0.5))
    else:
        print(f"Unknown dataset {args.dataset}")

    
    
    transform_train = transforms.Compose([ #DisPFL and cider
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    
    transform_standard = transforms.Compose([transforms.ToTensor(), normalize,]) #mnist fmnist
    
    #train_set, test_set, classes = load_dataset(args, transform_standard)
    transforms_dict = {
        0: transforms.Compose(
            [transforms.ToTensor(), normalize,]
        ),
        1: transforms.Compose(
            [transforms.RandomRotation([90, 90]), transforms.ToTensor(), normalize,]
        ),
        2: transforms.Compose(
            [transforms.RandomRotation([180,180]), transforms.ToTensor(), normalize,]
        ),
        3: transforms.Compose(
            [transforms.RandomRotation([270,270]), transforms.ToTensor(), normalize,]
        ),
    }
    




    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    #     AddGaussianNoise(0., noise_level, net_id, total)])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     AddGaussianNoise(0., noise_level, net_id, total)])


    total = args.num_clusters  #to customize noise level acording to clusters
    noise_level = args.noise_level #initial noise level
    #net_id is fized here to either 0/1/2/3  
    
    transforms_dict_noise = {
        0: transforms.Compose(
            [transforms.ToTensor(), AddGaussianNoise(0., noise_level, 0, total)] #, normalize,
        ),
        1: transforms.Compose(
            [transforms.ToTensor(), AddGaussianNoise(0., noise_level, 1, total)]
        ),
        2: transforms.Compose(
            [transforms.ToTensor(), AddGaussianNoise(0., noise_level, 2, total)]
        ),
        3: transforms.Compose(
            [transforms.ToTensor(), AddGaussianNoise(0., noise_level, 3, total)]
        ),
    }


    if args.clustering == 'rotation':  #total data in all clusters will be same (for MNIST itwill be 60000, but diff rotation) 
        train_datasets = {}
        test_datasets = {}
        for r in range(args.num_clusters):
            if args.dataset == 'mnist':
                train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                      transform=transforms_dict[r])

                test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=transforms_dict[r])
            
            elif args.dataset == 'fmnist':
                train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                      transform=transforms_dict[r])

                test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=transforms_dict[r])

            elif args.dataset == 'cifar10':
                train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                      transform=transforms_dict[r])
                test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transforms_dict[r])

            elif args.dataset == 'cifar100':
                train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                      transform=transforms_dict[r])
                test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=transforms_dict[r])


            train_datasets[r] = train_dataset
            test_datasets[r] = test_dataset

        return train_datasets, test_datasets
    
    elif args.clustering == 'noise':  #total data in all clusters will be same (for MNIST itwill be 60000, but diff rotation) 
        train_datasets = {}
        test_datasets = {}
        for r in range(args.num_clusters):
            if args.dataset == 'mnist':
                train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                      transform=transforms_dict_noise[r])

                test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=transforms_dict_noise[r])
            
            elif args.dataset == 'fmnist':
                train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                      transform=transforms_dict_noise[r])

                test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=transforms_dict_noise[r])

            elif args.dataset == 'cifar10':
                train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                      transform=transforms_dict_noise[r])
                test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transforms_dict_noise[r])

            elif args.dataset == 'cifar100':
                train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                      transform=transforms_dict_noise[r])
                test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=transforms_dict_noise[r])


            train_datasets[r] = train_dataset
            test_datasets[r] = test_dataset

        return train_datasets, test_datasets

    else:  # no clustering or label based clustering(standard transformation only)
            if args.dataset == 'mnist':
                train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                      transform=transform_standard)

                test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=transform_standard)
            
            elif args.dataset == 'fmnist':
                train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                      transform=transform_standard)

                test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=transform_standard)


            elif args.dataset == 'cifar10':
                train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                      transform=transform_standard)  # transform_train
                test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transform_standard)

            elif args.dataset == 'cifar100':
                train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                      transform=transform_standard)  # transform_train
                test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=transform_standard)


            return train_dataset, test_dataset
          
    
    
    
    # # sample training data amongst users
    # if args.iid:
    #     # Sample IID user data from Mnist
    #     user_groups = dist_iid(train_dataset, args.num_clients)
    # else:
    #     # Sample Non-IID user data from Mnist
    #     if args.unequal:
    #         # Chose uneuqal splits for every user
    #         user_groups = dist_dirichlet(train_dataset, args.num_clients, args.alpha, args.num_classes, args.global_seed)
    #     else:
    #         # Chose euqal splits for every user
    #         user_groups = dist_pathological(train_dataset, args.num_clients, args.shard_size)

   
    #train dataset is used as validation set(to be split for server)
    #return train_dataset, test_dataset


if __name__ == '__main__':
    pass