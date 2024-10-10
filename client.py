import torch.nn.functional as F

from copy import deepcopy
from utils.torch_utils import *


class Client(object):
    r"""Implements one clients

    Attributes
    ----------
    learners_ensemble
    n_learners

    train_iterator

    val_iterator

    test_iterator

    train_loader

    n_train_samples

    n_test_samples

    samples_weights

    local_steps

    logger

    tune_locally:

    Methods
    ----------
    __init__
    step
    write_logs
    update_sample_weights
    update_learners_weights

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False
    ):

        self.learners_ensemble = learners_ensemble
        self.n_learners = len(self.learners_ensemble)
        self.tune_locally = tune_locally

        if self.tune_locally:
            self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)
        else:
            self.tuned_learners_ensemble = None

        self.binary_classification_flag = self.learners_ensemble.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        ##########################################################
        # self.num_classes = 10 #TODO: change this
        # self.sample_per_class = torch.zeros(self.num_classes)
        # self.test_sample_per_class = torch.zeros(self.num_classes)

        # for x, y in self.train_iterator:
        #     for yy in y:
        #         self.sample_per_class[yy.item()] += 1
        # print(f'Train dist: {self.sample_per_class}')
        # self.id_labels = [i for i in range(self.num_classes) if self.sample_per_class[i]>0]
        # #print(f'ID Labels: {self.id_labels}')
        
        # for x, y in self.test_iterator:
        #     for yy in y:
        #         self.test_sample_per_class[yy.item()] += 1
        # print(f'Test dist: {self.test_sample_per_class}')

        # print(f'Client data count: Train: {len(self.train_iterator.dataset)} Test: {len(self.test_iterator.dataset)}\n')
        ###########################################################


        # self.train_loader = iter(self.train_iterator)
        # self.test_loader = iter(self.test_iterator)
        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)



        self.samples_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners

        self.local_steps = local_steps
        self.device = self.learners_ensemble.learners[0].device

        self.counter = 0
        self.logger = logger

    # def get_next_batch(self):
    #     try:
    #         batch = next(self.train_loader)
    #     except StopIteration:
    #         self.train_loader = iter(self.train_iterator)
    #         batch = next(self.train_loader)

    #     return batch



    def get_next_batch(self):
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch





    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1
        self.update_sample_weights()
        self.update_learners_weights()

        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights
                )
        # print('Theta updated')        

        
        # self.update_prototypes()  #not common for all

        # TODO: add flag arguments to use `free_gradients`
        # self.learners_ensemble.free_gradients()

        return client_updates

    def write_logs(self):
        if self.tune_locally:
            self.update_tuned_learners()

        if self.tune_locally:
            train_loss, train_acc = self.tuned_learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.tuned_learners_ensemble.evaluate_iterator(self.test_iterator)
        else:
            # print('Client evaluating on val iterator')
            train_loss, train_acc = self.learners_ensemble.evaluate_iterator(self.val_iterator)
            # print('Client evaluating on test iterator')
            test_loss, test_acc = self.learners_ensemble.evaluate_iterator(self.test_iterator, test_flag=True)
            


        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc

    def update_sample_weights(self):
        pass

    def update_learners_weights(self):
        pass

    def update_tuned_learners(self):
        print('Updating client tuned learners')

        if not self.tune_locally:
            return
        for learner_id, learner in enumerate(self.tuned_learners_ensemble):
            copy_model(source=self.learners_ensemble[learner_id].model, target=learner.model)
            learner.fit_epochs(self.train_iterator, self.local_steps, weights=self.samples_weights[learner_id])





class MixtureClient(Client):

    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator) #[n_lerners x iterator_samples]    criteria(y_pred,y) for each lrnr       
        #print(f'Dimensions of learners_weights:{self.learners_ensemble.learners_weights.shape}, all_losses:{all_losses.shape}')
        self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T  
        #print(f'Sample weights dimension: {self.samples_weights.shape}')

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1) #[nn_lrnr,n_sample].mean =mx1




class MixtureClientCont(Client):
    
    def __init__(self, learners_ensemble, train_iterator, val_iterator, test_iterator, logger, local_steps, tune_locally=False):
        super().__init__(learners_ensemble, train_iterator, val_iterator, test_iterator, logger, local_steps, tune_locally)
        
        # Initialize prototypes from local data when the client is created
        self.init_prototypes_from_local_data()
        print("Prototypes initialized from local data in the client.")
        

    #define own step() including prototype update
    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        Perform one step for the client with the updated global prototypes.
        """
        
        # if self.counter == 0:
        #     self.init_prototypes_from_local_data()
        
        
        self.counter += 1
        self.update_sample_weights() #q

        self.update_learners_weights() #pi


        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights
                )
        
        # print('Theta updated')        
        
        #single set from m set of features
        self.compute_prototypes()
        # print('Local prototype updated. One set form m component fetures.')

        return client_updates


    
    def init_prototypes_from_local_data(self):
        """
        Initialize class prototypes using the local data available on the client.
        Prototypes are initialized as the mean of features for each class, aggregated from the m models in the ensemble.
        """
        num_classes = self.learners_ensemble.learners[0].num_classes
        feature_dim = self.learners_ensemble.learners[0].feat_dim  # Assuming this attribute exists
        device = self.learners_ensemble.device

        prototypes = torch.zeros(num_classes, feature_dim).to(device)
        class_counts = torch.zeros(num_classes).to(device)

        with torch.no_grad():
            for batch in self.train_iterator:
                x, y, _ = batch
                x = x.to(device)
                y = y.to(device)

                # Extract and aggregate features from the ensemble of learners
                features = self.learners_ensemble.get_features(self.train_iterator,torch.arange(len(y)))    #mean features for every sample from learners (learners model)

                # Accumulate features for each class
                for class_idx in range(num_classes):
                    class_mask = (y == class_idx)
                    if class_mask.any():
                        class_features = features[class_mask]
                        prototypes[class_idx] += class_features.sum(dim=0)
                        class_counts[class_idx] += class_mask.sum()

        # Avoid division by zero
        class_counts = class_counts.unsqueeze(1).clamp(min=1.0)

        # Compute the mean for each class prototype
        prototypes /= class_counts

        # Normalize prototypes
        prototypes = F.normalize(prototypes, dim=1)

        # Update the learner's ensemble with the initialized prototypes
        for class_idx in range(num_classes):
            self.learners_ensemble.update_prototype(class_idx, 0, prototypes[class_idx])

        # print("Prototypes initialized from local data.")



    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator) #[n_lerners x iterator_samples]    criteria(y_pred,y) for each lrnr       
        #print(f'Dimensions of learners_weights:{self.learners_ensemble.learners_weights.shape}, all_losses:{all_losses.shape}')
        self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T  
        #print(f'Sample weights dimension: {self.samples_weights.shape}')
        # print('q() updated...')

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1) #[nn_lrnr,n_sample].mean =mx1
        # print('pi updated...')

    # def update_prototypes(self):
    #     """M-step: Update prototypes on the client based on the posterior probabilities."""
    #     # Iterate over each class and update its prototype
    #     for class_idx in range(self.learners_ensemble.num_classes):
    #         class_indices = (self.train_iterator.dataset.targets == class_idx)
    #         class_features = self.learners_ensemble.get_features(class_indices)
    #         if len(class_features) > 0:
    #             # Compute the sum of class features
    #             feature_sum = class_features.sum(dim=0)
    #             # Normalize the sum to get the prototype
    #             new_prototype = F.normalize(feature_sum, dim=0)
    #             # Update the prototype in the learner's ensemble
    #             self.learners_ensemble.update_prototype(class_idx, new_prototype)
    #     print('mu updated...')


   
    #every model m has its local prototype 
    # def update_prototypes(self):
    #     """M-step: Update prototypes on the client based on the posterior probabilities."""
    #     # Iterate over each class and each mixture component to update its prototype
    #     for class_idx in range(self.learners_ensemble.num_classes):
    #         for m in range(self.n_learners):  #mixture component
    #             # Get the indices of samples belonging to the current class
    #             class_indices = (self.train_iterator.dataset.targets == class_idx)
                
    #             # Get the posterior probabilities q_t(z_t^i = m) for the current mixture component
    #             posteriors = self.samples_weights[m, class_indices]
                
    #             # Get the features for the samples corresponding to the current class
    #             class_features = self.learners_ensemble.get_features(class_indices)
                
    #             if len(class_features) > 0:
    #                 # Compute the weighted sum of class features using the posteriors as weights
    #                 weighted_features_sum = (posteriors.unsqueeze(1) * class_features).sum(dim=0)
                    
    #                 # Normalize by the sum of the posteriors
    #                 sum_of_posteriors = posteriors.sum()
    #                 if sum_of_posteriors > 0:
    #                     weighted_features_sum /= sum_of_posteriors
                    
    #                 # Normalize the result to get the prototype
    #                 new_prototype = F.normalize(weighted_features_sum, dim=0)
                    
    #                 # Update the prototype in the learner's ensemble for the current mixture component
    #                 self.learners_ensemble.update_prototype(class_idx, m, new_prototype)
        
    #     print('mu updated...')

    
    def compute_prototypes(self):
        """M-step: Update prototypes on the client based on the posterior probabilities."""
        # Initialize a dictionary to store prototypes for each class
        num_classes = self.learners_ensemble.learners[0].num_classes
        class_prototypes = {class_idx: torch.zeros(self.learners_ensemble.learners[0].feat_dim).to(self.device) for class_idx in range(num_classes)}
        class_weights = {class_idx: 0.0 for class_idx in range(num_classes)}

        # Iterate over each class and each mixture component to compute weighted prototypes
        for class_idx in range(num_classes):
            for m in range(self.n_learners):  # mixture component
                # Get the indices of samples belonging to the current class
                class_indices = (self.train_iterator.dataset.targets == class_idx)
                
                # Get the posterior probabilities q_t(z_t^i = m) for the current mixture component
                posteriors = self.samples_weights[m, class_indices]
                # print(f'Posterior of class {class_idx}: {posteriors.shape}')
                
                # Get the features for the samples corresponding to the current class
                class_features = self.learners_ensemble.get_features(self.train_iterator,class_indices)
                # print(f'Class features shape: {class_features.shape}')
                
                if len(class_features) > 0:
                    # Compute the weighted sum of class features using the posteriors as weights
                    # print(f'Posterior unsquueze shape: {posteriors.unsqueeze(1).shape}')
                    weighted_features_sum = (posteriors.unsqueeze(1).to(self.device) * class_features).sum(dim=0)
                    # print(f'Weighted feature sum: {weighted_features_sum.shape}')
                    
                    # Update the prototype sum and weight sum for this class
                    class_prototypes[class_idx] += weighted_features_sum
                    class_weights[class_idx] += posteriors.sum()
        
        # Normalize the prototypes across all models
        for class_idx in range(num_classes):
            if class_weights[class_idx] > 0:
                class_prototypes[class_idx] /= class_weights[class_idx]
                class_prototypes[class_idx] = F.normalize(class_prototypes[class_idx], dim=0)
                # Update the prototype in the learner's ensemble
                self.learners_ensemble.update_prototype(class_idx, 0, class_prototypes[class_idx])  # Assuming 0 is the index for the aggregated prototype
        
        # print('mu updated from m features...')    

    
    
    def update_prototypes_from_server(self, global_prototypes):
        """
        Updates the local prototypes based on the aggregated prototypes from the server.
        
        Parameters:
        - global_prototypes: Tensor of shape [num_classes, feature_dim] from the server.
        """
        for class_idx in range(self.learners_ensemble.learners[0].num_classes):
            # Update the local prototype using the server's global prototype for each class
            self.learners_ensemble.update_prototype(class_idx, 0, global_prototypes[class_idx])
        
        # print('Local prototypes received and saved from server.')




class AgnosticFLClient(Client):
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False
    ):
        super(AgnosticFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."

    def step(self, *args, **kwargs):
        self.counter += 1

        batch = self.get_next_batch()
        losses = self.learners_ensemble.compute_gradients_and_loss(batch)

        return losses


class FFLClient(Client):
    r"""
    Implements client for q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            q=1,
            tune_locally=False
    ):
        super(FFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."
        self.q = q

    def step(self, lr, *args, **kwargs):

        hs = 0
        for learner in self.learners_ensemble:
            initial_state_dict = self.learners_ensemble[0].model.state_dict()
            learner.fit_epochs(iterator=self.train_iterator, n_epochs=self.local_steps)

            client_loss, _ = learner.evaluate_iterator(self.train_iterator)
            client_loss = torch.tensor(client_loss)
            client_loss += 1e-10

            # assign the difference to param.grad for each param in learner.parameters()
            differentiate_learner(
                target=learner,
                reference_state_dict=initial_state_dict,
                coeff=torch.pow(client_loss, self.q) / lr
            )

            hs = self.q * torch.pow(client_loss, self.q-1) * torch.pow(torch.linalg.norm(learner.get_grad_tensor()), 2)
            hs /= torch.pow(torch.pow(client_loss, self.q), 2)
            hs += torch.pow(client_loss, self.q) / lr

        return hs / len(self.learners_ensemble)
