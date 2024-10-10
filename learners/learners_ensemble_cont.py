import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnersEnsembleCont(object):
    """
    Iterable Ensemble of Learners.

    Attributes
    ----------
    learners
    learners_weights
    model_dim
    is_binary_classification
    device
    metric

    Methods
    ----------
    __init__
    __iter__
    __len__
    compute_gradients_and_loss
    optimizer_step
    fit_epochs
    evaluate
    gather_losses
    free_memory
    free_gradients

    """
    def __init__(self, learners, learners_weights):
        self.learners = learners
        self.learners_weights = learners_weights

        self.model_dim = self.learners[0].model_dim
        self.is_binary_classification = self.learners[0].is_binary_classification
        self.device = self.learners[0].device
        self.metric = self.learners[0].metric

        self.prototypes = torch.zeros(self.learners[0].num_classes, 1, self.learners[0].feat_dim).to(self.device) # torch.zeros(self.num_classes, len(self.learners), self.feature_dim).to(self.device)
        
    ########################################################################################
    def get_features(self, iterator, indices):
        """
        Extract and aggregate features for the given indices from all learners in the ensemble.
        
        :param indices: Indices of the samples to extract features for.
        :return: Tensor of aggregated features.
        """
        features = []

        # Iterate over each learner and extract features
        for learner in self.learners:
            learner_features = learner.get_features(iterator, indices)
            features.append(learner_features)
        
        # Stack features from all learners and average them
        aggregated_features = torch.stack(features).mean(dim=0)
        
        return aggregated_features

    #single prototype
    # def update_prototype(self, class_idx, new_prototype):
    #     """Update the prototype for a specific class."""
    #     self.prototypes[class_idx] = new_prototype
    
    #m prototypes
    def update_prototype(self, class_idx, component_idx, new_prototype):
        """Update the prototype for a specific class and mixture component."""
        self.prototypes[class_idx][component_idx] = new_prototype


    # def get_num_classes(self):
    #     """Return the number of classes."""
    #     return self.prototypes.size(0)
    ########################################################################################

    def optimizer_step(self):
        """
        perform one optimizer step, requires the gradients to be already computed
        """
        for learner in self.learners:
            learner.optimizer_step()

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        losses = []
        for learner_id, learner in enumerate(self.learners):
            loss = learner.compute_gradients_and_loss(batch, weights=weights)
            losses.append(loss)

        return losses

    # def fit_batch(self, batch, weights):
    #     """
    #     updates learners using  one batch.

    #     :param batch: tuple of (x, y, indices)
    #     :param weights: tensor with the learners_weights of each sample or None
    #     :type weights: torch.tensor or None
    #     :return:
    #         client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
    #         and the updated parameters for each learner in the ensemble.

    #     """
    #     client_updates = torch.zeros(len(self.learners), self.model_dim)

    #     for learner_id, learner in enumerate(self.learners):
    #         old_params = learner.get_param_tensor()
    #         if weights is not None:
    #             learner.fit_batch(batch=batch, weights=weights[learner_id])
    #         else:
    #             learner.fit_batch(batch=batch, weights=None)

    #         params = learner.get_param_tensor()

    #         client_updates[learner_id] = (params - old_params)

    #     return client_updates.cpu().numpy()

    # def fit_epochs(self, iterator, n_epochs, weights=None): #sample weights
    #     """
    #     perform multiple training epochs, updating each learner in the ensemble

    #     :param iterator:
    #     :type iterator: torch.utils.data.DataLoader
    #     :param n_epochs: number of epochs
    #     :type n_epochs: int
    #     :param weights: tensor of shape (n_learners, len(iterator)), holding the weight of each sample in iterator
    #                     for each learner ins ensemble_learners
    #     :type weights: torch.tensor or None
    #     :return:
    #         client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
    #         and the updated parameters for each learner in the ensemble.

    #     """
    #     client_updates = torch.zeros(len(self.learners), self.model_dim)

    #     for learner_id, learner in enumerate(self.learners):
    #         old_params = learner.get_param_tensor()
    #         if weights is not None:
    #             learner.fit_epochs(iterator, n_epochs, weights=weights[learner_id])
    #         else:
    #             learner.fit_epochs(iterator, n_epochs, weights=None)
    #         params = learner.get_param_tensor()

    #         client_updates[learner_id] = (params - old_params)

    #     return client_updates.cpu().numpy()


    def fit_batch(self, batch, weights):
        """
        Updates learners using one batch, with prototype and CE loss.
        """
        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):
            old_params = learner.get_param_tensor()

            # Call fit_batch for each learner, passing global_prototypes and weights
            learner.fit_batch(batch=batch, global_prototypes=self.prototypes, weights=weights[learner_id])

            params = learner.get_param_tensor()
            client_updates[learner_id] = (params - old_params)

        return client_updates.cpu().numpy()



    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        Perform multiple training epochs, using prototype and CE loss.
        """
        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):
            old_params = learner.get_param_tensor()

            # Call fit_epochs for each learner, passing global_prototypes and weights
            learner.fit_epochs(iterator, n_epochs, global_prototypes=self.prototypes, weights=weights[learner_id])

            params = learner.get_param_tensor()
            client_updates[learner_id] = (params - old_params)

        return client_updates.cpu().numpy()



    def evaluate_iterator(self, iterator):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """
        if self.is_binary_classification:
            criterion = nn.BCELoss(reduction="none")
        else:
            criterion = nn.NLLLoss(reduction="none")

        for learner in self.learners:
            learner.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)
                n_samples += y.size(0)

                y_pred = 0.
                for learner_id, learner in enumerate(self.learners):
                    rep = learner.model.base(x)  ###############################
                    if self.is_binary_classification:
                        y_pred += self.learners_weights[learner_id] * torch.sigmoid(learner.model.head(rep))  ###############################
                    else:
                        y_pred += self.learners_weights[learner_id] * F.softmax(learner.model.head(rep), dim=1) #############################

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)
                    global_loss += criterion(y_pred, y).sum().item()
                    y_pred = torch.logit(y_pred, eps=1e-10)
                else:
                    global_loss += criterion(torch.log(y_pred), y).sum().item()

                global_metric += self.metric(y_pred, y).item()

            return global_loss / n_samples, global_metric / n_samples





    def gather_losses(self, iterator):
        """
        gathers losses for all sample in iterator for each learner in ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor (n_learners, n_samples) with losses of all elements of the iterator.dataset

        """
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(len(self.learners), n_samples)
        for learner_id, learner in enumerate(self.learners):
            all_losses[learner_id] = learner.gather_losses(iterator, self.prototypes)

        return all_losses





    def free_memory(self):
        """
        free_memory: free the memory allocated by the model weights

        """
        for learner in self.learners:
            learner.free_memory()



    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        for learner in self.learners:
            learner.free_gradients()



    def __iter__(self):
        return LearnersEnsembleIterator(self)




    def __len__(self):
        return len(self.learners)




    def __getitem__(self, idx):
        return self.learners[idx]





class LearnersEnsembleIterator(object):
    """
    LearnersEnsemble iterator class

    Attributes
    ----------
    _learners_ensemble
    _index

    Methods
    ----------
    __init__
    __next__

    """
    def __init__(self, learners_ensemble):
        self._learners_ensemble = learners_ensemble.learners
        self._index = 0

    def __next__(self):
        while self._index < len(self._learners_ensemble):
            result = self._learners_ensemble[self._index]
            self._index += 1

            return result

        raise StopIteration
