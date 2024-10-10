import torch
from utils.losses_decood import *


class LearnerCont:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    is_binary_classification (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True

    Methods
    ------
    compute_gradients_and_loss:

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_epoch:

    fit_batches: perform successive optimizer steps over successive batches

    fit_epochs:

    evaluate_iterator: evaluate `model` on an iterator

    gather_losses:

    get_param_tensor: get `model` parameters as a unique flattened tensor

    free_memory: free the memory allocated by the model weights

    free_gradients:
    """

    def __init__(
            self, 
            model,
            criterion,
            metric,
            device,
            optimizer,
            lr_scheduler=None,
            is_binary_classification=False,
            num_classes = 10,
            feat_dim = 512
    ):

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_binary_classification = is_binary_classification
        self.feat_dim = feat_dim
        
        self.num_classes = num_classes
        ###########
        self.comp_loss = CompLoss(num_classes, device, feat_dim)                         
        ############
        self.model_dim = int(self.get_param_tensor().shape[0])
        

    # def set_pcl(self, args, train_ierator):
    
    #     self.loss_dis = DisLoss(self.num_classes, self.device,self.feat_dim, self.model, self.ierator, temperature = args.tau).cuda() #should be separate for different learners since 
    #     self.loss_comp = CompLoss(args,temperature = args.tau).cuda()
            
    
    def optimizer_step(self):
        """
         perform one optimizer step, requires the gradients to be already computed.
        """
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    
    # def get_features(self, iterator, indices):
    #     """
    #     Extract features for the given indices by passing the data through the base model.
        
    #     :param indices: Indices of the samples to extract features for.
    #     :return: Tensor of extracted features.
    #     """
    #     # Get the dataset and corresponding samples
    #     dataset = iterator.dataset

    #     x = dataset.data[indices].to(self.device).type(torch.float32)

    #     # If the dataset requires additional processing, handle it here
    #     if isinstance(dataset.data, torch.Tensor):
    #         x = dataset.data[indices].to(self.device).type(torch.float32)
    #     else:
    #         x = torch.stack([dataset[i][0] for i in indices]).to(self.device).type(torch.float32)
        

    #     print(x.shape)
    #     if x.shape[1] != 3:
    #         # Assuming x is (N, C, H, W) where C != 3, adjust this as needed
    #         # If your input has a different channel dimension, adjust the code here.
    #         x = x.permute(0, 3, 1, 2) # Convert (N, H, W, C) -> (N, C, H, W)
    #     print(x.shape)    


    #     # Pass the data through the base model to get features
    #     with torch.no_grad():
    #         self.model.eval()
    #         features = self.model.base(x)
    #         features = F.normalize(features, dim=1)  # Normalize features if needed

    #     return features



    def get_features(self, iterator, indices):
        """
        Extract features for the given indices by passing the data through the base model.
        
        :param indices: Indices of the samples to extract features for.
        :return: Tensor of extracted features.
        """
        # Get the dataset and corresponding samples
        dataset = iterator.dataset

        # Extract input data for the given indices
        if isinstance(dataset.data, torch.Tensor):
            x = dataset.data[indices].to(self.device).type(torch.float32)
        else:
            x = torch.stack([dataset[i][0] for i in indices]).to(self.device).type(torch.float32)

        # If the input has 3 dimensions (grayscale), add a channel dimension
        if len(x.shape) == 3:  # For grayscale images like FEMNIST
            x = x.unsqueeze(1)  # Convert from [batch_size, H, W] to [batch_size, 1, H, W]
        
        # For datasets like CIFAR-10 (if x has 4 dimensions but channels are last)
        if x.shape[1] != 1 and x.shape[1] != 3:  # Ensure that channels are in the correct position
            x = x.permute(0, 3, 1, 2)  # Convert (N, H, W, C) -> (N, C, H, W)

        print(f"Input shape after adjustments: {x.shape}")

        # Pass the data through the base model to get features
        with torch.no_grad():
            self.model.eval()  # Disable dropout and batch norm
            features = self.model.base(x)
            features = F.normalize(features, dim=1)  # Normalize features if needed

        return features





    # def compute_gradients_and_loss(self, batch, global_prototypes=None, weights=None):
    #     """
    #     Compute the gradients and loss over one batch, using updated global prototypes for prototype loss.
        
    #     :param batch: tuple of (x, y, indices)
    #     :param global_prototypes: the prototypes received from the server
    #     :param weights: tensor with the weights of each sample or None
    #     :return: loss
    #     """
    #     x, y, _ = batch
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     # Get features from the model
    #     features = self.model.base(x)

    #     # Compute compactness loss using the updated prototypes
    #     comp_loss = self.comp_loss(features, global_prototypes, y)

    #     # Optionally compute other losses (like cross-entropy) and combine
    #     ce_loss = self.criterion(self.model.head(features), y)

    #     # Combine the losses (you can adjust weights here if needed)
    #     loss = ce_loss + comp_loss

    #     # Compute gradients
    #     loss.backward()
        
    #     return loss


    def compute_gradients_and_loss(self, batch, global_prototypes = None, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        self.model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        #y_pred = self.model(x)        ###################################################################################################################
        
        rep = self.model.base(x)       ###################################################################################################################
        #y_pred = self.model.head(rep) ###################################################################################################################
        

        rep = F.normalize(rep, dim=1)                  
        output = self.model.head(rep)
        loss_CE = self.criterion(output, y)  #CE loss

        learner_id = 0 #signle set of prototype is used and kept in 0
        relevant_prototypes = global_prototypes[:, learner_id, :]  # Shape: [num_classes, feature_dim]
        
        lc = self.comp_loss(rep, relevant_prototypes, y)
                        
        loss_vec = loss_CE + lc ##################loss_ce+0.2 lc

        #loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        return loss.detach()

    


    def fit_batch(self, batch, global_prototypes = None, weights=None):
        """
        perform an optimizer step over one batch drawn from `iterator`

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        #y_pred = self.model(x) ######################################################################################
        
        rep = self.model.base(x) #####################################################################################
        y_pred = self.model.head(rep) ################################################################################


        learner_id = 0  # Ensure that each learner has a unique id
        relevant_prototypes = global_prototypes[:, learner_id, :]  # Shape: [num_classes, feature_dim]

        comp_loss = self.comp_loss(rep, relevant_prototypes, y)   ################

        loss_vec = self.criterion(y_pred, y) + comp_loss     ############## + comp_loss
        metric = self.metric(y_pred, y) / len(y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.detach(), metric.detach()



    def fit_epoch(self, iterator, global_prototypes = None, weights=None):
        """
        theta_update = q(x) * gradient
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            #y_pred = self.model(x) ########################################################################################################################
        
            rep = self.model.base(x) #######################################################################################################################
            y_pred = self.model.head(rep) ##################################################################################################################

            learner_id = 0  # Ensure that each learner has a unique id
            relevant_prototypes = global_prototypes[:, learner_id, :]  # Shape: [num_classes, feature_dim]
            comp_loss = self.comp_loss(rep, relevant_prototypes, y) ##############
            
            loss_ce = self.criterion(y_pred, y)

            loss_vec = loss_ce  + comp_loss       ########################## 3 + comp_loss

            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()
            
            print(f'loss_ce: {loss_ce.mean()} loss_comp: {comp_loss.mean()} weighted_loss_mean: {loss}')

            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).detach()
        
        print('\n')

        return global_loss / n_samples, global_metric / n_samples
    



    def gather_losses(self, iterator, global_prototypes):   ####only CE loss
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)

                #y_pred = self.model(x) ###################################################################################################
        
                rep = self.model.base(x) ##################################################################################################
                y_pred = self.model.head(rep) #############################################################################################

                learner_id = 0  # Ensure that each learner has a unique id
                relevant_prototypes = global_prototypes[:, learner_id, :]  # Shape: [num_classes, feature_dim]
                comp_loss = self.comp_loss(rep, relevant_prototypes, y) ##############


                all_losses[indices] = self.criterion(y_pred, y).squeeze() + comp_loss   ################## + comp_loss

        return all_losses



    def evaluate_iterator(self, global_prototypes, iterator):  ####only CE Loss
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, _ in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            with torch.no_grad():
                #y_pred = self.model(x) ########################################################################################################################
        
                rep = self.model.base(x) ########################################################################################################################
                y_pred = self.model.head(rep) ###################################################################################################################

                learner_id = 0  # Ensure that each learner has a unique id
                relevant_prototypes = global_prototypes[:, learner_id, :]  # Shape: [num_classes, feature_dim]
                comp_loss = self.comp_loss(rep, relevant_prototypes, y) ##############

                
                global_loss += self.criterion(y_pred, y).sum().detach() + comp_loss.sum().detach() ##########
                global_metric += self.metric(y_pred, y).detach()

            n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples



    def fit_epochs(self, iterator, n_epochs, global_prototypes, weights=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch(iterator, global_prototypes,  weights)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()



    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)


    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)


    def free_memory(self):
        """
        free the memory allocated by the model weights

        """
        del self.optimizer
        del self.model

    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        self.optimizer.zero_grad(set_to_none=True)

