"""
Client implementation for DFedDG2: DECOOD + von Mises-Fisher (vMF) prototype learning.

Each client in the decentralized federated network:
  - Maintains a backbone + classification head model (``model.base`` / ``model.head``).
  - Trains locally using a combination of cross-entropy and DECOOD losses.
  - Computes per-class prototype embeddings from the final local training epoch.
  - Exchanges prototypes with graph neighbors via vMF-likelihood-weighted gossip
    (handled externally by ``comm_utils/comm_vmf_gossip.py``).

Loss variants (controlled by ``args.decood_loss_code``):
  - ``'CD'``  : CompLoss + DisLoss only (no cross-entropy).
  - ``'ECD'`` : CE + λ*(CompLoss + DisLoss).
  - ``'ECP'`` : CE + 0.2*CompLoss + 0.1*ProtoLoss  (FedProto-style; lp currently inactive).
  - ``'EC'``  : CE + λ*CompLoss  ← recommended default.

ID vs. OOD classes:
  - ID (in-distribution) labels: classes present in this client's local training data.
  - OOD labels: classes absent locally; their prototypes are seeded from neighbors via
    the DisLoss buffer once a neighbor has contributed a non-default value.
"""

import os
import copy
import torch
import random
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from utils.losses_decood import *
from utils.imb_losses import *
from sklearn import metrics
# from data.data_utils import DatasetSplit
from lib.data_utils import DatasetSplit
from sklearn.preprocessing import label_binarize



class ClientDecoodVMF():
    """
    Federated client implementing DECOOD + vMF prototype learning.

    The client maintains a backbone/head split model and computes per-class
    prototype embeddings from local training data. Prototypes are shared with
    graph neighbors and aggregated using vMF-likelihood-weighted gossip.

    ID classes (labels present in local training data) are learned directly.
    OOD classes (labels absent locally) have their prototypes seeded from
    neighbor contributions via the DisLoss buffer once a neighbor has provided
    a non-default value.

    Args:
        id (int): Unique client identifier.
        args (Namespace): Global experiment configuration. Key fields:
            ``comm``, ``device``, ``local_bs``, ``local_epochs``, ``num_classes``,
            ``feat_dim``, ``normalize``, ``tau``, ``LAMBDA``, ``decood_loss_code``,
            ``proto_m``, ``lr``, ``momentum``, ``weight_decay``,
            ``learning_rate_decay``, ``test_on_cosine``, ``use_imb_loss``,
            ``data_type``, ``model``.
        adj: Adjacency row for this client (used externally by comm scripts).
        dataset_train: Full training dataset (subset selected by ``user_data_idx_train``).
        user_data_idx_train (list): Sample indices for this client's training split.
        dataset_test: Full test dataset (subset selected by ``user_data_idx_test``).
            If None, test split is carved from training data (90/10).
        user_data_idx_test (list | None): Sample indices for this client's test split.
        cluster (int | None): Optional cluster ID (currently unused).
    """
    def __init__(self, id, args, adj, dataset_train=None, user_data_idx_train=None, dataset_test=None, user_data_idx_test=None, cluster=None):
        self.id = id
        self.comm = args.comm
        self.device = args.device
        self.is_bayes = args.is_bayes
        self.adj = adj
        self.local_bs = args.local_bs
        self.local_epochs = args.local_epochs
        self.save_folder_name = args.save_folder_name
        self.global_seed = args.global_seed
        self.grad = []                            # Model gradients after local update (reserved for gradient-based methods)
        self.test_on_cosine = args.test_on_cosine # If True, use cosine similarity for prototype-based inference
        self.feat_dim = args.feat_dim
        self.neighbors_id = None                  # Fixed neighbor list (used by Penz-style algorithms)
        self.data_type = args.data_type           # Data modality: 'image', 'pc' (point cloud), etc.
        self.use_imb_loss = args.use_imb_loss
        self.cluster = cluster                    # Cluster assignment (reserved; not used in current algorithm)
        self.clients_in_same_cluster = []

        # Dataset — if a separate test set is provided, use it directly;
        # otherwise carve val (10%) and test (10%) from the training indices.
        if dataset_test is not None:
            self.train_loader, self.val_loader, self.test_loader = self.train_val_test(
                dataset_train, list(user_data_idx_train), dataset_test, list(user_data_idx_test))
        else:
            self.train_loader, self.val_loader, self.test_loader = self.train_val_test(
                dataset_train, list(user_data_idx_train), None, None)

        self.train_size = len(self.train_loader.dataset)
        self.unc = -1  # Sentinel: -1 means uncertainty not yet computed for this round

        self.num_classes = args.num_classes
        self.sample_per_class = torch.zeros(self.num_classes)       # Per-class train sample counts
        self.test_sample_per_class = torch.zeros(self.num_classes)  # Per-class test sample counts

        print(f'Cluster:{self.cluster}')
        for x, y, _ in self.train_loader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        print(f'Client id: {self.id}, Train dist: {self.sample_per_class}')

        # ID labels: classes that appear in this client's local training data
        self.id_labels = [i for i in range(self.num_classes) if self.sample_per_class[i] > 0]

        for x, y, _ in self.test_loader:
            for yy in y:
                self.test_sample_per_class[yy.item()] += 1
        print(f'Client id: {self.id}, Test dist: {self.test_sample_per_class}')
        print(f'Client data count: Train: {len(self.train_loader.dataset)} Test: {len(self.test_loader.dataset)}\n')

        # OOD labels: classes absent from local training data
        self.ood_labels = [l for l in range(args.num_classes) if l not in self.id_labels]

        # Model: backbone (base) + classification head; model0 reserved for gradient computation
        self.model = copy.deepcopy(args.model)
        self.model0 = copy.deepcopy(args.model)

        # Prototypes
        self.local_protos = None       # Per-class mean embeddings computed after local training
        self.global_protos = None      # Aggregated prototypes received from neighbors
        self.aggregated_protos = None  # Temporary buffer for gossip-aggregated prototypes (pre-sync)
        self.aggregated_kappas = None  # Temporary buffer for gossip-aggregated kappa values
        self.LAMBDA = args.LAMBDA      # Weight for DECOOD losses (lc, ld) in ECD/EC variants

        self.normalize = args.normalize
        self.loss_mse = nn.MSELoss(reduction='sum')
        self.loss_cs = nn.CosineSimilarity(dim=1)
        self.loss_CE = nn.CrossEntropyLoss()

        self.tau = args.tau  # Temperature for contrastive losses and prototype-based inference

        # Loss functions and optimizer
        self.decood_loss_code = args.decood_loss_code
        self.loss_dis = DisLoss(args, self.model, self.train_loader, temperature=args.tau).cuda()
        self.loss_comp = CompLoss(args, temperature=args.tau).cuda()
        self.kappa_hats = self.loss_dis.kappa_hat  # Per-class vMF concentration estimates

        if self.ood_labels:
            # Store the default (unupdated) DisLoss prototype for OOD class detection
            self.ood_init_proto = copy.deepcopy(self.loss_dis.prototypes[self.ood_labels[0]])

        self.learning_rate_decay = args.learning_rate_decay
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.lr,
            momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
        )
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay
        )

        # Aggregation state
        self.req_avg = False   # Flag: whether this client is awaiting an averaged model
        self.n_k = 1
        self.agg_count = 1     # Number of nodes whose parameters have been folded in so far

        # Per-round performance history (local and global evaluation)
        self.unc = 0
        self.l_test_acc_hist = []
        self.l_test_auc_hist = []
        self.l_test_unc_hist = []
        self.g_test_acc_hist = []
        self.g_test_auc_hist = []
        self.g_test_unc_hist = []

        

    def train_val_test(self, dataset_train, train_idxs, dataset_test, test_idxs):
        """
        Build train, validation, and test DataLoaders for this client.

        Two modes depending on whether a separate test dataset is provided:

        - **Separate test set** (``dataset_test`` is not None):
            All training indices are used for training; ``test_idxs`` indexes
            into ``dataset_test``. No validation loader is created (returns None).

        - **Single dataset** (``dataset_test`` is None):
            Training indices are split 80 / 10 / 10 into train / val / test.

        Training indices are shuffled before splitting so that repeated runs
        produce different orderings and averaged results are less order-dependent.

        Args:
            dataset_train: The full training dataset object.
            train_idxs (list[int]): Indices into ``dataset_train`` for this client.
            dataset_test: The full test dataset object, or None.
            test_idxs (list[int] | None): Indices into ``dataset_test``, or None.

        Returns:
            tuple: (trainloader, validloader, testloader)
                ``validloader`` is None when a separate test dataset is used.
        """
        random.shuffle(train_idxs)  # Shuffle so repeated calls yield different orderings

        if dataset_test is not None:
            idxs_train = train_idxs[:int(len(train_idxs))]  # Use all training indices
            idxs_test = test_idxs
        else:
            idxs_train = train_idxs[:int(0.8 * len(train_idxs))]                              # 80% train
            idxs_val   = train_idxs[int(0.8 * len(train_idxs)):int(0.9 * len(train_idxs))]   # 10% val
            idxs_test  = train_idxs[int(0.9 * len(train_idxs)):]                              # 10% test


        trainloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_train),
                                 batch_size=self.local_bs, shuffle=True, drop_last=False)
        
        if (dataset_test is not None) and (test_idxs is not None):
            validloader = None
            print('No validation set.')
            testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, idxs_test),
                                batch_size=self.local_bs, shuffle=False, drop_last=False)
        else:
            validloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_val),
                                 batch_size=self.local_bs, shuffle=False, drop_last=True)
        
            testloader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs_test),
                                batch_size=self.local_bs, shuffle=False, drop_last=True)

        
        return trainloader, validloader, testloader
    



    def update(self):
        """
        Run one round of local training (``local_epochs`` epochs over training data).

        Each batch:
          1. Extracts features via ``model.base``; optionally L2-normalizes them.
          2. Computes cross-entropy (CE) loss via ``model.head``.
          3. Computes DisLoss (ld) and CompLoss (lc) against EMA prototypes.
          4. Combines losses according to ``decood_loss_code``:
               - 'CD'  → lc + ld
               - 'ECD' → CE + λ*(lc + ld)
               - 'ECP' → CE + 0.2*lc + 0.1*lp  (lp currently inactive)
               - 'EC'  → CE + λ*lc
          5. Clips gradients to max-norm 1.0 before the optimizer step.

        After all epochs, per-class prototype embeddings are computed as the mean
        of final-epoch representations. OOD-class prototypes are adopted from the
        DisLoss buffer if a neighbor has already pushed a non-default value there.

        Side effects:
            Updates ``self.model`` weights, ``self.local_protos``, and ``self.kappa_hats``.
        """
        trainloader = self.train_loader

        self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        print(f'Local epoch: {max_local_epochs}')

        protos = defaultdict(list)
        avg_comp_loss = 0
        avg_dis_loss = 0
        avg_CE_loss = 0
        avg_lp_loss = 0

        for epoch in range(max_local_epochs):
            for (x, y, indices) in trainloader:

                x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                if torch.isnan(x).any():
                    print(f"NaN in input batch {i}")

                rep = self.model.base(x)

                if torch.isnan(rep).any() or torch.isinf(rep).any():
                    print("NaN or Inf detected in base model output!")

                if self.normalize:
                    rep = F.normalize(rep + 1e-8, dim=1)  # 1e-8: prevents zero-norm vectors before normalization

                output = self.model.head(rep)
                if self.use_imb_loss:
                    loss_CE = balanced_softmax_loss(y, output, self.sample_per_class)
                else:
                    loss_CE = self.loss_CE(output, y)

                # DisLoss updates EMA prototypes and returns dispersion loss
                ld = self.loss_dis(rep, y, self.loss_dis.prototypes)
                # CompLoss pulls features toward their class prototype
                lc = self.loss_comp(rep, self.loss_dis.prototypes, y)
                
                
                # NOTE: FedProto prototype alignment loss (lp) is disabled.
                # The 'ECP' variant references lp but it is not computed here.
                # Kept as a loss_code option for potential future re-enabling.

                if self.decood_loss_code == 'CD':
                    loss = lc + ld
                elif self.decood_loss_code == 'ECD':
                    loss = loss_CE + self.LAMBDA * (lc + ld)
                elif self.decood_loss_code == 'ECP':
                    loss = loss_CE + .2 * lc + .1 * lp  # lp currently inactive
                elif self.decood_loss_code == 'EC':
                    loss = loss_CE + self.LAMBDA * lc

                avg_CE_loss += loss_CE.data
                avg_comp_loss += lc.data
                avg_dis_loss += ld.data

                # Collect embeddings from the final epoch to compute local prototypes
                if epoch == max_local_epochs - 1:
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Prevents exploding gradients
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        avg_CE_loss = avg_CE_loss / (len(self.train_loader.dataset) * max_local_epochs)
        avg_comp_loss = avg_comp_loss / (len(self.train_loader.dataset) * max_local_epochs)
        avg_dis_loss = avg_dis_loss / (len(self.train_loader.dataset) * max_local_epochs)
        avg_lp_loss = avg_lp_loss / (len(self.train_loader.dataset) * max_local_epochs)

        print(f'Client {self.id} Avg Training : LCE:{avg_CE_loss}  LC: {avg_comp_loss}   LD: {avg_dis_loss}   LP: {avg_lp_loss}')

        # Compute per-class representative prototypes from final-epoch embeddings
        self.local_protos = agg_func(self.normalize, protos, self.num_classes, self.device)
        
        # Update OOD class prototypes from DisLoss prototypes if available
        for ood_class in self.ood_labels:
            ood_proto = self.loss_dis.prototypes[ood_class]
            if not torch.equal(ood_proto, self.ood_init_proto):  # it has been updated from default value by neighbor client
                # print(f'OOD proto updated for class {ood_class} from neighbors.')
                self.local_protos[ood_class] = ood_proto.clone().detach()
        
        self.kappa_hats = self.loss_dis.kappa_hat  # Per-class vMF concentration estimates (updated by DisLoss)


    

    def count_params(self, prototypes=None):
        """
        Count communication cost: prototype parameters vs. full model parameters.

        Useful for comparing the bandwidth of prototype-only gossip against
        full model parameter sharing (e.g., FedAvg).

        Note:
            The ``prototypes`` argument is accepted for API compatibility but is
            ignored; ``self.local_protos`` is always used as the prototype source.

        Args:
            prototypes: Unused. Kept for interface compatibility.

        Returns:
            tuple[int, int]: (total_proto_params, model_params)
                - ``total_proto_params``: Total number of scalar values in all local prototypes.
                - ``model_params``: Total number of trainable model parameters.
        """
        model_params = sum(p.numel() for p in self.model.parameters())
        prototypes = self.local_protos

        total_proto_params = 0
        for proto in prototypes.values():
            total_proto_params += proto.numel()

        return total_proto_params, model_params



    
    def init_local_proto(self):
        """
        Compute initial local prototypes without performing any training.

        Runs the frozen model in eval mode over ``local_epochs`` passes of the
        training data, collecting per-class embeddings, then averages them into
        ``self.local_protos``. Used to warm-start prototypes before the first
        communication round.

        Side effects:
            Sets ``self.local_protos``.
        """
        trainloader = self.train_loader
        self.model.eval()

        protos = defaultdict(list)

        for epoch in range(self.local_epochs):
            for (x, y, indices) in trainloader:
                x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                if self.normalize:
                    rep = F.normalize(rep, dim=1)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.local_protos = agg_func(self.normalize, protos, self.num_classes, self.device)

    

    def save_aggregated_protos(self, global_protos):
        """
        Store gossip-aggregated prototypes in a temporary buffer.

        Called by the communication script after aggregating neighbor prototypes.
        The buffer is committed to active use by ``set_global_protos`` and
        ``set_dis_loss_protos`` in the post-gossip step.

        Args:
            global_protos (dict[int, Tensor]): Aggregated prototype per class label.
        """
        self.aggregated_protos = global_protos

    def save_aggregated_kappas(self, global_kappas):
        """
        Store gossip-aggregated vMF concentration parameters in a temporary buffer.

        Args:
            global_kappas (dict[int, float | Tensor]): Aggregated kappa per class label.
        """
        self.aggregated_kappas = global_kappas

    def set_global_protos(self):
        """
        Commit buffered aggregated prototypes to ``self.global_protos``.

        Should be called after ``save_aggregated_protos`` to make the
        gossip-aggregated prototypes available for use in the next training round.
        """
        self.global_protos = self.aggregated_protos

    def set_dis_loss_protos(self):
        """
        Inject buffered aggregated prototypes into the DisLoss prototype buffer.

        Overwrites the EMA prototypes in ``self.loss_dis`` with the
        gossip-aggregated values so that the next local training round starts
        from the globally informed prototype positions.
        """
        for c in range(self.num_classes):
            self.loss_dis.prototypes[c] = self.aggregated_protos[c].data

    def print_protos(self):
        """Print the current DisLoss EMA prototype for each class (debug utility)."""
        print('Dis loss proto:\n')
        for c in range(self.num_classes):
            print(f'Class: {c} , prototype: {self.loss_dis.prototypes[c]}')

    
 
    
    def avg_model(self, run_agg_model_sd, run_agg_n_k):
        """
        Finalize model averaging by dividing accumulated state dict by node count.

        Divides every parameter tensor in ``run_agg_model_sd`` by ``run_agg_n_k``
        (the total number of nodes whose parameters were summed), then loads the
        result into this client's model.

        Args:
            run_agg_model_sd (dict): Accumulated (summed) model state dict.
            run_agg_n_k (int): Total number of nodes contributing to the sum.
        """
        for key in run_agg_model_sd.keys():
            run_agg_model_sd[key] /= run_agg_n_k

        self.model.load_state_dict(run_agg_model_sd)
        self.req_avg = False




    def agg(self, run_agg_model_sd, run_agg_n_k, rec_params):
        """
        Accumulate a received neighbor's model parameters into the running sum.

        Used during MST-based aggregation (``get_mst_agg_model``). Adds the
        received state dict element-wise to the running accumulator and increments
        the node count. The caller is responsible for calling ``avg_model`` to
        divide by the final count.

        Args:
            run_agg_model_sd (dict): Running accumulated state dict (modified in-place).
            run_agg_n_k (int): Current count of nodes accumulated so far.
            rec_params (tuple): (state_dict, n_k) from the receiving neighbor,
                as returned by ``get_mst_agg_model``.

        Returns:
            tuple[dict, int]: Updated (run_agg_model_sd, run_agg_n_k).
        """
        rec_model_sd = rec_params[0]
        run_agg_n_k += rec_params[1]  # Accumulate node count from the received subtree

        for key in run_agg_model_sd.keys():
            run_agg_model_sd[key] += rec_model_sd[key]

        return run_agg_model_sd, run_agg_n_k
    
    


    def get_mst_agg_model(self, visited, clients, adj):
        """
        Recursively aggregate model parameters over the MST via DFS.

        Traverses the spanning tree rooted at this client using depth-first search.
        At each unvisited neighbor, recursively collects and accumulates that
        subtree's parameters. If this client has no unvisited neighbors, its own
        state dict is returned unchanged.

        Args:
            visited (list[bool]): Shared visited array; updated in-place to avoid cycles.
            clients (list[ClientDecoodVMF]): All clients indexed by ID.
            adj (array-like): Adjacency matrix where adj[i][j] == 1 indicates an edge.

        Returns:
            tuple[dict, int]: (accumulated_state_dict, accumulated_node_count)
        """
        visited[self.id] = True

        run_agg_model_sd = copy.deepcopy(self.model.state_dict())
        run_agg_n_k = copy.deepcopy(len(self.train_loader))

        for i in range(len(visited)):
            if adj[self.id][i] == 1 and not visited[i]:
                run_agg_model_sd, run_agg_n_k = self.agg(
                    run_agg_model_sd, run_agg_n_k,
                    clients[i].get_mst_agg_model(visited, clients, adj)
                )

        return run_agg_model_sd, run_agg_n_k




    def performance_test(self, data_loader=None, saveFlag=0):
        """
        Evaluate the model on a data loader using prototype-based nearest-neighbour classification.

        For each sample, cosine similarity is computed against every local prototype.
        The class with the highest similarity is the predicted label.

        If ``saveFlag=1``, per-sample predictions, probabilities, and entropy-based
        uncertainty estimates are saved to a JSON file named
        ``predictions_uncertainties_client_{id}.json``.

        Falls back to returning (0, 1e-5, 0) if ``self.local_protos`` is not yet
        initialized (i.e., before the first training round).

        Args:
            data_loader: DataLoader to evaluate on. Defaults to ``self.test_loader``.
            saveFlag (int): Set to 1 to save per-sample predictions and uncertainties.

        Returns:
            tuple[float, float, float]: (accuracy, 0, 0)
                The second and third values are placeholder slots for AUC and UNC
                (not computed in this method).
        """
        if data_loader is None:
            data_loader = self.test_loader

        self.model.eval()

        test_acc = 0
        test_num = 0
        results = []  # To store predictions and uncertainties

        if self.local_protos is not None:
            with torch.no_grad():
                for (x, y, indices) in data_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)

                    if self.normalize:
                        rep = F.normalize(rep, dim=1)  # Normalize if required

                    # Initialize with +inf so that missing prototype slots don't win argmax
                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in self.local_protos.items():
                            output[i, j] = F.cosine_similarity(r.unsqueeze(0), pro.unsqueeze(0))

                    predictions = torch.argmax(output, dim=1)  # Get predicted class indices
                    test_acc += (torch.sum(predictions == y)).item()
                    test_num += y.shape[0]

                    if saveFlag == 1:
                        # Compute probabilities (softmax)
                        probabilities = F.softmax(output / self.tau, dim=1)  # Adjust with temperature if needed

                        # Compute entropy-based uncertainty for each prediction
                        for i in range(y.shape[0]):
                            predicted_class = predictions[i].item()
                            
                            # Get the predicted class probability
                            predicted_probability = probabilities[i, predicted_class].item()

                            # Shannon entropy as uncertainty proxy; 1e-8 prevents log(0)
                            entropy = -torch.sum(probabilities[i] * torch.log(probabilities[i] + 1e-8)).item()

                            # Store results
                            results.append({
                                "sample_index": indices[i].item(),
                                "true_label": y[i].item(),
                                "predicted_label": predicted_class,
                                "predicted_probability": predicted_probability,
                                "uncertainty": entropy
                            })

            # Save predictions and uncertainties to a file if saveFlag is set
            if saveFlag == 1:
                client_id = self.id if hasattr(self, 'id') else 'unknown_client'
                save_path = f"predictions_uncertainties_client_{client_id}.json"
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"Predictions and uncertainties saved to {save_path}")
                print(f'Saved uncertainties of client {client_id}')    

            return test_acc / test_num, 0, 0
        else:
            return 0, 1e-5, 0



    def get_local_unc(self):
        """
        Compute per-class vMF concentration (kappa) from local training data.

        Runs the model in eval mode over ``local_epochs`` passes, collects
        unnormalized feature vectors per class, averages them, then estimates
        kappa using the Hornik & Grün (2014) formula:

            R̂ = ||μ_unnorm||
            κ̂ = R̂ * (feat_dim - R̂²) / (1 - R̂²)

        Note: ``agg_func`` is called with ``normalize=False`` to preserve the
        pre-normalization norm needed for the kappa computation.

        Returns:
            dict[int, Tensor]: Per-class kappa estimates.
        """
        trainloader = self.train_loader
        self.model.eval()

        protos = defaultdict(list)
        kappa_hat = defaultdict(list)
        R_hat = defaultdict(list)

        for epoch in range(self.local_epochs):
            for (x, y, indices) in trainloader:
                x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                if self.normalize:
                    rep = F.normalize(rep, dim=1)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        # Compute unnormalized mean prototypes — normalization would destroy R̂
        new_protos = agg_func(False, protos, self.num_classes, self.device)

        for [label, proto] in protos.items():
            R_hat[label] = torch.norm(proto).detach()
            R_hat_sqr = R_hat[label] * R_hat[label]
            kappa_hat[label] = (R_hat[label] * (self.feat_dim - R_hat_sqr)) / (1 - R_hat_sqr)
            print(f'Label: {label}, R_hat[label]: {R_hat[label].detach()}, kappa_hat[label]: {kappa_hat[label].detach()}')

        return kappa_hat 
        

        
    
    def get_mis_classification(self, rec_label=None, rec_proto=None):
        """
        Measure misclassification rate when a received neighbor prototype is substituted.

        Replaces the local prototype for ``rec_label`` with ``rec_proto``, then
        evaluates nearest-neighbour classification on the training set using either
        cosine similarity or MSE distance (controlled by ``self.test_on_cosine``).

        If ``rec_label`` is None, evaluates using the unmodified local prototypes
        (useful as a baseline misclassification rate).

        Args:
            rec_label (int | None): Class label whose prototype is being replaced.
            rec_proto (Tensor | None): The received prototype tensor to substitute.

        Returns:
            float: Fraction of training samples misclassified (misclassification rate).
                   Returns None if ``self.local_protos`` is not initialized.
        """
        data_loader = self.train_loader

        self.model.eval()

        test_loss = 0
        test_num = 0
        
        if self.local_protos is not None:
            #create space
            tmp_local_protos = copy.deepcopy(self.local_protos)
            if rec_label is not None:
                tmp_local_protos[rec_label] = rec_proto
                print(f'Testing received proto of label :{rec_label} @ client : {self.id} ')
            else:
                print('Evaluating misclassification on local protos')

            # print('Normalize before testing ID')
            with torch.no_grad():
                # for x, y in data_loader:
                for (x, y, indices) in data_loader:
                    x = x[0]
                    
                    x = x.to(self.device)
                    y = y.to(self.device)
                    rep = self.model.base(x)

                    if self.normalize:
                        rep = F.normalize(rep, dim=1)                                                #Close it if normalize in the loss ###################*************


                    # Initialize with +inf; valid prototype slots will be overwritten below
                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in tmp_local_protos.items():
                            if type(pro) != type([]):
                                if self.test_on_cosine == False:
                                    output[i, j] = self.loss_mse(r, pro)   # MSE: lower = closer
                                else:
                                    output[i, j] = F.cosine_similarity(r.unsqueeze(0), pro.unsqueeze(0))

                    # argmin used for MSE mode (lower distance = better match)
                    test_loss += (torch.sum(torch.argmin(output, dim=1) != y)).item()
                    test_num += y.shape[0]

            return test_loss/test_num




    def performacne_train(self):  # Note: method name has a typo ('performacne'); kept for backward compatibility
        """
        Evaluate training-set loss with optional FedProto prototype alignment.

        Runs the model in eval mode over the training data and computes the
        cross-entropy loss. If ``self.global_protos`` is available, an MSE
        alignment term (weighted by ``self.lamda``) is added to pull local
        representations toward the global prototype of each sample's class.

        Note:
            This method calls ``self.load_train_data()`` which is not defined on
            this class; it may be a leftover from an earlier base class. Verify
            before calling.

        Returns:
            tuple[float, int]: (total_loss, total_samples)
        """
        trainloader = self.load_train_data()

        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            # for x, y in trainloader:
            for (x, y, indices) in trainloader:
                x = x[0]
                
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                if self.normalize:
                    rep = F.normalize(rep, dim=1)        #Close it if normalize in the loss ###################


                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

     
        return losses, train_num




def agg_func(normalize, protos, num_classes=10, device='cpu'):
    """
    Average per-class prototype lists into a single representative prototype tensor.

    For each class label, stacks all collected embedding vectors, computes their
    mean, and optionally applies L2 normalization along dim=0.

    Args:
        normalize (bool): If True, applies ``F.normalize(..., dim=0)`` to each
            averaged prototype. Should match the normalization used during training.
        protos (dict[int, list[Tensor]]): Mapping from class label to a list of
            feature tensors collected during local training.
        num_classes (int): Total number of classes. Accepted for API consistency
            but not used internally (only labels present in ``protos`` are processed).
        device (str): Unused; tensors retain their original device. Accepted for
            API consistency.

    Returns:
        dict[int, Tensor]: The input ``protos`` dict with lists replaced by
            single averaged (and optionally normalized) prototype tensors.

    Note:
        Modifies ``protos`` in-place and also returns it.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data

            if normalize:
                protos[label] = F.normalize(proto / len(proto_list), dim=0)
            else:
                protos[label] = proto / len(proto_list)
        else:
            if normalize:
                protos[label] = F.normalize(proto_list[0], dim=0)
            else:
                protos[label] = proto_list[0]

    return protos