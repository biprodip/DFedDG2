"""
Main script for DFedDG2 decentralized federated learning experiments.

This script:
- Parses command-line arguments
- Sets seeds and device
- Builds the communication topology
- Initializes clients and datasets
- Runs the selected communication method
- Saves per-trial and aggregated metrics

The goal here is to keep behavior compatible with the original script while
making the code slightly more professional and maintainable.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.model_manager import *

from comm_utils.topology_manager import *
from comm_utils.decentralized import *
from comm_utils.comm_decood import *
from comm_utils.comm_vmf_gossip import *
from comm_utils.comm_gossip import *
from comm_utils.mst import *
from client_manager import *
from utils import *
from comm_mst import *

# Add local lib folder to path
lib_dir = (Path(__file__).parent / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from lib.pcl_utils import *  # noqa: E402  (import after sys.path manipulation)

LOGGER = logging.getLogger(__name__)
DEFAULT_SEED = 1234


def seed_worker(worker_id: int) -> None:
    """
    Seed worker processes for DataLoader to ensure reproducibility.

    Parameters
    ----------
    worker_id : int
        Worker id passed from DataLoader.
    """
    # Make worker seed depend on worker_id but deterministic across runs.
    base_seed = DEFAULT_SEED
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for DECOOD / FedProto experiments."""
    parser = argparse.ArgumentParser(description="Arguments for DECOOD / FedProto")

    # General model specific
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--global_seed", default=1234, type=int, help="Global seed")
    parser.add_argument(
        "--algorithm",
        default="FedProto",
        help="Algorithm name: FedConCIDER/FedGradProto/Decood/FedProto",
    )
    parser.add_argument(
        "--comm",
        default="vanilla_proto",
        help="Communication method: comm_GH/vanilla_proto/gossip/clus/mst/grad_proto_decood/comm_mst/comm_vmf_gossip",
    )
    parser.add_argument(
        "--num_trials", type=int, default=3, help="Number of trials to run"
    )
    parser.add_argument(
        "--num_clients", default=20, type=int, help="Total number of clients"
    )
    parser.add_argument(
        "--mst_comm",
        default=False,
        action="store_true",
        help="Whether to use MST communication",
    )
    parser.add_argument(
        "--topo", default="sparse", help="Topology type: ring/sparse/fc"
    )
    parser.add_argument(
        "--dynamic_topo",
        type=int,
        default=1,
        help="Create adjacency matrix each round (1) or keep fixed (0)",
    )
    parser.add_argument(
        "--sparse_neighbors",
        default=5,
        type=int,
        help="Number of neighbors in sparse topology (also to create ring/fc)",
    )
    parser.add_argument(
        "--test_on_cosine",
        default=False,
        type=bool,
        help="Use cosine-based performance testing (otherwise MSE)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device type override (e.g. 'cuda', 'cpu'); if None, auto-detect",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="GPU device index for torch.cuda.set_device (e.g. 0, 1, ...)",
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        type=bool,
        help="Whether to disable CUDA even if available",
    )
    parser.add_argument(
        "--weighted_adj_mat",
        default=1,
        type=int,
        help="Use adjacency weights proportional to class sample distribution",
    )

    # Model
    parser.add_argument(
        "--model", default=None, help="Model: assigned based on algorithm and dataset"
    )
    parser.add_argument(
        "--backbone",
        default="mobilenet_proj",
        help="Backbone: mobilenet/mobilenet_proj/resnet_proj/CNNMnist",
    )
    parser.add_argument(
        "--head", default=None, help="Head: assigned based on algorithm and dataset"
    )
    parser.add_argument(
        "--feat_dim", default=512, type=int, help="Feature (embedding) dimension"
    )
    parser.add_argument(
        "--normalize",
        default=True,
        type=bool,
        help="Whether to L2-normalize features",
    )
    parser.add_argument(
        "--out_channel",
        default=64,
        type=int,
        help="Model_hetero experiments (CNN out channels)",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="Optimizer momentum parameter"
    )
    parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="Weight decay parameter"
    )
    parser.add_argument(
        "--use_imb_loss",
        default=True,
        type=bool,
        help="Use imbalance loss instead of standard cross entropy",
    )

    # Client specific
    parser.add_argument(
        "--num_rounds", default=20, type=int, help="Number of communication rounds"
    )
    parser.add_argument("--optimizer", default="sgd", help="Optimizer type")
    parser.add_argument(
        "--lr", default=0.1, type=float, help="Initial learning rate"
    )
    parser.add_argument(
        "--local_bs", default=32, type=int, help="Local batch size for training"
    )
    parser.add_argument(
        "--test_bs", default=32, type=int, help="Batch size for testing"
    )
    parser.add_argument(
        "--learning_rate_decay",
        default=0.9,
        type=float,
        help="Learning rate decay factor per round/epoch",
    )
    parser.add_argument(
        "--local_epochs", default=3, type=int, help="Number of local epochs"
    )

    # Dataset and point cloud specific
    parser.add_argument("--dataset", default="cifar10", help="Dataset name")
    parser.add_argument(
        "--num_classes", default=10, type=int, help="Number of classes"
    )
    parser.add_argument(
        "--dist", default="iid", help="Data distribution: iid/path/dir"
    )
    parser.add_argument(
        "--num_samples",
        default=0,
        type=int,
        help="Number of samples to be kept from full trainset for constrained experiments "
        "(0 means use all available)",
    )
    parser.add_argument(
        "--data_type",
        default="img",
        help="Data type: img/pc/text/tabular (image / point-cloud / text / tabular)",
    )
    parser.add_argument(
        "--num_point", type=int, default=1024, help="Point number for point clouds"
    )
    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="Use normals for PC"
    )
    parser.add_argument(
        "--process_data",
        action="store_true",
        default=False,
        help="Save processed data offline for PC",
    )
    parser.add_argument(
        "--use_uniform_sample",
        action="store_true",
        default=False,
        help="Use uniform sampling for PC",
    )

    # OOD
    parser.add_argument("--ood_dataset", default="svhn", help="OOD dataset name")
    parser.add_argument(
        "--ood_train_method",
        default="OE",
        help="OOD training method: energy/OE",
    )
    parser.add_argument(
        "--proto_bs_id", default=5, type=int, help="Proto batch size for in-distribution"
    )
    parser.add_argument(
        "--proto_bs_ood", default=5, type=int, help="Proto batch size for OOD"
    )
    parser.add_argument(
        "--ood_header_epoch",
        default=100,
        type=int,
        help="OOD header epoch (FedGH Proto)",
    )
    parser.add_argument(
        "--id_header_epoch",
        default=1,
        type=int,
        help="ID header epoch (FedGH Proto)",
    )

    # Clustering and Penz params
    parser.add_argument(
        "--clustering",
        default="noise",
        help="Clustering type: 'rotation'/'noise'/'label'/'None'",
    )
    parser.add_argument(
        "--noise_level",
        default=0.2,
        type=float,
        help="Noise level for 'noise' clustering",
    )
    parser.add_argument(
        "--num_clusters", default=1, type=int, help="Number of clusters"
    )
    parser.add_argument(
        "--num_sel_clients",
        default=4,
        type=int,
        help="Number of selected clients",
    )
    parser.add_argument(
        "--isDFL",
        default=True,
        type=bool,
        help="Whether decentralized federated learning (DFL)",
    )
    parser.add_argument(
        "--clustered_agg",
        default=False,
        type=bool,
        help="Whether clustered aggregation is used",
    )
    parser.add_argument(
        "--neighbour_selection",
        default="loss_based",
        help="Neighbour selection in Penz: grad_based/loss_based",
    )
    parser.add_argument(
        "--neighbour_exploration",
        default="greedy",
        help="Neighbour exploration type",
    )
    parser.add_argument(
        "--topk", default=7, type=int, help="Top-k parameter for neighbour selection"
    )
    parser.add_argument(
        "--submod",
        default=False,
        type=bool,
        help="Whether to use submodular selection",
    )
    parser.add_argument(
        "--n_samplings",
        default=1,
        type=int,
        help="Number of samplings for submodular selection",
    )
    parser.add_argument(
        "--confidence_level",
        default=0.95,
        type=float,
        help="Confidence level for selection",
    )
    parser.add_argument(
        "--grad_thres",
        default=0.0,
        type=float,
        help="Gradient threshold for screening",
    )
    parser.add_argument(
        "--sub_mod_sel_ratio",
        default=0.7,
        type=float,
        help="Sub-modular selection ratio",
    )
    parser.add_argument(
        "--fed_avg_sel",
        default=0.7,
        type=float,
        help="FedAvg selection ratio",
    )
    parser.add_argument(
        "--n_neighbor", default=4, type=int, help="Number of neighbors"
    )
    parser.add_argument(
        "--subset_ratio",
        default=0.5,
        type=float,
        help="Subset ratio for neighbor exploration",
    )
    parser.add_argument(
        "--avg_error_thres",
        default=0.01,
        type=float,
        help="Average error threshold",
    )

    # Proto-approach params
    parser.add_argument(
        "--decood_loss_code",
        default="CD",
        help="Decood loss type: ECD/ECP/EC/CD",
    )
    parser.add_argument(
        "--LAMBDA",
        default=0.2,
        type=float,
        help="Weight for prototype contrastive loss in L = L_CE + LAMBDA * L_PL",
    )
    parser.add_argument(
        "--training_mode",
        default="SupCon",
        help="Training mode (e.g., SupCon, CE, etc.)",
    )
    parser.add_argument(
        "--posterior_type",
        default="soft",
        help="Posterior type (e.g., soft/hard)",
    )
    parser.add_argument(
        "--proto_m",
        default=0.5,
        type=float,
        help="Prototype running-average parameter (used in DECOOD losses)",
    )
    parser.add_argument(
        "--lamda_fed_proto",
        default=1.0,
        type=float,
        help="Lambda for FedProto regularization (0.1/1/2/3 etc.)",
    )
    parser.add_argument(
        "--unequal", default=False, type=bool, help="Whether unequal data splits"
    )
    parser.add_argument(
        "--shard_size", default=500, type=int, help="Shard size for pathological setup"
    )
    parser.add_argument(
        "--no_shard_per_client",
        default=4,
        type=int,
        help="Number of shards per client",
    )
    parser.add_argument(
        "--tau",
        default=0.1,
        type=float,
        help="Temperature parameter for contrastive/prototype loss",
    )
    parser.add_argument(
        "--sel_on_kappa",
        default=True,
        type=bool,
        help="Kappa-based selection of prototypes for aggregation",
    )

    # Others
    parser.add_argument("--th", default=0.1, type=float, help="Generic threshold")
    parser.add_argument(
        "--is_bayes",
        default=False,
        type=bool,
        help="Whether Bayesian modeling is used",
    )
    parser.add_argument(
        "--p", default=0.1, type=float, help="Generic p-parameter"
    )
    parser.add_argument(
        "--mu", default=0.75, type=float, help="Mu parameter for FedProx-like methods"
    )
    parser.add_argument(
        "--m_in",
        default=-25,
        type=int,
        help="M_in parameter (e.g., for energy-based OOD)",
    )
    parser.add_argument(
        "--m_out",
        default=-7,
        type=int,
        help="M_out parameter (e.g., for energy-based OOD)",
    )

    # FedPCL arguments
    parser.add_argument(
        "--percent",
        type=float,
        default=1.0,
        help="Percentage of dataset to train with",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10,
        help="Number of training samples in total (FedPCL)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of test samples per dataset",
    )
    parser.add_argument(
        "--feature_iid",
        type=int,
        default=0,
        help="Default set to feature non-IID (0). Set to 1 for feature IID.",
    )
    parser.add_argument(
        "--label_iid",
        type=int,
        default=1,
        help="Default set to label non-IID (0). Set to 1 for label IID.",
    )
    parser.add_argument(
        "--test_ep",
        type=int,
        default=10,
        help="Number of test episodes for evaluation",
    )
    parser.add_argument(
        "--save_protos",
        type=int,
        default=1,
        help="Whether to save prototypes or not",
    )

    # Local arguments (FedPCL code parameters)
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=10,
        help="Number of samples per class (local)",
    )
    parser.add_argument(
        "--t", type=float, default=2.0, help="Coefficient of local loss"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet distribution parameter for non-IID splits",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        help="Local model type (e.g. 'cnn')",
    )

    # Directories
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../FedPCL/data/",
        help="Data directory (default: '../FedPCL/data/')",
    )
    parser.add_argument(
        "--save_folder_name",
        default="results/",
        help="Folder to save experiment results",
    )
    parser.add_argument(
        "--params_dir",
        default="params/",
        help="Directory to save client parameters / stats",
    )

    args = parser.parse_args()
    return args


def generate_filename(args: argparse.Namespace) -> str:
    """
    Generate a filename based on dataset, feature IID, and label IID settings.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    str
        Path to a .pkl file under data/.
    """
    feature_label_pattern = (
        f"feature_{'iid' if args.feature_iid else 'non_iid'}_"
        f"label_{'iid' if args.label_iid else 'non_iid'}_{args.dataset}.pkl"
    )
    filepath = os.path.join("data", feature_label_pattern)
    return filepath


def load_data(filename: str) -> Any:
    """
    Load the dataset and user-group data from a file.

    Parameters
    ----------
    filename : str
        Path to a pickle file.

    Returns
    -------
    Any
        Unpickled content of the file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def main() -> int:
    """Main entry point for running decentralized FL experiments."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    args = parse_arguments()

    # Ensure save directories exist
    os.makedirs(args.save_folder_name, exist_ok=True)
    os.makedirs(args.params_dir, exist_ok=True)

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.device is not None:
        device_str = args.device
    else:
        device_str = "cuda" if use_cuda else "cpu"
    args.device = torch.device(device_str)

    # DataLoader worker generator
    g = torch.Generator()
    g.manual_seed(0)

    train_kwargs = {
        "batch_size": args.local_bs,
        "num_workers": 1,
        "worker_init_fn": seed_worker,
        "generator": g,
    }
    test_kwargs = {
        "batch_size": args.test_bs,
        "num_workers": 1,
        "worker_init_fn": seed_worker,
        "generator": g,
    }

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Set global seeds
    LOGGER.info("Training on device: %s", args.device)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Build topology / mixing matrix
    LOGGER.info("Topology type: %s", args.topo)
    if args.topo in ("ring", "fc"):
        LOGGER.info("Using custom %s topology", args.topo)
        adj_mat = topology_manager(args)
    else:
        LOGGER.info("Using Erdos-Renyi graph for adjacency / mixing matrix")
        # Mixing is computed from adj_mat internally if required
        adj_mat = get_mixing_matrix(args, n=args.num_clients, p=0.5, seed=3)

    LOGGER.debug("Adjacency / mixing matrix:\n%s", adj_mat)

    # Model and head split
    args.model, args.feat_dim = get_model(args)
    LOGGER.info("Using feature dim: %s", args.feat_dim)

    if args.algorithm != "DisPFL":
        if args.backbone == "mobilenet":
            # MobileNet default feature dimension
            args.head = copy.deepcopy(args.model.classifier)
            args.model.classifier = nn.Identity()
        elif args.backbone == "mobilenet_proj":
            # MobileNet with custom projection dimension
            args.head = copy.deepcopy(args.model.classifier[-1])
            args.model.classifier[-1] = nn.Identity()
        elif args.backbone in ("CNNMnist", "resnet18_proj", "resnet34_proj"):
            args.head = copy.deepcopy(args.model.fc[-1])
            args.model.fc[-1] = nn.Identity()
        else:
            LOGGER.warning("Backbone model name '%s' unrecognized!", args.backbone)

        args.model = BaseHeadSplit(args.model, args.head)

    LOGGER.info(
        "\nAlgorithm: %s\nDataset: %s\nDist: %s\nCommunication: %s\nRounds: %d\n",
        args.algorithm,
        args.dataset,
        args.dist,
        args.comm,
        args.num_rounds,
    )

    # Acc matrices over trials
    acc_mtx = torch.zeros([args.num_trials, args.num_clients])
    global_test_acc_mtx = torch.zeros([args.num_trials])

    for trial in range(args.num_trials):
        LOGGER.info("Starting trial %d / %d", trial + 1, args.num_trials)

        # Dataset initialization
        # NOTE: prepare_* functions are expected to come from imported utilities.
        if args.feature_iid and args.label_iid == 0 and args.dist == "dir":
            # Feature IID, label non-IID
            if args.dataset == "digit":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_mnistm_noniid(args.num_clients, args=args)
            elif args.dataset == "office":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_caltech_noniid(args.num_clients, args=args)
            elif args.dataset == "domainnet":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_real_noniid(args.num_clients, args=args)
            else:
                raise ValueError(f"Unsupported dataset '{args.dataset}' for this setup.")
        elif args.feature_iid == 0 and args.label_iid and args.dist == "dir":
            # Feature non-IID, label IID
            if args.dataset == "digit":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_digits(args.num_clients, args=args)
            elif args.dataset == "office":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_office(args.num_clients, args=args)
            elif args.dataset == "domainnet":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_domainnet(args.num_clients, args=args)
            else:
                raise ValueError(f"Unsupported dataset '{args.dataset}' for this setup.")
        elif args.feature_iid == 0 and args.label_iid == 0 and args.dist == "dir":
            # Feature non-IID, label non-IID
            if args.dataset == "digit":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_digits_noniid(args.num_clients, args=args)
            elif args.dataset == "office":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_office_noniid(args.num_clients, args=args)
            elif args.dataset == "domainnet":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_domainnet_noniid(args.num_clients, args=args)
            else:
                raise ValueError(f"Unsupported dataset '{args.dataset}' for this setup.")
        elif args.feature_iid and args.label_iid == 0 and args.dist == "path":
            # Pathological setup
            if args.dataset == "digit":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_mnistm_noniid_path(args.num_clients, args=args)
            elif args.dataset == "domainnet":
                (
                    train_dataset_list,
                    test_dataset_list,
                    user_groups,
                    user_groups_test,
                ) = prepare_data_real_noniid_path(args.num_clients, args=args)
            else:
                raise ValueError(f"Unsupported dataset '{args.dataset}' for this setup.")
        else:
            raise ValueError(
                f"Unsupported combination of feature_iid={args.feature_iid}, "
                f"label_iid={args.label_iid}, dist='{args.dist}'."
            )

        test_loader = None  # Global test loader (unused in current code path)

        # Create clients
        clients = []
        client_id = 0
        for i in range(args.num_clients):
            client = get_clients(
                args.algorithm,
                client_id,
                args,
                adj_mat,
                train_dataset_list[i],
                user_groups[i],
                test_dataset_list[i],
                user_groups_test[i],
                0,
            )
            clients.append(client)
            client_id += 1

        # Adjust adjacency matrix weights proportional to local sample counts if enabled
        if args.weighted_adj_mat:
            LOGGER.info("Updating adjacency matrix with local sample count weights.")
            adj_mat = update_adjacency_matrix(
                adj_mat, clients, args.num_classes, alpha=None
            )

        # Communication / training
        if args.comm == "comm_mst":
            (
                avg_l_acc_hist,
                avg_g_acc,
                avg_l_auc,
                avg_g_auc,
                avg_l_unc,
                avg_g_unc,
            ) = comm_mst(args, adj_mat, clients, debug=False, test_loader=test_loader)
        elif args.comm == "comm_vmf_gossip":
            LOGGER.info("Using vMF gossip communication.")
            avg_l_acc_hist = comm_vmf_gossip(
                args, adj_mat, clients, debug=False, test_loader=test_loader
            )
        else:
            # Keep behavior identical: if other comm methods are implemented elsewhere,
            # they should be triggered by args.comm via imported utilities.
            LOGGER.warning(
                "Communication method '%s' is not explicitly handled here. "
                "Make sure it is implemented in imported modules.",
                args.comm,
            )
            # You might have other comm_* functions that are called elsewhere.

        # Trial-specific evaluation and saving
        aggregated_trial_test_acc = 0.0
        total_test_samples = 0

        for c in clients:
            # Save client performance for this trial
            acc_mtx[trial, c.id] = c.l_test_acc_hist[-1]  # local performance

            aggregated_trial_test_acc += (
                c.l_test_acc_hist[-1] * len(c.test_loader.dataset)
            )
            total_test_samples += len(c.test_loader.dataset)

            # Save client models / prototypes / histories
            filename = os.path.join(
                args.params_dir,
                "id_loaders_{}_{}_{}_client_{}.pkl".format(
                    args.algorithm, args.dataset, args.comm, c.id
                ),
            )
            with open(filename, "wb") as f:
                if args.algorithm in ["FedAvg", "Gossip", "DisPFL", "Penz"]:
                    pickle.dump([c.l_test_acc_hist], f)
                else:
                    pickle.dump(
                        [c.local_protos, c.global_protos, c.l_test_acc_hist], f
                    )

        LOGGER.info("Client performance saved to: %s", filename)

        aggregated_trial_test_acc /= total_test_samples
        global_test_acc_mtx[trial] = aggregated_trial_test_acc

        # Save average local accuracy history for this trial
        avg_local_filename = os.path.join(
            args.save_folder_name,
            "avg_local_acc_{}_{}_{}_feat_iid_{}_label_iid_{}_iid_{}_COMM_{}.pkl".format(
                args.num_clients,
                args.dataset,
                args.algorithm,
                args.feature_iid,
                args.label_iid,
                args.dist,
                args.comm,
            ),
        )
        with open(avg_local_filename, "wb") as f:
            pickle.dump(avg_l_acc_hist, f)

    # Save client uncertainties in JSON / other formats
    for cl in clients:
        cl.performance_test(data_loader=None, saveFlag=1)

    LOGGER.info("Counting parameters of first client model...")
    count_params(clients[0], args.comm)

    # Save acc_mtx across trials
    acc_mtx_filename = os.path.join(
        args.save_folder_name,
        "acc_mtx_{}_{}_{}_feat_iid_{}_label_iid_{}_iid_{}_COMM_{}.pkl".format(
            args.num_clients,
            args.dataset,
            args.algorithm,
            args.feature_iid,
            args.label_iid,
            args.dist,
            args.comm,
        ),
    )
    with open(acc_mtx_filename, "wb") as f:
        pickle.dump(acc_mtx, f)

    # Per-client mean / std across trials
    client_test_acc = torch.mean(acc_mtx, dim=0)
    client_test_std = torch.std(acc_mtx, dim=0)
    LOGGER.info(
        "Individual mean accuracy of each client (over all trials): %s",
        client_test_acc,
    )
    LOGGER.info(
        "Individual std accuracy of each client (over all trials): %s",
        client_test_std,
    )

    # Unweighted performance
    LOGGER.info(
        "The unweighted test accuracy of all clients in final trial: %s",
        acc_mtx[args.num_trials - 1, :],
    )

    acc_avg = torch.zeros([args.num_trials])
    for i in range(args.num_trials):
        acc_avg[i] = torch.mean(acc_mtx[i, :])

    LOGGER.info(
        "The avg test accuracy (unweighted) of all clients over all trials: %.2f ± %.2f",
        torch.mean(acc_avg),
        torch.std(acc_avg),
    )

    # Weighted performance
    mean_global_test_acc = torch.mean(global_test_acc_mtx)
    std_global_test_acc = torch.std(global_test_acc_mtx)
    LOGGER.info(
        "The avg test accuracy (weighted by client test sample size) "
        "over all trials: %.2f ± %.2f",
        mean_global_test_acc,
        std_global_test_acc,
    )

    return 0


if __name__ == "__main__":
    ret = main()
    print("Execution finished with code:", ret)