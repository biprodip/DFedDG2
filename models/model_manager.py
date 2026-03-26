"""model_manager.py — Backbone model factory for DFedDG2.

Provides ``get_model``, which instantiates the requested backbone architecture,
optionally replaces its classification head with a two-layer projection head,
moves the model to ``args.device``, and returns the model together with the
actual feature (embedding) dimension used by downstream loss functions.

Supported backbones (``args.backbone``)
---------------------------------------
'CNNMNIST'       : Custom CNN for MNIST with projection head.
'CNNFashionMNist': Custom CNN for FashionMNIST.
'cifarnet'       : Lightweight CIFAR-10 network.
'mobilenet'      : MobileNetV2 (pretrained) with default head replaced by a
                   linear classifier.
'mobilenet_proj' : MobileNetV2 (pretrained) with a two-layer linear projection
                   head (feat_dim → num_classes).
'resnet18_proj'  : ResNet-18 (pretrained) with a two-layer projection head.
'resnet34_proj'  : ResNet-34 (pretrained) with a two-layer projection head.
dataset-based    : PointNet for 'modelnet10'/'modelnet40' (3-D point clouds).

Note: The old dataset-keyed implementation is retained below as a commented-out
block for reference.  The current implementation dispatches on ``args.backbone``
instead of ``args.dataset``.
"""

import logging
from models.models import *
import torchvision.models as models_torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, ResNet18_Weights

LOGGER = logging.getLogger(__name__)



def get_model(args):
    """Instantiate and return the backbone model for a federated client.

    Dispatches on ``args.backbone`` to build the correct architecture, replaces
    the classification / fully-connected head with a projection head where
    applicable, and moves the model to ``args.device``.

    Args:
        args: Namespace with at least the following fields:
            ``backbone`` (str)   — architecture identifier (see module docstring).
            ``device``           — torch device (e.g., ``'cuda'`` or ``'cpu'``).
            ``num_classes`` (int)— number of output classes.
            ``feat_dim`` (int)   — embedding dimension for projection heads.
            ``out_channel`` (int)— output channels for CNNMNIST (optional).
            ``use_normals`` (bool)— whether to use normal vectors for PointNet.
            ``dataset`` (str)    — used as fallback to detect 3-D datasets.

    Returns:
        tuple[nn.Module, int]:
            - model: The constructed model, already on ``args.device``.
            - feature_dim: The dimension of the penultimate (embedding) layer
              output, used to configure loss functions such as CompLoss/DisLoss.

    Raises:
        UnboundLocalError: If ``args.backbone`` is not recognised and
            ``args.dataset`` is not a 3-D dataset; ``model`` will be undefined.
    """
    # define DL model
    if args.backbone == 'CNNMNIST':
        model = CNNMnist(in_features=1, num_classes=args.num_classes, out_channel=args.out_channel, proj_dim = args.feat_dim).to(args.device)
        feature_dim = args.feat_dim
    elif args.backbone=='CNNFashionMNist':
        model = CNNFashion_Mnist().to(args.device)
        feature_dim = args.feat_dim  #to be corrected
    elif args.backbone=='cifarnet':
        #cifarnet default feature dimension
        args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        feature_dim = args.feat_dim
    elif args.backbone=='mobilenet':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(args.device)
        feature_dim = list(model.classifier.parameters())[0].shape[1]   #default feature dimension extraction
        model.classifier = nn.Linear(feature_dim, args.num_classes).to(args.device)
    elif args.backbone=='mobilenet_proj': #mobilenet with projection head
        #mobilenet custom projection dimension head
        model = models_torch.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(args.device)
        feature_dim = list(model.classifier.parameters())[0].shape[1]
        model.classifier = nn.Sequential(
                    nn.Linear(feature_dim, args.feat_dim),  # Reducing dimension to 512
                    # nn.ReLU(inplace=True),        # Adding non-linearity
                    # nn.Linear(args.feat_dim, args.feat_dim),  # Reducing dimension to 512
                    # nn.Dropout(p=0.001),
                    nn.Linear(args.feat_dim, args.num_classes)  # Final layer for classification
            ).to(args.device)
        feature_dim = args.feat_dim
    elif args.backbone=='resnet18_proj': #resnet18 with projection head
         model = models_torch.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device)
         feature_dim = list(model.fc.parameters())[0].shape[1]  #512
         model.fc = nn.Sequential(
                    nn.Linear(feature_dim, args.feat_dim),  # Reducing dimension to 512
                    # nn.ReLU(inplace=True),        # Adding non-linearity
                    # nn.Linear(args.feat_dim, args.feat_dim),  # Reducing dimension to 512
                    # nn.Dropout(p=0.1),
                    nn.Linear(args.feat_dim, args.num_classes)  # Final layer for classification
            ).to(args.device)
         feature_dim = args.feat_dim
    elif args.backbone=='resnet34_proj': #resnet34 with projection head S
         model = models_torch.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(args.device)
         feature_dim = list(model.fc.parameters())[0].shape[1]
         model.fc = nn.Sequential(
                    nn.Linear(feature_dim, args.feat_dim),  # Reducing dimension to 512
                    # nn.ReLU(inplace=True),        # Adding non-linearity
                    # nn.Linear(args.feat_dim, args.feat_dim),  # Reducing dimension to 512
                    # nn.Dropout(p=0.1),
                    nn.Linear(args.feat_dim, args.num_classes)  # Final layer for classification
            ).to(args.device)
         feature_dim = args.feat_dim
    elif args.dataset in ['modelnet10', 'modelnet40']:
        model = PointNet(k=args.num_classes, normal_channel=args.use_normals).to(args.device)
        feature_dim = args.feat_dim
    else:
        LOGGER.warning("Defined Backbone not Found.")
    return model, feature_dim