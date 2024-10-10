from models.models import *
# from models.mobilenet_v2 import *
import torchvision.models as models_torch
from torchvision.models import ResNet18_Weights

def get_model(args):
    #define DL model
    if args.dataset in 'mnist':
        model = CNNMnist(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        feature_dim = args.feat_dim
    elif 'fmnist' in args.dataset: 
        model = CNNFashion_Mnist().to(args.device)
        feature_dim = args.feat_dim        
        #args.model = resnet10(num_classes=args.num_classes).to(args.device)
    elif args.dataset in ['cifar10','digits','office','domainnet']:
        # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        # model = resnet18(num_classes=args.num_classes).to(args.device)  #raw resnet from models
        # feature_dim = args.feat_dim

         model = models_torch.resnet18(pretrained=True).to(args.device) 
         feature_dim = list(model.fc.parameters())[0].shape[1]
         model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
         
        # model = models_torch.mobilenet_v2(pretrained=True).to(args.device)
        # feature_dim = list(model.classifier.parameters())[0].shape[1]
        # model.classifier = nn.Linear(feature_dim, args.num_classes).to(args.device)
    elif args.dataset in 'digits':
         model = models_torch.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device) 
         feature_dim = list(model.fc.parameters())[0].shape[1]
         model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)         
    elif args.dataset in 'office':
         model = models_torch.resnet18(pretrained=True).to(args.device) 
         feature_dim = list(model.fc.parameters())[0].shape[1]
         model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
    elif args.dataset in 'domainnet':
         model = models_torch.resnet18(pretrained=True).to(args.device) 
         feature_dim = list(model.fc.parameters())[0].shape[1]
         model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
    elif args.dataset in 'cifar100':
        #  model = models_torch.resnet50(pretrained=True).to(args.device)
        #  feature_dim = list(model.fc.parameters())[0].shape[1]
        #  model.fc = nn.Linear(feature_dim,args.num_classes).to(args.device)

        # Load ResNet50 with pretrained weights
        model = models_torch.resnet50(pretrained=True).to(args.device)
        feature_dim = model.fc.in_features
        model.fc = nn.Sequential(
                    nn.Linear(feature_dim, 512),  # Reducing dimension to 512
                    nn.ReLU(inplace=True),        # Adding non-linearity
                    nn.Linear(512, args.num_classes)  # Final layer for classification
            ).to(args.device)
        feature_dim = 512  # Set the feature dimension to 512

        # model = models_torch.mobilenet_v2(pretrained=True).to(args.device)
        # feature_dim = list(model.classifier.parameters())[0].shape[1]
        # model.classifier = nn.Linear(feature_dim, args.num_classes).to(args.device)

        # model = resnet34(num_classes=args.num_classes).to(args.device)
        # feature_dim = args.feat_dim

    elif args.dataset in ['modelnet10', 'modelnet40']:
        model = PointNet(k=args.num_classes, normal_channel=args.use_normals).to(args.device)
        feature_dim = args.feat_dim
    return model, feature_dim
