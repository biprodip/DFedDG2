from models.models import *
# from models.mobilenet_v2 import *
import torchvision.models as models_torch
from torchvision.models import ResNet18_Weights,MobileNet_V2_Weights




def get_model(args):
    #define DL model
    if args.backbone=='CNNMNIST':
        print('Backbone: CNNMNist')
        model = CNNMnist(in_features=1, num_classes=args.num_classes, out_channel=args.out_channel, proj_dim = args.feat_dim).to(args.device)
        feature_dim = args.feat_dim
    elif args.backbone=='CNNFashionMNist': 
        print('Backbone: CNNFashionMNist')
        model = CNNFashion_Mnist().to(args.device)
        feature_dim = args.feat_dim  #to be corrected      
    elif args.backbone=='cifarnet': 
        #cifarnet default feature dimension 
        print('Backbone: CifarNet')
        args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        feature_dim = args.feat_dim
    elif args.backbone=='mobilenet':
        print('Backbone: MobileNet_Default_Feature_Dimension')
        model = models_torch.mobilenet_v2(pretrained=True).to(args.device)
        feature_dim = list(model.classifier.parameters())[0].shape[1]   #default feature dimension extraction
        model.classifier = nn.Linear(feature_dim, args.num_classes).to(args.device)
    elif args.backbone=='mobilenet_proj': #mobilenet with projection head          
        #mobilenet custom projection dimension head
        print('Backbone: MobileNet_Projection_Head')
        # model = models_torch.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(args.device)
        model = models_torch.mobilenet_v2(pretrained=True).to(args.device)
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
         print('Backbone: ResNet18_Projection_Head')
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
         print('Backbone: ResNet34_Projection_Head')
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
        print('Backbone: ModeleNet40_Projection_Head')
        model = PointNet(k=args.num_classes, normal_channel=args.use_normals).to(args.device)
        feature_dim = args.feat_dim
    else:
        print('Defined Backbone not Found.')
    return model, feature_dim



# def get_model(args):
#     #define DL model
#     if args.dataset in 'mnist':
#         print('Mnist CNN model used...')
#         model = CNNMnist(in_features=1, num_classes=args.num_classes, out_channel=args.out_channel, proj_dim = args.feat_dim).to(args.device)
#         feature_dim = args.feat_dim
#     elif 'fmnist' in args.dataset: 
#         model = CNNFashion_Mnist().to(args.device)
#         feature_dim = args.feat_dim        
#         #args.model = resnet10(num_classes=args.num_classes).to(args.device)
#     elif args.dataset in ['cifar10']:
#         # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
#         # model = resnet18(num_classes=args.num_classes).to(args.device)  #raw resnet from models
#         # feature_dim = args.feat_dim

#         #  model = models_torch.resnet18(pretrained=True).to(args.device) 
#         #  feature_dim = list(model.fc.parameters())[0].shape[1]
#         #  model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
         
#         model = models_torch.mobilenet_v2(pretrained=True).to(args.device)
#         feature_dim = list(model.classifier.parameters())[0].shape[1]
#         model.classifier = nn.Linear(args.feat_dim, args.num_classes).to(args.device)
#         feature_dim = args.feat_dim
#     elif args.dataset in 'digit':
#         # model = models_torch.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device) 
#         # feature_dim = list(model.fc.parameters())[0].shape[1]
#         # model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)         
        
#         ##default mobile net
#         # model = models_torch.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(args.device)
#         # feature_dim = list(model.classifier.parameters())[0].shape[1]
#         # model.classifier = nn.Linear(feature_dim, args.num_classes).to(args.device)
        
#         #mobilenet custom projection dimension head
#         model = models_torch.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(args.device)
#         feature_dim = list(model.classifier.parameters())[0].shape[1]
#         model.classifier = nn.Sequential(
#                     nn.Linear(feature_dim, args.feat_dim),  # Reducing dimension to 512
#                     # nn.ReLU(inplace=True),        # Adding non-linearity
#                     # nn.Linear(args.feat_dim, args.feat_dim),  # Reducing dimension to 512
#                     # nn.Dropout(p=0.1),
#                     nn.Linear(args.feat_dim, args.num_classes)  # Final layer for classification
#             ).to(args.device)
#         feature_dim = args.feat_dim    
        

#     elif args.dataset in 'office':
#         # model = models_torch.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device) 
#         # feature_dim = list(model.fc.parameters())[0].shape[1]
#         # model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)         

#         # model = models_torch.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(args.device)
#         # feature_dim = list(model.classifier.parameters())[0].shape[1]
#         # model.classifier = nn.Linear(feature_dim, args.num_classes).to(args.device)

#         model = models_torch.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(args.device)
#         feature_dim = list(model.classifier.parameters())[0].shape[1]
#         model.classifier = nn.Sequential(
#                     nn.Linear(feature_dim, args.feat_dim),  # Reducing dimension to 512
#                     # nn.ReLU(inplace=True),        # Adding non-linearity
#                     # nn.Linear(args.feat_dim, args.feat_dim),  # Reducing dimension to 512
#                     # nn.Dropout(p=0.1),
#                     nn.Linear(args.feat_dim, args.num_classes)  # Final layer for classification
#             ).to(args.device)
#         feature_dim = args.feat_dim

#     elif args.dataset in 'domainnet':
#         # model = models_torch.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(args.device) 
#         # feature_dim = list(model.fc.parameters())[0].shape[1]
#         # model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)         

#         # model = models_torch.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(args.device)
#         # feature_dim = list(model.classifier.parameters())[0].shape[1]
#         # model.classifier = nn.Linear(feature_dim, args.num_classes).to(args.device)

#         # Mobilenet custom head 
#         model = models_torch.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(args.device)
#         feature_dim = list(model.classifier.parameters())[0].shape[1]
#         model.classifier = nn.Sequential(
#                     nn.Linear(feature_dim, args.feat_dim),  # Reducing dimension to 512
#                     # nn.ReLU(inplace=True),        # Adding non-linearity
#                     # nn.Linear(args.feat_dim, args.feat_dim),  # Reducing dimension to 512
#                     # nn.Dropout(p=0.1),
#                     nn.Linear(args.feat_dim, args.num_classes)  # Final layer for classification
#             ).to(args.device)
#         feature_dim = args.feat_dim

#     elif args.dataset in 'cifar100':
#         #  model = models_torch.resnet50(pretrained=True).to(args.device)
#         #  feature_dim = list(model.fc.parameters())[0].shape[1]
#         #  model.fc = nn.Linear(feature_dim,args.num_classes).to(args.device)

#         # Load ResNet50 with pretrained weights
#         model = models_torch.resnet50(pretrained=True).to(args.device)
#         feature_dim = model.fc.in_features
#         model.fc = nn.Sequential(
#                     nn.Linear(feature_dim, args.feat_dim),  # Reducing dimension to 512
#                     nn.ReLU(inplace=True),        # Adding non-linearity
#                     nn.Linear(args.feat_dim, args.num_classes)  # Final layer for classification
#             ).to(args.device)
#         feature_dim = args.feat_dim  # Set the feature dimension to 512

#         # model = models_torch.mobilenet_v2(pretrained=True).to(args.device)
#         # feature_dim = list(model.classifier.parameters())[0].shape[1]
#         # model.classifier = nn.Linear(feature_dim, args.num_classes).to(args.device)

#         # model = resnet34(num_classes=args.num_classes).to(args.device)
#         # feature_dim = args.feat_dim

#     elif args.dataset in ['modelnet10', 'modelnet40']:
#         model = PointNet(k=args.num_classes, normal_channel=args.use_normals).to(args.device)
#         feature_dim = args.feat_dim
#     return model, feature_dim
