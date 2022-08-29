import torch
import logging
import torchvision
from torch import nn

from layers import Flatten, L2Norm, GeM


CHANNELS_NUM_IN_LAST_CONV = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "vgg16": 512,
    }


class GeoLocalizationNet_Ensemble(nn.Module):
    def __init__(self, backbone, fc_output_dim):
        super().__init__()
        self.backbone1, self.backbone2, self.backbone3, self.backbone4, features_dim, features_dim1 = get_backbone(backbone)
        
        ## ENS-F: aggregation after the ensemble of features extracture obtained from backbones

        #self.aggregation = nn.Sequential(
        #        L2Norm(),
        #        GeM(),
        #        Flatten(),
        #        nn.Linear(features_dim, fc_output_dim),
        #        L2Norm()
        #    )

        # ENS-D: feature extractors and descriptors, then final ensembles  
        
        self.aggregation1 = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim1, fc_output_dim),
                L2Norm()
            )

        self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )


    ## forward ENS-F:

    #def forward(self, x):
    #    feat1 = self.backbone1(x)
    #    feat2 = self.backbone2(x)
    #    feat3 = self.backbone3(x)

    #    features = [feat1, feat2, feat3]
    #    # concateno i descrittori:
    #    features = torch.cat(features, dim= 1)
    #    x = self.aggregation(features)
    #    return x

    ## forward ENS-D:

    def forward(self,x):
      feature1 = self.backbone1(x)
      aggregation18 = self.aggregation1(feature1)
      
      feature2 = self.backbone2(x)
      aggregation50 = self.aggregation(feature2)
      
      feature3 = self.backbone3(x)
      aggregation101 = self.aggregation(feature3)

      aggregations = [aggregation18, aggregation50, aggregation101]
      x = torch.cat(aggregations, dim=1)

      # mean instead of concatenation, but leads to worst results
      #x = (aggregation18 + aggregation50 + aggregation101)/3

      return x


def get_backbone(backbone_name):
    backbone1 = torchvision.models.resnet18(pretrained=True)
    backbone2 = torchvision.models.resnet50(pretrained=True)
    backbone3 = torchvision.models.resnet101(pretrained=True)
    backbone4 = torchvision.models.vgg16(pretrained=True)
    
    #resnet 18
    for name1, child1 in backbone1.named_children():
            if name1 == "layer3":  # Freeze layers before conv_3
                break
            for params1 in child1.parameters():
                params1.requires_grad = False
    logging.debug(f"Train only layer3 and layer4 of the {'resnet18'}, freeze the previous ones")
    layers1 = list(backbone1.children())[:-2]  # Remove avg pooling and FC layer
  
    #resnet 50
    for name2, child2 in backbone2.named_children():
            if name2 == "layer3":  # Freeze layers before conv_3
                break
            for params2 in child2.parameters():
                params2.requires_grad = False
    logging.debug(f"Train only layer3 and layer4 of the {'resnet50'}, freeze the previous ones")
    layers2 = list(backbone2.children())[:-2]  # Remove avg pooling and FC layer
     
    #resnet 101
    for name3, child3 in backbone3.named_children():
            if name3 == "layer3":  # Freeze layers before conv_3
                break
            for params3 in child3.parameters():
                params3.requires_grad = False
    logging.debug(f"Train only layer3 and layer4 of the {'resnet101'}, freeze the previous ones")
    layers3 = list(backbone3.children())[:-2]  # Remove avg pooling and FC layer
    
    #vgg16
    layers4 = list(backbone4.features.children())[:-2]  # Remove avg pooling and FC layer
    for l in layers4[:-5]:
        for p in l.parameters(): p.requires_grad = False
    logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
        
    backbone1 = torch.nn.Sequential(*layers1)
    backbone2 = torch.nn.Sequential(*layers2)
    backbone3 = torch.nn.Sequential(*layers3)
    backbone4 = torch.nn.Sequential(*layers4)
   

    # ENS-D: use of resnet 18, 50, 101
    features_dim = 2048      # 4608 for ENS-F
    features_dim1 = 512
    
    return backbone1, backbone2, backbone3, backbone4, features_dim, features_dim1
