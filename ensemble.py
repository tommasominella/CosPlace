import torch
import logging
import torchvision
from torch import nn

from model.layers import Flatten, L2Norm, 

class MyEnsemble(nn.Module):
    def __init__(self, firstModel, secondModel):
        super(MyEnsemble, self).__init__()
        self.firstModel = modelA
        self.secondModel = modelB
        
        
        self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )
    

    def forward(self, x1, x2):
        x1 = self.firstModel(x1)
        x2 = self.secondModel(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.aggregation(x)
        return x
