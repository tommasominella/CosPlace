import torch
import logging
import torchvision
from torch import nn


class MyEnsemble(nn.Module):
    def __init__(self, firstModel, secondModel):
        super(MyEnsemble, self).__init__()
        self.firstModel = modelA
        self.secondModel = modelB
        self.classifier = nn.Linear(in_features, n_classes) #define accordingly
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1 = self.firstModel(x1)
        x2 = self.secondModel(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(self.relu(x))
        return x
