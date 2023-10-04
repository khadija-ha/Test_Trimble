import torch
from torchvision import models
from torch import nn

class CustomModel:
    def __init__(self, class_names):
        self.model = models.densenet161(pretrained=True)
        
        # Freeze all layers except the final layer
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.classifier.in_features
        
        # Modify the final layer for binary classification
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, len(class_names)),
            nn.Sigmoid()
        )


    def get_model(self):
        return self.model
