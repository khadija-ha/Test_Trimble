import torch
from torchvision import models
from torch import nn

class CustomModel:
    def __init__(self, class_names):

        # Load a pre-trained DenseNet-161 model
        self.model = models.densenet161(pretrained=True)
        #load a pre-trained ResNeXt-50 model
        #self.model = models.resnext50_32x4d(pretrained=True)
        
        
        # Get the number of input features for the final fully connected layer
        num_ftrs = self.model.classifier.in_features
        #num_ftrs = model.fc.in_features
        
        # Replace the final fully connected layer 
        self.model.classifier = nn.Linear(num_ftrs, len(class_names))
       # nn.Sigmoid()  # Utiliser nn.Sigmoid() pour la classification binaire


    def get_model(self):
        return self.model
