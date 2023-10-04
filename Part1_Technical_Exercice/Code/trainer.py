import time
import copy
import torch
import logging
import os
from torch import nn, optim
from torch.optim import lr_scheduler
from Visualisation import VisualisationHandler


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, dir_checkpoint='checkpoints/'):
        # Initialize the Trainer class with model, criterion, optimizer, scheduler, data loaders, and other parameters
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.num_epochs = num_epochs
        self.dir_checkpoint = dir_checkpoint
        self.visualizer = VisualisationHandler()

    def train_model(self):
        # Function to train the model
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())  # Initialize the best model weights
        best_acc = 0.0  # Initialize the best validation accuracy
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
                running_loss = 0.0    # Initialize running loss
                running_corrects = 0  # Initialize running correct predictions                
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1) # Get predicted class labels
                        labels = torch.tensor([[1.0 - label, label] for label in labels], dtype=torch.float32)
                        labels = labels.to(self.device)  # Déplacez les étiquettes sur le même dispositif que les sorties
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()  # Update model weights
                    running_loss += loss.item() * inputs.size(0)
                    correct_preds = torch.sum(preds == torch.argmax(labels, dim=1))
                    running_corrects += correct_preds

                if phase == 'train':
                    self.scheduler.step()  # Adjust learning rate  
                epoch_loss = running_loss / self.dataset_sizes[phase]  # Calculate epoch loss
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase] # Calculate epoch accuracy
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
               
                if phase == 'train':
                    self.visualizer.updateTrainLoss_EvolutionFigure(epoch, epoch_loss) # Update loss visualization
                else:
                    epoch_acc = epoch_acc.item()
                    self.visualizer.updateValLoss_EvolutionFigure(epoch, epoch_loss)
                    self.visualizer.updateMainCriteriaFigure(epoch, epoch_acc) # Update main criteria visualization
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())  # Save the best model weights

            print()
            if True:
                try:
                    os.mkdir(self.dir_checkpoint) # Create a checkpoint directory if it doesn't exist
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(self.model.state_dict(), self.dir_checkpoint + f'CP_epoch{epoch + 1}.pth') # Save a checkpoint
                logging.info(f'Checkpoint {epoch + 1} saved !')
 
        time_elapsed = time.time() - since # Calculate training time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  
        print('Best val Acc: {:4f}'.format(best_acc))
        self.model.load_state_dict(best_model_wts) # Load the best model weights

        return self.model
