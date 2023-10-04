from dataset import CustomDataset
from trainer import Trainer
from model import CustomModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# Define your mean, std, and data_dir

def main():
    # Define mean and std values for data normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Specify the directory where the dataset is located
    data_dir = '/media/khadija/data_ssd1/trimble/aug/'
    # Create CustomDataset instance
    dataset = CustomDataset(data_dir, mean, std)
    #image_datasets = dataset.get_datasets()
    dataloaders = dataset.get_dataloader()
    dataset_sizes = dataset.get_dataset_sizes()
    class_names = dataset.get_class_names()
    # Create CustomModel instance
    model_obj = CustomModel(class_names)
    model = model_obj.get_model()
    # Create Trainer instance
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define the loss criterion (BCE) for training
    criterion = nn.BCELoss()
    # Define the optimizer (Stochastic Gradient Descent - SGD)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Define a step learning rate scheduler
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # Create a Trainer instance for training
    trainer = Trainer(model, criterion, optimizer, step_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=50)
    # Train the model
    model = trainer.train_model()
    # Save the model
    torch.save(model.state_dict(), '/home/khadija/Test_Trimble/Part1_Technical_Exercice/Models/model_trimble.pth')

if __name__ == "__main__":
    main()
