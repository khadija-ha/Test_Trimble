import os
import torch
from torchvision import datasets, transforms

class CustomDataset:
    def __init__(self, data_dir, mean, std):
        # Define data transformations for training and validation sets
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
                #transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally (optional)
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                #transforms.Normalize(mean, std)  # Normalize the pixel values (optional)
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
                #transforms.CenterCrop(224),  # Center crop images to 224x224 pixels (optional)
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                #transforms.Normalize(mean, std)  # Normalize the pixel values (optional)
            ]),
        }

        # Load image datasets for 'train' and 'val' from the specified data directory
        self.image_datasets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
            for x in ['train', 'val']
        }
    
    def get_datasets(self):
        # Return the 'train' and 'val' datasets
        return self.image_datasets

    def get_dataloader(self, batch_size=2, num_workers=10):
        # Create data loaders for 'train' and 'val' datasets
        dataloaders = {
            x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
            for x in ['train', 'val']
        }
        return dataloaders

    def get_dataset_sizes(self):
        # Get the sizes (number of samples) of 'train' and 'val' datasets
        dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        return dataset_sizes

    def get_class_names(self):
        # Get the class names from the 'train' dataset
        class_names = self.image_datasets['train'].classes
        return class_names
