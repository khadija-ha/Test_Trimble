import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from evaluator import Evaluator  
from model import CustomModel


def main():
    # Path to the pre-trained model
    model_path = '/data1/Test_Trimble/Part1_Technical_Exercice/Models/model_trimble.pth'
    # Folder containing test images
    test_folder = '/media/khadija/data_ssd1/trimble/dataset1/test_images/'   
    # List of supported image file extensions
    valid_extensions = ['.jpg', '.jpeg', '.png']
    # Check if CUDA (GPU) is available 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create an instance of the CustomModel class 
    custom_model = CustomModel(class_names=["field", "road"])
    model = custom_model.get_model()
    # Load the model weights from the specified path
    model.load_state_dict(torch.load(model_path))
    # Move the model to the GPU if CUDA is available, else use the CPU
    model = model.to(device)
    # Create an instance of the Evaluator class with the loaded model and device
    evaluator = Evaluator(model, device)    
    # Perform model evaluation on the test images
    evaluator.test_model(test_folder, valid_extensions)

if __name__ == "__main__":
    main()