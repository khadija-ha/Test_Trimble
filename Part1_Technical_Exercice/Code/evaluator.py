import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from Visualisation import VisualisationHandler


class Evaluator:
    def __init__(self, model, device):
        # Initialize the Evaluator class
        self.model = model.to(device)
        self.device = device
        self.visualizer = VisualisationHandler()

    def test_model(self, test_folder, valid_extensions):
        self.model.eval()# Set the model to evaluation mode
        preprocessTransforms = self.load_transforms()# Define a set of image preprocessing transformations

        for filename in os.listdir(test_folder):
            if any(filename.endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(test_folder, filename)
                image = Image.open(image_path)
                image = preprocessTransforms(image)
                image = image.unsqueeze(0) # Add a batch dimension
                with torch.no_grad():
                    image = image.to(self.device)
                    outputs = self.model(image)  # Forward pass through the model
                    # Get the predicted class and confidence score
                    print(outputs)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    predicted_class = torch.argmax(outputs).item()
                    confidence_score = probabilities[0, predicted_class]
                    # Display the image and predicted class using matplotlib
                    plt.imshow(image.cpu().squeeze().permute(1, 2, 0).numpy())
                    plt.title(f'Predicted Class: {self.get_class_name(predicted_class)}, Confidence: {confidence_score:.2f}')
                    plt.axis('off')
                    plt.show()

    def load_transforms(self):
        # Define a series of image preprocessing transformations using torchvision
        return transforms.Compose(
            [
               # transforms.Resize(256),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def get_class_name(self, class_index):
        # Define a list of class names corresponding to class indices
        class_names = ["field", "road"]
        if class_index >= 0 and class_index < len(class_names):
            return class_names[class_index]
        else:
            return f"Unknown Class ({class_index})"



