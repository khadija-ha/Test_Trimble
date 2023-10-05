import onnx
import torch
from model import CustomModel

def convert():
    # Define class names
    class_names = ["field", "road"]
    # Create an instance
    custom_model = CustomModel(class_names)
    # Load pre-trained weights
    model_weights_path = '/data1/Test_Trimble/Part1_Technical_Exercice/Models/model_trimble.pth'
    custom_model.model.load_state_dict(torch.load(model_weights_path))
    # Set the model to evaluation mode
    custom_model.model.eval()
    # Define batch size and create a random input tensor
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)  
    # Define the path to save the ONNX model
    onnx_path =  '/data1/Test_Trimble/Part1_Technical_Exercice/Models/model_trimble.onnx'
    # Export the custom model to ONNX format
    torch.onnx.export(
        custom_model.model,
        input_tensor,
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    # Load the ONNX model and check its correctness
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("Custom model was successfully converted to ONNX format.")

if __name__ == "__main__":
    convert()
