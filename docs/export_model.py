import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exercise_answers.answers_2 import Net

def export_model():
    # Initialize the model
    model = Net()
    
    # Load the trained model state
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'exercise_answers', 'model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        print("Please run answers_2.py first to train the model")
        return
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create dummy input tensor
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Export the model
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.onnx')
    torch.onnx.export(model,
                     dummy_input,
                     output_path,
                     export_params=True,
                     opset_version=12,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})
    
    print(f"Model exported successfully to {output_path}")

if __name__ == '__main__':
    export_model()