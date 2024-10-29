import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# TIP: Use PyTorch documentation to fill in the right responses

# Define custom transforms
class ContrastEnhancement(object):
    def __call__(self, img):
        ############## WRITE YOUR CODE BELOW ###############
        # TODO: what should it return to adjust the contrast?
        return                                              
        ############## WRITE YOUR CODE ABOVE ###############

class NoiseReduction(object):
    def __call__(self, img):
        ################### WRITE YOUR CODE BELOW ####################
        # TODO: what should it return to reduce noise?
        return                                                        
        ################### WRITE YOUR CODE ABOVE ####################
        # HINT: Reducing noise can also be interpreted as reducing background details
        #       What's it called when an images details are reduced?

# Define the preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),

    ########## WRITE YOUR CODE BELOW ##########
    # TODO: find a way to normalize the data using PyTorch
    
    ContrastEnhancement(),
    NoiseReduction(),
    ########## WRITE YOUR CODE ABOVE ##########
])

# Load the MNIST dataset
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST() # TODO: Fill in the arguments for test_dataset

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to display images
def display_images(images, labels):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Test the preprocessing
def test_preprocessing():
    # Get a batch of images and labels
    images, labels = next(iter(train_loader))
    
    print("Batch shape:", images.shape)
    print("Labels shape:", labels.shape)
    
    # Convert tensors to numpy arrays for display
    images_np = images.numpy()
    labels_np = labels.numpy()
    
    # Display the preprocessed images
    display_images(images_np, labels_np)
    
    # Print statistics
    print("Image statistics:")
    print(f"Min value: {images.min():.4f}")
    print(f"Max value: {images.max():.4f}")
    print(f"Mean: {images.mean():.4f}")
    print(f"Std: {images.std():.4f}")

# Run the test
test_preprocessing()