import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# To be used as a library, this file has been reorganized

# Define custom transforms
class ContrastEnhancement(object):
    def __call__(self, img):
        return transforms.functional.adjust_contrast(img, 2)

class NoiseReduction(object):
    def __call__(self, img):
        return transforms.functional.gaussian_blur(img, kernel_size=3)

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
def test_preprocessing(train_loader):
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


def prepare_data():
    # Define the preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ContrastEnhancement(),
        NoiseReduction(),
    ])

    # Load the MNIST dataset
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Run the test
    test_preprocessing(train_loader)

    return [train_dataset, test_dataset, train_loader, test_loader]