"""Utility tools for the MUPPET benchmark framework.

This module provides utility functions for data preprocessing and loading operations
commonly used in the benchmarking pipeline. It includes image loading utilities
with standard preprocessing for ImageNet-style models.

Functions:
    load_imagenet_image: Load and preprocess images for ImageNet-style models
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_imagenet_image(image_path: str) -> torch.Tensor:
    """Loads an image from a file, preprocesses it, and returns it as a PyTorch tensor.

    This function resizes the image to 224x224, normalizes it using ImageNet's mean and
    standard deviation, and returns it as a tensor of shape (1, 3, 224, 224).

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The image tensor of shape (1, 3, 224, 224), ready for model input.
    """
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert("RGB")

    # Define the transformation pipeline
    loader = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=np.array([0.485, 0.456, 0.406]),
                std=np.array([0.229, 0.224, 0.225]),
            ),
            transforms.Lambda(lambda x: torch.unsqueeze(x, 0)),
        ]
    )

    return loader(image)
