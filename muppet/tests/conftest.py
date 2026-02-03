"""Test configuration and shared fixtures for MUPPET test suite.

This module provides common pytest fixtures and configuration for testing MUPPET components.
It includes mock models, dummy data generators, and shared test utilities used across
multiple test modules to ensure consistent testing environments.

The tests verify:
- Mock segmentation and classification model functionality
- Dummy data generation for various testing scenarios
- VGG model fixture setup and preprocessing pipelines
- Test image loading and transformation utilities
- Shared test utilities for model and data validation
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from muppet import DEVICE


class DummySegmentationModelClass:
    """Dummy segmentation model class for testing purposes.

    A mock segmentation model that provides configurable behavior
    for testing segmentation-related functionality without requiring
    actual model weights or complex computations.
    """

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        perturbation_level: float = 0.0,
    ):
        """Initialize the dummy segmentation model for testing.

        Args:
            num_classes: Number of segmentation classes to output.
            threshold: Threshold for segmentation decisions.
            perturbation_level: Level of perturbation to apply to outputs.
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.perturbation_level = perturbation_level

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Test prediction method for mock model with thresholding."""
        if x.ndim != 4:
            x = np.expand_dims(x, 0)  # ensure batch dim
        batch_size, channels, h, w = x.shape
        outputs = np.zeros(
            (batch_size, self.num_classes, h, w), dtype=np.float32
        )
        for c in range(self.num_classes):
            channel_idx = c % channels
            outputs[:, c] = (x[:, channel_idx] > self.threshold).astype(
                np.float32
            )

        # Only add noise if perturbation_level > 0
        if self.perturbation_level > 0:
            outputs += np.random.randn(*outputs.shape) * self.perturbation_level

        return outputs

    def __call__(self, x):
        """Test callable interface for mock model."""
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
            out = self.predict(arr)
            return torch.from_numpy(out).to(x.device)
        return self.predict(x)

    def shape_input(
        self,
        x: np.ndarray,
        original_shape: tuple = None,
        channel_first: bool = True,
    ) -> np.ndarray:
        """Test method for input shape handling."""
        if x.ndim == 3:
            return np.expand_dims(x, 0)
        elif x.ndim == 4:
            return x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


class DummyClassificationModelClass:
    """A dummy classification model that computes class scores by thresholding input channels, simulating image classification.
    For each class, it selects a corresponding input channel, applies a binary threshold, and aggregates the resulting mask
    using either the spatial mean or max. This is functionally equivalent to applying the DummySegmentationModel followed by
    mean or max pooling over the spatial dimensions. Useful for testing metrics without requiring real classifiers.
    """

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        agg_method: str = "mean",
    ):
        """Initialize the dummy classification model for testing.

        Args:
            num_classes: Number of classification classes to output.
            threshold: Threshold for binary classification decisions.
            agg_method: Aggregation method for spatial dimensions, either 'mean' or 'max'.
        """
        self.num_classes = num_classes
        self.threshold = threshold
        assert agg_method in ["mean", "max"], (
            "Aggregation must be 'mean' or 'max'."
        )
        self.agg_method = agg_method

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Test prediction method for mock segmentation model with aggregation."""
        batch_size, channels, h, w = x.shape
        outputs = np.zeros((batch_size, self.num_classes), dtype=x.dtype)

        for c in range(self.num_classes):
            channel_idx = c % channels
            # Create binary segmentation mask for each class
            seg_mask = (x[:, channel_idx] > self.threshold).astype(x.dtype)

            if self.agg_method == "mean":
                outputs[:, c] = seg_mask.mean(axis=(1, 2))  # scalar per image
            elif self.agg_method == "max":
                outputs[:, c] = seg_mask.max(axis=(1, 2))  # scalar per image

        return outputs

    def shape_input(
        self,
        x: np.ndarray,
        original_shape: tuple = None,
        channel_first: bool = True,
    ) -> np.ndarray:
        """Test method for segmentation model input shape handling."""
        if x.ndim == 3:
            return np.expand_dims(x, 0)
        elif x.ndim == 4:
            return x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


@pytest.fixture(scope="session")
def model_vgg():
    """Fixture to get VGG16 model and load it to the device."""
    model = models.vgg16(pretrained=True)
    model.to(DEVICE)
    model.eval()
    return model


@pytest.fixture
def cat_image_for_vgg():
    """Fixture to get a cat image."""
    # Load the image
    img = Image.open("muppet/tests/data/cat.jpg")
    # Preprocess the image
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    img_tensor = preprocess(img)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    # Move to device
    img_tensor = img_tensor.to(DEVICE)
    return img_tensor


@pytest.fixture
def create_dummy_inputs():
    """Fixture to create dummy inputs for segmentation testing.

    Returns:
        function: A function that creates dummy inputs for testing.
    """

    def __create_dummy_inputs(n_classes):
        """Creates dummy inputs for testing segmentation metrics.

        Args:
            n_classes (int): Number of classes for the segmentation task.

        Returns:
            tuple: A tuple containing:
                - x: Input image tensor of shape (C, H, W)
                - y: Output heatmaps tensor of shape (n_classes, H, W)
                - a: Attribution heatmap tensor of shape (1, H, W)
        """
        np.random.seed(42)  # for reproducibility

        # Input image with 3 channels, values roughly between 0.25 and 0.75
        x = (np.random.rand(3, 224, 224) * 0.5 + 0.25).astype(np.float32)

        # Output heatmaps for n_classes, each with random but meaningful values between 0.2 and 0.8
        y = (np.random.rand(n_classes, 224, 224) * 0.6 + 0.2).astype(np.float32)

        # Attribution heatmap for 1 channel, values between 0.3 and 1.0
        a = (np.random.rand(1, 224, 224) * 0.7 + 0.3).astype(np.float32)

        return x, y, a

    return __create_dummy_inputs


@pytest.fixture
def DummySegmentationModel():
    """Fixture to create dummy segmentation models for testing.

    Returns:
        function: A function that creates dummy segmentation models.
    """

    def __make_dummy_segmentation_model(
        num_classes: int, perturbation_level: float = 0.0
    ):
        """Create a dummy segmentation model for testing.

        Args:
            num_classes (int): Number of output classes for the model.
            perturbation_level (float): Level of perturbation to apply in the model.

        Returns:
            DummySegmentationModelClass: A dummy segmentation model instance.
        """
        return DummySegmentationModelClass(
            num_classes=num_classes, perturbation_level=perturbation_level
        )

    return __make_dummy_segmentation_model


@pytest.fixture
def dummy_classification_model():
    """Fixture to create dummy classification models for testing.

    Returns:
        function: A function that creates dummy classification models.
    """

    def __make_dummy_classification_model(num_classes: int):
        """Create a dummy classification model for testing.

        Args:
            num_classes (int): Number of output classes for the classification model.

        Returns:
            DummyClassificationModelClass: A dummy classification model instance.
        """
        return DummyClassificationModelClass(num_classes=num_classes)

    return __make_dummy_classification_model


class DummySegmentationModell(nn.Module):
    """Dummy PyTorch segmentation model for testing purposes.

    A simple convolutional model that simulates segmentation behavior
    using a single convolutional layer to generate per-class logits.
    """

    def __init__(self, num_classes: int):
        """Initialize the dummy segmentation PyTorch model.

        Args:
            num_classes: Number of segmentation classes for the model to output.
        """
        super().__init__()
        self.num_classes = num_classes
        # A simple linear layer can simulate a logit output
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=self.num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Test forward pass for mock PyTorch model."""
        # Example forward pass returning logits
        return self.conv(x)


@pytest.fixture
def dummy_seg_model():
    """Fixture to create dummy segmentation models for testing.

    Returns:
        function: A function that creates dummy segmentation models.
    """

    def __make_dummy_seg_model(num_classes: int):
        return DummySegmentationModell(num_classes=num_classes)

    return __make_dummy_seg_model
