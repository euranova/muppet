"""Segmentation model wrappers for the MUPPET benchmark framework.

This module provides unified wrapper classes for semantic segmentation models
with a consistent PyTorch-based interface for loading, training, and inference.

Classes:
    SegmentationModel: Unified wrapper for semantic segmentation models
"""

import torch

from muppet.benchmark.models.base import AbstractModel


class SegmentationModel(AbstractModel):
    """A unified wrapper for semantic segmentation models, providing a consistent API for PyTorch models.

    This class handles model loading from Hydra-style configurations and offers a
    PyTorch-like interface with `.fit()` and `.forward()` methods, enabling
    seamless pipeline integration for segmentation tasks. It focuses specifically on computer vision segmentation tasks.

    Since this implementation currently only supports PyTorch models, it omits
    the hyperparameter tuning logic present in other model wrappers.
    """

    def __init__(
        self,
        model,
        name: str,
        pretrained: bool = True,
    ):
        """Initialize the SegmentationModel wrapper instance.

        Args:
            model (Union[DictConfig, Any]): The model configuration (e.g., from Hydra) or an already
                instantiated PyTorch model object.
            name (str): The name of the model.
            pretrained (bool, optional): Whether to use a pretrained version of the model. Defaults to True.

        Note:
            The `postprocessing_func` is automatically set to `torch.softmax(x, dim=1)`
            to convert the model's raw logits into class probabilities for each pixel.
        """
        super().__init__(
            model,
            name,
            pretrained,
            postprocessing_func=lambda x: torch.softmax(x, dim=1),
        )
        assert isinstance(self.model, torch.nn.Module), (
            "The model must be a PyTorch Module."
        )

    def fit(self, train_loader):
        """Trains or Fine-tunes the segmentation model using the provided training data.

        This method currently raises a `NotImplementedError`, as the training loop
        logic for PyTorch models needs to be implemented.

        Args:
            train_loader (DataLoader): The data loader containing the training samples.
        """
        if isinstance(self.model, torch.nn.Module):
            # TODO: implement pytorch trainer (lightning ?!) and train model
            raise NotImplementedError(
                "PyTorch fit function not implemented yet."
            )

    def forward(self, x):
        """Performs a forward pass for inference on the segmentation model.

        This method calls the PyTorch model directly and extracts the 'out' key
        from the output dictionary, which is a common pattern for segmentation
        models from torchvision or similar libraries.

        Args:
            x (torch.Tensor): The input tensor, typically an image.

        Returns:
            torch.Tensor: The output tensor from the model, representing the
            segmentation map.
        """
        return self.model(x)["out"]
