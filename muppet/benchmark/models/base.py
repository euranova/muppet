"""Base model classes for the MUPPET benchmark framework.

This module provides abstract base classes for creating unified model wrappers
that support different frameworks (PyTorch, scikit-learn, XGBoost) with a
consistent API. It enables seamless integration across various model types
through standardized training and inference interfaces.

Classes:
    AbstractModel: Abstract base class for unified model wrappers
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch

from muppet import DEVICE


class AbstractModel(torch.nn.Module, ABC):
    """A flexible model wrapper that unifies training and inference for models from different frameworks:
    PyTorch, scikit-learn, and XGBoost.

    This class supports model configuration through Hydra. If the provided model config contains
    a `_target_` key (Hydra-style), it loads and instantiates the model dynamically. It allows both
    standard training and Optuna-based hyperparameter optimization (for non-PyTorch models).
    It enables seamless integration across various model types through standardized training and inference interfaces.

    It behaves like a standard PyTorch module, with `.fit()` for training and `.forward()` for inference,
    enabling easy integration into a PyTorch-style pipeline.
    """

    def __init__(
        self,
        model,
        name: str,
        pretrained: bool = True,
        postprocessing_func: Callable = lambda x: x,
    ) -> None:
        """Initialize the AbstractModel wrapper instance.

        Args:
            model (Union[DictConfig, Any]): Model configuration or model instance.
            name (str): Name of the model.
            pretrained (bool, optional): Whether to use a pretrained version (if supported).
                                        Default is True.
            postprocessing_func (Callable, optional): A function to apply to the model's raw outputs.
                                                      Defaults to a no-op lambda function.
        """
        super().__init__()
        self.model = model
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.to(DEVICE)

        self.name = name
        self.pretrained = pretrained
        self.postprocessing_func = postprocessing_func

    @abstractmethod
    def fit(self, train_loader):
        """Trains the model on the provided data.

        This method must be implemented by subclasses. It should handle the model's training
        loop and optimization.

        Args:
            train_loader (DataLoader): The data loader containing training samples.
        """
        raise NotImplementedError(
            "This method must be implemented explicitly in subclasses."
        )

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Performs a forward pass through the model for inference.

        This method must be implemented by subclasses. It defines the model's behavior
        during inference.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        raise NotImplementedError(
            "This method must be implemented explicitly in subclasses."
        )

    def infer_model(self, dataloader) -> tuple[np.ndarray, np.ndarray]:
        """Runs inference using the wrapped model on the provided DataLoader.

        This function processes the input data in batches, performs a forward pass through
        the model without computing gradients, and collects both the input data and the
        predicted outputs after post-processing.

        Args:
            dataloader (DataLoader): A DataLoader providing batches of input data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - inputs (np.ndarray): The concatenated input data as a NumPy array.
                - predictions (np.ndarray): The concatenated post-processed predictions as a NumPy array.
        """
        all_inputs = []
        all_predictions = []

        for batch in dataloader:
            inputs = batch[0].to(DEVICE)
            with torch.no_grad():
                outputs = self(inputs)
            predictions = self.postprocessing_func(outputs)

            all_predictions.append(predictions.detach().cpu().numpy())
            all_inputs.append(batch[0].detach().cpu().numpy())

        inputs_array = np.concatenate(all_inputs, axis=0)
        predictions_array = np.concatenate(all_predictions, axis=0)

        return inputs_array, predictions_array
