"""Base generator classes for producing perturbation values in MUPPET XAI framework.

This module defines the foundation for generators used by perturbators to create realistic
replacement values for masked regions during the perturbation process. Generators are
essential components that enable sophisticated perturbation strategies beyond simple
replacement with zeros or noise.

In the MUPPET four-step framework (generate masks → apply perturbations → calculate
attributions → aggregate results), generators support the perturbation step by providing
contextually appropriate replacement values. This is crucial for maintaining data realism
and producing meaningful explanations.

The module contains:
    Generator: Abstract base class for generators that don't require training, suitable
        for simple statistical sampling or rule-based value generation.
    TrainableGenerator: Extended abstract class with built-in training infrastructure
        for neural network-based generators that learn data distributions.

Key Design Principles:
    - Generators focus solely on producing replacement values
    - Training is handled transparently with early stopping and validation splits
    - Deterministic sampling support through optional seed parameters
    - Extensible architecture for domain-specific perturbation strategies

Note:
    Generators are typically not used directly but are embedded within perturbator
    implementations. They enable advanced explanation methods like conditional sampling,
    learned imputations, and distribution-aware perturbations.
"""

#
# Created on Wed May 24 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import time
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from muppet import DEVICE, logger


class Generator(ABC):
    """Abstract base class for data generators in perturbation methods.

    Generators create synthetic data to replace masked or perturbed
    regions in input examples. They provide the core imputation
    functionality for creating meaningful perturbations.

    """

    def __init__(
        self,
    ) -> None:
        """Abstract class for generators that don't need to be trained on data.

        Attributes:
            device: The used device. Will get updated from the main explainer after initialization.

        """
        self.device = DEVICE
        super().__init__()

    @abstractmethod
    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Responsible for generating the perturbed values. It is called by the `Perturbator.perturbate` method.

        Fully customizable and must be implemented in child generator that is required by a perturbator.

        For deterministic sampling at inference time, take advantage of the passing a seed parameter
        in order to fix the seed, something like torch.manual_seed(seed), as in `GaussianFeatureGenerator`.

        """
        raise NotImplementedError


class TrainableGenerator(torch.nn.Module, Generator):
    """Abstract base class for trainable neural network generators.

    Extends the basic Generator with PyTorch neural network capabilities
    and built-in training infrastructure. Supports complex learned
    perturbation strategies through gradient-based optimization

    Implementing subclass of this trainable perturbator only requires to
    implement the `run_epoch` method.

    """

    def __init__(
        self,
        lr: float,
        num_epochs: int,
    ) -> None:
        """Abstract class for generators with the train method implemented.

        Args:
            lr (float): Learning rate
            num_epochs (int): Number of epochs

        Attributes:
            device: The used device. Will get updated from the main explainer after initialization.

        """
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = DEVICE
        super().__init__()

    def train_generator(
        self,
        train_loader: DataLoader,
        validation_ratio: float = 0.2,
    ) -> Tuple[list, list]:
        """Train the model.

        Args:
            train_loader (DataLoader): The train data loader

        Returns:
            The training results trends history

        """
        stime = time.time()
        self.to(self.device)

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)

        best_loss = np.inf

        train_loss_trends = list()
        validation_loss_trends = list()

        train_size = int((1 - validation_ratio) * len(train_loader.dataset))
        trainset_dataset, validation_dataset = torch.utils.data.random_split(
            train_loader.dataset,
            [train_size, len(train_loader.dataset) - train_size],
            generator=torch.Generator(
                device=torch.get_default_device()
            ).manual_seed(42),
        )
        trainset_loader = DataLoader(
            trainset_dataset.dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
        )
        validation_loader = DataLoader(
            validation_dataset.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
        )

        for epoch in range(self.num_epochs + 1):
            train_loss = self.run_epoch(
                dataloader=trainset_loader, in_train=True
            )
            validation_loss = self.run_epoch(
                dataloader=validation_loader,
                in_train=False,
            )

            train_loss_trends.append(train_loss)
            validation_loss_trends.append(validation_loss)

            if epoch % 10 == 0:
                logger.info(f"\nEpoch {epoch}")
                logger.info(f"Model Training Loss ===> {train_loss}")
                logger.info(f"Model Validation Loss ===> {validation_loss}")

            if validation_loss < best_loss:
                best_loss = validation_loss
                # Add early stopping
            elif validation_loss > 1.2 * best_loss:
                # Note: maybe save the model at this epoch later on
                break

        logger.info(
            f"Model validation loss = {best_loss:.6f}  | Exucution time: {time.time() - stime}"
        )

        return train_loss_trends, validation_loss_trends

    @abstractmethod
    def run_epoch(
        self,
        dataloader: DataLoader,
        in_train: bool,
    ) -> float:
        """Run one training epoch.
        This is a customizable method that depends on the nature of the generator!
        """
        raise NotImplementedError
