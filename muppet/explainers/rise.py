"""RISE (Randomized Input Sampling for Explanation) explainer for black-box models.

This module implements RISE, a model-agnostic explanation method that generates
importance maps by probing the model with randomly masked versions of the input.
RISE is particularly effective for image classification tasks and works entirely
through black-box access to the model, making it broadly applicable.

MUPPET Component Integration:
    - **Explorer**: `RandomMasksExplorer` - generates random binary masks with configurable sparsity
    - **Perturbator**: `SetToZeroPerturbator` - applies zero-masking to occlude input regions
    - **Attributor**: `ClassScoreAttributor` - extracts model confidence scores for target class
    - **Aggregator**: `WeightedSumAggregator` - computes weighted average of masks using confidence scores

Classes:
    RISEExplainer: Implementation of the RISE method for black-box model explanation.

References:
    Petsiuk, Vitali, Abir Das, and Kate Saenko. "RISE: Randomized input sampling for
    explanation of black-box models." arXiv preprint arXiv:1806.07421 (2018).
    https://ui.adsabs.harvard.edu/abs/2018arXiv180607421P/abstract
"""
#
# Created on Fri Jun 09 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

from typing import Union

import torch

from muppet.components.aggregator.mask import WeightedSumAggregator
from muppet.components.attributor import ClassScoreAttributor
from muppet.components.convention import AttributionConvention
from muppet.components.explorer.mask import RandomMasksExplorer
from muppet.components.perturbator.simple import SetToZeroPerturbator
from muppet.explainers.base import MuppetExplainer


class RISEExplainer(MuppetExplainer):
    """RISE (Randomized Input Sampling for Explanation) explainer implementation.

    Implements the RISE method that generates importance maps through random masking
    and statistical aggregation of model responses. The core principle of RISE is to
    generate a large number of random masks, apply them to the input, evaluate the
    masked inputs with the model, and then compute a weighted average of the masks
    where the weights are the model's confidence scores.

    Key advantages of RISE:
    - Model-agnostic: works with any black-box model
    - Simple and interpretable: directly measures prediction changes under occlusion
    - Flexible: supports both constructive and destructive explanation modes

    This approach provides a statistical estimation of pixel importance without requiring
    any knowledge of the model's internal structure. The method generates smooth,
    intuitive heatmaps that highlight the most important regions for the model's
    prediction. The statistical nature of the approach means more masks generally
    lead to better approximations of the true importance.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        nmasks: int = 800,
        mask_dim: int = 7,
        mask_proba: float = 0.1,
        seed: int | None = None,
        convention: Union[AttributionConvention, str] = "destructive",
    ) -> None:
        """Initialize the RISE explainer for black-box model explanation.

        Args:
            model (torch.nn.Module): The black-box model to explain its predictions.
            nmasks (int): Number of random masks to generate.
            mask_dim (int): The size of the squared grade (down-scaled mask).
            mask_proba (float): The probability of setting, independently, each value of the (downscaled) mask to 0 meaning there will be no perturbation at this position.
            seed (int, optional): Seed to initialize for reproducible results.
            convention: choose if the explainer finds important features by identifying features that destroy (destructive) efficiently the model's prediction from the input,
                or by identifying features that build (constructive) efficiently the model's response from a completly perturbed input
        """
        # Parameters
        self.nmasks = nmasks
        self.mask_dim = mask_dim
        self.mask_proba = mask_proba

        explorer = RandomMasksExplorer(
            nmasks=self.nmasks,
            mask_dim=self.mask_dim,
            mask_proba=self.mask_proba,
            seed=seed,
        )
        perturbator = SetToZeroPerturbator()
        attributor = ClassScoreAttributor(convention=convention)
        aggregator = WeightedSumAggregator(convention=convention)

        # Initialize the explainer with these modules
        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )
