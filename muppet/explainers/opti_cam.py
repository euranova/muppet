"""Opti-CAM (Optimizing CAM) explainer for CNN interpretability.

This module implements Opti-CAM, a method that optimizes Class Activation Mapping (CAM)
by learning the optimal linear combination of feature maps from the last convolutional
layer. Unlike traditional CAM methods that use fixed weights, Opti-CAM optimizes these
weights per input image to maximize the model's response for a given class.

MUPPET Component Integration:
    - **Explorer**: `GradientCAMExplorer` - optimizes linear combination coefficients for feature maps
    - **Perturbator**: `SetToZeroPerturbator` - applies zero-masking perturbations for evaluation
    - **Attributor**: `MaskRegularizedScoreAttributor` - computes optimization loss for coefficient learning
    - **Aggregator**: `LearntMaskAggregator` - returns the optimized feature combination as explanation

Classes:
    OptiCAMExplainer: Implementation of Opti-CAM for CNN interpretability.

References:
    Zhan, Hanwei, et al. "Optimizing saliency maps for interpretability."
    arXiv preprint arXiv:2301.07002 (2023).
    https://arxiv.org/pdf/2301.07002.pdf
"""

#
# Created on Tue Dec 5 2023
#
# Copyright (c) 2023 Léo Beaumont @Euranova
#
from typing import Union

import torch

from muppet.components.aggregator import LearntMaskAggregator
from muppet.components.attributor.differentiable import (
    MaskRegularizedScoreAttributor,
)
from muppet.components.convention import AttributionConvention
from muppet.components.explorer.gradient import GradientCAMExplorer
from muppet.components.memory import FeaturesCombinationPremise
from muppet.components.perturbator import SetToZeroPerturbator
from muppet.explainers.base import MuppetExplainer


class OptiCAMExplainer(MuppetExplainer):
    """OptiCAM (Optimized Class Activation Mapping) explainer implementation.

    Implements Opti-CAM that learns optimal feature map combinations through gradient-based
    optimization to generate improved class activation maps. Opti-CAM addresses limitations
    of gradient-based CAM methods by directly optimizing the linear combination coefficients
    through gradient ascent.

    The method works by:
    1. Extracting feature maps from the last convolutional layer
    2. Learning optimal linear combination weights through gradient optimization
    3. Computing the weighted sum of feature maps as the final saliency map
    4. Applying post-processing (ReLU) to ensure positive importance scores

    This results in more accurate and focused saliency maps that better highlight the
    regions responsible for the model's predictions.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_iter: int = 100,
        lr: float = 0.2,
        nb_premises_at_startup: int = 1,
        convention: Union[AttributionConvention, str] = "constructive",
    ) -> None:
        """Initialize the Opti-CAM explainer for CNN interpretability.

        Args:
            model (torch.nn.Module): Convolutional neural network (CNN)
            max_iter (int): The number of iterations for SGD.
            lr (float): Learning rate.
            nb_premises_at_startup (int): How many premises generated at startup. Leave to 1 in most cases, unless you want to separately optimize several trajectories.
            convention: choose if the explainer finds important features by identifying features that destroy (destructive) efficiently the model's prediction from the input,
                or by identifying features that build (constructive) efficiently the model's response from a completly perturbed input
        """
        self.coefficients = None

        explorer = GradientCAMExplorer(
            max_iter=max_iter,
            lr=lr,
            premise_class=FeaturesCombinationPremise,
            nb_premises_at_startup=nb_premises_at_startup,
        )
        perturbator = SetToZeroPerturbator()
        attributor = MaskRegularizedScoreAttributor(convention=convention)
        aggregator = LearntMaskAggregator(convention=convention)

        # Initialize the main explainer
        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )
