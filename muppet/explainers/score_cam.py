"""Score-CAM (Score-weighted Class Activation Mapping) explainer for CNNs.

This module implements Score-CAM, a gradient-free method for generating class activation
maps that overcomes limitations of gradient-based CAM approaches. Instead of using
gradients to weight feature maps, Score-CAM evaluates the contribution of each activation
map by using it as a mask and measuring the resulting change in the model's confidence.

MUPPET Component Integration:
    - **Explorer**: `CAMExplorer` - extracts feature maps from the model's last convolutional layer
    - **Perturbator**: `SetToZeroPerturbator` - applies feature map-based masking to input
    - **Attributor**: `ClassScoreAttributor` - evaluates model confidence on masked inputs
    - **Aggregator**: `WeightedSumAggregator` - combines feature maps weighted by their scores with ReLU post-processing

Classes:
    ScoreCAMExplainer: Implementation of Score-CAM for CNN explanation.

References:
    Wang, Haofan, et al. "Score-CAM: Score-weighted visual explanations for convolutional
    neural networks." Proceedings of the IEEE/CVF conference on computer vision and
    pattern recognition workshops. 2020.
    https://arxiv.org/pdf/1910.01279.pdf
"""

#
# Created on Wed Nov 29 2023
#
# Copyright (c) 2023 Léo Beaumont @Euranova
#
from typing import Union

import torch

from muppet.components.aggregator.mask import WeightedSumAggregator
from muppet.components.attributor.classification import ClassScoreAttributor
from muppet.components.convention import AttributionConvention
from muppet.components.explorer.feature import CAMExplorer
from muppet.components.perturbator.simple import SetToZeroPerturbator
from muppet.explainers.base import MuppetExplainer


class ScoreCAMExplainer(MuppetExplainer):
    """ScoreCAM (Score-weighted Class Activation Mapping) explainer implementation.

    Implements ScoreCAM method that generates visual explanations by weighting feature maps
    based on their contribution scores to the final prediction, providing parameter-free
    class activation mappings.

    Score-CAM addresses several issues with traditional gradient-based methods:
    - Eliminates gradient saturation problems
    - Provides more reliable importance scores
    - Removes dependence on gradient computations
    - Offers better visual explanation quality

    The method works by:
    1. Extracting feature maps from the last convolutional layer
    2. Using each feature map as a mask to perturb the input image
    3. Evaluating the masked input to get confidence scores
    4. Computing a weighted combination of feature maps using these scores

    This approach provides a more direct measurement of each feature map's contribution
    to the final prediction, leading to more accurate and interpretable visualizations.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        convention: Union[AttributionConvention, str] = "constructive",
    ) -> None:
        """Initialize the Score-CAM explainer for CNN explanation.

        Args:
            model (torch.nn.Module): The convolutional model to explain its predictions.
            convention: choose if the explainer finds important features by identifying
                features that destroy (destructive) efficiently the model's prediction
                from the input, or by identifying features that build (constructive)
                efficiently the model's response from a completly perturbed input
        """
        # Parameters

        explorer = CAMExplorer(model=model)
        perturbator = SetToZeroPerturbator()
        attributor = ClassScoreAttributor(convention=convention)
        aggregator = WeightedSumAggregator(
            convention=convention,
            post_proc=lambda x: torch.nn.functional.relu(x),
        )

        # Initialize the explainer with these modules
        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )
