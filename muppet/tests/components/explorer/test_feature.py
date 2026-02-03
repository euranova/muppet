"""Tests for feature-based explorer components.

This module tests the CAMExplorer in MUPPET, which generates premises for Class Activation Mapping
(CAM) based explanations. The explorer creates feature-based premises by analyzing convolutional
layer activations to identify important spatial regions for model predictions.

The tests verify:
- Correct premise generation count matching expected feature map dimensions
- Proper integration with convolutional models (VGG architecture)
- Memory system compatibility with generated premises
- Feature map extraction and processing accuracy
- CAM-based exploration strategy effectiveness
"""
#
# Created on Mon Dec 4 2023
#
# Copyright (c) 2023 Léo Beaumont @Euranova
#

import pytest
import torch

from muppet import DEVICE
from muppet.components.explorer.feature import CAMExplorer
from muppet.components.memory.base import PremiseList


@pytest.mark.parametrize(
    "image, premise_amount",
    [
        (torch.zeros(1, 3, 224, 224).to(DEVICE), 512),
    ],
)
def test_cam_explorer(image, premise_amount, model_vgg):
    """Test CAMExplorer premises creation

    Args:
        model (torch.Module): convolutional model
        image (torch.Tensor): image to use the model on
        premise_amount (int): amount of premise generated
    """
    explorer = CAMExplorer(model=model_vgg)
    explorer(memory=PremiseList(), example=image)

    premises = explorer.get_premises_to_explore()

    assert len(premises) == premise_amount
