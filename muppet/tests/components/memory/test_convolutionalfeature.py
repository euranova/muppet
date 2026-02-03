"""Tests for convolutional feature premise components.

This module tests the ConvolutionalFeaturePremise in MUPPET, which creates premises from
convolutional neural network feature maps. The premise generates masks based on activation
patterns in specific channels of feature maps for CAM-style explanations.

The tests verify:
- Correct mask generation from convolutional activation channels
- Proper normalization and thresholding of activation values
- Expected mask patterns for different activation distributions
- Channel-specific mask creation with various activation signatures
- Integration with upsampled feature map dimensions
"""
#
# Created on Mon Dec 4 2023
#
# Copyright (c) 2023 Léo Beaumont @EuraNova
#

import pytest
import torch

from muppet.components.memory import ConvolutionalFeaturePremise

fake_activations = torch.tensor(
    [
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # (b=1, c=4, w=3, h=3)
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[1, -1, 0], [0, 1, 1], [-1, -1, -1]],
        ]
    ]
)

diagonal_mask = torch.tensor(
    [[[[0, 1, 1], [1, 0, 1], [1, 1, 0]]]]  # (b=1, c=1, w=3, h=3)
)

last_expected_mask = torch.tensor([[[[0, 1, 1 / 2], [1 / 2, 0, 0], [1, 1, 1]]]])


@pytest.mark.parametrize(
    "activations, channel, expected_mask",
    [
        (fake_activations, 0, torch.ones(1, 1, 3, 3)),
        (fake_activations, 1, diagonal_mask),
        (fake_activations, 2, 1 - diagonal_mask),
        (fake_activations, 3, last_expected_mask),
    ],
)
def test_convolutional_feature_premise_mask_generation(
    activations, channel, expected_mask
):
    """Test ConvolutionalFeaturePremise mask generation functionality.

    Args:
        activations (torch.Tensor): up-sampled activation of the last convolutional layer
        channel (int): channel of the activation to use
        expected_mask: Expected mask to be generated
    """
    key = (activations, channel)
    premise = ConvolutionalFeaturePremise(key)

    assert premise.mask.equal(expected_mask)
