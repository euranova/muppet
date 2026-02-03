"""Tests for features combination premise components.

This module tests the FeaturesCombinationPremise in MUPPET, which creates premises from
combinations of multiple features or activations. The premise generates masks by combining
key-activation pairs and performs upscaling to match target dimensions.

The tests verify:
- Correct mask generation from key-activation combinations
- Proper upscaling of masks to target spatial dimensions
- Mask-heatmap consistency for feature combination premises
- Expected behavior with different activation patterns and keys
- Integration with device management and tensor operations
"""
#
# Created on Wed Dec 6 2023
#
# Copyright (c) 2023 Léo Beaumont @EuraNova
#

import pytest
import torch

from muppet.components.memory import FeaturesCombinationPremise

fake_activation = (torch.rand(1, 1, 3, 3) > 0.5) * 1.0


@pytest.mark.parametrize(
    "key, activations, upscaled_mask_shape, expected_result",
    [
        (
            torch.rand(1, 10),
            torch.zeros(1, 10, 3, 3),
            (3, 3),
            torch.ones(1, 1, 3, 3),
        ),
        (torch.rand(1, 1), fake_activation, (3, 3), 1 - fake_activation),
    ],
)
def test_features_combination_premise_mask_gen(
    key, activations, upscaled_mask_shape, expected_result
):
    """Test features combination premise mask generation.

    Validates that the FeaturesCombinationPremise correctly generates
    masks based on key-activation pairs and upscaling requirements,
    ensuring proper mask-heatmap consistency.

    Args:
        key: Key tensor identifying the feature combination.
        activations: Activation tensor for the feature combination.
        upscaled_mask_shape: Target shape for mask upscaling.
        expected_result: Expected mask tensor output.

    Returns:
        None: Test passes if generated mask matches expected result.
    """
    premise = FeaturesCombinationPremise(
        key=key,
        activations=activations,
        upscaled_mask_shape=upscaled_mask_shape,
    )
    premise.device = torch.get_default_device()
    assert premise.get_mask().equal(expected_result)
    assert torch.equal(premise.mask, premise.heatmap)
