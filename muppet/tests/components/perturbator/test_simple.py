"""Tests for simple perturbator components.

This module tests simple perturbation methods in MUPPET, including BlurPerturbator and
SetToZeroPerturbator for image data. These perturbators apply basic transformations like
blurring and masking to create perturbed versions of input images.

The tests verify:
- Correct blur perturbation application based on mask patterns
- SetToZero perturbation behavior with different mask configurations
- Batch processing capability for large-scale perturbations
- Memory management and batch size handling for out-of-memory scenarios
- Expected perturbation outputs with various mask and input combinations
"""
#
# Created on Tue Jul 04 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import pytest
import torch

from muppet.components.perturbator.simple import (
    BlurPerturbator,
    SetToZeroPerturbator,
)


@pytest.mark.parametrize(
    "masks, expected_perturbation, perturbator_class",
    [
        (
            torch.zeros((1, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            torch.ones((1, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            BlurPerturbator,
        ),
        (
            torch.zeros((10, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            torch.ones((10, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            BlurPerturbator,
        ),
        (
            torch.ones((1, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            torch.ones((1, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            BlurPerturbator,
        ),
        (
            torch.zeros((1, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            torch.ones((1, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            SetToZeroPerturbator,
        ),
        (
            torch.zeros((10, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            torch.ones((10, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            SetToZeroPerturbator,
        ),
        (
            torch.ones((1, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            torch.zeros((1, 1, 1, 20, 20)),  # (N=1, b=1, c=1, w=20, h=20)
            SetToZeroPerturbator,
        ),
    ],
)
def test_blur_perturbator(masks, expected_perturbation, perturbator_class):
    """Test the BlurPerturbator with a mask full of zeros."""
    x = torch.ones((1, 1, 20, 20))  # (b=1, c=1, w=20, h=20)

    perturber = perturbator_class()
    perturbed_x = perturber.perturbate(x, masks)  # (N=1, x.shape)

    assert torch.round(perturbed_x, decimals=3).equal(
        torch.round(expected_perturbation, decimals=3)
    )


def test_batch_perturbator_computation():
    """Test that the perturbator handles out-of-memory errors gracefully."""
    x = torch.ones((1, 3, 224, 224))  # (b=1, c=1, w=224, h=224)
    nb_samples = 100
    masks = torch.ones(
        (nb_samples, 1, 1, 224, 224)
    )  # (N=10000, c=1, w=224, h=224)

    perturber = SetToZeroPerturbator(max_batch_size=33)
    perturbed_x = perturber(x, masks)  # (N=1, x.shape)

    assert perturbed_x.shape == torch.Size((nb_samples, 1, 3, 224, 224))
