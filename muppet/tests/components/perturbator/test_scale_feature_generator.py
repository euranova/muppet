"""Tests for scale feature perturbator components.

This module tests the ScaleFeaturePerturbator in MUPPET, which applies scaling-based
perturbations to input features. The perturbator uses generators to create scaled
versions of inputs while preserving original values based on binary mask patterns.

The tests verify:
- Correct mask-based scaling perturbation application
- Original value preservation where masks are active
- Generator scaling application where masks are inactive
- Tensor shape consistency after scaling perturbations
- Expected output values with various input and mask combinations
"""

import pytest
import torch

from muppet.components.perturbator.scale_feature_generator import (
    ScaleFeaturePerturbator,
)


class GeneratorScale:
    """Mock generator class for testing scale feature perturbation.

    A test double that simulates generative model behavior for
    scale feature perturbation testing with simplified methods.
    """

    def train_generator(self):
        """Test stub for generator training."""
        pass

    def generate(self, x_t: torch.Tensor, data_scaled) -> torch.Tensor:
        """Test generator method for scaled feature perturbation."""
        return data_scaled * x_t + 2


@pytest.mark.parametrize(
    "x_t, masks, expected_output",
    [
        (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[0, 1, 0], [1, 0, 1]]),
            torch.tensor([[2.0, 4.0, 2.0], [3.0, 2.0, 5.0]]),
        ),
        (
            torch.tensor([-1.0, 0.0, 1.0]),
            torch.tensor([[1, 0, 1], [0, 1, 0]]),
            torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]),
        ),
        (
            torch.tensor([0.5, 1.5, 2.5]),
            torch.tensor([[1, 1, 1], [0, 0, 1]]),
            torch.tensor([[2.5, 3.5, 4.5], [2.0, 2.0, 4.5]]),
        ),
    ],
)
def test_scale_feature_perturbator(x_t, masks, expected_output):
    """Test scale feature perturbator with different input configurations.

    Validates that the perturbator correctly applies mask-based scaling
    perturbations, preserving original values where masks are active
    and applying generator scaling otherwise.

    Args:
        x_t: Input tensor to be perturbed.
        masks: Binary mask tensor controlling perturbation application.
        expected_output: Expected tensor output after perturbation.

    Returns:
        None: Test passes if perturbed output matches expected values.
    """
    assert masks.size(0) >= 2  # Ensure masks contains at least 2 masks
    g = GeneratorScale()
    perturbator = ScaleFeaturePerturbator(generator=g)
    perturbed_x = perturbator.perturbate(x_t, masks)
    expected_output = expected_output.unsqueeze(1)
    assert perturbed_x.shape == expected_output.shape
    assert torch.equal(perturbed_x, expected_output)
