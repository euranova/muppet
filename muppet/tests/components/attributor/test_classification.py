"""Tests for classification-based attribution components.

This module tests the classification attribution functionality in MUPPET,
focusing on score-based attributors that compute feature importance scores
from model predictions. It validates both constructive and destructive
attribution conventions and mask-regularized scoring methods.

The tests verify:
- Class score attribution accuracy and sign conventions
- Constructive vs destructive attribution modes
- Mask-regularized attribution calculation
- Attribution loss computation and expected values
- Memory management and premise handling for attributions
"""

#
# Created on Tue Jun 27 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#
import pytest
import torch

from muppet.components.attributor import ClassScoreAttributor
from muppet.components.attributor.differentiable import (
    MaskRegularizedScoreAttributor,
)
from muppet.components.memory import FeaturesCombinationPremise
from muppet.components.memory.base import PremiseList

ones_mask = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])

x_perturbed = torch.zeros((1, 1, 3, 2, 2))  # (mb, b, 3, w, h)


def model(x):
    """Simple mock model for testing purposes.

    Returns a fixed prediction tensor that favors the second class
    to enable consistent testing of attribution calculations.

    Args:
        x: Input tensor (unused in this mock implementation).

    Returns:
        torch.Tensor: Fixed prediction tensor of shape (1, 2) with
            higher confidence for class 1.
    """
    # expected to predict the second class
    return torch.tensor([[0.001, 2.2]])


class FakePremise:
    """Fake premise class for testing classification attributors.

    A mock premise that stores attribution values for testing
    classification attribution components in isolation.
    """

    def __init__(self, att=None) -> None:
        """Initialize fake premise for testing classification attributors.

        Args:
            att: Optional attribution value to store in the premise.
        """
        self.attribution = att


def test_class_score_attributor_constructive():
    """Test attributor by verifying that the calculated score is what
    is expected.
    """
    # one example of full of ones
    x = torch.ones((1, 3, 2, 2))  # (b, 3, w, h)
    expected_prediction = torch.softmax(model(x).detach(), dim=1)[0, 1].item()

    # create a fake memory to be used by attributor
    memory = PremiseList()
    memory.register_premises([FakePremise()])

    attributor = ClassScoreAttributor(convention="constructive")

    attributor.calculate_attribution(
        x=x, perturbed_inputs=x_perturbed, model=model, memory=memory
    )

    assert expected_prediction == memory.get_premises()[0].attribution.item()


fake_score = torch.ones(1)


@pytest.mark.parametrize(
    "perturbed_score, expected_result",
    [
        (fake_score, -fake_score),
        (-fake_score, fake_score),
        (0 * fake_score, 0 * fake_score),
    ],
)
def test_simple_score_attributor(perturbed_score, expected_result):
    """Test if the Loss calculated is really $-f(x')$."""
    fake_premise = FeaturesCombinationPremise(
        torch.tensor([]), torch.tensor([]), (0, 0)
    )
    attributor = MaskRegularizedScoreAttributor()
    assert attributor.calculate_attribution_loss(
        fake_premise, perturbed_score
    ).equal(expected_result)


def test_class_score_attributor_destructive():
    """Test attributor by verifying that the calculated score is what
    is expected.
    """
    # one example of full of ones
    x = torch.ones((1, 3, 2, 2))  # (b, 3, w, h)
    expected_prediction = -torch.softmax(model(x).detach(), dim=1)[0, 1].item()

    # create a fake memory to be used by attributor
    memory = PremiseList()
    memory.register_premises([FakePremise()])

    attributor = ClassScoreAttributor(convention="destructive")

    attributor.calculate_attribution(
        x=x, perturbed_inputs=x_perturbed, model=model, memory=memory
    )

    assert expected_prediction == memory.get_premises()[0].attribution.item()
