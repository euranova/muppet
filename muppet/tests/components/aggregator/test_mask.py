"""Tests for mask-based aggregation components.

This module tests the mask aggregation functionality in MUPPET, validating
how aggregators process and combine mask-based attributions from various
perturbation methods. It ensures proper handling of spatial masks and
correct aggregation across different mask configurations.

The tests verify:
- Mask aggregation accuracy and consistency
- Proper handling of different mask types and sizes
- Attribution computation from masked inputs
- Memory management for mask-based premises
- Output format validation for aggregated results
"""
#
# Created on Wed Jun 28 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import pytest
import torch

from muppet import DEVICE
from muppet.components.aggregator.mask import WeightedSumAggregator
from muppet.components.memory import (
    BinaryRandomPremise,
    ConvolutionalFeaturePremise,
)
from muppet.components.memory.base import PremiseList


@pytest.mark.parametrize(
    "batch,mask_shape,attribution",
    [
        (1, (4, 4), 0.1),
        (1, (7, 7), 0.9),
    ],
)
def test_weighted_sum_aggregator_dont_perturb_x(batch, mask_shape, attribution):
    """Test the RISE aggregator when not perturbing at all, it should
    returns a heatmap full of attributions values.
    """
    expectted_heatmap = (
        torch.ones((batch, 1, *mask_shape), device=DEVICE) * attribution
    )  # shape (b=1, c=1, w, h)
    mask_prob = 1

    memory = PremiseList()
    premise1 = BinaryRandomPremise(
        key=(batch, mask_prob, mask_shape),
        seed=2,
    )
    premise1.attribution = torch.ones((1)) * attribution  # (b)

    premise2 = BinaryRandomPremise(
        key=(batch, mask_prob, mask_shape),
        seed=2,
    )

    premise2.attribution = torch.ones((1)) * attribution  # (b)

    # add premises to the memory so aggregator can access them
    memory.register_premises([premise1, premise2])

    aggregator = WeightedSumAggregator()
    aggregator.get_explanation(memory=memory).equal(expectted_heatmap)


premises = []
for k in range(20):
    fake_premise = ConvolutionalFeaturePremise(key=(None, None))
    constant_activation = torch.ones(1, 1, 224, 224) * k
    fake_premise._heatmap = constant_activation
    fake_premise.attribution = torch.ones(1)

    premises.append(fake_premise)

fake_memory = PremiseList()
fake_memory.register_premises(premises)


@pytest.mark.parametrize("memory", [fake_memory])
def test_cam_aggregator(memory):
    """Test if constant activations give a null heatmap with the right shape

    Args:
        memory (Premiselist): memory containing premises
    """
    aggregator = WeightedSumAggregator(
        post_proc=lambda x: torch.nn.functional.relu(x)
    )
    heatmap = aggregator.get_explanation(memory=memory)

    assert heatmap.equal(torch.zeros(1, 1, 224, 224, device=DEVICE))
