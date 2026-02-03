"""Tests for distribution-based aggregation components.

This module tests the distribution aggregation functionality in MUPPET,
specifically focusing on Monte Carlo KL divergence-based aggregation methods.
It validates that distribution aggregators correctly compute attributions
from premise lists and handle probabilistic aggregation scenarios.

The tests verify:
- Proper attribution calculation using Monte Carlo methods
- KL divergence aggregation accuracy
- Memory premise handling and retrieval
- Expected output shapes and numerical values
- Consistency across multiple aggregation runs
"""
#
# Created on Mon May 15 2023
#
# Copyright (c) 2023 Ismail Bachchar @EuraNova
#

import numpy as np
import torch

from muppet.components.aggregator.distribution import MonteCarloKLAggregator
from muppet.components.memory import TimeStepPremise
from muppet.components.memory.base import PremiseList


def test_proba_aggregator_scores():
    """Simple test that ProbaAggregator calculates the expected attributions."""
    custom_keys = [
        ({"timestep": 4, "feature": 0}, (1, 5)),
        ({"timestep": 4, "feature": 0}, (1, 5)),
        ({"timestep": 3, "feature": 0}, (1, 5)),
        ({"timestep": 3, "feature": 0}, (1, 5)),
        ({"timestep": 2, "feature": 0}, (1, 5)),
        ({"timestep": 2, "feature": 0}, (1, 5)),
        ({"timestep": 1, "feature": 0}, (1, 5)),
        ({"timestep": 1, "feature": 0}, (1, 5)),
    ]
    custom_attribution = torch.tensor([1.0]).unsqueeze(0).repeat(8, 1)
    expected_attribution = 2.0 / (1 + np.exp(-5)) - 1

    memory = PremiseList()
    memory.register_premises([TimeStepPremise(key=i) for i in custom_keys])
    for i, j in zip(memory.get_premises(), custom_attribution):
        i.attribution = j

    aggregator = MonteCarloKLAggregator(num_sampling=2)

    r = aggregator.get_explanation(memory=memory)  # (b=1, f, t)

    # t=0 is always 0, not explained
    assert r[0, 0, 0].item() == 0
    # make sure all scores are similar
    for i in r[0, 0, 2:]:
        assert i.equal(r[0, 0, 1])
        assert round(i.item(), 4) == round(expected_attribution, 4)
