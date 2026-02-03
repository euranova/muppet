"""Tests for timestep-based explorer components.

This module tests the RepeatedTimestepExplorer in MUPPET, which generates time-series specific
premises for temporal data explanation. The explorer creates timestep-based masks and premises
for analyzing sequential data and time-dependent model predictions.

The tests verify:
- Correct generation of timestep-based premise keys
- Expected number of premises for different sequence lengths
- Monte Carlo sampling integration with timestep exploration
- Binary mask premise independence across temporal dimensions
- Proper handling of time series data shapes and structures
"""
#
# Created on Fri May 12 2023
#
# Copyright (c) 2023 Ismail Bachchar @EuraNova
#

import pytest
import torch

from muppet.components.explorer.timestep import RepeatedTimestepExplorer
from muppet.components.memory import BinaryRandomPremise

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


def test_repeated_timestep_explorer_generated_keys():
    """Test that RepeatedTimestepExplorer explorer calculates generates the expected premises keys."""
    signal_length = 5
    explorer = RepeatedTimestepExplorer(num_sampling=2)
    explorer.example = torch.zeros(
        1,
        1,
        signal_length,
    )  # (b, f, t) set needed example. Usually this is done by the main-explainer at runtime

    expected_num_premises = (signal_length - 1) * 2

    for new_premises in explorer:
        premise_keys = [i.key for i in new_premises]

    assert premise_keys == custom_keys
    assert len(premise_keys), expected_num_premises


@pytest.mark.parametrize(
    "signal_length,num_sampling", [(100, 10), (200, 2), (10, 1)]
)
def test_repeated_timestep_explorer_length(signal_length, num_sampling):
    """Test that RepeatedTimestepExplorer explorer calculates generates the expected number of premises for different sequences and Monte
    Carlo samples.

    Args:
        signal_length (int): The sequence length.
        num_sampling (int): Monte Carlo number of samples.

    """
    explorer = RepeatedTimestepExplorer(num_sampling=num_sampling)
    explorer.example = torch.zeros(
        1,
        1,
        signal_length,
    )  # (b, f, t) set needed input shape. Usually this is done by the main-explainer at runtime
    for new_premises in explorer:
        premise_keys = [i.key for i in new_premises]

    assert len(premise_keys) == (signal_length - 1) * num_sampling


def test_binarymask_premise_independent_rows():
    """Ensure that binary mask can generate independent rows."""
    batch = (3, 4)
    mask_shape = (3, 256)
    mask_prob = 0.5

    premise = BinaryRandomPremise(
        key=(batch, mask_prob, mask_shape),
        seed=42,
    )

    m = premise.mask[0, 0]

    assert m.shape == mask_shape
    assert not torch.equal(m[0], m[1])
    assert not torch.equal(m[1], m[2])
