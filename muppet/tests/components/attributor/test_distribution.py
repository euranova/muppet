"""Tests for distribution-based attribution components.

This module tests the distribution attribution functionality in MUPPET,
specifically focusing on probability shift attributors that compute
attribution based on distribution changes in model outputs. It validates
temporal attribution methods for time series data.

The tests verify:
- Probability shift attribution calculation
- Temporal attribution handling with timestep exploration
- Memory management for temporal premises
- Attribution accuracy for time series perturbations
- Expected attribution values and consistency across timesteps
"""
#
# Created on Mon May 15 2023
#
# Copyright (c) 2023 Ismail Bachchar @EuraNova
#

import torch

from muppet.components.attributor.distribution import ProbaShiftAttributor
from muppet.components.explorer.timestep import RepeatedTimestepExplorer
from muppet.components.memory.base import PremiseList

custom_perturbations = torch.tensor(
    [
        [[[1, 1, 1, 1, 100]]],
        [[[1, 1, 1, 1, 100]]],
        [[[1, 1, 1, 100, torch.nan]]],
        [[[1, 1, 1, 100, torch.nan]]],
        [[[1, 1, 100, torch.nan, torch.nan]]],
        [[[1, 1, 100, torch.nan, torch.nan]]],
        [[[1, 100, torch.nan, torch.nan, torch.nan]]],
        [[[1, 100, torch.nan, torch.nan, torch.nan]]],
    ]
)

x = torch.ones(1, 1, 5)


def _model(x):
    return torch.tensor([0.5, 0.3, 0.2]).reshape((1, 3))


def test_proba_attributor_expected_attributions():
    """Test that ProbaAttributor calculates and save into memory the expected attributions."""
    memory = PremiseList()
    explorer = RepeatedTimestepExplorer(num_sampling=2)
    # set the input shape. It's done by the main explainer at runtime
    explorer.example = x

    for new_premises in explorer:
        memory.register_premises(new_premises)

    attributor = ProbaShiftAttributor(padding=None)
    attributor(
        x=x,
        perturbed_inputs=custom_perturbations,
        model=_model,
        memory=memory,
    )

    # check that all saved premises have a zero attribution
    for i in memory.get_premises():
        assert i.attribution == 0
