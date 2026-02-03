"""Tests for timestep premise components.

This module tests the TimeStepPremise in MUPPET, which creates time-series specific premises
for temporal data explanations. The premise generates masks that highlight specific timesteps
and features in sequential data for time-dependent model analysis.

The tests verify:
- Correct mask generation for different timestep positions
- Proper handling of NaN values for future timesteps
- Feature-specific mask creation in multivariate time series
- Mask-heatmap consistency for timestep premises
- Expected mask patterns with various timestep and feature combinations
"""
#
# Created on Mon May 15 2023
#
# Copyright (c) 2023 Ismail Bachchar @EuraNova
#

import pytest
import torch

from muppet.components.memory import TimeStepPremise


@pytest.mark.parametrize(
    "timestep, mask_shape, feature, expected_mask",
    [
        (
            4,
            (1, 5),
            0,
            torch.tensor([[[0.0, 0.0, 0, 0, 1]]]),
        ),  # (b=1, f=1, t=5)
        (3, (1, 5), 0, torch.tensor([[[0.0, 0.0, 0, 1, torch.nan]]])),
        (2, (1, 5), 0, torch.tensor([[[0.0, 0.0, 1, torch.nan, torch.nan]]])),
        (
            1,
            (1, 5),
            0,
            torch.tensor([[[0.0, 1.0, torch.nan, torch.nan, torch.nan]]]),
        ),
        (
            2,
            (2, 5),
            1,
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 0, torch.nan, torch.nan],
                        [0.0, 0.0, 1, torch.nan, torch.nan],
                    ]
                ]
            ),
        ),
        (
            2,
            (2, 5),
            0,
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 1, torch.nan, torch.nan],
                        [0.0, 0.0, 0, torch.nan, torch.nan],
                    ]
                ]
            ),
        ),
    ],
)
def test_time_step_premise_mask_generation(
    timestep, mask_shape, feature, expected_mask
):
    """Test TimeStepPremise mask generation functionality.

    Args:
        timestep (int): Timestep number.
        expected_mask (torch.Tensor): Expected mask to be generated.

    """
    premise = TimeStepPremise(
        key=({"timestep": timestep, "feature": feature}, mask_shape)
    )

    assert premise.mask.nan_to_num(torch.inf).equal(
        expected_mask.nan_to_num(torch.inf)
    )

    assert torch.equal(
        premise.mask.nan_to_num(torch.inf),
        premise.heatmap.nan_to_num(torch.inf),
    )
