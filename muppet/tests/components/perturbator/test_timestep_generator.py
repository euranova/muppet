"""Tests for timestep generator perturbator components.

This module tests the GeneratorSamplingPertubator in MUPPET, which applies generator-based
perturbations to time series data at specific timesteps. The perturbator uses generators
to create synthetic values while preserving temporal structure and feature relationships.

The tests verify:
- Correct timestep-specific perturbation generation
- Multi-timestep perturbation capability with NaN handling
- Multiple feature perturbation in multivariate time series
- Generator integration with mask-based selective perturbation
- Expected perturbation patterns with complex generator logic
"""
#
# Created on Mon May 15 2023
#
# Copyright (c) 2023 Ismail Bachchar @EuraNova
#

import torch

from muppet.components.perturbator.timestep_generator import (
    GeneratorSamplingPertubator,
)


class Generator:
    """Mock generator class for testing timestep perturbation.

    A test double that simulates generative model behavior for
    timestep perturbation testing with deterministic outputs.
    """

    def __init__(self) -> None:
        """Initialize mock generator for timestep perturbation testing.

        Sets up a trained generator with deterministic behavior for testing.
        """
        self.is_trained = True
        pass

    def generate(self, past, current, features_to_perturb):
        """Test generator method for simple timestep perturbation."""
        return current + 99


custom_mask = torch.tensor(
    [
        [[[0, 0, 0, 0, 1]]],
        [[[0, 0, 0, 0, 1]]],
        [[[0, 0, 0, 1, torch.nan]]],
        [[[0, 0, 0, 1, torch.nan]]],
        [[[0, 0, 1, torch.nan, torch.nan]]],
        [[[0, 0, 1, torch.nan, torch.nan]]],
        [[[0, 1, torch.nan, torch.nan, torch.nan]]],
        [[[0, 1, torch.nan, torch.nan, torch.nan]]],
    ]
)  # (N=9, b=1, f=1, t=5)

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
)  # (N=8, b=1, f=1, t=5)


def test_generator_sampling_pertubator_perturbations():
    """Test that GeneratorSamplingPertubator calculates the expected perturbations."""
    g = Generator()
    x = torch.ones((1, 1, 5))  # (b=1, f=1, t=5)

    perturbator = GeneratorSamplingPertubator(generator=g, train_loader=None)
    perturbations = perturbator(x=x, masks=custom_mask)

    assert perturbations.nan_to_num(nan=0).equal(
        custom_perturbations.nan_to_num(nan=0)
    )


custom_mask_mult = torch.tensor(
    [
        [[[0, 0, 0, 1, 1]]],
        [[[0, 0, 0, 1, torch.nan]]],
    ]
)

custom_perturbations_mult = torch.tensor(
    [
        [[[1, 1, 1, 100, 100]]],
        [[[1, 1, 1, 100, torch.nan]]],
    ]
)


def test_perturbator_multitimestep():
    """Ensure the Generator can perturbate multiple timesteps."""
    g = Generator()
    x = torch.ones((1, 1, 5))  # (b=1, f=1, t=5)

    perturbator = GeneratorSamplingPertubator(generator=g, train_loader=None)
    perturbations = perturbator(x=x, masks=custom_mask_mult)

    assert perturbations.nan_to_num(nan=0).equal(
        custom_perturbations_mult.nan_to_num(nan=0)
    )


class ComplexGenerator:
    """Complex mock generator class for advanced timestep perturbation testing.

    An extended test double that simulates more sophisticated generative
    model behavior with configurable training state and complex generation logic.
    """

    def __init__(self) -> None:
        """Initialize complex mock generator for advanced perturbation testing.

        Sets up a trained generator with complex generation logic for testing
        sophisticated perturbation scenarios.
        """
        self.is_trained = True
        pass

    def generate(self, past, current, features_to_perturb):
        """Test generator method for complex timestep perturbation with selective feature masking."""
        copied_current = current.clone()

        mask = torch.ones_like(copied_current, dtype=torch.bool)
        do_not_perturb_feats = sorted(
            set(range(copied_current.shape[-1])) - set(features_to_perturb)
        )
        mask[:, do_not_perturb_feats] = False
        print(f"mask: {mask}")
        new_values = [99] * (
            copied_current.shape[1] - len(do_not_perturb_feats)
        )  # Example new values

        copied_current[mask] = copied_current[mask] + torch.tensor(new_values)

        return copied_current


custom_mask_mult_features = torch.tensor(
    [
        [
            [
                [0, 0, 0, 1, 1],
                [0, 1, 0, 0, torch.nan],
                [torch.nan, 0, 0, 1, torch.nan],
            ]
        ]
    ]
)

custom_perturbations_mult_features = torch.tensor(
    [
        [
            [
                [1, 1, 1, 100, 100],
                [1, 100, 1, 1, torch.nan],
                [torch.nan, 1, 1, 100, torch.nan],
            ]
        ]
    ]
)


def test_perturbator_multiple_features():
    """Ensure the Generator can perturbate multiple timesteps."""
    g = ComplexGenerator()
    x = torch.ones((1, 3, 5))  # (b=1, f=3, t=5)
    from muppet.components.perturbator.timestep_generator import (
        GeneratorSamplingPertubator,
    )

    perturbator = GeneratorSamplingPertubator(generator=g, train_loader=None)
    perturbations = perturbator(x=x, masks=custom_mask_mult_features)

    assert perturbations.nan_to_num(nan=0).equal(
        custom_perturbations_mult_features.nan_to_num(nan=0)
    )
