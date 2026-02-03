"""Tests for Gaussian-based generator components.

This module tests Gaussian generators in MUPPET, including conditional Gaussian feature generators
for time series data and standard Gaussian tabular generators. These components create synthetic
samples using Gaussian distributions for perturbation-based explanation methods.

The tests verify:
- Conditional Gaussian feature generation with different feature subsets
- Proper handling of multivariate time series data generation
- Standard Gaussian tabular data generation around instances or dataset statistics
- Generated sample shape consistency and tensor properties
- Error handling for invalid feature perturbation configurations
"""
#
# Created on Thu May 25 2023
#
# Copyright (c) 2023 Ismail Bachchar @EuraNova
#

import pytest
import torch

from muppet import DEVICE
from muppet.components.perturbator.generator.conditional_timestep_generator import (
    ConditionalGaussianFeatureGenerator,
)
from muppet.components.perturbator.generator.tabular_generator import (
    StandardGaussianTabularGenerator,
)


@pytest.mark.parametrize(
    "num_features, features_to_sample",
    [
        (2, {0}),
        (2, {1}),
        (2, {}),
        (2, {0, 1}),
        (3, {0, 1}),
        (1, {0}),
        (1, {}),
    ],
)
def test_gaussian_feature_generator_inference(num_features, features_to_sample):
    """Test the multivariate case of GaussianFeatureGenerator model's inference method with different features to explain (S).

    Args:
        features_to_sample (set): Set of features to explain (preserve).

    """
    x = torch.ones((2, num_features, 5))  # (b, f, t)
    x_4 = x[:, :, 4]

    generator = ConditionalGaussianFeatureGenerator(
        feature_size=num_features,
        hidden_size=100,
        latent_size=50,
        mid_layer_size=10,
        prediction_size=1,
        num_samples=1,
        cov_noise_level=0.01,
        max_noise_correction=5,
        lr=0.0001,
        num_epochs=10,
        timesteps_divide_num=1,
        seed=None,
    )
    if num_features - len(features_to_sample) <= 0:
        with pytest.raises(AssertionError):
            sample = generator.generate(
                past=x,
                current=x_4,
                features_to_perturb=features_to_sample,
            )
    else:
        sample = generator.generate(
            past=x,
            current=x_4,
            features_to_perturb=features_to_sample,
        )

        assert sample.shape == (2, num_features)


# Test StandardGaussianTabularGenerator

train_data = torch.tensor(
    [[1.0, 2.0, 1, 2], [2.0, 3.0, 0, 3], [3.0, 4.0, 1, 3], [4.0, 5.0, 0, 1]]
)  # Including both numerical and categorical data
# categorical_features_indices = [2, 3]  # Indices indicating which features are categorical


@pytest.mark.parametrize(
    "x_t, data_scaled , sample_around_instance",
    [
        (
            torch.tensor([[1.0, 2.0, 1, 2]]),
            torch.randn((3, 1, 4)),
            True,
        ),  # Testing around instance with categorical feature
        (
            torch.tensor([[2.0, 3.0, 0, 3]]),
            torch.randn((3, 1, 4)),
            False,
        ),  # Testing with learned statistics
    ],
)
def test_StandardGaussianTabularGenerator(
    x_t, sample_around_instance, data_scaled
):
    """Test standard Gaussian tabular generator with different configurations.

    Validates the generator's ability to produce tabular samples either
    around specific instances or using learned dataset statistics,
    ensuring proper tensor shapes and types.

    Args:
        x_t: Target instance tensor for conditional generation.
        sample_around_instance: Whether to sample around the given instance.
        data_scaled: Scaled data tensor for generation context.

    Returns:
        None: Test passes if generated samples have correct properties.
    """
    generator = StandardGaussianTabularGenerator(
        train_data=train_data, sample_around_instance=sample_around_instance
    )
    generator.train_generator()
    generated_sample = generator.generate(
        x_t, data_scaled=data_scaled.to(DEVICE)
    )

    assert generated_sample.shape == data_scaled.shape
    assert isinstance(generated_sample, torch.Tensor)
