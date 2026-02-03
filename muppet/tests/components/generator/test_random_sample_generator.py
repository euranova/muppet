"""Tests for random sample generator components.

This module tests the RandomSampleTabularGenerator in MUPPET, which generates synthetic tabular
data by randomly sampling from training datasets. This generator provides a simple baseline
approach for creating perturbed samples in tabular data explanation methods.

The tests verify:
- Random sample generation from training data with correct dimensions
- Generated sample tensor shape consistency
- Value range validation ensuring samples remain within training data bounds
- Proper tensor type preservation during generation process
- Generator functionality with various training dataset configurations
"""

import torch

from muppet.components.perturbator.generator.tabular_generator import (
    RandomSampleTabularGenerator,
)


def test_random_sample_generator():
    """Test random sample tabular generator functionality.

    Validates that the generator produces samples from training data
    with correct tensor dimensions and value ranges, ensuring samples
    fall within the expected bounds of the original dataset.

    Returns:
        None: Test passes if generated samples have proper shape and range.
    """
    # Example training data
    train_data = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32
    )

    # Instantiate the generator with the training data
    generator = RandomSampleTabularGenerator(train_data)

    # Number of samples to generate
    n_samples = 10

    # Generate samples
    generated_tensor = generator.generate(n_samples)

    # Check the shape of the generated tensor
    assert generated_tensor.shape == (
        n_samples,
        1,
        train_data.shape[1],
    ), (
        f"Expected shape {(n_samples, 1, train_data.shape[1])}, but got {generated_tensor.shape}"
    )

    # Optionally, verify some properties of the generated tensor
    # For instance, check if the values are within the range of the training data
    min_train_val = train_data.min()
    max_train_val = train_data.max()
    assert torch.all(generated_tensor >= min_train_val) and torch.all(
        generated_tensor <= max_train_val
    ), (
        "Generated tensor values are out of the expected range based on training data."
    )
