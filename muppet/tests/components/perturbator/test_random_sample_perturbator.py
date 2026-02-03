"""Tests for random sample perturbator components.

This module tests the RandomSamplePerturbator in MUPPET, which applies perturbations to tabular
data using random sampling techniques. The perturbator selectively replaces feature values
based on binary masks while preserving original values where masks are active.

The tests verify:
- Correct mask-based perturbation logic preserving original values
- Proper integration with random sample generators
- Generated tensor shape consistency after perturbation
- Value substitution accuracy based on mask activation patterns
- Perturbation behavior with different instance and mask configurations
"""

import torch

from muppet.components.perturbator.generator.tabular_generator import (
    RandomSampleTabularGenerator,
)
from muppet.components.perturbator.scale_feature_generator import (
    RandomSamplePerturbator,
)


def test_random_sample_perturbator():
    """Test random sample perturbator functionality.

    Validates that the perturbator correctly applies random sampling
    perturbations based on binary masks, preserving original values
    where masks are active and substituting generated values otherwise.

    Returns:
        None: Test passes if perturbations follow mask-based logic correctly.
    """
    # Example training data
    train_data = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32
    )

    # Instantiate the generator with the training data
    generator = RandomSampleTabularGenerator(train_data)

    # Number of samples to generate
    n_samples = 10

    # Create RandomSamplePerturbator with the generator
    perturbator = RandomSamplePerturbator(generator)

    # Define a mask tensor and instance
    masks = torch.randint(
        0, 2, (n_samples, train_data.shape[1])
    ).float()  # Random binary masks
    instance = torch.tensor(
        [[4, 5, 6]], dtype=torch.float32
    )  # Example instance

    # Apply perturbations
    perturbed_tensor = perturbator.perturbate(instance, masks)

    generated_tensor = perturbator.generated_tensor

    # Check the shape of tensors
    assert perturbed_tensor.shape == (
        n_samples,
        1,
        train_data.shape[1],
    ), (
        f"Expected shape {(n_samples, 1, train_data.shape[1])}, but got {perturbed_tensor.shape}"
    )

    # Check the values in perturbed_tensor
    for i in range(n_samples):
        for j in range(train_data.shape[1]):
            if masks[i, j] == 0:
                assert perturbed_tensor[i, 0, j] == generated_tensor[i, 0, j], (
                    f"Sample {i}, feature {j}: Expected value from generated tensor but got {perturbed_tensor[i, 0, j]}"
                )
            else:
                assert perturbed_tensor[i, 0, j] == instance[0, j], (
                    f"Sample {i}, feature {j}: Expected original instance value but got {perturbed_tensor[i, 0, j]}"
                )
