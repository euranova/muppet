"""Tests for mask-based explorer components.

This module tests various mask generation explorers in MUPPET, including random masks, segmented
binary masks, normal distribution masks, and feature permutation explorers. These components
generate different types of masks for perturbation-based explanation methods.

The tests verify:
- Random mask generation with specified probabilities and patterns
- Segmented binary mask creation for image regions
- Normal distribution mask generation for tabular data
- Feature permutation mask creation with repetition handling
- Mask shape consistency and tensor property validation
"""
#
# Created on Tue Jun 27 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import numpy as np
import pytest
import torch

from muppet.components.explorer.mask import (
    BinaryFeaturePermutationsExplorer,
    RandomMasksExplorer,
    RandomNormalExplorer,
    SegmentedBinaryRandomMasksExplorer,
)

zeros_mask = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])  # (b=1, c=1, w=2, h=2)
ones_mask = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])  # (b=1, c=1, w=2, h=2)


@pytest.mark.parametrize(
    "proba, expected_mask",
    [
        (1, zeros_mask),
        (0, ones_mask),
    ],
)
def test_random_masks_explorer(proba, expected_mask):
    """Test generated masks with corresponding probability"""
    explorer = RandomMasksExplorer(nmasks=2, mask_proba=proba, seed=1)

    # Set the example based on the example_shape
    explorer.example = torch.zeros(1, 1, 2, 2)  # (b, c, w, h)

    premises = explorer.get_premises_to_explore()

    for p in premises:
        assert p.mask.equal(expected_mask)


@pytest.mark.parametrize(
    "proba,expected_mask",
    [
        (0, zeros_mask),
        (1, ones_mask),
    ],
)
def test_segmented_binary_random_masks_explorer(proba, expected_mask):
    """Test segmented binary random masks explorer with different probabilities.

    Verifies that the explorer generates masks with expected patterns
    based on the masked probability parameter for segmented regions.

    Args:
        proba: Probability of masking segments (0 or 1 for deterministic tests).
        expected_mask: Expected mask tensor pattern for the given probability.

    Returns:
        None: Test passes if generated masks match expectations.
    """
    explorer = SegmentedBinaryRandomMasksExplorer(
        nmasks=2, masked_proba=proba, n_segments=10
    )

    example = torch.rand((1, 3, 2, 2))
    explorer.example = example
    premises = explorer.get_premises_to_explore()
    for p in premises:
        assert p.mask.equal(expected_mask)


# Test RandomNormalExplorer for tab data


def test_random_normal_explorer():
    """Test random normal explorer for tabular data.

    Verifies that the explorer generates random normal masks for tabular
    data with correct tensor properties and consistent key-mask relationships.

    Returns:
        None: Test passes if masks have proper shape and key consistency.
    """
    explorer = RandomNormalExplorer(nmasks=2, seed=1)

    example = torch.rand((1, 5))
    explorer.example = example
    premises = explorer.get_premises_to_explore()
    for p in premises:
        assert isinstance(p.mask, torch.Tensor)
        assert p.mask.shape == (5,)
        assert torch.equal(p.key, p.mask), (
            "The mask is not identical to the key."
        )


@pytest.mark.parametrize(
    "num_features, n_repeats, max_permutations, expected_unique_masks",
    [
        (3, 3, 20, 8),  # Case 1: Small number of features, requires repeats
        (
            10,
            4,
            1024,
            1024,
        ),  # Case 2: Large number of features, no repeats needed
    ],
)
def test_binary_feature_permutations_explorer(
    num_features, n_repeats, max_permutations, expected_unique_masks
):
    """Test binary feature permutations explorer with different configurations.

    Validates the explorer's ability to generate binary feature permutation
    masks with proper repetition handling and uniqueness constraints for
    both small and large feature spaces.

    Args:
        num_features: Number of features in the input tensor.
        n_repeats: Number of repetitions for each unique permutation.
        max_permutations: Maximum number of permutations to generate.
        expected_unique_masks: Expected number of unique mask patterns.

    Returns:
        None: Test passes if permutations meet uniqueness and repeat requirements.
    """
    # Set a seed for reproducibility
    seed = 42

    # Create an instance of the BinaryFeaturePermutationsExplorer
    explorer = BinaryFeaturePermutationsExplorer(
        n_repeats=n_repeats, seed=seed, max_permutations=max_permutations
    )

    # Create a random example tensor with the specified number of features
    explorer.example = torch.rand((1, num_features)).to(
        explorer.device
    )  # Shape: (b=1, f=num_features)

    # Get the premises (binary feature permutations)
    premises = explorer.get_premises_to_explore()

    # Assertions to check the number of generated premises
    assert len(premises) == max_permutations, (
        f"Expected {max_permutations} premises, but got {len(premises)}"
    )

    # Convert masks to a numpy array for easier uniqueness check
    masks = np.array(
        [
            premise.mask.cpu().numpy().astype(int).flatten()
            for premise in premises
        ]
    )

    # Check uniqueness using numpy
    unique_masks = np.unique(masks, axis=0)

    # Ensure that the number of unique masks matches the expected value
    assert unique_masks.shape[0] == expected_unique_masks, (
        f"Expected {expected_unique_masks} unique masks, but got {unique_masks.shape[0]}"
    )

    # For Case 1, check if the repeats are correctly placed
    if num_features == 3:
        mask_counts = {tuple(mask): 0 for mask in unique_masks}
        for mask in masks:
            mask_counts[tuple(mask)] += 1

        for count in mask_counts.values():
            assert count == n_repeats or count == n_repeats - 1, (
                "Repeats not correctly applied"
            )

    print(
        f"Test passed for num_features={num_features}, n_repeats={n_repeats}, max_permutations={max_permutations}."
    )
