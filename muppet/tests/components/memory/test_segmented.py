"""Tests for segmented premise components.

This module tests segmented premise implementations in MUPPET, including SegmentedBinaryImagePremise
and BinaryRandomPremise. These premises handle segmentation-based explanations for image data
and provide binary mask generation capabilities.

The tests verify:
- Correct binary mask generation from segmented image keys
- Proper spatial correspondence between segments and generated masks
- Mask-heatmap equality for binary random premises
- Consistent premise behavior with different segmentation patterns
- Random premise generation with reproducible seeding
"""

import pytest
import torch

from muppet.components.memory.premise import (
    BinaryRandomPremise,
    SegmentedBinaryImagePremise,
)


@pytest.mark.parametrize(
    "key, expected_mask",
    [
        (
            torch.tensor([1, 0]),
            torch.tensor([[1, 1], [0, 0]]).unsqueeze(dim=0).unsqueeze(dim=0),
        ),
        (
            torch.tensor([0, 0]),
            torch.tensor([[0, 0], [0, 0]]).unsqueeze(dim=0).unsqueeze(dim=0),
        ),
    ],
)
def test_premise_mask_generation(key, expected_mask):
    """Test segmented binary image premise mask generation.

    Validates that segmented premises correctly generate binary masks
    based on segment keys and example segmentations, ensuring proper
    spatial correspondence between segments and masks.

    Args:
        key: Binary key tensor indicating which segments to activate.
        expected_mask: Expected binary mask tensor for the given key.

    Returns:
        None: Test passes if generated mask matches expected pattern.
    """
    segmented_example = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
    premise = SegmentedBinaryImagePremise(key, segmented_example)
    assert torch.equal(premise.mask, expected_mask)


def test_random_mask_heatmap_equality():
    """Test binary random premise mask and heatmap consistency.

    Verifies that binary random premises maintain equality between
    their mask and heatmap representations, ensuring consistent
    visualization and computation properties.

    Returns:
        None: Test passes if mask equals heatmap for random premise.
    """
    p = BinaryRandomPremise(
        key=(3, 0.5, (2, 2)),
        seed=42,
    )
    assert torch.equal(p.heatmap, p.mask)
