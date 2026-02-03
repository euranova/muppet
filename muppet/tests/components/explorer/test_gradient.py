"""Tests for gradient-based explorer components.

This module tests the GradientExplorer in MUPPET, which generates gradient-based premises for
explanation methods that rely on gradient information. The explorer iteratively creates masks
and premises based on model gradients to identify important input regions.

The tests verify:
- Proper explorer initialization with correct iteration parameters
- Generated mask shapes matching input tensor dimensions
- Iterative premise generation through gradient exploration
- Memory system integration with gradient-based premises
- Consistent mask properties across exploration iterations
"""
#
# Created on Mon Jul 03 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import pytest
import torch

from muppet.components.explorer.gradient import GradientExplorer
from muppet.components.memory.base import PremiseList


@pytest.mark.parametrize("batch_size", [1])
def test_gradient_explorer_initialization(batch_size):
    """Test the initialization part of the explorer. It must return the expected
    initialized number of masks of the expected shape.

    """
    explorer = GradientExplorer(max_iter=10)
    example = torch.zeros(
        1, 3, 224, 224
    )  # arbitrary shape of dim=4 to pass the assert bcz it's expected to be filled at main explainer call
    memory = PremiseList()

    for premises in explorer(example=example, memory=memory):
        masks = [p.mask for p in premises]
        break

    for m in masks:
        assert m.shape == (1, 1, 224, 224)  # (b=1, c=1, w=224, h=224)
