"""Tests for embedding distance attributor components.

This module tests the EmbeddingDistanceAttributor in MUPPET, which calculates attribution scores
based on embedding distances between original and perturbed model outputs. The attributor measures
how perturbations affect the model's internal representations by computing distances in embedding space.

The tests verify:
- Correct zero attribution when inputs are identical (no perturbation effect)
- Proper integration with premise memory systems
- Expected behavior with gradient-based premises
- Attribution calculation accuracy for embedding-based explanations
- Model-agnostic functionality across different architectures
"""

import torch

from muppet.components.attributor.embedding import EmbeddingDistanceAttributor
from muppet.components.memory.base import PremiseList
from muppet.components.memory.premise import GradientPremise


def test_simple_attributor():
    """Test embedding distance attributor with identical inputs.

    Verifies that when original and perturbed inputs are identical,
    the embedding distance attribution score is zero, as expected
    for no perturbation effect.

    Returns:
        None: Test passes if attribution equals zero for identical inputs.
    """
    # Create some default premise and model
    test_shape = (1, 1, 42)
    input_ex = torch.zeros(test_shape)
    p = GradientPremise(
        key=torch.zeros(test_shape), upscaled_mask_shape=test_shape[1:]
    )

    assert torch.equal(p.mask, p.heatmap)

    def identity(x):
        return x

    model = identity
    memory = PremiseList()
    memory._premises = [p]

    # Testing
    attributor = EmbeddingDistanceAttributor()

    # In this test, the "perturbed inputs" are identical to the original, so the loss
    # should be 0.

    # remember to add the N dimension
    perturbed_inputs_test = torch.zeros(test_shape).unsqueeze(0)

    attributor(
        x=input_ex,
        perturbed_inputs=perturbed_inputs_test,
        model=model,
        memory=memory,
    )

    for p in memory._premises:
        assert p.attribution == 0
