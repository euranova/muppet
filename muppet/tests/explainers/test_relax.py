"""Tests for RELAX explainer components.

This module tests the RELAXExplainer in MUPPET, which provides explanations using
the RELAX method for neural network models. It validates proper functioning with
different perturbation strategies and mock image representation models.

The tests verify:
- Correct heatmap generation with expected spatial dimensions
- Integration with mock image representation models
- Different perturbation method behavior (SetToZero vs Blur)
- Deterministic output consistency across multiple runs
- Proper seed-based reproducibility for explanation generation
"""

import torch
import torch.nn.functional as F

# Muppet imports
from muppet import DEVICE
from muppet.components.perturbator.simple import BlurPerturbator
from muppet.explainers import RELAXExplainer


class MockImageRepresentation(torch.nn.Module):
    """Mock image representation model for RELAX explainer testing.

    A simplified neural network that simulates image classification
    behavior using random projections for testing RELAX explanations.
    """

    def __init__(self):
        """Initialize mock image representation model for RELAX testing.

        Creates a random projection matrix for testing purposes.
        """
        super().__init__()
        self.projection = torch.rand(size=(500, 224 * 224 * 3)).to(DEVICE)

    def forward(self, x):
        """Test forward pass for mock image representation model."""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1).to(DEVICE)
        return F.linear(x_flat, self.projection)


def test_RELAX_explainer(cat_image_for_vgg):
    """Test MUPPET's heatmap (with RELAX explainer).

    Args:
        model (torch.Module): model to explain

        image (torch.Tensor): image on which model is used
    """
    representation_model = MockImageRepresentation()

    explainer = RELAXExplainer(
        model=representation_model,
        nmasks=10,
        mask_dim=10,
        mask_proba=0.3,
        seed=42,
    )
    heatmap_settozero = explainer(example=cat_image_for_vgg)
    heatmap_settozero2 = explainer(example=cat_image_for_vgg)
    explainer.perturbator = BlurPerturbator()
    heatmap_blur = explainer(example=cat_image_for_vgg)

    assert heatmap_settozero.shape[-1] == cat_image_for_vgg.shape[-1]
    assert heatmap_settozero.shape[-2] == cat_image_for_vgg.shape[-2]
    assert heatmap_settozero.shape[0] == 1
    assert heatmap_blur.shape[-1] == cat_image_for_vgg.shape[-1]
    assert heatmap_blur.shape[-2] == cat_image_for_vgg.shape[-2]
    assert heatmap_blur.shape[0] == 1

    assert heatmap_settozero.equal(heatmap_settozero2)
    assert not heatmap_blur.equal(heatmap_settozero)
