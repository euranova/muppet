"""Tests for model-based aggregation components.

This module tests the model aggregation functionality in MUPPET, specifically
focusing on surrogate model-based aggregators like Ridge regression and
segmented image model aggregators. It validates the proper fitting of
surrogate models to premise data and attribution calculation.

The tests verify:
- Surrogate model fitting and prediction accuracy
- Segmented image aggregation with spatial segments
- Tabular data aggregation using linear models
- Model aggregator device handling and tensor operations
- Expected attribution values and shapes from trained models
"""

import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from muppet import DEVICE
from muppet.components.aggregator import (
    ModelAggregator,
    SegmentedImageModelAggregator,
)
from muppet.components.explorer import SegmentedBinaryRandomMasksExplorer
from muppet.components.memory import (
    KeyBasedMaskPremise,
    SegmentedBinaryImagePremise,
)
from muppet.components.memory.base import PremiseList


def test_segmented_image_model(model_vgg, cat_image_for_vgg):
    """Test the ModelAggregator when not perturbing at all, it should
    returns a heatmap full of attributions values.
    """
    example = cat_image_for_vgg
    _, _, h, w = example.shape

    with torch.no_grad():
        true_output = F.softmax(
            model_vgg(example).detach(), dim=1
        )  # (b, nclasses)

    predicted_class = torch.argmax(true_output, dim=1)  # (b=1)

    expected_heatmap = torch.zeros(1, 1, h, w).to(DEVICE)
    explorer = SegmentedBinaryRandomMasksExplorer(n_segments=3)
    explorer.example = example
    segmented_example = explorer.get_segmented_tensor_from_example()
    memory = PremiseList()
    premise1 = SegmentedBinaryImagePremise(
        key=torch.tensor([0, 0, 0]), segmented_example=segmented_example
    )
    premise2 = SegmentedBinaryImagePremise(
        key=torch.tensor([0, 0, 0]), segmented_example=segmented_example
    )

    premise1.attribution = {"attribution": predicted_class, "similarity": 1}

    premise2.attribution = {"attribution": predicted_class, "similarity": 1}

    # add premises to the memory so aggregator can access them
    memory.register_premises([premise1, premise2])
    memory.segmented_example = segmented_example
    surrogate_model = Ridge(alpha=1)
    aggregator = SegmentedImageModelAggregator(surrogate_model)
    aggregator.device = DEVICE
    heatmap = aggregator.get_explanation(
        memory=memory,
    )

    assert heatmap.equal(expected_heatmap)


tab = torch.ones(1, 3).float()


def test_tab_aggreg_model():
    """Test the ModelAggregator when not perturbing at all, it should
    returns a heatmap full of attributions values.
    """
    expected_heatmap = torch.tensor([[0.0132, 0.0132, 0.0132]]).to(
        DEVICE
    )  # Update this with actual values from Ridge

    memory = PremiseList()
    premise1 = KeyBasedMaskPremise(
        key=torch.tensor([0.1, 0.2, 0.3]).float(), seed=1
    )
    premise2 = KeyBasedMaskPremise(
        key=torch.tensor([0.4, 0.5, 0.6]).float(), seed=2
    )

    # # Use non-zero masks for premises
    # premise1._mask = torch.tensor([0.1, 0.2, 0.3]).float()
    # premise2._mask = torch.tensor([0.4, 0.5, 0.6]).float()

    premise1.attribution = {
        "attribution": torch.tensor([0.7]),
        "similarity": 1,
    }
    premise2.attribution = {
        "attribution": torch.tensor([0.8]),
        "similarity": 1,
    }

    # add premises to the memory so aggregator can access them
    memory.register_premises([premise1, premise2])
    surrogate_model = Ridge(alpha=1)
    aggregator = ModelAggregator(surrogate_model)
    aggregator.device = torch.get_default_device()
    heatmap = (
        aggregator.get_explanation(
            memory=memory,
        )
        .float()
        .to(DEVICE)
    )

    assert torch.allclose(heatmap, expected_heatmap, atol=1e-2)
