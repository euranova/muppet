"""Tests for image classification explainer components.

This module tests various image classification explainers in MUPPET, including OptiCAM,
MP (Mask Perturbation), RISE, and ScoreCAM. It validates that these explainers produce
consistent outputs and handle different convention modes properly.

The tests verify:
- Correct output shape generation matching input image dimensions
- Deterministic behavior with consistent results across multiple runs
- Proper integration with VGG models and image preprocessing
- Convention-specific behavior (destructive vs constructive modes)
- Mean squared error consistency between repeated explanations
"""

import pytest
import torch
import torch.nn.functional as F

# Muppet imports
from muppet.explainers import (
    MPExplainer,
    OptiCAMExplainer,
    RISEExplainer,
    ScoreCAMExplainer,
)


@pytest.mark.parametrize(
    "explainer_constructor,explainer_kwargs",
    [
        (OptiCAMExplainer, {"max_iter": 5, "convention": "destructive"}),
        (OptiCAMExplainer, {"max_iter": 5, "convention": "constructive"}),
        (MPExplainer, {"max_iter": 5, "convention": "destructive"}),
        (MPExplainer, {"max_iter": 5, "convention": "constructive"}),
        (
            RISEExplainer,
            {
                "nmasks": 10,
                "mask_dim": 5,
                "seed": 42,
                "convention": "destructive",
            },
        ),
        (
            RISEExplainer,
            {
                "nmasks": 10,
                "mask_dim": 5,
                "seed": 42,
                "convention": "constructive",
            },
        ),
        (ScoreCAMExplainer, {"convention": "destructive"}),
        (ScoreCAMExplainer, {"convention": "constructive"}),
    ],
)
def test_image_classif_explainer(
    explainer_constructor: type,
    explainer_kwargs: dict,
    model_vgg: torch.nn.Module,
    cat_image_for_vgg: torch.Tensor,
) -> None:
    """Test image classification explainer for correct output shape and determinism."""
    explainer = explainer_constructor(model_vgg, **explainer_kwargs)
    heatmap_muppet = explainer(example=cat_image_for_vgg)
    heatmap_muppet2 = explainer(example=cat_image_for_vgg)
    assert heatmap_muppet.shape[-1] == cat_image_for_vgg.shape[-1]
    assert heatmap_muppet.shape[-2] == cat_image_for_vgg.shape[-2]
    assert heatmap_muppet.shape[0] == 1
    assert F.mse_loss(heatmap_muppet, heatmap_muppet2) < 1e-4
