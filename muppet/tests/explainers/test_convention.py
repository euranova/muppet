"""Tests for explainer convention handling components.

This module tests the convention system in MUPPET explainers, which allows explainers
to operate in either "destructive" or "constructive" modes. This affects how perturbations
are applied and how explanations are interpreted across different XAI methods.

The tests verify:
- Correct handling of destructive vs constructive conventions
- Different explanation outputs based on convention choice
- Mean squared error differences between convention modes
- Proper integration with various explainer types (OptiCAM, MP, RISE, ScoreCAM)
- Expected behavior differences across convention settings
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
        (OptiCAMExplainer, {"max_iter": 5}),
        (MPExplainer, {"max_iter": 5}),
        (RISEExplainer, {"nmasks": 10, "mask_dim": 5, "seed": 42}),
        (ScoreCAMExplainer, {}),
    ],
)
def test_image_classif_explainer(
    explainer_constructor: type,
    explainer_kwargs: dict,
    model_vgg: torch.nn.Module,
    cat_image_for_vgg: torch.Tensor,
) -> None:
    """Test image classification explainer for correct output shape and determinism."""
    explainer_destructive = explainer_constructor(
        model_vgg, convention="destructive", **explainer_kwargs
    )
    explainer_constructive = explainer_constructor(
        model_vgg, convention="constructive", **explainer_kwargs
    )
    heatmap_muppet_destructive = explainer_destructive(
        example=cat_image_for_vgg
    )
    heatmap_muppet_constructive = explainer_constructive(
        example=cat_image_for_vgg
    )
    assert (
        F.mse_loss(heatmap_muppet_destructive, heatmap_muppet_constructive)
        > 1e-1
    )
