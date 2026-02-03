"""Tests for similarity-based attributor components.

This module tests the SimilarityAttributor in MUPPET, which calculates attribution scores based
on similarity functions like LIME and Kernel SHAP. The attributor measures feature importance by
evaluating how perturbations affect model predictions through different similarity metrics.

The tests verify:
- Correct attribution calculation for LIME similarity with image data
- Proper attribution calculation for LIME similarity with tabular data
- Expected behavior with Kernel SHAP similarity function
- Integration with different premise types (segmented and key-based)
- Accurate similarity score computation and storage in memory
"""

import torch

from muppet.components.attributor import (
    SimilarityAttributor,
    kernel_shap_similarity,
    lime_similarity,
)
from muppet.components.memory.base import PremiseList
from muppet.components.memory.premise import (
    KeyBasedMaskPremise,
    SegmentedBinaryImagePremise,
)

image = torch.ones(1, 3, 2, 2).float()
tab = torch.ones(1, 3).float()


def _model(x):
    return torch.tensor([[0.5, 0.5]])


def test_proba_attributor_expected_attributions_lime_image():
    """Test that ProbaAttributor calculates and
    save into memory's premises the expected attributions.
    """
    segmented_example = torch.tensor(
        [[[1, 1], [0, 0]], [[0, 0], [1, 1]]]
    ).float()
    memory = PremiseList()
    premise = SegmentedBinaryImagePremise(
        torch.tensor([0, 0]), segmented_example
    )
    memory.register_premises([premise])

    attributor = SimilarityAttributor(similarity_fun=lime_similarity)
    attributor(
        x=image,
        perturbed_inputs=image.unsqueeze(dim=0),
        model=_model,
        memory=memory,
    )
    assert (
        torch.equal(premise.attribution["attribution"], torch.tensor([0.5]))
        and premise.attribution["similarity"] == 1.0
    )


def test_proba_attributor_expected_attributions_lime_tab():
    """Test that ProbaAttributor calculates and
    save into memory's premises the expected attributions.
    """
    memory = PremiseList()
    premise = KeyBasedMaskPremise((1, 1, 1), seed=None)
    memory.register_premises([premise])

    attributor = SimilarityAttributor(similarity_fun=lime_similarity)
    attributor(
        x=tab,
        perturbed_inputs=tab.unsqueeze(dim=0),
        model=_model,
        memory=memory,
    )
    assert (
        torch.equal(premise.attribution["attribution"], torch.tensor([0.5]))
        and premise.attribution["similarity"] == 1.0
    )


def test_proba_attributor_expected_attributions_kernel_shap():
    """Test that ProbaAttributor calculates and
    saves into memory's premises the expected attributions for kernel SHAP.
    """
    memory = PremiseList()
    premise = KeyBasedMaskPremise(
        torch.tensor([[1, 1, 1]], dtype=torch.float32), seed=None
    )
    memory.register_premises([premise])

    # Define or load your test input tensor and model
    tab = torch.tensor([1.0, 0.0, 1.0])  # Example tensor, adjust as needed

    attributor = SimilarityAttributor(similarity_fun=kernel_shap_similarity)
    attributor(
        x=tab,
        perturbed_inputs=tab.unsqueeze(dim=0),
        model=_model,
        memory=memory,
    )

    # Define expected attribution and similarity values
    expected_attribution = torch.tensor(
        [0.5]
    )  # Adjust according to your model and test setup
    expected_similarity = (
        1.0  # Adjust according to your kernel SHAP implementation
    )

    # Verify the results
    assert torch.equal(premise.attribution["attribution"], expected_attribution)
    assert premise.attribution["similarity"] == expected_similarity
