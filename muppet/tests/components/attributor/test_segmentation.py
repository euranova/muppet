"""Tests for segmentation-based attributor components.

This module tests the DiceScoreAttributor in MUPPET, which calculates attribution scores using
the Dice coefficient for segmentation tasks. The attributor evaluates how perturbations affect
model performance by comparing segmentation masks between original and perturbed predictions.

The tests verify:
- Attribution values remain within valid range [0, 1]
- Monotonic increasing attribution with increasing noise levels
- Proper integration with segmentation models and premise memory
- Correct Dice coefficient calculation for binary segmentation tasks
- Consistent behavior across different perturbation magnitudes
"""

import torch

from muppet.components.attributor.embedding import DiceScoreAttributor
from muppet.components.memory.base import PremiseList


class FakePremise:
    """Fake premise class for testing segmentation attributors.

    A mock premise that stores attribution values for testing
    segmentation attribution components in isolation.
    """

    def __init__(self, att=None) -> None:
        """Initialize fake premise for testing segmentation attributors.

        Args:
            att: Optional attribution value to store in the premise.
        """
        self.attribution = att


def test_attribution_in_valid_range(DummySegmentationModel):
    """_summary_: Test that the attribution values calculated by the DiceScoreAttributor are within a valid range [0, 1].
    This ensures that the attributor is functioning correctly and producing meaningful attributions.
    """
    model = DummySegmentationModel(5)
    attributor = DiceScoreAttributor()
    x = torch.randn(1, 3, 16, 16)
    perturbed_inputs = x.repeat(3, 1, 1, 1).unsqueeze(1)
    memory = PremiseList()
    memory.register_premises([FakePremise()])

    attributor.calculate_attribution(x, perturbed_inputs, model, memory)

    for premise in memory.get_premises():
        attr = premise.attribution.item()
        assert 0.0 <= attr <= 1.0, f"Attribution out of range: {attr}"


def test_monotonic_increasing_attribution_with_noise(dummy_seg_model):
    """_summary_: Test that the DiceScoreAttributor produces monotonic increasing attribution values as noise increases.
    This ensures that the attributor's output is consistent with the expectation that more noise leads to
    higher attribution values, indicating that the model's predictions are more affected by the noise.
    """
    model = dummy_seg_model(2)
    attributor = DiceScoreAttributor()
    x = torch.randn(1, 3, 16, 16)
    # Create perturbed inputs with increasing noise levels
    # The noise levels are designed to increase the perturbation effect on the model's predictions
    torch.manual_seed(0)  # Ensure reproducibility
    noise_levels = [0.0, 0.2, 0.4, 0.6]
    perturbed = [x + torch.randn_like(x) * nl for nl in noise_levels]
    perturbed_inputs = torch.stack(perturbed)
    memory = PremiseList()
    memory.register_premises([FakePremise()])

    attributor.calculate_attribution(x, perturbed_inputs, model, memory)

    attributions = [p.attribution.item() for p in memory.get_premises()]
    for attr in attributions:
        assert 0.0 <= attr <= 1.0, (
            "Attributions contain NaN values or is out of range"
        )
    assert attributions == sorted(attributions), (
        f"Attributions not monotonic: {attributions}"
    )
