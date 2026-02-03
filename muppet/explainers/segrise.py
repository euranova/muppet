"""SegRISE explainer for semantic segmentation models.

This module implements SegRISE, an adaptation of the RISE (Randomized Input Sampling
for Explanation) or RELAX method to semantic segmentation tasks. Unlike
the original RISE which was developed for classification and uses the reference
class probability as weight, SegRISE uses the Dice similarity coefficient to measure
the overlap between original and perturbed segmentation outputs.

MUPPET Component Integration:
    - **Explorer**: `RandomMasksExplorer` - generates random binary masks for input perturbation
    - **Perturbator**: `SetToZeroPerturbator` - applies zero-masking to occlude input regions
    - **Attributor**: `DiceScoreAttributor` - calculates Dice similarity between original and perturbed segmentations
    - **Aggregator**: `WeightedSumAggregator` - computes weighted average of masks using Dice scores

Classes:
    SegRISEExplainer: RISE and RELAX adaptation for segmentation models.

Note:
    The model should output segmentation maps (not class probabilities) for the
    Dice score calculation to be meaningful.
"""

#
# Created on Fri Jun 09 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#
import torch

from muppet.components.aggregator.mask import WeightedSumAggregator
from muppet.components.attributor.embedding import DiceScoreAttributor
from muppet.components.explorer.mask import RandomMasksExplorer
from muppet.components.memory.base import PremiseList
from muppet.components.perturbator.simple import SetToZeroPerturbator
from muppet.explainers.base import MuppetExplainer


class SegRISEExplainer(MuppetExplainer):
    """SegRISE explainer specialized for image segmentation tasks.

    Adapts the RISE method for semantic segmentation by generating pixel-level explanations.
    The key innovation of SegRISE is replacing classification-based attribution with
    segmentation-appropriate metrics. The Dice score provides a more meaningful measure
    of how perturbations affect segmentation quality compared to simple class probabilities,
    making it better suited for pixel-level prediction tasks.

    SegRISE methodology:
    1. Generate random masks like standard RISE
    2. Apply masks to input images (zero-masking perturbation)
    3. Compare original and perturbed segmentation maps using Dice score
    4. Weight masks by their Dice similarity scores
    5. Aggregate weighted masks into final importance heatmap

    This approach identifies which regions of the input are most critical for maintaining
    good segmentation performance, providing insights into what the model relies on for
    accurate pixel-level predictions.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        nmasks: int = 800,
        mask_dim: int = 7,
        mask_proba: float = 0.1,
        seed: int = None,
    ) -> None:
        """Initialize the SegRISE explainer for segmentation model explanation.

        Args:
            model (torch.nn.Module): The black-box model to explain its predictions.
            nmasks (int): Number of random masks to generate.
            mask_dim (int): The size of the squared grade (down-scaled mask).
            mask_proba (float): The probability of setting, independently, each value of the (downscaled) mask to 0 meaning there will be no perturbation at this position.
            seed (int, optional): Seed to initialize for reproducible results.
        """
        # Parameters
        self.nmasks = nmasks
        self.mask_dim = mask_dim
        self.mask_proba = mask_proba

        explorer = RandomMasksExplorer(
            nmasks=self.nmasks,
            mask_dim=self.mask_dim,
            mask_proba=self.mask_proba,
            seed=seed,
        )
        perturbator = SetToZeroPerturbator()
        attributor = DiceScoreAttributor()
        aggregator = WeightedSumAggregator()
        memory = PremiseList()

        # Initialize the explainer with these modules
        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
            memory=memory,
        )
