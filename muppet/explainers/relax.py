"""RELAX explainer for embedding-based model explanations.

This module implements the RELAX explainer, a perturbation-based XAI method that
generates explanations by analyzing how perturbations affect the model's
embeddings. It can be seen as a extention of RISE, from classification to
representation learning task.

MUPPET Component Integration:
    - **Explorer**: `RandomMasksExplorer` - generates random binary masks for input perturbation
    - **Perturbator**: `SetToZeroPerturbator` - applies zero-masking to perturb input regions
    - **Attributor**: `EmbeddingDistanceAttributor` - measures changes in model embeddings
    - **Aggregator**: `WeightedSumAggregator` - combines perturbations weighted by embedding changes

Classes:
    RELAXExplainer: Implementation of the RELAX method for embedding-based explanations.

Refenrecs:
    Wickstrøm, Kristoffer K., et al. "RELAX: Representation learning explainability."
    International Journal of Computer Vision 131.6 (2023): 1584-1610.
    https://arxiv.org/pdf/2112.10161

"""
#
# Created on Tue Jul 25 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import torch

from muppet.components.aggregator import WeightedSumAggregator
from muppet.components.attributor.embedding import EmbeddingDistanceAttributor
from muppet.components.explorer.mask import RandomMasksExplorer
from muppet.components.perturbator import SetToZeroPerturbator
from muppet.explainers.base import MuppetExplainer


class RELAXExplainer(MuppetExplainer):
    """RELAX explainer implementation for embedding-based explanations.

    Implements the RELAX method that explains representation learning models
    based on the RISE method principles.

    Key characteristics:
    - Uses random masking perturbations similar to RISE
    - Analyzes embedding output changes

    By measuring how perturbations affect these embeddings, RELAX can identify
    which parts of the input are most critical for the embedding understanding.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        nmasks: int,
        mask_dim: int,
        mask_proba: float,
        seed: int | None = None,
    ):
        """Initialize the RELAX explainer for representation learning explanations.

        Args:
            nmasks (int): Number of random masks to generate.
            mask_dim (int): The size of the squared grade (down-scaled mask).
            mask_proba (float): The probability of setting, independently, each pixel of the mask to 1.
            seed (int, optional): Random seed for reproducibility.
        """
        self.nmasks = nmasks
        self.mask_dim = mask_dim
        self.mask_proba = mask_proba

        # existing components
        explorer = RandomMasksExplorer(
            nmasks=self.nmasks,
            mask_dim=self.mask_dim,
            mask_proba=self.mask_proba,
            seed=seed,
        )
        perturbator = SetToZeroPerturbator()

        attributor = EmbeddingDistanceAttributor()

        aggregator = WeightedSumAggregator()

        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )
