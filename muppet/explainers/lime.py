"""LIME (Local Interpretable Model-Agnostic Explanations) explainer implementations.

This module implements Local Interpretable Model-Agnostic Explanations (LIME) for both
image and tabular data. LIME explains predictions by learning locally faithful surrogate
models around individual instances. The method generates random perturbations of the
input, evaluates the model on these perturbations, and fits an interpretable model
(typically linear regression) to approximate the black-box model's behavior locally.

MUPPET Component Integration:
    **For Images (LIMEImageExplainer)**:
    - **Explorer**: `SegmentedBinaryRandomMasksExplorer` - generates random masks over image segments/superpixels
    - **Perturbator**: `SetToZeroPerturbator` - masks out image regions by setting them to zero
    - **Attributor**: `SimilarityAttributor` - calculates similarity-weighted attributions based on distance from original
    - **Aggregator**: `SegmentedImageModelAggregator` - fits surrogate model and maps segment importance back to pixels

    **For Tabular Data (LIMETabularExplainer)**:
    - **Explorer**: `RandomNormalExplorer` - generates random binary feature masks
    - **Perturbator**: `ScaleFeaturePerturbator` - scales features using generator-based perturbations
    - **Attributor**: `SimilarityAttributor` - weights perturbations by similarity to original instance
    - **Aggregator**: `ModelAggregator` - fits Ridge regression to learn feature importance

Classes:
    LIMEImageExplainer: LIME implementation for image classification models.
    LIMETabularExplainer: LIME implementation for tabular data.

References:
    Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. "Why should I trust you?
    Explaining the predictions of any classifier." Proceedings of the 22nd ACM SIGKDD
    international conference on knowledge discovery and data mining. 2016.
    https://arxiv.org/pdf/1602.04938v3.pdf
"""
#
# Created on Wed Dec 13 2023
#
# Copyright (c) 2023 Jérémy Rozier @Euranova
#

import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from muppet import DEVICE
from muppet.components.aggregator.model import (
    ModelAggregator,
    SegmentedImageModelAggregator,
)
from muppet.components.attributor.similarity import (
    SimilarityAttributor,
    lime_similarity,
)
from muppet.components.explorer.mask import (
    RandomNormalExplorer,
    SegmentedBinaryRandomMasksExplorer,
)
from muppet.components.memory.base import PremiseList
from muppet.components.perturbator.generator.tabular_generator import (
    StandardGaussianTabularGenerator,
)
from muppet.components.perturbator.scale_feature_generator import (
    ScaleFeaturePerturbator,
)
from muppet.components.perturbator.simple import SetToZeroPerturbator
from muppet.explainers.mp import MuppetExplainer


class LIMEImageExplainer(MuppetExplainer):
    """LIME implementation for image classification models.

    Implements the Local Interpretable Model-Agnostic Explanations (LIME) method for image
    classification. LIME explains predictions through superpixel-based perturbations and
    segmented surrogate model fitting.

    LIME's key principle is local fidelity - the explanation should accurately represent
    the model's behavior in the neighborhood of the specific instance being explained.
    This allows LIME to work with any type of model (model-agnostic) while providing
    human-interpretable explanations through simple surrogate models.

    The method generates random masks to perturb the input data, then fits a model
    on the perturbed dataset to create a surrogate model of the explained model locally
    faithful around the input. For images, LIME generates a heatmap to identify the
    areas of pixels which made the model take the decision for the input data.


    """

    def __init__(
        self,
        model: torch.nn.Module,
        surrogate_model=Ridge(alpha=1, fit_intercept=True),
        nmasks: int = 500,
        masked_proba: float = 0.5,
        n_segments: int = 100,
    ) -> None:
        """Initialize the LIME Image explainer.

        Args:
            model (torch.nn.Module): The black-box model to explain its predictions.
            surrogate_model: The regressor model for learning the surrogate model in the aggregator.
            nmasks (int): Number of random masks to generate.
            masked_proba (float): The probability of masking each super-pixel of the image.
            n_segments (int): Number of segments to divide the image into.
        """
        self.nmasks = nmasks
        self.mask_proba = masked_proba

        explorer = SegmentedBinaryRandomMasksExplorer(
            nmasks=self.nmasks,
            masked_proba=self.mask_proba,
            n_segments=n_segments,
        )
        perturbator = SetToZeroPerturbator()
        attributor = SimilarityAttributor(similarity_fun=lime_similarity)
        aggregator = SegmentedImageModelAggregator(surrogate_model)
        memory = PremiseList()

        super().__init__(
            model, explorer, perturbator, attributor, aggregator, memory
        )


class LIMETabularExplainer(MuppetExplainer):
    """LIME implementation for tabular data.

    Implements the Local Interpretable Model-Agnostic Explanations (LIME) method for tabular
    data. LIME explains predictions by perturbing individual features and learning linear
    surrogate models.

    LIME's key principle is local fidelity - the explanation should accurately represent
    the model's behavior in the neighborhood of the specific instance being explained.
    This allows LIME to work with any type of model (model-agnostic) while providing
    human-interpretable explanations through simple surrogate models.

    The method generates random masks to perturb the input data, then fits a model
    on the perturbed dataset to create a surrogate model of the explained model locally
    faithful around the input. The final explanation is a vector indicating the contribution
    of each feature to the model's prediction.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        generator=StandardGaussianTabularGenerator,
        surrogate_model=Ridge(alpha=1, fit_intercept=True),
        nmasks: int = 500,
        sample_around_instance: bool = True,
        seed: int = 1,
        similarity_fun=lime_similarity,
    ) -> None:
        """Initialize the LIME Tabular explainer.

        Args:
            model: The black-box model to explain its predictions.
            train_loader: DataLoader containing training data for fitting the generator.
            generator: Generator for creating perturbed tabular data samples.
            surrogate_model: The regressor model for learning the surrogate model.
            nmasks: Number of random masks to generate.
            sample_around_instance: Whether to sample data around the instance being explained.
            seed: Random seed for reproducibility.
            similarity_fun: Similarity function for LIME kernel weighting.
        """
        self.nmasks = nmasks
        self.train_data = torch.concat([data[0] for data in train_loader]).to(
            DEVICE
        )
        self.sample_around_instance = sample_around_instance
        self.similarity_fun = similarity_fun

        # Instantiate modules
        explorer = RandomNormalExplorer(nmasks=nmasks, seed=seed)

        # If no generator is provided, initialize a standard Gaussian tabular generator
        generator = StandardGaussianTabularGenerator(
            train_data=self.train_data,
            sample_around_instance=sample_around_instance,
        )

        # perturbator initialization & generator training if required
        perturbator = ScaleFeaturePerturbator(generator=generator)
        attributor = SimilarityAttributor(similarity_fun=similarity_fun)
        aggregator = ModelAggregator(surrogate_model=surrogate_model)

        # Initiate the explainer with these modules
        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )
