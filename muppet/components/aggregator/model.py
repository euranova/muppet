"""Model-based aggregators using surrogate models for local explanations.

This module provides aggregators that use surrogate models (like Ridge regression) to
fit local linear approximations of the complex model's behavior. This approach is
fundamental to LIME-style explanations, where interpretable models explain individual
predictions by learning from perturbations in the local neighborhood.

The aggregators support both tabular data (returning coefficients directly) and
segmented image data (mapping coefficients back to pixel space using superpixels
as in LIME-image). The surrogate models are fitted using weighted samples based
on similarity to the original input.

Classes:
    ModelAggregator: Base aggregator using surrogate models for local explanations.
    SegmentedImageModelAggregator: Specialized aggregator for image data using superpixels.
"""
#
# Created on Wed Nov 29 2023
#
# Copyright (c) 2023 Jérémy Rozier @Euranova
#

from typing import List

import torch
from sklearn.linear_model import Ridge

from muppet.components.aggregator.base import Aggregator
from muppet.components.memory.base import PremiseList
from muppet.components.memory.premise import (
    KeyBasedMaskPremise,
    Premise,
    SegmentedBinaryImagePremise,
)


class ModelAggregator(Aggregator):
    """Base aggregator using surrogate models for local explanations.

    This aggregator fits a surrogate model to provide local linear approximations
    of the complex model's behavior. It is fundamental to LIME-style explanations,
    where interpretable models explain individual predictions by learning from
    perturbations in the local neighborhood.

    """

    def __init__(
        self, surrogate_model=Ridge(alpha=1, fit_intercept=True)
    ) -> None:
        """Initialize the model aggregator.

        Args:
            surrogate_model: The model to use in explanation. Defaults to Ridge regression.
                Must have model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit(). It must be an inherently interpretable model,
                specifically a ((regularized)(Linear|Logistic)regression) model.
        """
        self.surrogate_model = surrogate_model
        self.convention = "perturbed_input_similarity"
        super().__init__()

    def fit(self, list_premises: List[Premise]) -> None:
        """Fits the surrogate model with premises."""
        assert "similarity" in list_premises[0].attribution.keys(), (
            "The attribution attribute for the premises must be a dictionary with this format {'attribution': attribution ; 'similarity':similarity}"
        )

        keys, perturbed_scores, similarities = zip(
            *[
                (
                    self._prepare_key(p),
                    p.attribution["attribution"],
                    p.attribution["similarity"],
                )
                for p in list_premises
            ]
        )
        # for p in list_premises:
        #     print(f'{self._prepare_key(p)}, {p.attribution["similarity"] }, {p.attribution["attribution"]}')

        perturbed_scores = (
            torch.stack(perturbed_scores).squeeze(dim=1).to(self.device)
        )
        keys = torch.stack(keys).to(self.device)

        try:
            self.surrogate_model.fit(
                keys.numpy(),
                perturbed_scores.numpy(),
                sample_weight=similarities,
            )
        except TypeError:
            self.surrogate_model.fit(
                keys.cpu().numpy(),
                perturbed_scores.cpu().numpy(),
                sample_weight=similarities,
            )

    def _prepare_key(self, premise: Premise) -> torch.Tensor:
        """Prepares the key based on the type of premise.

        For fitting the surrogate model, the key must be
        appropriately transformed to be understandable by the model.
        Specifically, for image premises, the key is prepared by
        inverting the mask to align with the surrogate model's requirements.
        This ensures faithful implementation similar to LIME.

        Classical surrogate models like Ridge and Lasso require an
        input vector with one element per feature, indicating the
        status (perturbed or not) of each feature. For example, in
        LIME Image, we produce a vector indicating the presence or
        absence of each superpixel, while in classical LIME,
        the vector represents the amplitude of the applied perturbation
        """
        if isinstance(premise, SegmentedBinaryImagePremise):
            return 1 - premise.key
        elif isinstance(premise, KeyBasedMaskPremise):
            return premise.key
        else:
            raise ValueError("Unknown premise type")

    def get_coefs(self, memory: PremiseList) -> torch.Tensor:
        """Method which fits a linear model with the data contained in premises
        and returns the learned coefficients of the surrogate model
        """
        list_premises = memory.get_premises()
        self.fit(list_premises)
        return torch.tensor(self.surrogate_model.coef_)

    def get_explanation(self, memory: PremiseList) -> torch.Tensor:
        """Calculate final heatmap.

        This method is meant to be overridden by subclasses to handle different types of data.

        Args:
            memory (Premiselist): A Premiselist where premises are saved.

        Returns:
            torch.Tensor: The heatmap or coefficients depending on the data type.
        """
        # By default, return coefficients directly for tabular data
        return self.get_coefs(memory).clone().detach().unsqueeze(0)


class SegmentedImageModelAggregator(ModelAggregator):
    """Specialized aggregator for image data using superpixels and surrogate models.

    This aggregator extends ModelAggregator to handle segmented image data by mapping
    surrogate model coefficients back to pixel space using superpixels. It transforms
    the learned feature importance values from the superpixel level back to a spatial
    heatmap that highlights important image regions.

    """

    def get_explanation(self, memory: PremiseList) -> torch.Tensor:
        """Calculate final heatmap for segmented image data.

        Args:
            memory (Premiselist): A Premiselist where premises are saved. Every premise provides the attribution where mask's weight is stored.

        Returns:
            torch.Tensor: Final heatmap map of same shape as input x (b=1, c=1, h, w) highlighting the most important parts of the input example.

            Where
            b is batch dimension, expected to be set to 1 as only one example is being explained for the moment,

            w is the width,

            h is the height.

        """
        segmented_example = memory.get_premises()[0].segmented_example.to(
            self.device
        )
        coefs = self.get_coefs(memory).clone().detach().to(self.device)

        s, h, w = segmented_example.shape
        heatmap = torch.matmul(
            coefs,
            segmented_example.view(
                s,
                h * w,
            ).double(),  # (1, s)*(s, h * w) => (1, h*w)
        )
        heatmap = heatmap.unsqueeze(0)
        min_values = heatmap.min(dim=1).values
        max_values = heatmap.max(dim=1).values
        mask_diff_max_min = min_values != max_values
        heatmap[~mask_diff_max_min] = 0

        # Adds a dimension for broadcasting
        min_values = min_values.unsqueeze(dim=1)
        max_values = max_values.unsqueeze(dim=1)

        heatmap[mask_diff_max_min] = (
            ((heatmap - min_values)[mask_diff_max_min])
            / (max_values - min_values)[mask_diff_max_min]
        )

        heatmap = heatmap.view(
            1, 1, *segmented_example.shape[1:]
        )  # (1, 1, h, w)
        return heatmap
