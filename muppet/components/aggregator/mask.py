"""Mask-based aggregators for image and spatial data explanations.

This module provides aggregators that work with mask-based perturbations for generating
explanation for image classification models. It implements weighted aggregation
methods that combine multiple perturbation masks with their corresponding attribution
weights to produce final saliency maps.

The module supports both weighted sum aggregation (for methods like RISE) and learned
mask aggregation (for gradient-based optimization methods) with support for different
attribution conventions (constructive vs destructive).


Classes:
    WeightedSumAggregator: Aggregates masks using weighted sum based on model predictions.
    LearntMaskAggregator: Returns normalized learned masks from optimization-based methods.
"""
#
# Created on Fri Jun 09 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

from typing import Callable, Union

import torch

from muppet.components.aggregator.base import Aggregator
from muppet.components.convention import AttributionConvention
from muppet.components.memory.base import PremiseList


class WeightedSumAggregator(Aggregator):
    """Weighted sum aggregator for mask-based explanations.

    This aggregator multiplies the weight of every perturbation by its mask and sums up
    all the masks. The weight is equal to the model's class probability. It is commonly
    used in methods like RISE.

    """

    def __init__(
        self,
        post_proc: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
        convention: Union[AttributionConvention, str] = "destructive",
    ) -> None:
        """Initialize the weighted sum aggregator.

        Args:
            post_proc: Apply the post_proc function to the calculated heatmap (example: ReLU).
                None by default, meaning no post processing is done.
            convention: The attribution convention to use, either "constructive" or "destructive".
        """
        self.post_proc = post_proc
        self.convention = convention
        super().__init__()

    def get_explanation(
        self,
        memory: PremiseList,
    ) -> torch.Tensor:
        """Calculate final heatmap by multiplying the weight of every perturbation by its mask and sum up all the masks.

        Args:
            memory (Premiselist): A simple list where premises are saved. Every premise provides the attribution where mask's weight is stored.

        Returns:
            torch.Tensor: Final heatmap map of same shape as input x (b=1, c, w, h) highlighting the most important parts of the input example.
                Where
                   - b is batch dimension, expected to be set to 1 as only one example is being explained for the moment,
                   - w is the width,
                   - h is the height.

        """
        masks = torch.stack(
            [premise.heatmap for premise in memory.get_premises()]
        ).to(self.device)  # (N, b=1, c=1, w, h)
        weights = torch.stack(
            [premise.attribution for premise in memory.get_premises()]
        ).to(self.device)  # (N, b=1)

        example_shape = masks.shape[1:]

        N = masks.size(0)
        b = masks.size(1)  # ==1
        w = masks.size(-2)
        h = masks.size(-1)

        masks = masks.view(N, h * w)  # (N, w*h)
        weights = weights.transpose(0, 1)  # (b=1, N)
        # weights = torch.nn.functional.softmax(weights, dim=-1)

        score_saliency_map = torch.matmul(
            weights, masks
        )  # (b=1, N) x (N, w*h) = (b=1, w*h)
        score_saliency_map = score_saliency_map.view(
            (b, w, h)
        )  # (b=1, w*h) => (b=1, w, h)

        score_saliency_map = score_saliency_map.view(
            *example_shape
        )  # (b=1, c=1, w, h)

        if self.post_proc is not None:
            # score_saliency_map = self.post_proc(score_saliency_map)
            # when post_proc finction return constant values (ex ReLU with only negative values)
            # bypass the post proc function
            score_saliency_map = (
                score_saliency
                if not torch.all(
                    (score_saliency := self.post_proc(score_saliency_map)) == 0
                )
                else score_saliency_map
            )
            assert score_saliency_map.shape == (
                b,
                1,
                w,
                h,
            ), "Post processing operation mustn't change saliency map shape."

        min_ssm = score_saliency_map.min()
        max_ssm = score_saliency_map.max()

        if min_ssm == max_ssm:
            return (
                score_saliency_map * 0
            )  # if heatmap is constant, set it to 0 everywhere

        else:
            return (score_saliency_map - min_ssm) / (
                max_ssm - min_ssm
            )  # else, normalise it between 0 and 1


class LearntMaskAggregator(Aggregator):
    """Learned mask aggregator for optimization-based explanation methods.

    This aggregator returns the normalized learned mask from gradient-based optimization
    methods. It normalizes the mask between 0 and 1 and applies convention-based
    transformations as needed.
    """

    def __init__(self, convention: str = "destructive") -> None:
        """Initialize the learnt mask aggregator.

        Args:
            convention (str): The attribution convention, either 'constructive' or 'destructive'.
                If "constructive", the heatmap is reversed using 1-heatmap.
        """
        self.convention = convention
        super().__init__()

    def get_explanation(
        self,
        memory: PremiseList,
    ) -> torch.Tensor:
        """Returns the learnt mask.

        Args:
            memory (Premiselist): List of one premise with the mask to be optimized.

        Returns:
            torch.Tensor: The final heatmap. Shape (b=1, *x.shape[1:]).

        """
        heatmap = memory.get_premises()[0].mask.detach()  # x.shape
        heatmap = heatmap[0]  # b=1 => (x.shape[1:]) (c=1, w, h)

        # get mask's min and max values
        minn = heatmap.min()
        maxx = heatmap.max()
        max_min = maxx - minn

        heatmap = (heatmap - minn) / max_min

        if self.convention == "constructive":
            heatmap = 1 - heatmap

        return heatmap.unsqueeze(dim=0)  # (b=1, c=1, w, h)
