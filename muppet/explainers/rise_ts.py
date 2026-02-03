"""RISE explainer variants for time series data.

This module implements time series adaptations of the RISE (Randomized Input Sampling
for Explanation) method. While the original RISE was designed for image data, these
variants extend the approach to handle multivariate time series by adapting the masking
strategy and perturbation methods to respect the temporal structure of the data.

MUPPET Component Integration:
    **For both explainers**:
    - **Explorer**: `RandomMasksExplorer` - generates 2D random masks for (features, time) dimensions
    - **Attributor**: `ClassScoreAttributor` - extracts model confidence for destructive analysis
    - **Aggregator**: `WeightedSumAggregator` - computes weighted average of masks

    **Perturbator differences**:
    - **RISETimeseriesExplainer**: `SetToZeroPerturbator` - simple zero-masking
    - **RISETimeseriesGenerativePerturbationExplainer**: `GeneratorSamplingPertubator` - realistic value sampling

Classes:
    RISETimeseriesExplainer: Basic RISE adaptation for time series using zero-masking.
    RISETimeseriesGenerativePerturbationExplainer: Advanced RISE variant with generative perturbations.

Note:
    Time series input format expected: (batch_size, 1, num_series, time_length)
    where num_series is the number of individual time series and time_length
    is the temporal dimension.
"""

import torch

from muppet.components.aggregator.mask import WeightedSumAggregator
from muppet.components.attributor import ClassScoreAttributor
from muppet.components.explorer.mask import RandomMasksExplorer
from muppet.components.perturbator.generator.tabular_generator import (
    GaussianSamplingGenerator,
)
from muppet.components.perturbator.simple import SetToZeroPerturbator
from muppet.components.perturbator.timestep_generator import (
    GeneratorSamplingPertubator,
)
from muppet.explainers.base import MuppetExplainer


class RISETimeseriesExplainer(MuppetExplainer):
    """RISE explainer specialized for time series data.

    Adapts the RISE method for time series analysis by using temporal masking strategies.
    The module provides adaptations for time series with 2D masking covering both time
    and feature dimensions, and preservation of temporal structure during perturbation.

    Both methods maintain RISE's core principle of statistical estimation through random
    sampling, but adapt the masking and perturbation strategies to be more appropriate
    for temporal data where complete removal (zero-masking) may be less realistic than
    replacing with plausible alternative values.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        nb_series: int | None = None,
        nmasks: int = 800,
        mask_dim: int = 7,
        mask_proba: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize the RISE explainer for time series data.

        Args:
            model (torch.nn.Module): The black-box model to explain its predictions.
            nb_series (int | None): The number of timeseries (n) that will be provided
                when asking for an explanation. This needs to be fixed in advance so the masks
                have the correct shape.
            nmasks (int): Number of random masks to generate.
            mask_dim (int): The size of the squared grade (down-scaled mask).
            mask_proba (float): The probability of setting, independently, each pixel of the mask to 1.
            seed (int, optional): Seed to initialize for reproducible results.
        """
        self.nmasks = nmasks
        self.mask_dim = mask_dim
        self.mask_proba = mask_proba
        self.nb_series = nb_series

        self.premise_kwargs = {"modality": "timeseries"}

        explorer = RandomMasksExplorer(
            nmasks=self.nmasks,
            mask_dim=(self.nb_series, self.mask_dim),
            mask_proba=self.mask_proba,
            seed=seed,
        )

        perturbator = SetToZeroPerturbator()
        attributor = ClassScoreAttributor(convention="destructive")
        aggregator = WeightedSumAggregator()

        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )


class RISETimeseriesGenerativePerturbationExplainer(MuppetExplainer):
    """RISE explainer for time series using generative perturbations.

    Extends RISE for time series by using generative models to create realistic
    perturbations instead of simple masking. Provides more naturalistic explanations
    for temporal data analysis by using generative perturbations based on learned
    distributions.

    This variant uses generative perturbations based on samples drawn from a Normal
    distribution whose parameters were estimated from the previous measurements of
    the time series.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        nb_series: int,
        nmasks: int = 800,
        mask_dim: int = 7,
        mask_proba: float = 0.1,
        seed: int | None = None,
        generator=None,
    ) -> None:
        """Initialize the RISE explainer with generative perturbations for time series data.

        Args:
            model (torch.nn.Module): The black-box model to explain its predictions.
            nb_series (int): The number of timeseries (n) that will be
                provided when asking for an explanation. This needs to be fixed in advance so the masks
                have the correct shape.
            nmasks (int): Number of random masks to generate.
            mask_dim (int): The size of the squared grade (down-scaled mask).
            mask_proba (float): The probability of setting, independently, each pixel of the mask to 1.
            seed (int, optional): Seed to initialize for reproducible results.
            generator: Generator used, by default it is the generator using the Normal distribution
        """
        self.nmasks = nmasks
        self.mask_dim = mask_dim
        self.mask_proba = mask_proba
        self.nb_series = nb_series

        self.premise_kwargs = {"modality": "timeseries"}

        explorer = RandomMasksExplorer(
            nmasks=self.nmasks,
            mask_dim=(self.nb_series, self.mask_dim),  # self.mask_dim,
            mask_proba=self.mask_proba,
            seed=seed,
        )

        # re-initialize the generator if not provided
        if generator is None:
            generator = (
                GaussianSamplingGenerator()
            )  # and train the mean and std on the appropriate series

        self.generator = generator

        # perturbator initialization & generator training if required
        perturbator = GeneratorSamplingPertubator(
            generator=self.generator,
            train_loader=None,
        )

        # These modules below are the same as classical RISE
        attributor = ClassScoreAttributor(convention="destructive")
        aggregator = WeightedSumAggregator()

        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )
