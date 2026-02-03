"""FIT (Feature Importance in Time) explainer for time series models.

This module implements the Instance-wise Feature Importance in Time (FIT) method
for explaining time series classification models. FIT evaluates the importance of
observations by quantifying the shift in the predictive distribution over time using
KL divergence. Unlike traditional XAI methods, FIT specifically addresses the temporal
nature of time series data and controls for time-dependent distribution shifts.

MUPPET Component Integration:
    - **Explorer**: `RepeatedTimestepExplorer` - generates timestep-wise masks
        for temporal perturbation
    - **Perturbator**: `ConditionalSamplingGeneratorPertubator` - applies
        conditional sampling based perturbations using trained generators
    - **Attributor**: `ProbaShiftAttributor` - calculates distributional
        shifts using probability differences
    - **Aggregator**: `MonteCarloKLAggregator` - aggregates KL divergences from
        Monte Carlo sampling

Classes:
    FITExplainer: Implementation of the FIT method for time series explanation.

Technical Details
    Workflow:
        1. For a given input example and a set of features to explain $S$, FIT calculates saliency map showing the importance of $S$ at every time step.
            It does so by perturbing the other features not presented in $\overline{S}$ ($S$ compliment) by values sampled from a conditional distribution on $S$ fitted on
            the historical data (up to the current explained time step $t$).

            The score of $S$ at time step $t$ is calculated by the a difference-measure between:

            - **Temporal shift between $X$ at time $t$ against itself at $t-1$**: $P(y/X_{0:t}) || P(y/X_{0:t-1})$,

            - **Unexplained distribution shift  between $X$ and $X'$ at time $t$**: $P(y/X_{0:t}) || P(y/X'_{0:t})$,
            where $X'_{0:t}=X_{0:t-1, x_{S, t}}$ means the values of features in $\overline{S}$ at time $t$ are perturbed and imputed by the generator.

        2. Supports univariate and multivariate time series,
        4. Implements only KL divergence as the difference-measure. More measures will be added later.


References:
    Tonekaboni, Sana, et al. "Instance-wise feature importance in time."
    Advances in Neural Information Processing Systems 33 (2020): 21757-21767.
    https://papers.nips.cc/paper/2020/hash/08fa43588c2571ade19bc0fa5936e028-Abstract.html
"""
#
# Created on Tue Apr 18 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import torch
from torch.utils.data import DataLoader

from muppet import logger
from muppet.components.aggregator.distribution import MonteCarloKLAggregator
from muppet.components.attributor.distribution import ProbaShiftAttributor
from muppet.components.explorer.timestep import RepeatedTimestepExplorer
from muppet.components.perturbator.generator.conditional_timestep_generator import (
    ConditionalGaussianFeatureGenerator,
)
from muppet.components.perturbator.timestep_generator import (
    ConditionalSamplingGeneratorPertubator,
)
from muppet.explainers.base import MuppetExplainer


class FITExplainer(MuppetExplainer):
    """FIT (Feature Importance in Time) explainer implementation.

    Implements the FIT method that evaluates the importance of observations by quantifying
    the shift in the predictive distribution over time using KL divergence. The method
    specifically addresses the temporal nature of time series data and controls for
    time-dependent distribution shifts.

    The FIT method quantifies feature importance by comparing:
    1. The temporal shift between predictions at consecutive time steps
    2. The output distributional shift between original and perturbed inputs

    This approach provides instance-specific explanations that highlight the most important
    time points and features throughout the entire time series sequence.

    The method works by training conditional generators to create realistic perturbations
    and evaluating feature importance through distributional shift analysis using KL divergence.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        num_sampling: int = 100,
        generator: ConditionalGaussianFeatureGenerator | None = None,
        padding: str | None = None,
        hidden_size: int = 100,
        latent_size: int = 50,
        mid_layer_size: int = 50,
        prediction_size: int = 1,
        num_samples: int = 1,
        cov_noise_level: float = 1e-4,
        max_noise_correction: int = 20,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        timesteps_divide_num: int = 1,
        seed: int | None = None,
    ) -> None:
        """Initialize the FIT explainer for time series explanation.

        Args:
            model (torch.nn.Module): The blackbox model to explain.
                It must output the probability distribution over the set of classes.
            train_loader (DataLoader): The training data loader.
            num_sampling (int): Number of Monte-Carlo sampling of the perturbed values.
            generator (GaussianFeatureGenerator, optional): The generator is used to impute the perturbed values,
                therefore, it must have the `self.generate()` method implemented.
                Defaults to None: Meaning create a `GaussianFeatureGenerator` and train it on the provided train and test datasets.
            padding (str, optional): Either "left" or "right" in order to choose how to
                apply the padding when the black-box model only accepts full length input,
                otherwise None is chosen which means no padding will be applied, assuming the model doesn't require it.
                Defaults to None.
            hidden_size (int): Hidden layer size for the generator network.
            latent_size (int): Latent space dimension for the generator.
            mid_layer_size (int): Middle layer size for the generator network.
            prediction_size (int): Prediction output size.
            num_samples (int): Number of samples to generate.
            cov_noise_level (float): The noise to add to the covariance to make it positive definite (PD).
            max_noise_correction (int): Maximum number of covariance PD correction iterations.
                After exceeding this number the identity matrix will be used as the covariance.
            learning_rate (float): Training learning rate used with Adam optimizer.
            num_epochs (int): Training number of epochs.
            timesteps_divide_num (int): Used to divide the time series.
                E.g, when set to 1, it means predict only at time $t=T$ using $X_{0:T-1}$.
            seed (int, optional): The seed value to be used for a deterministic sampling using the generator.
                If a custom generator is given, therefore, it's
                expected to handle the reproducibility if it's needed!.
        """
        one_training_sample = next(iter(train_loader))[0]

        # explainer parameters
        self.num_sampling = num_sampling  # L
        padding = self._set_padding(value=padding)

        # Instantiate modules
        explorer = RepeatedTimestepExplorer(num_sampling=num_sampling)

        # re-initialize the generator if not provided
        if generator is None:
            # retrieve the number of channels in the timeseries
            self.feature_size = one_training_sample.shape[1]
            assert self.feature_size is not None, (
                "If training the generator is desired, you must provide the feature_size argument."
            )
            logger.info("The generator will be trained on the provided data!")
            # initialized with is_trained set to false
            generator = ConditionalGaussianFeatureGenerator(
                feature_size=self.feature_size,
                hidden_size=hidden_size,
                latent_size=latent_size,
                mid_layer_size=mid_layer_size,
                prediction_size=prediction_size,
                num_samples=num_samples,
                cov_noise_level=cov_noise_level,
                max_noise_correction=max_noise_correction,
                lr=learning_rate,
                num_epochs=num_epochs,
                timesteps_divide_num=timesteps_divide_num,
                seed=seed,
            )
        # add is_trained attribute to generator if it's provided
        else:
            generator.is_trained = True

        # perturbator initialization & generator training if required
        perturbator = ConditionalSamplingGeneratorPertubator(
            generator=generator,
            train_loader=train_loader,
        )
        attributor = ProbaShiftAttributor(padding=padding)
        aggregator = MonteCarloKLAggregator(
            num_sampling=self.num_sampling,
        )

        # Initiate the explainer with these modules
        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )

    # class helpful methods
    def _set_padding(
        self,
        value: str | None,
    ) -> str | None:
        """Constraints the choices of padding value. Accepts only left or right. If none of them is passed then choose None meaning no padding will be applied.

        Args:
        Args:
            value (str): str value must be "left" or "right" if not provided then no padding will be applied

        Returns:
            value (str): value if it's valid otherwise None
        """
        if value in ["left", "right"]:
            return value

        logger.info(
            "The passed padding value is neither 'left' nor 'right', then it is considered as None (not provided)!"
        )
        return None

    def set_generator_seed(
        self,
        seed: float,
    ) -> None:
        """Re-set the generator's seed. Used to control the reproducibility when perturbing.

        Args:
            seed (float): the seed to set.

        """
        self.perturbator.generator.seed = seed
        return None
