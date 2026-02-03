"""Time series perturbators using trainable generators for temporal explanations.

This module implements specialized perturbators for time series data that leverage
trainable generators to create realistic temporal perturbations. These perturbators
are essential for explaining sequential models where temporal dependencies and
patterns are crucial for understanding model behavior.

In the MUPPET framework's perturbation step, these perturbators go beyond simple
masking by using learned generative models to impute missing values in time series
data. This preserves temporal coherence and realistic data characteristics, leading
to more meaningful explanations for sequential models.

The module contains:
    GeneratorSamplingPerturbator: Uses trainable generators to impute values at
        multiple timesteps simultaneously, suitable for complex temporal patterns
    ConditionalSamplingGeneratorPerturbator: Performs conditional sampling for
        single-timestep perturbations, ideal for feature-specific temporal explanations

Key Technical Features:
    - Automatic generator training with validation splits and early stopping
    - Temporal dependency preservation through learned representations
    - Feature-wise conditional sampling for fine-grained explanations
    - NaN handling for padding when model cannot handle unequal length signals

These perturbators are designed for explaining sequential models in domains like:
    - Financial time series forecasting
    - Sensor data analysis and anomaly detection
    - Medical signal processing and diagnosis

The generators learn to capture complex temporal patterns during training, enabling
realistic counterfactual scenarios that maintain temporal consistency while revealing
feature importance across time.
"""

#
# Created on Tue Apr 18 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import torch
from torch.utils.data import DataLoader

from muppet import logger
from muppet.components.perturbator.base import TrainablePerturbator
from muppet.components.perturbator.generator.base import TrainableGenerator


class GeneratorSamplingPertubator(TrainablePerturbator):
    """Perturbator using generative models for time series imputation.

    Employs trainable generators to create realistic substitutes for
    perturbed time series segments. Learns temporal patterns from
    training data to generate contextually appropriate perturbations.

    """

    def __init__(
        self,
        generator: TrainableGenerator,
        train_loader: DataLoader,
        max_batch_size: int = 100,
    ) -> None:
        """Perturbator that uses a generative model - generator - in order to impute the missing measurements.

        Args:
            generator (TrainableGenerator): The generator to be used to impute the perturbed values through its inference method '__call__'.
            train_loader (DataLoader): Training data.
            max_batch_size (int): Max Batch size to use. Default to 100.
        """
        super().__init__(
            generator=generator,
            train_loader=train_loader,
            max_batch_size=max_batch_size,
        )

    def perturbate(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Perturbate the input x according to the masks. The generator will be
        used to perturbate each position covered by the mask.

        Args:
            x (torch.Tensor): The input example. Shape (b=1, f, t)
            masks (torch.Tensor): Encodes what parts of x to be perturbed. Shape (N, x.shape)

        Returns:
            torch.Tensor: Perturbed version of x. (N, x.shape)

        """
        N = masks.size(0)
        # match inputs and masks shape to multiply them
        x_t = x.unsqueeze(dim=0).repeat(
            N, *[1 for _ in x.shape]
        )  # (b=1, f, t) => (N, b=1, f, t)

        # the value 0 will be replaced/imputed and the rest stays as it is
        perturbations = x_t * (1 - masks)  # (N, x.shape)

        # release temp
        del x_t

        # loop through the received masks
        for idx, mask in enumerate(masks):
            # Iterate over all possible timesteps
            # Never perturb step 0, there is nothing to base the perturbation on

            for time_step in range(1, mask.shape[-1]):
                # Do not perturb if all values are 0 or NaN
                if (
                    torch.equal(
                        mask[0, :, time_step],
                        torch.zeros_like(mask[0, :, time_step]),
                    )
                    or torch.isnan(mask[0, :, time_step]).all()
                ):
                    continue

                features_to_perturb_at_this_step = torch.where(
                    mask[0, :, time_step].flatten() == 1
                )[0].tolist()

                # get x0:t
                past = x[:, :, :time_step]
                # get x_at_t
                current = x[:, :, time_step]

                # impute the perturbed values using the generator
                sampled_values_at_time_step = self.generator.generate(
                    past=past,
                    current=current,
                    features_to_perturb=features_to_perturb_at_this_step,
                )
                # Returns a vector of shape [nb_series], the same as current.
                # The values at the positions of features_to_not_perturb_at_this_step
                # should be the same as in current, and the rest should have been
                # perturbed.

                # Returns a vector of shape [nb_series], the same as current.
                # The values at the positions of features_to_not_perturb_at_this_step
                # should be the same as in current, and the rest should have been
                # perturbed.
                assert (
                    perturbations[idx, :, :, time_step].shape == current.shape
                )
                assert sampled_values_at_time_step.shape == current.shape
                perturbations[idx, :, :, time_step] = (
                    sampled_values_at_time_step
                )

            perturbations[idx, mask.isnan()] = float("nan")
        logger.debug(
            f"Calculated perturbations: {str([i for i in perturbations])}"
        )

        return perturbations  # (N, x.shape)


class ConditionalSamplingGeneratorPertubator(TrainablePerturbator):
    """Conditional perturbator for advanced time series explanations.

    Uses conditional generators to create perturbations that respect
    feature dependencies and temporal relationships. Enables sophisticated
    perturbations for time series models.

    """

    def __init__(
        self,
        generator: TrainableGenerator,
        train_loader: DataLoader,
        max_batch_size: int = 100,
    ) -> None:
        """Perturbator that uses a GAN model - generator - in order to impute the missing measurements.

        Args:
            generator (TrainableGenerator): The generator to be used to impute the perturbed values through its inference method '__call__'.
            train_loader (DataLoader): Training data.
            max_batch_size (int): Max Batch size to use. Default to 100.
        """
        super().__init__(
            generator=generator,
            train_loader=train_loader,
            max_batch_size=max_batch_size,
        )

    def perturbate(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Perturbate the input x according to the masks. The generator will be
        used to perturbate each position covered by the mask.

        Args:
            x (torch.Tensor): The input example. Shape (b=1, f, t)
            masks (torch.Tensor): Encodes what parts of x to be perturbed. Shape (N, x.shape)

        Returns:
            torch.Tensor: Perturbed version of x. (N, x.shape)

        """
        N = masks.size(0)
        # match inputs and masks shape to multiply them
        x_t = x.unsqueeze(dim=0).repeat(
            N, *[1 for _ in x.shape]
        )  # (b=1, f, t) => (N, b=1, f, t)

        # the value 0 will be replaced/imputed and the rest stays as it is
        perturbations = x_t * (1 - masks)  # (N, x.shape)

        # release temp
        del x_t

        # loop through the received masks
        for idx, mask in enumerate(masks):
            # get the time step to be perturbed
            _, feature, time_step = (
                int(el) for el in torch.where(mask == 1)
            )  # mask shape is (b=1, f, t)
            # get x0:t
            past = x[:, :, :time_step]
            # get x_at_t
            current = x[:, :, time_step]
            # impute the perturbed values using the generator
            sampled_values_at_time_step = self.generator.generate(
                past=past,
                current=current,
                features_to_perturb={feature},
                # perturb the current feature
            )

            assert torch.sum(perturbations[idx, :, feature, time_step]) == 0, (
                f"The perturbator is expecting all the values at time step={time_step} to be 0, but found {perturbations[idx, :, :, time_step]}"
            )
            # update the value at time_step=t from 0 to the sampled one
            perturbations[idx, :, :, time_step] = sampled_values_at_time_step
        logger.debug(
            f"Calculated perturbations: {str([i for i in perturbations])}"
        )

        return perturbations  # (N, x.shape)
