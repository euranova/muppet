"""Scale-based perturbators for tabular data using statistical generators.

This module implements perturbators that use statistical generators to create scaled
perturbations of tabular data. These perturbators are designed for the MUPPET XAI
framework's perturbation step, where input features are selectively replaced with
generated values based on learned or observed data distributions.

The perturbators in this module specialize in tabular data explanations by leveraging
sophisticated generators that understand feature distributions, correlations, and
data types. This enables more realistic perturbations compared to simple masking
approaches, leading to better explanation quality for tabular models.

The module contains:
    ScaleFeaturePerturbator: Uses Gaussian-based generators for continuous tabular
        features with statistical scaling and normalization
    RandomSamplePerturbator: Employs frequency-based sampling from training data
        distributions for mixed categorical/numerical features

Key Features:
    - Statistical distribution preservation through generator training
    - Mixed data type handling (numerical and categorical)
    - Instance-centered perturbations for local explanations
    - Configurable sampling strategies and scaling approaches
    - Memory-efficient batch processing with automatic size adjustment

These perturbators are essential for tabular explanation methods like LIME and SHAP,
where the quality of counterfactual examples directly impacts explanation fidelity
and interpretability. They enable realistic "what-if" scenarios by generating
plausible alternative feature values.
"""

import torch

from muppet.components.perturbator.base import Perturbator
from muppet.components.perturbator.generator.tabular_generator import (
    RandomSampleTabularGenerator,
    StandardGaussianTabularGenerator,
)


class ScaleFeaturePerturbator(Perturbator):
    """Perturbator for tabular data using Gaussian generators.

    Specializes in perturbing tabular data by applying statistical
    generators that maintain feature distributions. Designed for
    structured data explanations where preserving data realism is crucial.
    """

    def __init__(
        self,
        generator: StandardGaussianTabularGenerator,
        max_batch_size: int = 100,
    ) -> None:
        """Perturbator for tabular data based on a Gaussian generator.

        This perturbator is designed to modify tabular data by generating
        controlled variations using a Gaussian-based generator, which has
        been trained on continuous input features.


        Args:
            generator (StandardGaussianTabularGenerator): An instance of a Gaussian
                generator for tabular data, which, once trained on the input data,
                generates continuous values based on a Gaussian distribution.
            max_batch_size (int): Max Batch size to use. Default to 100.
        """
        self.generator = generator
        super().__init__(max_batch_size=max_batch_size)

    def perturbate(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Perturbs the input tensor `x` using the provided masks.

        This method applies perturbations to the input `x` based on the given `masks`.
        The masks determine which features of the input should be perturbed (1 for
        perturbed, 0 for not perturbed). The generator is used to produce substitute
        values for the masked features.

        Args:
            x (torch.Tensor): The input tensor to be perturbed, with shape (1, f),
                where `f` is the number of features in the data.
            masks (torch.Tensor): A tensor of masks containing 0s and 1s to determine
                which features in `x` will be perturbed. Its shape is
                (b, *shape), where `shape` corresponds to the shape of `x`.

        Returns:
            torch.Tensor: A tensor containing the generated perturbed values,
            with shape (number_of_masks, *x.shape).
        """
        data_scaled = masks.unsqueeze(1)
        # Generate perturbed samples using the generator
        sampled_values_tensor = self.generator.generate(
            x, data_scaled
        )  # x and masks are passed to generate, shape (number_masks, *x.shape)

        return sampled_values_tensor


class RandomSamplePerturbator(Perturbator):
    """A class to perturb tabular data using generated samples and binary masks.

    Employs frequency-based sampling from training data distributions for mixed categorical/numerical features

    """

    def __init__(
        self,
        generator: RandomSampleTabularGenerator,
        max_batch_size: int = 100,
    ) -> None:
        """Initializes the RandomSamplePerturbator with a generator.

        Args:
            generator (RandomSampleTabularGenerator): An instance of
                RandomSampleTabularGenerator that will be used to generate random samples.
        """
        self.generator = generator
        super().__init__(max_batch_size=max_batch_size)

    def perturbate(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Perturb the input tensor using generated samples and binary masks.

        Args:
            x (torch.Tensor): The input tensor to be perturbed. Should have shape (1, num_features).
            masks (torch.Tensor): A tensor containing binary masks with shape (number_masks, 1, num_features).
                Each mask is used to determine where to apply the perturbation.

        Returns:
            torch.Tensor: A tensor containing the perturbed samples with shape (number_masks, 1, num_features).
                Perturbations are applied according to the masks: where masks are 1, the value from
                the generated tensor is used; where masks are 0, the original value `x` is preserved.
        """
        self.generated_tensor = None
        n_samples, n_features = masks.shape
        masks = masks.unsqueeze(1)

        # Generate samples using the provided generator
        self.generated_tensor = self.generator.generate(n_samples).to(
            self.device
        )  # x and masks are passed to generate

        # Ensure masks and generated_tensor ar compatible

        # Calculate perturbations
        perturbations = (1 - masks) * self.generated_tensor + masks * x
        return perturbations
