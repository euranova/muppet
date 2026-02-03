"""Simple perturbation strategies for basic masking operations.

Provides simple **perturbator** implementations for the MUPPET XAI framework,
implementing the second step of the perturbation process. These modules apply
straightforward transformations to input data based on binary masks from explorers.

Simple perturbators are fast, interpretable, and computationally efficient.

Classes:
    SetToZeroPerturbator: Masks features by **setting them to zero**.
    BlurPerturbator: Applies Gaussian blur to masked regions in image data, simulating
        information removal while maintaining spatial structure

These simple perturbators are building blocks for many explanation methods including RISE.
They provide baseline perturbation strategies that can be compared against more sophisticated
approaches or used when interpretability and speed are prioritized over realism.
"""
#
# Created on Fri Jun 09 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import torch
from torchvision.transforms import GaussianBlur

from muppet.components.perturbator.base import Perturbator


class SetToZeroPerturbator(Perturbator):
    """Simple perturbator that sets masked features to zero.

    Provides basic perturbation by multiplying input features with
    the complement of the mask (1-mask). This creates binary
    perturbations where features are either preserved or zeroed out.
    """

    def __init__(self, max_batch_size: int = 100) -> None:
        """Initialize the SetToZeroPerturbator.

        Args:
            max_batch_size (int): Maximum batch size for processing. Defaults to 100.
        """
        super().__init__(max_batch_size=max_batch_size)

    def perturbate(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Set masked features to zero by multiplying with (1-mask).

        Args:
            x (torch.Tensor): The input example. Shape (b, *input_dims).
            masks (torch.Tensor): The perturbation masks. Shape (N, *mask_shape).

        Returns:
            torch.Tensor: Perturbed examples with masked features set to zero.
        """
        # repeat the input example across N masks
        x_t = x.unsqueeze(dim=0).repeat(
            masks.size(0), *[1 for _ in x.shape]
        )  # (*x.shape) => (N, *x.shape)

        perturbations = x_t * (1 - masks)  # (N, *x.shape)

        return perturbations


class BlurPerturbator(Perturbator):
    """Perturbator that applies Gaussian blur to masked regions.

    Creates perturbations by blurring masked areas instead of
    zeroing them. Maintains spatial information while reducing
    detail, useful for image explanations where complete occlusion
    is too harsh.

    This perturbator simulates information removal while maintaining spatial structure
    by applying Gaussian blur to masked regions in image data. It provides a more
    realistic perturbation than simple masking for image explanations.

    """

    def __init__(
        self,
        add_noise: bool = False,
        kernel_size: tuple[int, int] = (11, 11),
        sigma: int = 5,
        max_batch_size: int = 100,
    ) -> None:
        """Initialize the BlurPerturbator.

        Args:
            add_noise (bool): Whether to add normal noise to perturbations. Defaults to False.
            kernel_size (tuple[int, int]): Gaussian kernel size for blurring. Defaults to (11, 11).
            sigma (int): Gaussian sigma parameter. Defaults to 5.
            max_batch_size (int): Maximum batch size for processing. Defaults to 100.
        """
        self.add_noise = add_noise
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.blurred_input = None
        super().__init__(max_batch_size=max_batch_size)

    def reinitialize(self):
        """Return BlurPerturbator to its original state."""
        self.blurred_input = None

    def perturbate(
        self,
        x: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the input perturbations by `$Input*(1-Mask) + BluredInput*Mask$`

        Args:
            x (torch.Tensor): The input examples. Shape (b=1, *x.shape[1:]) E.g (b=1, c, w, h)

            masks (torch.Tensor): The generated masks to use for perturbing x.
                Shape (N, *mask_shape), len(mask_shape)==x.dim(). E.g mask_shape =(b=1, c=1, w, h)
                - b is batch dimension, expected to be set to 1 as only one example is being explained for the moment,
                - c is the channel dimensions,
                - w is the width,
                - h is the height,
                - N the number of perturbation masks.

        Returns:
            x' (torch.Tensor): Perturbed version of x. Shape (N, *x.shape)

        """
        # blur the input example once for all
        if self.blurred_input is None:
            self.blurred_input = GaussianBlur(
                kernel_size=self.kernel_size, sigma=self.sigma
            )(x)  # E.g (b=1, c, w, h)

        N = masks.size(0)
        x = x.unsqueeze(dim=0).repeat(
            N, *[1 for _ in x.shape]
        )  # (*x.shape) => (N, *x.shape)
        x_b = self.blurred_input.unsqueeze(dim=0).repeat(
            N, *[1 for _ in self.blurred_input.shape]
        )  # (*x_b.shape) => (N, *x_b.shape)

        perturbations = x * (1 - masks) + x_b * masks  # (N, *x.shape)

        # add normal noise if requested
        if self.add_noise:
            noise = torch.randn(perturbations.shape)
            perturbations = perturbations + noise

        return perturbations
