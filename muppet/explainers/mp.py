"""MP (Meaningful Perturbation) explainer for interpretable explanations.

This module implements the Meaningful Perturbation method for generating interpretable
explanations of black-box models through optimized mask learning. Unlike sampling-based
methods, MP directly optimizes a continuous mask that identifies the most meaningful
regions of an input image for the model's prediction.

MUPPET Component Integration:
    - **Explorer**: `GradientExplorer` - uses gradient-based optimization to learn optimal masks iteratively
    - **Perturbator**: `BlurPerturbator` - applies Gaussian blur perturbations based on learned masks
    - **Attributor**: `MaskRegularizedScoreAttributor` - computes loss with TV and L1 regularization
    - **Aggregator**: `LearntMaskAggregator` - returns the final optimized mask as explanation

Classes:
    MPExplainer: Implementation of the Meaningful Perturbation method.

References:
    Fong, Ruth C., and Andrea Vedaldi. "Interpretable explanations of black boxes by
    meaningful perturbation." Proceedings of the IEEE international conference on
    computer vision. 2017.
    https://arxiv.org/pdf/1704.03296.pdf
"""
#
# Created on Fri Jun 30 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import torch

from muppet.components.aggregator.mask import LearntMaskAggregator
from muppet.components.attributor.differentiable import (
    MaskRegularizedScoreAttributor,
)
from muppet.components.explorer.gradient import GradientExplorer
from muppet.components.perturbator.simple import BlurPerturbator
from muppet.explainers.base import MuppetExplainer


class MPExplainer(MuppetExplainer):
    """Meaningful Perturbation explainer implementation.

    Implements the Meaningful Perturbation method that learns optimal explanation masks
    through gradient-based optimization with regularization constraints. The method works
    by learning a mask through gradient-based optimization that maximizes (constructive)
    or minimizes (destructive) the model's confidence while being constrained by
    regularization terms.

    Key features of MP include:
    - Direct optimization of explanation masks using gradient descent
    - Differentiable perturbation operations
    - Total Variation (TV) regularization for smooth, coherent explanations
    - L1 regularization for sparse, focused explanations
    - Support for both constructive and destructive explanation modes

    The optimization objective balances prediction preservation with mask sparsity
    and smoothness, resulting in compact and interpretable explanations.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        tv_beta: float = 3,
        lr: float = 0.2,
        max_iter: int = 100,
        l1_coeff: float = 0.01,
        tv_coeff: float = 0.02,
        mask_shape: tuple[int, int] = (28, 28),
        add_noise: bool = False,
        kernel_size: tuple[int, int] = (11, 11),
        sigma: int = 5,
        convention: str = "destructive",
    ) -> None:
        """Initialize the MP explainer for learning optimal explanation masks.

        Args:
            model (torch.nn.Module): Black-box model.
            tv_beta (float): Degree of the Total Variation (TV) denoising norm.
            lr (float): Learning rate.
            max_iter (int): The number of iterations for SGD.
            l1_coeff (float): L1 regularization coefficient.
            tv_coeff (float): Total Variation coefficient.
            mask_shape (tuple[int, int]): The down-scaled learning mask shape.
            add_noise (bool): Whether to add noise during perturbation.
            kernel_size (tuple[int, int]): Kernel size for blur perturbation.
            sigma (int): Standard deviation for Gaussian blur.
            convention (str): choose if the explainer finds important features
                by identifying features that destroy (destructive) efficiently the model's prediction from the input,
                or by identifying features that build (constructive) efficiently the model's response from a completly perturbed input
        """
        # Explorer pars
        self.lr = lr
        self.mask_shape = mask_shape

        # Perturbator pars
        self.add_noise = add_noise
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Attributor pars
        self.tv_beta = tv_beta
        self.max_iter = max_iter
        self.l1_coeff = l1_coeff
        self.tv_coeff = tv_coeff

        explorer = GradientExplorer(
            max_iter=self.max_iter, lr=self.lr, mask_shape=self.mask_shape
        )

        perturbator = BlurPerturbator(
            add_noise=self.add_noise,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
        )

        attributor = MaskRegularizedScoreAttributor(
            l1_coeff=self.l1_coeff,
            tv_coeff=self.tv_coeff,
            tv_beta=self.tv_beta,
            convention=convention,
        )
        aggregator = LearntMaskAggregator(convention=convention)

        # Initialize the main explainer
        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )
