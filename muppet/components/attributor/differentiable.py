"""Gradient-based and differentiable attributors for MUPPET XAI.

This module provides attribution methods that leverage gradient-based optimization and
differentiable loss functions. These attributors are designed for scenarios where the
attribution interacts with a optimisation-based exploration techniques.

Classes:
    DifferentiableAttributor: Abstract base class for differentiable attributors that
        use customizable loss functions to compute attributions through backpropagation.
    MaskRegularizedScoreAttributor: Concrete implementation that combines classification
        scores with mask regularization terms (L1 and Total Variation) to find minimal
        and smooth explanatory masks.
"""

from abc import abstractmethod
from typing import Union

import torch

from muppet.components.attributor.base import AttributionConvention, Attributor
from muppet.components.memory.base import Memory, Premise


class DifferentiableAttributor(Attributor):
    """Base class for gradient-based attribution methods.

    This class is used alongside gradient-based exploration methods. It loops through the premises
    to fill up their attributions by calling a customizable loss function. All needed
    arguments for the loss calculation must be initialized within the child class.

    This base class sets up the true class placeholder for gradient-based attribution methods
    that can benefit from backpropagation and differentiable loss functions.

    Attributes:
        true_class: The true class index calculated once from the original input x.
    """

    def __init__(self) -> None:
        """Initialize the differentiable attributor."""
        self.true_class = None
        super().__init__()

    def reinitialize(self):
        """Return DifferentiableAttributor to its original state."""
        self.true_class = None

    def calculate_attribution(
        self,
        x: torch.Tensor,
        perturbed_inputs: torch.Tensor,
        model: torch.nn.Module,
        memory: Memory,
    ) -> None:
        """Calculates the loss of an objective function defined by `calculate_attribution_loss` method.

        Args:
            x (torch.Tensor): Example to explain. Shape (1, *x.shape[1:]) E.g (b=1, c, w, h) for images
                - b is number of input examples,
                - c is the channel dimensions,
                - w is the width,
                - h is the height,
            perturbed_inputs (torch.Tensor): Perturbed versions of the example. Shape (N, x.shape) E.g (N, b, c, w, h)
                - N is the number of applied perturbations on the example.
            model (torch.nn.Module): The black-box model.
            memory (FlatList): Structure holding the premises where attributions will be saved.
        """
        # get premises from memory
        premises = memory.get_premises()

        # calculate the true class once
        if self.true_class is None:
            with torch.no_grad():
                true_output = torch.nn.Softmax(dim=1)(model(x)).to(
                    self.device
                )  # (b, nclasses)

            self.true_class = torch.argmax(true_output, dim=1)  # (b)

        probs = torch.nn.Softmax(dim=1)(
            model(perturbed_inputs[:, 0, :])  # number of masks per input is 1
        )  # (b, 1000)

        for i in range(len(premises)):
            loss = self.calculate_attribution_loss(
                premise=premises[i], output=probs[i, self.true_class[i]]
            )

            # save the loss to premise's attribution
            premises[i].attribution = loss.to(self.device)

        return

    @abstractmethod
    def calculate_attribution_loss(
        self, premise: Premise, output: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the optimization loss using premise element and model's output corresponding to the predicted class from original input example.

        Args:
            premise (Premise): The memory's element that represent the perturbation.

            output (torch.Tensor): The model's output for the corresponding input example.

        Raises:
            NotImplementedError: Must be implemented in child classes.

        """
        raise NotImplementedError


class MaskRegularizedScoreAttributor(DifferentiableAttributor):
    """Regularized mask attribution using L1 and total variation loss.

    This attributor calculates a loss function combining minimal mask penalty, total variation
    denoising, and true class probability from the perturbed input:
    Loss = λ|m| + λ'|tv(1-m)| + f(x')
    By default no regularization is applied on the mask.

    Attributes:
        l1_coeff: L1 regularization coefficient for mask sparsity.
        tv_coeff: Total Variation coefficient for smoothness regularization.
        tv_beta: Degree of the Total Variation denoising norm.
        convention: The attribution convention (constructive or destructive).
        true_class: The true class index calculated once from the original input x.

    """

    def __init__(
        self,
        l1_coeff: float = 0,
        tv_coeff: float = 0,
        tv_beta: float = 0,
        convention: Union[AttributionConvention, str] = "destructive",
    ) -> None:
        """Initialize the mask regularized score attributor.

        Args:
            l1_coeff: L1 regularization coefficient for the mask.
            tv_coeff: Total variation regularization coefficient for the mask.
            tv_beta: Beta parameter for total variation calculation.
            convention: Attribution convention, either 'destructive' or 'constructive'.
        """
        self.l1_coeff = l1_coeff
        self.tv_coeff = tv_coeff
        self.tv_beta = tv_beta

        self.true_class = None
        self.convention = convention
        super().__init__()

    def calculate_attribution_loss(self, premise, output) -> torch.Tensor:
        """Calculates the attribution/loss from the sum of the remise's mask mean, mask's TV norm and probability prediction
        corresponding to the true class.

        Args:
            premise: The premise element representing the perturbation.

            output: The true class predicted probability.

        Returns:
            loss: The calculated loss.

        """
        mask_loss = 0
        tv_denoise_loss = 0

        if self.l1_coeff != 0:
            mask_loss = self.l1_coeff * torch.mean(
                torch.abs(premise.key)
            )  # (b=1)

        if self.tv_coeff != 0:
            tv_denoise_loss = self.tv_coeff * self._tv_norm(
                (1 - premise.key), self.tv_beta
            )  # (b=1)

        if self.convention == "descructive":
            score_impact = output
        else:
            score_impact = -output

        return mask_loss + tv_denoise_loss + score_impact

    @staticmethod
    def _tv_norm(input: torch.Tensor, tv_beta: float) -> torch.Tensor:
        """Computes the Total Variation (TV) denoising term.

        Args:
            input: Tensor to calculate its TV. Shape (1, w, h).

            tv_beta: Degree of the Total Variation denoising norm. Where

            w is the width,

            h is the height.

        """
        col_grad = torch.mean(
            torch.abs((input[0, :, :-1] - input[0, :, 1:])).pow(tv_beta)
        )
        row_grad = torch.mean(
            torch.abs((input[0, :-1, :] - input[0, 1:, :])).pow(tv_beta)
        )

        return row_grad + col_grad
