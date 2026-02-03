"""Classification-based attributors for MUPPET XAI.

This module provides attribution methods for classification models.
These attributors calculate attribution scores based on class probabilities,
making them ideal for explaining image classification, text classification,
and other discrete classification tasks.

Classes:
    ClassScoreAttributor: Calculates attributions based on non perturbed input class
        probability prediction score, measuring how much each perturbation affects
        the model's confidence in the correct prediction. Supports both destructive
        and constructive attribution conventions.

Technical Details:
    The ClassScoreAttributor computes attributions by:
    1. Determining the true class from the original input
    2. Evaluating the model's probability for this class on each perturbation
    3. Converting probabilities to attribution scores based on the convention:
       - Destructive: Higher scores for perturbations that reduce class confidence
       - Constructive: Higher scores for perturbations that maintain class confidence

    This method is computationally efficient and provides intuitive explanations for
    classification models across various domains and architectures.
"""

#
# Created on Fri Jun 09 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

from typing import Union

import torch
import torch.nn.functional as F

from muppet.components.attributor.base import Attributor
from muppet.components.convention import AttributionConvention
from muppet.components.memory.base import PremiseList


class ClassScoreAttributor(Attributor):
    """Attribution based on probability score of the true class for classification tasks.

    This attributor calculates the probability score of the true class (calculated from
    the original example) and stores it into premise's attribution. Since in most cases
    we expect the most impactful perturbations to have the highest attribution, by default
    the attribution will be MINUS the probability of the true score.

    Attributes:
        true_class: True class index determined from the original input.
    """

    def __init__(
        self, convention: Union[AttributionConvention, str] = "destructive"
    ) -> None:
        """Initialize the class score attributor.

        Args:
            convention: The attribution convention, either 'destructive' or 'constructive'.
        """
        self.true_class = None
        self.convention = convention
        super().__init__()

    def reinitialize(self):
        """Reinitialize the classification attributor."""
        self.true_class = None
        return super().reinitialize()

    def calculate_attribution(
        self,
        x: torch.Tensor,
        perturbed_inputs: torch.Tensor,
        model: torch.nn.Module,
        memory: PremiseList,
    ) -> None:
        """Calculates the attribution of perturbed inputs.

        Args:
            x (torch.Tensor): Example to explain. (b=1, *x.shape[1:]).
            perturbed_inputs (torch.Tensor): The example's perturbations. Shape (N, *x.shape).
            model (torch.nn.Module): The black-box model.
            memory (Premiselist): Premises' memory where to save the attributions.

        Where b is the batch size (=1), N is the number of generated masks.

        """
        # calculate the true class if not already done
        if self.true_class is None:
            with torch.no_grad():
                true_output = F.softmax(model(x).detach(), dim=1)
            self.true_class = torch.argmax(true_output, dim=1)  # (b=1)

        # calculate the true class prediction of every perturbation
        for idx, premise in enumerate(memory.get_premises()):
            with torch.no_grad():
                logits = model(perturbed_inputs[idx].float()).detach()

            probs = F.softmax(logits, dim=1)  # (b=1, nclasses)
            premise.attribution = probs[:, self.true_class].squeeze(
                dim=-1
            )  # (b=1, 1) => (b)

            if self.convention == AttributionConvention.DESTRUCTIVE:
                premise.attribution = -premise.attribution

        return
