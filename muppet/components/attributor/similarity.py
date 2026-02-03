"""Similarity-based attributors for MUPPET XAI.

This module provides attribution methods that incorporate similarity measures between
original and perturbed inputs. These attributors are essential for local explanation
methods like LIME and SHAP, where the importance of perturbations is weighted by their
similarity to the original input.

Classes:
    SimilarityAttributor: Generic attributor that combines model predictions with
        configurable similarity functions for flexible local explanation methods.

Functions:
    lime_similarity: Gaussian kernel similarity function for LIME-style explanations.
    kernel_shap_similarity: SHAP kernel similarity function based on coalition size.

"""

#
# Created on Wed Nov 29 2023
#
# Copyright (c) 2023 Jérémy Rozier @Euranova
#

import math
from typing import Callable

import torch
import torch.nn.functional as F

from muppet.components.attributor.base import Attributor
from muppet.components.memory.base import Memory, Premise


class SimilarityAttributor(Attributor):
    """Attribution based on similarity measures between original and perturbed inputs.

    This attributor calculates similarities relative to a provided similarity function.
    The similarity function returns high values when inputs are highly different, making
    it suitable for LIME-style explanations where we need to weight samples by their
    distance from the original input.

    Similarity-based attribution combines two key components:
    1. **Model response**: How the model's prediction changes with perturbation
    2. **Input similarity**: How similar the perturbation is to the original input

    The SimilarityAttributor stores both values:
    ```python
    premise.attribution = {
        "attribution": model_prediction_change,
        "similarity": similarity_score
    }
    ```

    **LIME Similarity**: Uses Gaussian kernel with Euclidean distance:
    ```
    similarity = exp(-distance²/σ²)
    ```

    **SHAP Kernel**: Based on coalition size with theoretical guarantees:
    ```
    weight = (M-1) / (C(M,|S|) × |S| × (M-|S|))
    ```
    Where M is total features and |S| is coalition size.

    **Dice Score**: For segmentation, measures overlap between predictions:
    ```
    Dice = 2×|intersection| / (|pred| + |true|)
    ```

    These methods are particularly effective for:
    - **Local explanations**: LIME and SHAP-style interpretability
    - **Faithful approximations**: Ensuring explanations reflect local model behavior
    - **Segmentation analysis**: Understanding model performance on different regions
    - **Coalition-based methods**: Game-theoretic explanation approaches

    Attributes:
        predicted_class: The predicted class from the original input.
        similarity_fun: The similarity function used for calculations.
        convention: The attribution convention (perturbed_input_similarity).

    Example:
        Using LIME-style similarity weighting:

        ```python
        # Initialize with LIME similarity function
        attributor = SimilarityAttributor(similarity_fun=lime_similarity)

        # Use in LIME explainer
        explainer = LIMEExplainer(
            model=image_classifier,
            attributor=attributor,
            # ... other components
        )

        explanation = explainer.explain(image_tensor)
        ```
    """

    def __init__(
        self,
        similarity_fun: Callable[
            [torch.Tensor, torch.Tensor, "Premise"], torch.Tensor
        ],
    ) -> None:
        """Initialize the SimilarityAttributor.

        Args:
            similarity_fun: Function that calculates similarity between original and
                perturbed inputs. Takes (original_tensor, perturbed_tensor, premise)
                and returns similarity scores. Higher values indicate greater difference.
        """
        self.similarity_fun = similarity_fun
        self.predicted_class = None

        # similarity function return high values when input are highly different
        self.convention = "perturbed_input_similarity"
        super().__init__()

    def calculate_attribution(
        self,
        x: torch.Tensor,
        perturbed_inputs: torch.Tensor,
        model: torch.nn.Module,
        memory: Memory,
    ) -> None:
        """Calculates the attribution of perturbed inputs.

        Args:
            - x (torch.Tensor): Example to explain. (b=1, *x.shape[1:]).
            - perturbed_inputs (torch.Tensor): The example's perturbations. Shape (N, *x.shape).
            - model (torch.nn.Module): Given model we want to explain.
            - memory (Memory): Memory where the premises are stored.

        Where b is the batch size (=1), N is the number of generated masks.
        """
        perturbed_inputs_vectors = perturbed_inputs.view(
            perturbed_inputs.shape[0], -1
        )
        x_vector = x.view(1, -1)

        # Calculate the true class if not already done
        if self.predicted_class is None:
            with torch.no_grad():
                true_output = F.softmax(
                    model(x).detach(), dim=1
                )  # (b, nclasses)
            self.predicted_class = torch.argmax(true_output, dim=1)  # (b=1)

        # Calculate the true class prediction of every perturbation
        for idx, premise in enumerate(memory.get_premises()):
            with torch.no_grad():
                logits = model(perturbed_inputs[idx].float()).detach()
            probs = F.softmax(logits, dim=1)  # (b=1, nclasses)

            final = probs[:, self.predicted_class].squeeze(dim=-1)

            # Compute similarity for the current premise
            similarities = self.similarity_fun(
                x_vector, perturbed_inputs_vectors, premise
            )

            if (
                similarities.shape[0] <= 1
            ):  # Shape is [1] or [] indicating scalar
                similarity = similarities
            else:  # Assume it's a vector, and we need to index it
                similarity = similarities[idx]

            premise.attribution = {
                "attribution": final,
                "similarity": similarity.item(),  # Ensure it's a scalar for consistency
            }

        return


def lime_similarity(
    x_vector: torch.Tensor, perturbed_vector: torch.Tensor, premise: "Premise"
) -> torch.Tensor:
    """Example LIME similarity function using a Gaussian kernel for LIME method.
    This function computes similarities for all perturbations at once.
    """
    distances = torch.cdist(x_vector, perturbed_vector).view(-1)
    min_value = distances.min()
    max_value = distances.max()

    if min_value == max_value:
        distances = distances * 0
    else:
        distances = (distances - min_value) / (max_value - min_value)

    kernel = torch.sqrt(torch.exp(-(distances**2) / 0.25**2))
    return kernel


def kernel_shap_similarity(
    x_vector: torch.Tensor, perturbed_vector: torch.Tensor, premise: "Premise"
) -> torch.Tensor:
    """Calculates similarity based on kernel SHAP.

    Args:
        x_vector (torch.Tensor): The original input tensor of shape (1, N).
        perturbed_vector (torch.Tensor): The perturbed input tensor of shape (1, N).
        premise (Premise): The premise object containing the key tensor of shape (1, f).

    Returns:
        torch.Tensor: A tensor of shape [1] containing the similarity score.
    """
    N = x_vector.shape[1]  # Total number of features
    key_tensor = premise.key.squeeze(0)  # Convert (1, f) to (f,)
    S = (key_tensor == 1).sum().item()  # Number of 1 features in the key

    # Handle case where S is 0 (totally perturbed vector, no similarity)
    if S == 0:
        return torch.tensor([0.0], dtype=torch.float32)

    # Handle case where S equals N (no perturbation, full similarity)
    if S == N:
        return torch.tensor([1.0], dtype=torch.float32)

    # Calculate the combinatorial factor using the formula:
    binomial_coefficient = math.factorial(N) / (
        math.factorial(S) * math.factorial(N - S)
    )
    similarity = (N - 1) / (binomial_coefficient * S * (N - S))

    # Return similarity as a tensor with shape [1]
    return torch.tensor([similarity], dtype=torch.float32)
