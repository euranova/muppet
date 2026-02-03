"""Embedding-based attributors for MUPPET XAI.

This module provides attribution methods that work with embedding spaces and latent
representations. These attributors are designed for models that output vector embeddings
rather than discrete classifications, making them ideal for explaining representation
learning models, autoencoders, and embedding-based systems.

Classes:
    EmbeddingDistanceAttributor: Calculates attributions based on the L2 distance between
        original and perturbed embeddings, measuring how much each perturbation changes
        the model's internal representation of the input.
    DiceScoreAttributor: Specialized attributor for semantic segmentation task that uses
        Dice coefficient to measure segmentation quality changes caused by perturbations
        instead of the Lé distance of the vector embedding
"""

#
# Created on Tue Jul 25 2023
#
# Copyright (c) 2023 Quentin Ferré & Ismail Bahchar @Euranova
#

import torch
import torch.nn.functional as F

from muppet.components.attributor.base import Attributor
from muppet.components.memory.base import Memory, PremiseList


class EmbeddingDistanceAttributor(Attributor):
    """Attribution based on distance between original and perturbed embeddings.

    A perturbation's value is equal to how much it changed the original embedding
    (i.e., original model output). The end goal is to find perturbations that make
    the perturbed embedding as far away from the original embedding as possible.

    This attributor measures how perturbations affect the model's embedding representations
    by calculating L2 distances between original and perturbed embeddings. The model output
    is expected to have shape (batch, **embedding_dim).

    The `EmbeddingDistanceAttributor` computes attributions by:
    1. **Reference computation**: Computing the original input's embedding E₀ = model(x)
    2. **Distance measurement**: For each perturbation xᵢ, computing Eᵢ = model(xᵢ)
    3. **Attribution scoring**: Computing distance d(E₀, Eᵢ) = ||E₀ - Eᵢ||₂
    4. **Sign adjustment**: Applying negative sign to maximize distance (destructive convention)

    The L2 distance provides a natural measure of representation change:
    ```
    Attribution = -||embedding_original - embedding_perturbed||₂
    ```

    The method works with any model that outputs continuous vector representations,
    regardless of the embedding dimensionality or architecture (CNNs, transformers,
    autoencoders, etc.).

    Attributes:
        input_embedding: Stores the true embedding output for comparison.

    """

    def __init__(self) -> None:
        """Initialize the EmbeddingDistanceAttributor."""
        self.convention = "destructive"

        self.input_embedding = None  # Will be used to store true output later
        super().__init__()

    def calculate_attribution(
        self,
        x: torch.Tensor,
        perturbed_inputs: torch.Tensor,
        model: torch.nn.Module,
        memory: Memory,
    ) -> None:
        """Calculate the L2 distance as the attribution between x and its perturbations.
        Note that the expected shape for x and perturbed_inputs (1 for batch, nb_rows, embedding_dim).

        Args:
            x (torch.Tensor): The input example to be explained.

            perturbed_inputs (torch.Tensor): The calculated perturbations by the Perturbator.

            model (torch.nn.Module): The black-box model.

            memory (Memory, optional): The simple list memory structure.

        """
        # Calculate the original example's embedding if not already done
        if self.input_embedding is None:
            with torch.no_grad():
                self.input_embedding = model(x).detach()

        # For each premise we currently focus on in this step,
        # save the distance btwn example and its perturbed version
        for idx, premise in enumerate(memory.get_premises()):
            input_reshaped = perturbed_inputs[
                idx
            ].float()  # perturbed_inputs (N, **x.shape)

            embedding = model(
                input_reshaped
            )  # .detach() # Careful not to detach

            dist = self.similarity(embedding, self.input_embedding)
            premise.attribution = dist

        return

    # -------------------------- Loss components ----------------------------- #

    def similarity(self, embedding, true_embedding):
        """Calculate similarity between perturbed and original embeddings.

        Computes the negative L2 distance between embeddings to maximize
        the distance (higher score for more different embeddings).

        Args:
            embedding (torch.Tensor): The perturbed embedding.
            true_embedding (torch.Tensor): The original input embedding.

        Returns:
            torch.Tensor: Negative L2 distance (higher values indicate more difference).
        """
        assert true_embedding.shape == embedding.shape, (
            f"True embedding and embedding must have the same shape. Instead we got {true_embedding.shape} and {embedding.shape}"
        )

        inaccuracy_term = self._compo_similarity(
            embedding.reshape(embedding.shape[0], -1),
            true_embedding.reshape(true_embedding.shape[0], -1),  # (b, 1)
        )

        final_dist = (
            -1 * inaccuracy_term
        )  # We want to MAXIMIZE the distance, so multiply inaccuracy term by -1 !
        return final_dist

    @staticmethod
    def _compo_similarity(prediction: torch.Tensor, original: torch.Tensor):
        """Math formula of the similarity used to compare the two embeddings.
        Simply the Euclidian distance.

        Args:
            prediction (torch.Tensor): Perturbation embedding
            original (torch.Tensor): Original input embedding

        Returns:
            torch.Tensor: The L2 loss between the two embeddings.

        """
        error = torch.square(original - prediction)
        final = torch.mean(error, dim=-1)  # Don't mean on the batch axis
        return final


class DiceScoreAttributor(Attributor):
    """Attribution based on Dice score between probability distributions.

    This attributor calculates the Dice score between the predicted probability
    distribution of a perturbed input and the original example's output. The Dice
    score measures the overlap between the two distributions, providing a similarity
    measure for classification outputs.

    This attributor is specifically designed for segmentation tasks where it measures
    how perturbations affect segmentation quality by calculating Dice coefficient
    changes between original and perturbed predictions.

    Attributes:
        true_class: The true class index calculated from the original input.

    """

    def __init__(self) -> None:
        """Initialize the DiceScoreAttributor.

        Inferit from `Attributor` with `true_class` is initiated to None
        """
        self.true_class = None
        super().__init__()

    def reinitialize(self):
        """Reset the attributor to its initial state.

        Clears the cached true class to ensure fresh calculations
        for new inputs.
        """
        self.true_class = None
        return super().reinitialize()

    def calculate_attribution(
        self,
        x: torch.Tensor,
        perturbed_inputs: torch.Tensor,
        model: torch.nn.Module,
        memory: PremiseList,
    ) -> None:
        """Calculates the attribution (Dice score) for perturbed inputs.

        Args:
            x (torch.Tensor): Example input. Shape: (1, C, H, W)
            perturbed_inputs (torch.Tensor): Perturbed inputs. Shape: (N, 1, C, H, W)
            model (torch.nn.Module): Model to explain.
            memory (PremiseList): Memory structure to attach attributions to.
        """
        if self.true_class is None:
            with torch.no_grad():
                reference_output = model(x)  # (1, C, H, W)
                true_class = torch.argmax(reference_output, dim=1)  # (1, H, W)
                true_one_hot = (
                    F.one_hot(true_class, num_classes=reference_output.shape[1])
                    .permute(0, 3, 1, 2)
                    .float()
                )  # (1, 21, H, W)

        for idx, premise in enumerate(memory.get_premises()):
            with torch.no_grad():
                logits = model(perturbed_inputs[idx].float())  # (1, C, H, W)
                pred_class = torch.argmax(logits, dim=1)  # (1, H, W)
                pred_one_hot = (
                    F.one_hot(pred_class, num_classes=reference_output.shape[1])
                    .permute(0, 3, 1, 2)
                    .float()
                )  # (1, 21, H, W)

            intersection = (true_one_hot * pred_one_hot).sum(
                dim=[0, 2, 3]
            )  # sum over batch & spatial dims => (21,)
            union = true_one_hot.sum(dim=[0, 2, 3]) + pred_one_hot.sum(
                dim=[0, 2, 3]
            )  # (21,)

            perturbation_importance = (
                2 * (intersection) / (union + 10e-12)
            )  # (21,)
            attribution = (
                1 - perturbation_importance[1:].mean()
            )  # average over classes

            premise.attribution = attribution.unsqueeze(0)

        return
