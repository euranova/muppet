"""Mask-based Explorer Components for Perturbation Strategies.

This module provides a set of explorer implementations that generate
various types of perturbation masks for the MUPPET XAI framework. These explorers
implement different sampling and masking strategies to systematically probe model
behavior through diverse perturbation patterns, supporting multiple data modalities
and explanation approaches.

The mask-based exploration strategies generate sets of perturbation
masks using random sampling, segmentation or distribution-based methods.
These approaches provide broad coverage of the input space to understand model
sensitivity across different regions or features.

Classes:
    RandomMasksExplorer: Generates random binary masks for spatial image perturbations
        with configurable mask density and grid resolution.
    SegmentedBinaryRandomMasksExplorer: Creates segment-based random masks using
        superpixel segmentation (SLIC) for semantically meaningful image regions.
    RandomNormalExplorer: Generates masks from normal distribution for continuous
        perturbations, particularly suitable for tabular data.
    BinaryFeaturePermutationsExplorer: Enumerates all possible binary feature
        combinations (coalitions) for exhaustive tabular data analysis.

The mask exploration process:
    1. **Generate**: Create mask patterns based on the specific strategy
    2. **Sample**: Apply probabilistic or deterministic sampling rules
    3. **Package**: Wrap masks in premises with necessary metadata
    4. **Batch**: Return complete set of masks for single-iteration exploration

Technical Details:
    - **Random Masks**: Generate binary masks with configurable grid size and density
    - **Segmented Masks**: Use SLIC superpixel segmentation for semantic coherence
    - **Normal Masks**: Sample from Gaussian distribution for continuous perturbations
    - **Permutation Masks**: Enumerate all 2^n feature combinations with limits
    - **Reproducible**: Seed-based random generation for consistent results

"""
#
# Created on Fri Jun 09 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import itertools
import random
from typing import List

import torch
from skimage.segmentation import slic

from muppet.components.explorer.base import Explorer
from muppet.components.memory import BinaryRandomPremise, KeyBasedMaskPremise
from muppet.components.memory.premise import SegmentedBinaryImagePremise


class RandomMasksExplorer(Explorer):
    """Random mask explorer for general perturbation-based explanations.

    Generates a specified number of random binary masks for perturbing
    input examples. Each mask is devrived by upscaling a randomly sampled
    binary grid.

    Technical Details:
        - **Random Masks**: Generate binary masks with configurable grid size and density
        - **Reproducible**: Seed-based random generation for consistent results

    """

    def __init__(
        self,
        nmasks: int = 800,
        mask_dim: int | tuple[int, int] = 7,
        mask_proba: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the RandomMasksExplorer.

        Args:
            nmasks (int): Number of random masks to generate. Defaults to 800.
            mask_dim (int | tuple[int, int]): Grid size for base mask. Defaults to 7.
            mask_proba (float): Probability of masking each cell. Defaults to 0.5.
            seed (int | None): Random seed for reproducibility. Defaults to None.
        """
        self.nmasks = nmasks
        self.mask_dim = mask_dim
        self.mask_proba = mask_proba
        self.seed = seed

        super().__init__()

    def get_premises_to_explore(self) -> List[BinaryRandomPremise]:
        """Generate a `nmasks` number of premises where every one corresponds to the perturbation of the input example.
        Expects 4D input example. Shape (b=1, **, h, w).

        ** can be anything, either no dimension or an arbitrary number of dimensions. Usually it will be 1 dimension, the channel, so shape
        is (b=1,c,h,w).

        where
            - b is batch dimension, expected to be set to 1 as only one example is being explained for the moment,
            - c is the channel dimensions,
            - w is the width,
            - h is the height,

        Returns:
                List[BinaryRandomPremise]: Every premise includes the necessary information to generate its random mask from the key attribute.
        """
        mask_shape = (self.example.shape[-2], self.example.shape[-1])

        premises = []
        for i in range(self.nmasks):
            seed = self.seed
            if seed is not None:
                seed = seed + i
            premise = BinaryRandomPremise(
                key=(self.mask_dim, self.mask_proba, mask_shape),
                seed=seed,
                **self.premise_kwargs,
            )
            premises.append(premise)

        # tell the main explainer to stop the exploration
        self.stop = True

        return premises


class SegmentedBinaryRandomMasksExplorer(Explorer):
    """Segmented random mask explorer for image-based explanations.

    Generates random binary masks based on image segmentation. Uses
    superpixel segmentation to create meaningful perturbation regions,
    ensuring that semantically coherent areas are masked together.

    This explorer uses superpixel segmentation (SLIC) to create semantically meaningful
    image regions for perturbation. Instead of using random pixel-based masks, it
    preserves object boundaries and creates coherent masked regions that respect
    the natural structure of the image.

    Technical Details:
        - **Segmented Masks**: Use SLIC superpixel segmentation for semantic coherence
        - **Spatial coherence**: Segmentation-based masks preserve object boundaries
        - **Semantic Meaningful**: Masks respect natural image structure

    """

    def __init__(
        self,
        nmasks: int = 500,
        masked_proba: float = 0.5,
        n_segments: int = 100,
    ) -> None:
        """Initialize the SegmentedBinaryRandomMasksExplorer.

        Args:
            nmasks (int): Number of random masks to generate. Defaults to 500.
            masked_proba (float): Probability of masking each superpixel. Defaults to 0.5.
            n_segments (int): Approximate number of superpixels. Defaults to 100.
        """
        self.nmasks = nmasks
        self.mask_proba = masked_proba
        self.n_segments = n_segments

        super().__init__()

    def get_premises_to_explore(self) -> List[SegmentedBinaryImagePremise]:
        """Generate a `nmasks` number of premises where every one corresponds to the perturbation of the input example.
        Expects 4D input example. Shape (b=1, c, h, w).

        where
            - b is batch dimension, expected to be set to 1 as only one example is being explained for the moment,
            - c is the channel dimensions,
            - w is the width,
            - h is the height,

        Returns:
            List[BinaryRandomPremise]: Every premise includes the necessary information to generate its random mask from the key attribute.

        """
        segmented_example = self.get_segmented_tensor_from_example()

        # Add the segmented example to the premise_kwargs ; this way it will be
        # passed to every Premise at its creation and memorized by them as an attribute
        self.premise_kwargs["segmented_example"] = segmented_example

        premises = []
        binary_matrix = (
            torch.rand(self.nmasks, segmented_example.shape[0])
            < self.mask_proba
        ).long()  # shape of binary_matrix = (nmasks, s)

        for i in range(self.nmasks):
            binary_vector = binary_matrix[i]
            premise = SegmentedBinaryImagePremise(
                key=binary_vector,
                **self.premise_kwargs,
            )
            premises.append(premise)

        # tell the main explainer to stop the exploration
        self.stop = True

        return premises

    def get_segmented_tensor_from_example(self):
        """Static method to get the segmented tensor from an array of labels.

        Args:
            example (torch.Tensor): Image to explain with shape (b=1, c, h, w).
            n_segments(int): The (approximate) number of labels in the segmented output image.

        Returns:
            tensor: segmented example, tensor of shape (approaches value of n_segments, h, w) in which each slice of shape (h, w)
                contains 1 in the area of the superpixel and 0 elsewhere.
        """
        _, _, h, w = self.example.shape

        example_np = self.example[0].permute(1, 2, 0).cpu()
        labels = torch.from_numpy(
            slic(
                example_np,
                n_segments=self.n_segments,
            )
        )

        unique_labels = torch.unique(labels)
        segmented_example = labels.unsqueeze(dim=0).repeat(
            len(unique_labels), 1, 1
        )
        indexes = (
            torch.arange(1, len(unique_labels) + 1)
            .unsqueeze(dim=1)
            .unsqueeze(dim=2)
            .repeat(1, h, w)
        ).cpu()

        # Compare each element of segmented_example with indexes
        # If they match, set the value to 1; otherwise, set it to 0
        segmented_example = torch.where(segmented_example == indexes, 1, 0)

        return segmented_example


class RandomNormalExplorer(Explorer):
    """Random normal distribution explorer for continuous perturbations.

    Generates perturbation premises using random values sampled from a
    normal distribution. Provides continuous perturbations instead of
    binary masks, useful for tabular data where smooth
    noise-based modifications are preferred over binary masking.

    Technical Details:
        - **Normal Masks**: Sample from Gaussian distribution for continuous perturbations
        - **Continuous perturbation**: Smooth noise-based modifications
        - **Reproducible**: Seed-based sampling for consistent results

    Note:
        This explorer is ideal for tabular data or scenarios where continuous
        perturbations are more appropriate than binary masking strategies.
    """

    def __init__(self, nmasks: int = 800, seed: int = 1) -> None:
        """Initialize the RandomNormalExplorer.

        Args:
            nmasks (int): Number of random masks to generate. Defaults to 800.
            seed (int): Random seed for reproducibility. Defaults to 1.
        """
        self.nmasks = nmasks
        self.seed = seed
        self.stop = False
        self.current_iteration = 0
        super().__init__()

    def get_random_normal_key(self, seed: int) -> torch.Tensor:
        """Generates a random vector (key) based on a normal distribution centered at zero.
        This vector is used as a key for the premise.

        Args:
            seed (int): Seed for random number generation to ensure reproducibility.

        Returns:
            torch.Tensor: A random mask (key) generated from the normal distribution.
            Shape is (nmasks, *x.shape), where `x.shape` is the input feature dimensions.
        """
        torch.manual_seed(seed)  # Set seed for reproducibility
        mask_shape = (
            self.example.shape[1],
        )  # Shape based on input feature size
        key = torch.randn(
            mask_shape, device=self.device
        )  # Random key generated
        return key

    def get_premises_to_explore(self) -> List[KeyBasedMaskPremise]:
        """Generates the list of premises to be explored. Each premise is created by generating
        a random key for perturbation using a unique seed.

        Returns:
            List[KeyBasedMaskPremise]: A list of premises to explore. Each premise contains a key and seed.
        """
        premises = []
        for i in range(self.nmasks):
            current_seed = self.seed + i  # Ensure each seed is unique
            key = self.get_random_normal_key(current_seed)  # Generate a key
            premise = KeyBasedMaskPremise(
                key=key, seed=current_seed
            )  # Create a premise with the key and seed
            premises.append(premise)

        # Indicate that exploration is complete
        self.stop = True

        return premises


class BinaryFeaturePermutationsExplorer(Explorer):
    """Binary feature permutation explorer for combinatorial explanations.

    Generates all possible binary feature combinations (coalitions) for
    systematic exploration of feature interactions. Useful for kernel-SHAP-like
    explanations where all feature subsets should to be evaluated.

    This explorer enumerates all possible binary feature combinations (coalitions)
    for exhaustive tabular data analysis. It systematically explores every possible
    subset of features to understand their individual and collective contributions
    to model predictions.

    Technical Details:
        - **Permutation Masks**: Enumerate all 2^n feature combinations with
          configurable limit, permutation are randomly sampled if the possible
          number of permutation exceed the limit.

    Note:
        This explorer is ideal for tabular data with manageable feature counts.
        For datasets with many features, the number of combinations grows
        exponentially (2^n), so max_permutations helps limit computational cost.
    """

    def __init__(
        self,
        n_repeats: int = 1,
        seed: int | None = None,
        max_permutations: int = 900,
    ) -> None:
        """Initialize the BinaryFeaturePermutationsExplorer.

        Args:
            n_repeats (int): Number of times to repeat each permutation. Defaults to 1.
            seed (int | None): Random seed for reproducibility. Defaults to None.
            max_permutations (int): Maximum number of permutations. Defaults to 900.
        """
        self.n_repeats = n_repeats
        self.seed = seed
        self.max_permutations = max_permutations

        super().__init__()

    def _generate_permutations(self) -> List[torch.Tensor]:
        """Generates all possible binary feature permutations for the given number of features
        and converts them into PyTorch tensors.

        Returns:
            List[torch.Tensor]: List of PyTorch tensors where each tensor represents a binary mask.
        """
        num_features = self.example.shape[1]

        # Generate all binary permutations (coalitions) for the given number of features
        permutations = list(itertools.product([0, 1], repeat=num_features))

        # Convert each permutation into a PyTorch tensor
        perm_tensors = [
            torch.tensor(perm, dtype=torch.float32) for perm in permutations
        ]

        return perm_tensors

    def get_premises_to_explore(self) -> List[KeyBasedMaskPremise]:
        """Generates all possible binary feature permutations for the given number of features.
        Limits the total number of permutations to `max_permutations`.

        Returns:
            List[BinaryPremise]: Each premise includes the necessary information (binary mask)
            to perform perturbations on the input example.
        """
        # Generate the permutations as PyTorch tensors
        perm_tensors = self._generate_permutations()

        # Total number of possible permutations without repetitions
        num_permutations = len(perm_tensors)

        if num_permutations >= self.max_permutations:
            # Case 1: The number of available permutations is sufficient
            random.seed(self.seed)
            perm_tensors = random.sample(perm_tensors, self.max_permutations)
        else:
            # Case 2: The number of permutations is insufficient, supplement with repeats
            total_needed = self.max_permutations
            perm_tensors = perm_tensors * self.n_repeats  # Apply the repeats

            # Select a subset if necessary after supplementing with repeats
            if len(perm_tensors) > total_needed:
                perm_tensors = perm_tensors[:total_needed]

        # Initialize the list of premises
        premises = []
        for i, perm_tensor in enumerate(perm_tensors):
            seed = self.seed
            if seed is not None:
                seed = seed + i

            # Create a BinaryPremise with the generated binary mask as a PyTorch tensor
            premise = KeyBasedMaskPremise(
                key=perm_tensor,
                seed=seed,
                **self.premise_kwargs,  # Additional arguments specific to the premise
            )
            premises.append(premise)

        # Signal that the exploration is complete
        self.stop = True

        return premises
