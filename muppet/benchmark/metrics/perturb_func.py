"""Perturbation functions for the MUPPET benchmark framework.

This module provides perturbation functions used in explanation evaluation metrics.
These functions modify input data by adding noise or other transformations at
specific indices, enabling the assessment of explanation quality through
perturbation-based analysis.

Functions:
    multiplicative_noise: Apply multiplicative Gaussian noise to specified array indices
"""

import copy

import numpy as np
from quantus import Sequence, Tuple
from quantus.helpers.utils import (
    expand_indices,
)


def multiplicative_noise(
    arr: np.ndarray,
    indices: Tuple[
        slice, ...
    ],  # Alt. Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes: Sequence[int],
    perturb_mean: float = 0.0,
    perturb_std: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """Perturb by multiplicative noise the input at indices.


    Args:
        arr: np.ndarray
            Array to be perturbed.
        indices: int, sequence, tuple
            Array-like, with a subset shape of arr.
        indexed_axes: sequence
            The dimensions of arr that are indexed.
            These need to be consecutive, and either include the first or last dimension of array.
        perturb_mean (float):
            The mean of the multiplicative noise.
        perturb_std (float):
            The standard of the multiplicative noise.
        kwargs: optional
            Keyword arguments.

    Returns:
        arr_perturbed: np.ndarray
            The array which some of its indices have been perturbed.
    """
    indices = expand_indices(arr, indices, indexed_axes)
    noise = np.random.normal(
        loc=perturb_mean, scale=perturb_std, size=arr.shape
    )

    arr_perturbed = copy.copy(arr)
    arr_perturbed[indices] = (arr_perturbed * (1 + noise))[indices]

    return arr_perturbed
