"""Time series classification dataset loading utilities using the Aeon library.

This module provides functionality to load time series classification datasets
from the  timeseriesclassification.com repository via the Aeon library,
which is a unified interface to various time series datasets commonly used
in machine learning research.

Functions:
    load_aeon_as_tensor: Load and convert Aeon datasets to PyTorch tensors
"""

import torch
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder

from muppet import logger


def load_aeon_as_tensor(dataset_name):
    """Load a timeseries classification dataset, converts it into PyTorch tensors, and optionally splits the data
    into training and testing sets.

    Args:
        dataset_name (str): Name of the dataset to load. Must be a valid dataset from VALID_DATASETS.

    Returns:
        - X_tensor: PyTorch tensor containing the input data.
        - y_tensor: PyTorch tensor containing the labels.

    Raises:
        ValueError: If the specified dataset is not valid.
    """
    # Load the data (X, y)
    try:
        X, y = load_classification(dataset_name)
    except ValueError as exc:
        logger.error(
            f"Error while loading {dataset_name} time series classification dataset using aeon."
        )
        raise exc
    # Convert X directly to a PyTorch tensor (assuming X is already numeric)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Use LabelEncoder to encode y (even if it's numeric, for consistency)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Encode all labels
    y_tensor = torch.tensor(
        y_encoded, dtype=torch.int
    )  # Convert encoded labels to PyTorch tensor

    return X_tensor, y_tensor
