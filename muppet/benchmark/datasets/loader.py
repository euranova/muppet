"""Dataset loading utilities for the MUPPET benchmark framework.

This module provides functions for loading datasets from various sources including
OpenML, UCI ML Repository, local files, and specialized datasets like ImageNet.
It includes utilities for data preprocessing, missing value handling, and format
conversion to support the benchmark framework.

Functions:
    load_tabular_dataset: Load tabular datasets from OpenML, UCI ML, or local files
    download_imagenet: Download and preprocess ImageNet samples
    load_pickle_dataset: Load datasets from pickle files (for time series data)
"""

import json
import os
import pickle
from pathlib import Path
from typing import Union

import fiftyone.zoo as foz
import openml
import pandas as pd
import torch
from PIL import Image
from ucimlrepo import fetch_ucirepo

from muppet import logger


def load_tabular_dataset(
    dataset_param: Union[int, str],
    target_column: str | None = None,
    source: str = "openml",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a dataset based on the specified source: 'openml', 'uciml', or local 'file'.

    :param dataset_param:
        - If source='openml': OpenML dataset ID (int) or dataset name (str).
        - If source='uciml': UCIML dataset ID (int) or dataset name (str).
        - If source='file': File path (string).
    :param target_column: Column name to use as the target.
        - For OpenML/UCIML, if None, will attempt to use the dataset's default or a single target column.
        - For 'file', this must be specified.
    :param source: 'openml', 'uciml', or 'file'.
    :return: Tuple (X, y), where X is the feature DataFrame and y is the target Series.
    """

    def load_from_openml(
        identifier: Union[int, str], target_column: str | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Helper function to load a dataset from OpenML.
        :param identifier: OpenML dataset ID (int) or name (str).
        :param target_column: Target column name (optional; uses default-target if None).
        :return: Tuple (X, y).
        """
        logger.info(f"Fetching dataset from OpenML: {identifier}")
        dataset = openml.datasets.get_dataset(identifier)
        target_attr = target_column or dataset.default_target_attribute

        if not target_attr:
            logger.warning(
                f"Dataset {identifier} has no default target attribute."
            )
            df, _, categorical_mask, _ = dataset.get_data(
                dataset_format="dataframe"
            )
            if target_column is None:
                # Try to infer a single categorical column as target
                inferred_target = [
                    col
                    for col, is_cat in zip(df.columns, categorical_mask)
                    if is_cat
                ]
                if len(inferred_target) == 1:
                    target_attr = inferred_target[0]
                    logger.info(f"Inferred target column: {target_attr}")
                else:
                    raise ValueError(
                        f"Dataset {identifier} does not have a clear target column. "
                        f"Please specify `target_column`. "
                        f"Available columns: {df.columns.tolist()}"
                    )
            elif target_column not in df.columns:
                raise ValueError(
                    f"Provided target column '{target_column}' not found in dataset {identifier}. "
                    f"Available columns: {df.columns.tolist()}"
                )
            else:
                target_attr = target_column

        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe", target=target_attr
        )
        return X, y

    def load_from_uciml(
        identifier: Union[int, str], target_column: str | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Helper function to load a dataset from the UCIML repository using ucimlrepo.
        :param identifier: UCIML dataset ID (int) or dataset name (str).
        :param target_column: Target column name (optional; uses a single target column by default if only one exists).
        :return: Tuple (X, y).
        """
        logger.info(f"Fetching dataset from UCIML: {identifier}")

        # Decide whether to fetch by ID or by name
        if isinstance(identifier, int):
            dataset = fetch_ucirepo(id=identifier)
        else:
            dataset = fetch_ucirepo(name=str(identifier))

        # The ucimlrepo object typically exposes:
        #   dataset.data.features   -> pd.DataFrame of features
        #   dataset.data.targets    -> pd.DataFrame of targets (can be multiple columns)
        X = dataset.data.features
        y_all = dataset.data.targets

        # If there is exactly one target column, we can treat that as default
        if y_all.shape[1] == 1:
            default_target = y_all.columns[0]
        else:
            default_target = None

        if target_column is None:
            # If there's only one target column, use it
            if default_target is not None:
                y = y_all[default_target]
                logger.info(f"Using default target column: {default_target}")
            else:
                raise ValueError(
                    f"Dataset {identifier} has multiple target columns. "
                    f"Please specify `target_column`. "
                    f"Available target columns: {y_all.columns.tolist()}"
                )
        else:
            # If the user gave a specific target column
            if target_column in y_all.columns:
                y = y_all[target_column]
            elif target_column in X.columns:
                # If it's actually among the features, we move that column from X to y
                y = X[target_column]
                X = X.drop(columns=[target_column])
            else:
                raise ValueError(
                    f"Provided target column '{target_column}' not found in dataset {identifier}. "
                    f"Available feature columns: {X.columns.tolist()}, "
                    f"target columns: {y_all.columns.tolist()}"
                )

        return X, y

    def load_from_file(
        file_path: str, target_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Helper function to load a dataset from a local file (CSV).
        :param file_path: Path to the dataset file.
        :param target_column: Name of the target column in the dataset.
        :return: Tuple (X, y).
        """
        logger.info(f"Loading dataset from file: {file_path}")
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(
                "Unsupported file format. Only CSV files are supported."
            )

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in the dataset."
            )
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    # Main function logic
    source = source.lower()
    if source == "openml":
        X, y = load_from_openml(dataset_param, target_column)
    elif source == "uciml":
        X, y = load_from_uciml(dataset_param, target_column)
    elif source == "file":
        if not target_column:
            raise ValueError(
                "Target column must be specified when loading from a local file."
            )
        X, y = load_from_file(dataset_param, target_column)
    else:
        raise ValueError(
            f"Invalid source '{source}'. Expected 'openml', 'uciml', or 'file'."
        )

    if X is None or y is None:
        raise ValueError(
            "Failed to load dataset. Features or target is missing."
        )

    logger.info(
        f"Dataset loaded successfully. Features shape: {X.shape}, Target length: {len(y)}"
    )
    return X, y


def download_imagenet(
    path: str,
    num_images: int = 1000,
    image_size: tuple[int, int] = (224, 224),
) -> str:
    """Download imagenet samples

    Args:
        path (str): The path to check and populate with the dataset if empty or insufficient.
        num_images (int): The number of images to load into the dataset.
        image_size (tuple[int, int]): The size to which each image should be resized (default is 224x224).

    Note:
        The dataset loaded is from ImageNet 2012 and is available via the FiftyOneDataset API:
        https://docs.voxel51.com/api/fiftyone.zoo.datasets.base.html#fiftyone.zoo.datasets.base.FiftyOneDataset
    """
    # Check if the directory exists, if not create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Check the number of JPEG or JPG images in the directory
    jpeg_files = [
        f for f in os.listdir(path) if f.endswith(".jpeg") or f.endswith(".jpg")
    ]
    if len(jpeg_files) < num_images:
        logger.info(
            f"Insufficient JPEG or JPG images found in {path}. Loading dataset..."
        )

        # Load the ImageNet sample dataset
        dataset = foz.load_zoo_dataset("imagenet-sample")

        # Limit the dataset to the required number of images
        samples = dataset.take(num_images)

        # Iterate over the samples and save them as JPG images in the specified path
        for sample in samples:
            image_path = sample.filepath
            image = Image.open(image_path).convert(
                "RGB"
            )  # Ensure image is in RGB format
            image = image.resize(
                image_size
            )  # Resize image to the specified size
            save_path = os.path.join(path, os.path.basename(image_path))
            save_path = (
                os.path.splitext(save_path)[0] + ".jpg"
            )  # Ensure file extension is .jpg
            image.save(save_path, "JPEG")

        logger.info(f"Dataset loaded and {num_images} images saved in {path}")
    else:
        logger.info(f"Sufficient JPEG or JPG images already exist in {path}")

    return path


def load_pickle_dataset(
    folder_path: str,
    nb_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper function to load a dataset from a local file (CSV).
    :param file_path: Path to the dataset file.
    :param target_column: Name of the target column in the dataset.
    :return: Tuple (X, y).
    """
    logger.info(
        f"Loading dataset from folder: {folder_path}, \
        the folder is assumed to contain x_test.pkl and y_test.pkl files"
    )

    with open(os.path.join(folder_path, "x_test.pkl"), "rb") as f:
        x_test = pickle.load(f)

    with open(os.path.join(folder_path, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)

    X_test_tensor = (
        torch.tensor(x_test, dtype=torch.float32)
        .clone()
        .detach()
        .cpu()
        .squeeze(1)
    )
    y_test_tensor = (
        torch.tensor(y_test, dtype=torch.float32)
        .clone()
        .detach()
        .cpu()
        .squeeze(1)
    )

    if nb_samples is None:
        return X_test_tensor, y_test_tensor[
            :, -1
        ]  # Keep only class at the last sample
    else:
        return X_test_tensor[:nb_samples], y_test_tensor[
            :nb_samples, -1
        ]  # Keep only class at the last sample


def load_imagenet_labels() -> dict[int, str]:
    """Load imagenet labels from a file.

    Expected loaded data is a list of labels.

    Returns:
        dict[int, str]: a mapping of classes id vs labels
    """
    labels_path = Path(__file__).parent / "imagenet_labels.json"
    with open(labels_path) as file:
        labels = json.load(file)

    return {idx: label for idx, label in enumerate(labels)}
