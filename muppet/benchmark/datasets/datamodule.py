"""Data module for handling different data types in the MUPPET benchmark framework.

This module provides generic dataset classes for loading and preprocessing datasets
across multiple modalities (image, tabular, time series) using Hydra-based configuration.

Classes:
    DataModule: Main data module for handling multi-modal datasets
    ImageDataset: PyTorch dataset for image data
    TabularDataset: PyTorch dataset for tabular data
    TimeseriesDataset: PyTorch dataset for time series data
"""

from pathlib import Path
from typing import Any, Literal, Optional

import hydra
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split

from muppet import logger


class DataModule:
    """A generic data module for handling different data types (image, tabular, timeseries),
    using Hydra-based configuration for dynamic data loading and preprocessing.

    This module accepts a function path via Hydra (under '_func_') in the `loader` argument
    to dynamically load data (either a directory or a DataFrame/Series), depending on the data type.
    It supports dynamic data loading, automatic train/test splitting, and provides convenient
    access to benchmark data for evaluation purposes.

    Supported types:
    - image: Expects the loader to return a directory path containing image files.
    - tabular: Expects the loader to return a tuple (features: pd.DataFrame, target: pd.Series).
    - timeseries: Placeholder for future implementation.

    After calling `prepare_data()`, the `train_loader` and `test_loader`
    become available. The `benchmark_data` property provides access to all the input
    data from the `test_loader`, concatenated into a single tensor on the target device.
    This data can be used for benchmarking purposes, such as benchmarking different explainers
    on the entire test set at once.

    Note:
        Accessing `benchmark_data` before calling `prepare_data()` will raise a RuntimeError.
    """

    image_suffixes = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

    def __init__(
        self,
        name,
        type: Literal["image", "tabular", "timeseries"],
        loader: dict[str, Any],
        transform=None,
        test_size: int | float | None = None,
        train_size: Optional[int] = None,
        dataloader_kwargs: dict[str, Any] = {},
        labels_mapping: dict[int, str] | None = None,
    ):
        """Initialize the DataModule instance.

        Args:
            name (str): Name of the dataset.
            type (Literal["image", "tabular", "timeseries"]): Type of data to load.
            loader (dict[str, Any]): A dictionary containing '_func_' key pointing to the
                data loader function as a string path (resolved by Hydra), and additional args.
                This function could perform a download (if data doesn't already exist), or simply
                load data from path (it it already exists).
            transform (callable, optional): Transformation to apply to the data.
                It could be a torch transformation object or sklearn preprocessing object.
            test_size (int|None): Number of data to use as test set.
            train_size (int, optional): Number of data to use as training set. Can only be used
                together with test_size as an integer. When provided, the dataset will first be
                reduced to train_size + test_size total samples, then split according to the
                specified sizes. Both train_size and test_size must be positive integers less
                than the total dataset size, and their sum must not exceed the dataset size.
            dataloader_kwargs (dict[str, Any]): Additional keyword arguments for DataLoader.
        """
        self.name = name
        self.type = type
        self.loader = loader
        self.transform = transform
        self.test_size = test_size
        self.train_size = train_size
        self.dataloader_kwargs = dataloader_kwargs
        self.labels_mapping = labels_mapping

        # Parameter validation
        if train_size is not None:
            if not isinstance(test_size, int):
                raise ValueError(
                    "train_size can only be used when test_size is an integer"
                )
            if not isinstance(train_size, int):
                raise ValueError("train_size must be an integer when provided")
            if train_size <= 0:
                raise ValueError("train_size must be a positive integer")
            if test_size <= 0:
                raise ValueError(
                    "test_size must be a positive integer when train_size is provided"
                )

    def prepare_data(self):
        """Prepares the datasets and initializes train/test DataLoaders based on the type of data.
        Loads the data using the Hydra-specified function from the loader config.
        """
        load_module = self.loader.pop("_func_")
        load_module = hydra.utils.get_method(load_module)
        if self.type == "image":
            # expects the loader to return a directory path containing image files
            data_dir = load_module(**self.loader)
            image_files = [
                file.as_posix()
                for suffix in self.image_suffixes
                for file in Path(data_dir).glob(f"*{suffix}")
            ]
            dataset_full = ImageDataset(image_files, transform=self.transform)
            num_samples = len(image_files)
        elif self.type == "tabular":
            # expects the loader to return a tuple (features: pd.DataFrame, target: pd.Series).
            data = load_module(**self.loader)
            dataset_full = TabularDataset(data, self.transform)
            num_samples = len(data[0])
        elif self.type == "timeseries":
            # expects the loader to return a tuple (features: pd.DataFrame, target: pd.Series).
            data = load_module(**self.loader)
            dataset_full = TimeseriesDataset(data, self.transform)
            num_samples = len(data[0])
        else:
            raise ValueError(
                f"Only `image` or `tabular` is supported. `{self.type}` is given"
            )

        if isinstance(self.test_size, float):
            assert 0 <= self.test_size <= 1, (
                "Float test_size should be set between 0 and 1, and is considered as a ratio"
            )
            self.test_size = int(self.test_size * num_samples)

        # Additional validation now that we know the dataset size
        if self.train_size is not None:
            if self.test_size is None:
                raise ValueError(
                    "If train_size is set, test_size should also be set."
                )
            if self.train_size >= num_samples:
                raise ValueError(
                    "train_size must be less than the total dataset size"
                )
            if self.train_size + self.test_size > num_samples:
                raise ValueError(
                    "train_size + test_size must not exceed the total dataset size"
                )
            if self.test_size >= num_samples:
                raise ValueError(
                    "test_size must be less than the total dataset size when train_size is provided"
                )

        if self.test_size is None:
            self.test_size = num_samples
            logger.info("By default all samples are set as test samples")

        if self.test_size > num_samples:
            raise ValueError(
                "The testsize should be less or equal to the total number of samples"
            )

        # Handle dataset splitting
        if self.train_size is not None:
            # Use single random_split with three parts: train, test, remainder (discarded)
            remainder_size = num_samples - self.train_size - self.test_size
            lengths = [self.train_size, self.test_size, remainder_size]
        else:
            # Original logic for when train_size is not provided
            # all non test sample are considered train samples
            lengths = [num_samples - self.test_size, self.test_size, 0]

        self.train_dataset, self.test_dataset, _ = random_split(
            dataset_full,
            lengths=lengths,
            generator=torch.Generator(
                device=torch.get_default_device()
            ).manual_seed(42),
        )

        if len(self.train_dataset) > 0:
            self.train_loader = DataLoader(
                self.train_dataset,
                drop_last=False,
                shuffle=True,
                **self.dataloader_kwargs,
            )
        else:
            self.train_loader = None

        self.test_loader = DataLoader(
            self.test_dataset,
            drop_last=False,
            shuffle=False,
            **self.dataloader_kwargs,
        )


class ImageDataset(Dataset):
    """PyTorch Dataset for loading image files.
    A simple dataset class for loading images from file paths with optional transformations.
    """

    def __init__(self, image_files, transform=None):
        """Initialize ImageDataset instance.

        Args:
            image_files (List[str]): List of image file paths.
            transform (callable, optional): Transformations to apply to each image.
        """
        super().__init__()
        self.image_files = image_files
        self.image_transform = transform

    def __len__(self):
        """Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, index):
        """Get image and label at the specified index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: (image, label) where image is a PIL Image or tensor and label is a string.
        """
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        return image, "label"  # TODO: return also label from image path (?!)


class TabularDataset(Dataset):
    """PyTorch Dataset for loading tabular data.

    A PyTorch dataset for structured data by processing features and targets,
    applying optional transformations, and handling categorical target encoding.
    """

    def __init__(self, data: tuple[pd.DataFrame, pd.Series], transform=None):
        """Initialize the TabularDataset instance.

        Args:
            data (Tuple[pd.DataFrame, pd.Series]): Tuple containing (features, target) where features is a pandas
                DataFrame with input variables and target is a pandas Series with
                the prediction target values.
            transform (callable, optional): Optional transformation pipeline (e.g., sklearn preprocessors)
                to apply to the features. If provided, will be fitted and applied
                to the feature data.
        """
        super().__init__()
        self.features, self.target = data
        if transform is not None:
            self.features = transform.fit_transform(self.features)

        if not ptypes.is_numeric_dtype(self.target):
            self.target = LabelEncoder().fit_transform(self.target)

        self.features: np.ndarray = np.asarray(self.features.values)
        self.target: np.ndarray = np.asarray(self.target)

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the tabular dataset.
        """
        return len(self.features)

    def __getitem__(self, index):
        """Get features and target at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (features, target) where features are the input features and target is the label.
        """
        return self.features[index], self.target[index]


class TimeseriesDataset(TabularDataset):
    """Dataset class for time series data that extends TabularDataset.

    A specialized dataset implementation for time series data that inherits from
    TabularDataset but adds specific handling for temporal data structures.
    """

    def __init__(self, data: tuple[pd.DataFrame, pd.Series], transform=None):
        """Initialize TimeseriesDataset instance.

        Args:
            data (tuple[pd.DataFrame, pd.Series]): A tuple containing time series features (DataFrame) and target (Series).
            transform (callable, optional): Optional transformations to apply to the features.
        """
        super(Dataset).__init__()
        self.features, self.target = data

        if not ptypes.is_numeric_dtype(self.target):
            self.target = LabelEncoder().fit_transform(self.target)

        self.features: np.ndarray = np.asarray(self.features)
        self.target: np.ndarray = np.asarray(self.target)
