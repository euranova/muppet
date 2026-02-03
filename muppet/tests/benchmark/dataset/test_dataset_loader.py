"""Tests for dataset loaders in the benchmark framework.

This module contains comprehensive tests for various dataset loading functionalities
in the MUPPET benchmark framework. It validates the correct loading, preprocessing,
and handling of different data types including images (ImageNet), tabular data
(UCI ML Repository, OpenML), and time series data (SPIKE).

The tests cover:
- Image dataset downloading and validation
- Data module creation with various train/test split configurations
- Tabular dataset loading from multiple sources
- Time series dataset handling and tensor format validation
- Error handling for invalid dataset configurations

Each test ensures proper data format, dimensions, and preprocessing pipeline
functionality across different modalities supported by MUPPET.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image


def test_download_imagenet():
    """Test downloading ImageNet validation dataset.

    Verifies that the ImageNet loader correctly downloads a specified number
    of images with proper dimensions to a temporary directory.

    Returns:
        None: Test passes if images are downloaded with correct properties.
    """
    cfg = OmegaConf.load("muppet/benchmark/conf/dataset/imagenet.yaml")
    loader_cfg = cfg.loader
    with TemporaryDirectory() as tmpdir:
        load_module = loader_cfg.pop("_func_")
        load_module = hydra.utils.get_method(load_module)
        loader_cfg.num_images = 5
        loader_cfg.path = tmpdir
        data_dir = load_module(**loader_cfg)
        image_files = [file.as_posix() for file in Path(data_dir).glob("*.jpg")]
        assert len(image_files) == loader_cfg.num_images
        for img_path in image_files:
            assert Image.open(img_path).size == tuple(loader_cfg.image_size)


@pytest.mark.parametrize(
    "test_size",
    [0.5, 5, 20, None],
)
def test_datamodule_image(test_size: int | float):
    """Test image datamodule with various test size configurations.

    Validates that the image datamodule correctly handles different test_size
    parameters including float ratios, integer counts, and None values.

    Args:
        test_size: Size of test dataset as integer count, float ratio (0-1),
            or None for using all data as test set.

    Returns:
        None: Test passes if datamodule splits data correctly.
    """
    cfg = OmegaConf.load("muppet/benchmark/conf/dataset/imagenet.yaml")
    with TemporaryDirectory() as tmpdir:
        cfg.loader.path = tmpdir
        cfg.loader.num_images = 10
        cfg.test_size = test_size
        cfg.dataloader_kwargs.batch_size = 1
        datamodule = hydra.utils.instantiate(cfg)
        if isinstance(test_size, float):
            datamodule.prepare_data()
            assert 0 <= test_size <= 1
            test_size_ = int(test_size * cfg.loader.num_images)
            assert (
                len(datamodule.train_dataset)
                == cfg.loader.num_images - test_size_
            )
            assert len(datamodule.test_dataset) == test_size_
        elif isinstance(test_size, int) and test_size < cfg.loader.num_images:
            datamodule.prepare_data()
            assert (
                len(datamodule.train_dataset)
                == cfg.loader.num_images - test_size
            )
            assert len(datamodule.test_dataset) == test_size
        elif isinstance(test_size, int) and test_size > cfg.loader.num_images:
            with pytest.raises(ValueError):
                datamodule.prepare_data()
            return
        else:
            datamodule.prepare_data()
            assert datamodule.train_loader is None
            assert len(datamodule.test_dataset) == cfg.loader.num_images

        for image, _ in datamodule.test_loader:
            assert isinstance(image, torch.Tensor)
            assert image.shape == torch.Size(
                [
                    cfg.dataloader_kwargs.batch_size,
                    3,
                    *cfg.transform.transforms[0].size,
                ]
            )


@pytest.mark.parametrize(
    "train_size, test_size",
    [(5, 0.5), (3, 5), (5, 5), (50, 5), (7, 5), (None, None)],
)
def test_datamodule_image_train_test_size(train_size, test_size: int | float):
    """Test image datamodule with combined train and test size configurations.

    Validates the interaction between train_size and test_size parameters,
    ensuring proper error handling for invalid combinations and correct
    dataset splits for valid ones.

    Args:
        train_size: Size of training dataset as integer count or None.
        test_size: Size of test dataset as integer count, float ratio, or None.

    Returns:
        None: Test passes if datamodule handles size combinations correctly.
    """
    cfg = OmegaConf.load("muppet/benchmark/conf/dataset/imagenet.yaml")
    with TemporaryDirectory() as tmpdir:
        cfg.loader.path = tmpdir
        cfg.loader.num_images = 10
        cfg.test_size = test_size
        cfg.train_size = train_size
        cfg.dataloader_kwargs.batch_size = 1
        if isinstance(test_size, float) and isinstance(train_size, int):
            with pytest.raises(
                (ValueError, hydra.errors.InstantiationException)
            ):
                datamodule = hydra.utils.instantiate(cfg)
                datamodule.prepare_data()
            return
        elif test_size is None and test_size is None:
            datamodule = hydra.utils.instantiate(cfg)
            datamodule.prepare_data()
            assert len(datamodule.train_dataset) == 0
            assert len(datamodule.test_dataset) == cfg.loader.num_images
        elif train_size + test_size > cfg.loader.num_images:
            with pytest.raises(
                (ValueError, hydra.errors.InstantiationException)
            ):
                datamodule = hydra.utils.instantiate(cfg)
                datamodule.prepare_data()
            return
        else:
            datamodule = hydra.utils.instantiate(cfg)
            datamodule.prepare_data()
            assert (
                len(datamodule.train_dataset) + len(datamodule.test_dataset)
                <= cfg.loader.num_images
            )
            assert (
                len(datamodule.train_dataset) + len(datamodule.test_dataset)
                <= cfg.test_size + cfg.train_size
            )


@pytest.mark.parametrize(
    "dataset_param, target_column, source",
    [
        (602, "Class", "uciml"),
        (61, "class", "openml"),
    ],
)
def test_download_tabular_dataset(dataset_param, target_column, source):
    """Test downloading tabular datasets from different sources.

    Verifies that tabular dataset loaders can successfully download datasets
    from UCI ML Repository and OpenML with correct feature and label formats.

    Args:
        dataset_param: Dataset identifier (ID number) for the source.
        target_column: Name of the column containing target labels.
        source: Data source name ('uciml' or 'openml').

    Returns:
        None: Test passes if dataset is downloaded with proper structure.
    """
    cfg = OmegaConf.load("muppet/benchmark/conf/dataset/tabular_dataset.yaml")
    loader_cfg = cfg.loader
    loader_cfg.dataset_param = dataset_param
    loader_cfg.target_column = target_column
    loader_cfg.source = source
    load_module = loader_cfg.pop("_func_")
    load_module = hydra.utils.get_method(load_module)
    features, labels = load_module(**loader_cfg)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(labels, pd.Series)
    assert len(features) == len(labels)
    assert labels.name == loader_cfg.target_column


@pytest.mark.parametrize(
    "dataset_name, dataset_param, target_column, source, test_size",
    [
        ("dry_beans", 602, "Class", "uciml", 100),
        ("iris", 61, "class", "openml", 0.2),
    ],
)
def test_datamodule_tabular(
    dataset_name, dataset_param, target_column, source, test_size
):
    """Test tabular datamodule functionality with various datasets.

    Validates that tabular datamodules correctly load, preprocess, and
    create data loaders for different tabular datasets with proper
    tensor formats and dimensions.

    Args:
        dataset_name: Human-readable name of the dataset.
        dataset_param: Dataset identifier for the source.
        target_column: Name of the target label column.
        source: Data source ('uciml' or 'openml').
        test_size: Size of test set as integer count or float ratio.

    Returns:
        None: Test passes if datamodule creates proper tensor outputs.
    """
    cfg = OmegaConf.load("muppet/benchmark/conf/dataset/tabular_dataset.yaml")
    cfg.name = dataset_name
    cfg.test_size = test_size
    cfg.loader.dataset_param = dataset_param
    cfg.loader.target_column = target_column
    cfg.loader.source = source
    cfg.dataloader_kwargs.batch_size = 16
    datamodule = hydra.utils.instantiate(cfg)
    datamodule.prepare_data()
    features, labels = next(iter(datamodule.train_loader))
    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert features.ndim == 2
    assert labels.shape == torch.Size([cfg.dataloader_kwargs.batch_size])


def test_load_spike():
    """Test loading the SPIKE neuromorphic time series dataset.

    Verifies that the SPIKE dataset loader correctly loads neuromorphic
    time series data with expected tensor shapes and data types.

    Returns:
        None: Test passes if SPIKE data has correct dimensions.
    """
    cfg = OmegaConf.load("muppet/benchmark/conf/dataset/spike.yaml")
    loader_cfg = cfg.loader
    load_module = loader_cfg.pop("_func_")
    load_module = hydra.utils.get_method(load_module)
    features, labels = load_module(**loader_cfg)
    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert features.shape == torch.Size([2000, 3, 80])
    assert labels.shape == torch.Size([2000])


def test_datamodule_timeseries():
    """Test time series datamodule functionality.

    Validates that time series datamodules correctly prepare datasets
    and create data loaders with proper batch dimensions for temporal data.

    Returns:
        None: Test passes if datamodule produces correct tensor shapes.
    """
    cfg = OmegaConf.load("muppet/benchmark/conf/dataset/spike.yaml")
    datamodule = hydra.utils.instantiate(cfg)
    datamodule.prepare_data()
    assert len(datamodule.test_dataset) == cfg.test_size
    features, labels = next(iter(datamodule.train_loader))
    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert features.shape == torch.Size(
        [cfg.dataloader_kwargs.batch_size, 3, 80]
    )
    assert labels.shape == torch.Size([cfg.dataloader_kwargs.batch_size])
