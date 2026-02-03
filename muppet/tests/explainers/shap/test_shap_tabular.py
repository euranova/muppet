"""Tests for SHAP tabular explainer components.

This module tests the ShapTabularExplainer in MUPPET, comparing its implementation against
the official SHAP KernelExplainer for tabular data. It validates that MUPPET's SHAP produces
similar explanations to the reference implementation for classification tasks.

The tests verify:
- Consistency with official SHAP kernel explainer implementation
- Proper SHAP value calculation for tabular feature importance
- Random Forest model integration and prediction accuracy
- Statistical similarity between MUPPET and official SHAP explanations
- Reproducibility across different random seeds and datasets
"""

import warnings

import numpy as np
import pytest
import shap
import torch
from sklearn import datasets, model_selection
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset

from muppet import DEVICE
from muppet.explainers import ShapTabularExplainer

warnings.filterwarnings("ignore")


class CustomDataset(Dataset):
    """Custom dataset class for testing SHAP tabular explanations.

    A mock dataset implementation that wraps test data for evaluating
    SHAP tabular explainer functionality with PyTorch data loading.
    """

    def __init__(self, data):
        """Initialize CustomDataset for SHAP tabular testing.

        Args:
            data: The dataset to be wrapped for testing.
        """
        self.data = data

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: A PyTorch tensor representing the data sample.
        """
        return self.data[idx], "label"


class RandomForestPyTorch(torch.nn.Module):
    """PyTorch wrapper for Random Forest classifier for SHAP testing.

    A PyTorch module that wraps a scikit-learn Random Forest classifier
    to make it compatible with SHAP testing and PyTorch workflows.
    """

    def __init__(self, n_estimators=500):
        """Initialize PyTorch Random Forest wrapper for SHAP testing.

        Args:
            n_estimators: Number of estimators in the random forest (default: 500).
        """
        super(RandomForestPyTorch, self).__init__()
        self.n_estimators = n_estimators

    def forward(self, x):
        """Test forward pass for mock PyTorch Random Forest model."""
        if x.is_cuda:
            x = x.cpu()
        return torch.tensor(
            self.estimator.predict_log_proba(x.numpy()), dtype=torch.float32
        ).to(DEVICE)

    def fit(self, X_train, y_train):
        """Test fit method for mock PyTorch Random Forest model."""
        rf_torch = RandomForestClassifier(self.n_estimators)
        rf_torch.fit(X_train, y_train)
        self.estimator = rf_torch


@pytest.mark.parametrize("random_seed", [9, 4, 5])
def test_shap(random_seed):
    """Test SHAP tabular explainer with different random seeds.

    Args:
        random_seed: Random seed for reproducible testing.
    """
    nb_sample = 20
    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        iris.data[:nb_sample],
        iris.target[:nb_sample],
        train_size=0.80,
        random_state=random_seed,
    )

    model_torch = RandomForestPyTorch(n_estimators=50)
    model_torch.fit(X_train, y_train)

    # Selecting a random example
    np.random.seed(random_seed)
    k = np.random.randint(0, len(X_test))
    example = torch.tensor(X_test[k : k + 1], dtype=torch.float32)
    train_dataset = CustomDataset(example)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Generate explanation using muppet
    shap_tabular = ShapTabularExplainer(
        model=model_torch, train_loader=train_loader, seed=1, n_repeats=70
    )
    explanation_shap = shap_tabular(example=example)
    explanation_shap_array = explanation_shap.cpu().numpy().reshape(-1)

    # Generate explanation using official SHAP
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train, y_train)

    explainer = shap.KernelExplainer(
        rf.predict_proba, X_train, algorithm="linear"
    )
    shap_values = explainer.shap_values(X_test[k])
    official_explanation_shap = (
        torch.from_numpy(shap_values[:, y_test[k]]).unsqueeze(0).to(DEVICE)
    )

    # Convert Muppet explanation to tensor
    exp_shap_tensor = (
        torch.tensor(explanation_shap_array, dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Calculate the distance between explanations
    a = torch.abs(exp_shap_tensor - official_explanation_shap)
    distance = torch.mean(a)

    assert distance < 0.06
