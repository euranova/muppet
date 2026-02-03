"""Tests for LIME tabular explainer components.

This module tests the LIMETabularExplainer in MUPPET, comparing its implementation against
the official LIME implementation for tabular data. It validates that MUPPET's LIME produces
similar explanations to the reference implementation for classification tasks.

The tests verify:
- Consistency with official LIME tabular implementation
- Proper feature importance calculation for tabular data
- Random Forest model integration and prediction accuracy
- Statistical similarity between MUPPET and official LIME explanations
- Reproducibility across different random seeds
"""

import warnings

import lime.lime_tabular
import numpy as np
import pytest
import torch
from sklearn import datasets, model_selection
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset

from muppet.explainers import LIMETabularExplainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")


class CustomFakeDataset(Dataset):
    """Custom fake dataset class for testing LIME tabular explanations.

    A mock dataset implementation that wraps test data for evaluating
    LIME tabular explainer functionality in isolation.
    """

    def __init__(self, data):
        """Initialize CustomFakeDataset for testing.

        Args:
            data: The dataset to be wrapped for testing.
        """
        self.data = data

    def __len__(self):
        """Return the number of samples in the fake dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Get sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            The sample at the given index.
        """
        return self.data[idx]


class RandomForestPyTorch(torch.nn.Module):
    """PyTorch wrapper for Random Forest classifier for LIME testing.

    A PyTorch module that wraps a scikit-learn Random Forest classifier
    to make it compatible with PyTorch-based testing frameworks.
    """

    def __init__(self, n_estimators=50):
        """Initialize mock PyTorch Random Forest model for testing.

        Args:
            n_estimators (int): Number of estimators in the random forest.
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


@pytest.mark.parametrize("random_seed", [4, 5, 42])
def test_lime(random_seed):
    """Test LIME tabular explainer with different random seeds.

    Args:
        random_seed: Random seed for reproducible testing.
    """
    iris = datasets.load_iris()
    nb_samples = 20
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        iris.data[:nb_samples],
        iris.target[:nb_samples],
        train_size=0.80,
        random_state=random_seed,
    )

    model_torch = RandomForestPyTorch(n_estimators=50)
    model_torch.fit(X_train, y_train)

    # Selecting a random example
    np.random.seed(random_seed)
    k = np.random.randint(0, len(X_test))
    example = torch.tensor(X_test[k : k + 1], dtype=torch.float32)
    train_data = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    dataset = CustomFakeDataset(train_data)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Generate explanation using muppet
    lime_tabular = LIMETabularExplainer(
        model=model_torch,
        train_loader=train_loader,
        nmasks=600,
    )
    explanation = lime_tabular(example=example)
    explanation_array = explanation.cpu().numpy().reshape(4)

    # Generate explanation using official LIME
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train, y_train)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        discretize_continuous=False,
        sample_around_instance=True,
    )
    exp = explainer.explain_instance(
        X_test[k], rf.predict_proba, num_features=4, top_labels=1
    )
    key = list(exp.local_exp.keys())[0]
    official_explanation = [0] * 4
    for feature, weight in exp.local_exp[key]:
        official_explanation[feature] = weight

    official_explanation_tensor = torch.tensor(
        official_explanation, dtype=torch.float32
    )
    exp_muppet = torch.tensor(explanation_array, dtype=torch.float32)

    a = torch.abs(exp_muppet - official_explanation_tensor)
    distance = torch.mean(a)

    assert distance < 0.06
