"""Tests for FIT (Feature Importance in Time) explainer components.

This module tests the FITExplainer in MUPPET, which provides temporal explanations for
time series models using perturbation-based attribution methods. It validates proper
functioning with RNN-based classifiers and temporal data structures.

The tests verify:
- Correct explanation shape generation for time series data
- Integration with RNN and GRU-based classification models
- Proper handling of multivariate time series inputs
- Generator-based perturbation functionality for temporal data
- Expected heatmap dimensions matching input sequence lengths
"""
#
# Created on Fri Jun 16 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import os
import pickle as pkl

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from muppet.explainers import FITExplainer


class PredictOneModel(torch.nn.Module):
    """Simple test model that always predicts class 1.

    A deterministic mock model for testing FIT explainer functionality
    that returns fixed predictions regardless of input.
    """

    def __init__(self):
        """Initialize simple test model that always predicts class 1.

        Creates a deterministic model for testing FIT explainer functionality.
        """
        super(PredictOneModel, self).__init__()

    def forward(self, x):
        """Test forward pass returning fixed prediction."""
        return torch.tensor([1.0, 0.0]).reshape(1, 2)


class Classifier(torch.nn.Module):
    """Test classifier model with configurable RNN architecture.

    A recurrent neural network classifier for testing that supports
    different RNN types (GRU, LSTM) and configurable hidden dimensions.
    """

    def __init__(
        self,
        feature_size,
        n_state=2,
        hidden_size=200,
        rnn="GRU",
        regres=True,
        bidirectional=False,
        return_all=False,
    ):
        """Initialize RNN-based classifier for testing.

        Args:
            feature_size: Input feature dimension size.
            n_state: Number of output classes (default: 2).
            hidden_size: Hidden state size for RNN (default: 200).
            rnn: RNN type, either "GRU" or "LSTM" (default: "GRU").
            regres: Whether to use regression head (default: True).
            bidirectional: Whether to use bidirectional RNN (default: False).
            return_all: Whether to return all hidden states (default: False).
        """
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == "GRU":
            self.rnn = torch.nn.GRU(
                feature_size, self.hidden_size, bidirectional=bidirectional
            ).to(self.device)
        else:
            self.rnn = torch.nn.LSTM(
                feature_size, self.hidden_size, bidirectional=bidirectional
            ).to(self.device)

        self.regressor = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.hidden_size, self.n_state),
            torch.nn.Softmax(-1),
        )

    def forward(self, input, past_state=None, **kwargs):
        """Test forward pass for RNN-based classifier."""
        if input.ndim == 4:
            input = input[:, 0, :, :]
        input = input.permute(2, 0, 1).to(self.device)
        self.rnn.to(self.device)
        self.regressor.to(self.device)
        if not past_state:
            #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
            past_state = torch.zeros([1, input.shape[1], self.hidden_size]).to(
                self.device
            )
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(input, past_state)
        else:
            all_encodings, (encoding, state) = self.rnn(
                input, (past_state, past_state)
            )
        if self.regres:
            if not self.return_all:
                return self.regressor(encoding.view(encoding.shape[1], -1))
            else:
                reshaped_encodings = all_encodings.view(
                    all_encodings.shape[1] * all_encodings.shape[0], -1
                )
                return torch.t(
                    self.regressor(reshaped_encodings).view(
                        all_encodings.shape[0], -1
                    )
                )
        else:
            return encoding.view(encoding.shape[1], -1)


class Generator:
    """Mock generator class for FIT explainer testing.

    A test double that simulates generative model behavior for
    testing FIT explainer with synthetic data generation.
    """

    def generate(self, past, current, features_to_perturb):
        """Test generator method returning current state unchanged."""
        return current

    def eval(self):
        """Test evaluation mode method stub."""
        pass

    def to(self, a):
        """Test device transfer method stub."""
        pass


class CustomDataset(Dataset):
    """Custom dataset class for FIT explainer testing.

    A mock dataset implementation that provides test data
    for evaluating FIT explainer functionality.
    """

    def __init__(self, data):
        """Initializes the dataset with the given data.

        Args:
            data (np.ndarray): A NumPy array containing the dataset samples.
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
        # Convert the NumPy array sample to a PyTorch tensor
        # Ensure the data type is float32 as expected by many PyTorch models
        # sample = torch.from_numpy(self.data[idx])
        return self.data[idx], 0


@pytest.mark.parametrize(
    "nb_feat,expected_heatmap_size", [(1, 1), (2, 2), (3, 3)]
)
def test_dummy_fit_returned_shape(nb_feat, expected_heatmap_size):
    """Simple FIT running test to ensure that it returns the expected shape"""
    sig_length = 5
    x = torch.ones((1, nb_feat, sig_length))  # shape (b, f, t)
    num_samples = 1000
    fake_data = np.random.rand(num_samples, nb_feat, sig_length).astype(
        np.float32
    )
    dataset = CustomDataset(fake_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    mfit = FITExplainer(
        model=PredictOneModel(),
        train_loader=train_loader,
        num_sampling=3,
        generator=Generator(),
        seed=42,
    )
    r = mfit(example=x)

    assert r.shape == (1, expected_heatmap_size, sig_length)  # (b=1, f=1, t=3)


def test_fit_returned_shape():
    """Simple FIT running test to ensure that it returns the expected shape"""
    path_to_file = __file__
    path_to_project = "/".join(path_to_file.split("/")[:-4])
    data_path = os.path.join(
        path_to_project,
        "muppet/benchmark/datasets/timeseries/synthetic/spike/",
    )

    with open(os.path.join(data_path, "x_test.pkl"), "rb") as f:
        x_test_fit = torch.tensor(pkl.load(f), dtype=torch.float32)
    x_train_fit = x_test_fit[:64]
    x_test_fit = x_test_fit[64:]

    dataset = CustomDataset(x_train_fit[:64])
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # path to saved model
    model_path = os.path.join(
        path_to_project, "muppet/benchmark/models/spike_model.pt"
    )

    # load the pre-trained model
    fit_model = Classifier(feature_size=x_test_fit.shape[1])
    fit_model.load_state_dict(torch.load(model_path))
    fit_model.eval()

    # take only two samples from test set to explain
    example = torch.tensor(x_test_fit[:1, :])  # (b=1, f=3, t=80)

    # Predict on x
    predicted_classe = torch.nn.Softmax(dim=-1)(
        fit_model(example).detach()
    ).argmax(dim=-1)

    predicted_classe = predicted_classe[0].item()

    fit_explainer = FITExplainer(
        model=fit_model,
        num_sampling=3,
        generator=Generator(),
        seed=2023,
        train_loader=train_loader,
    )

    fit_explanations = fit_explainer(example=example)  # (b=1, f=2, t=80)

    assert fit_explanations.shape == (1, 3, 80)
