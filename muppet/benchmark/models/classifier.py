"""Classification model wrappers for the MUPPET benchmark framework.

This module provides unified wrapper classes for classification models across
different machine learning frameworks (PyTorch, scikit-learn, XGBoost) with
a consistent interface for training and inference.

Classes:
    Classifier: Unified wrapper for classification models with hyperparameter tuning
    GRUClassifier: PyTorch GRU-based classifier for sequence/time series data
"""

from typing import Any

import numpy as np
import optuna
import torch
from optuna.integration import OptunaSearchCV

from muppet.benchmark.models.base import AbstractModel


class Classifier(AbstractModel):
    """A unified wrapper for classification models, providing a consistent API for models
    from different frameworks like PyTorch, scikit-learn, and XGBoost.

    This class handles both standard training and hyperparameter optimization
    (via Optuna for non-PyTorch models). It dynamically loads models from
    Hydra-style configurations and provides a PyTorch-like interface with
    `.fit()` and `.forward()` methods for seamless pipeline integration.
    It supports hyperparameter optimization via Optuna and provides a consistent
    interface for training and inference.
    """

    def __init__(
        self,
        model,
        name: str,
        pretrained: bool = True,
        random_state: int = 105,
        n_trials: int = 20,
        cv: int = 5,
        scoring: str = "balanced_accuracy",
        study_name: str = "classifier",
        predict_proba_func: str = "predict_log_proba",
        params_tuning: dict[str, Any] | None = None,
        n_jobs: int = 1,
    ):
        """Initialize the Classifier wrapper instance.

        Args:
            model (Union[DictConfig, Any]): The model configuration (e.g., from Hydra) or an already
                instantiated model object.
            name (str): The name of the model.
            pretrained (bool, optional): Whether to use a pretrained version of the model. Defaults to True.
            random_state (int, optional): The random state for reproducibility, particularly in
                hyperparameter tuning. Defaults to 105.
            n_trials (int, optional): The number of trials for Optuna-based hyperparameter search.
                Defaults to 20.
            cv (int, optional): The number of cross-validation folds for hyperparameter tuning.
                Defaults to 5.
            scoring (str, optional): The scoring function used to evaluate models during tuning.
                Defaults to "balanced_accuracy".
            study_name (str, optional): The name for the Optuna study. Defaults to "classifier".
            predict_proba_func (str, optional): The name of the method to call on non-PyTorch models
                to get probabilistic outputs. Defaults to "predict_log_proba".
            params_tuning (dict[str, Any] | None, optional): A dictionary defining the hyperparameter
                search space for Optuna. If `None`, no tuning is performed. Defaults to None.
            n_jobs (int, optional): The number of parallel jobs for hyperparameter tuning. Defaults to 1.
        """
        super().__init__(
            model,
            name,
            pretrained,
            postprocessing_func=lambda x: torch.argmax(x, dim=-1),
        )
        assert isinstance(self.model, torch.nn.Module) or hasattr(
            self.model, "fit"
        ), "The model must be a PyTorch Module or have a .fit() method."
        self.random_state = random_state
        self.n_trials = n_trials
        self.cv: int = cv
        self.scoring = scoring
        self.study_name = study_name
        self.predict_proba_func = predict_proba_func
        self.params_tuning = params_tuning
        self.n_jobs = n_jobs

    def fit(self, train_loader):
        """Trains the classification model using the provided training data.

        For PyTorch models, this method currently raises a `NotImplementedError`, as the training
        logic needs to be added (e.g., using a training loop or a library like PyTorch Lightning).
        For scikit-learn and XGBoost models, it extracts data from the DataLoader, converts it to
        NumPy arrays, and either fits the model directly or, if `params_tuning` is provided,
        performs Optuna-based hyperparameter tuning and uses the best-found estimator.

        Args:
            train_loader (DataLoader): The data loader containing the training samples.
        """
        if isinstance(self.model, torch.nn.Module):
            # TODO: implement pytorch trainer (lightning ?!) and train model
            raise NotImplementedError(
                "PyTorch fit function not implemented yet."
            )
        else:  # for sklearn models / xgboost
            X, y = [], []
            for features, label in train_loader:
                X.append(features.numpy())
                y.append(label.numpy())
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y, axis=0)

            if self.params_tuning is None:
                # only fit with defaults params if any params are given
                self.model.fit(X, y)
            else:
                # tuning hyper-parameters and return the best model
                optuna_search = OptunaSearchCV(
                    estimator=self.model,  # type: ignore
                    param_distributions=self.params_tuning,  # type: ignore
                    cv=self.cv,
                    n_trials=self.n_trials,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    refit=True,
                    study=optuna.create_study(
                        direction="maximize",
                        study_name=self.study_name,
                    ),
                )
                optuna_search.fit(X, y)
                self.model = optuna_search.best_estimator_

    def forward(self, x):
        """Performs a forward pass for inference, returning probabilistic outputs.

        For PyTorch models, this method calls the model directly. For scikit-learn
        and XGBoost models, it converts the input tensor to a NumPy array, calls the
        method specified by `predict_proba_func` (e.g., `.predict_log_proba`), and
        converts the results back into a PyTorch tensor. This ensures a consistent
        tensor output regardless of the underlying framework.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A tensor containing the probabilistic outputs.
        """
        if isinstance(self.model, torch.nn.Module):
            return self.model(x)
        else:
            x_np = x.detach().cpu().numpy()
            return torch.tensor(
                getattr(self.model, self.predict_proba_func)(x_np),
                dtype=torch.float32,
            ).to(x.device)


class GRUClassifier(torch.nn.Module):
    """GRU-based neural network classifier for sequential data.

    A recurrent neural network classifier using GRU (Gated Recurrent Unit)
    layers for processing sequential data with configurable hidden dimensions
    and output classes. Sets up a recurrent neural network with GRU or LSTM layers for processing
    time series or sequential data, with configurable architecture parameters.
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
        pretrained_model_path: str = None,
    ):
        """Initialize the GRUClassifier instance.

        Args:
            feature_size: Number of input features per time step.
            n_state: Number of output classes for classification.
            hidden_size: Dimension of the hidden state in RNN layers.
            rnn: Type of RNN to use ("GRU" or "LSTM").
            regres: If True, adds regression/classification head on top of RNN.
            bidirectional: If True, uses bidirectional RNN processing.
            return_all: If True, returns outputs from all time steps; if False,
                only returns the final time step output.
            pretrained_model_path: Optional path to load pretrained model weights.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        self.name = "GRU Classifier"
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

        if pretrained_model_path is not None:
            self.load_state_dict(
                torch.load(pretrained_model_path, map_location=self.device)
            )
            self.eval()

    def forward(self, input, past_state=None, **kwargs):
        """Forward pass through the RNN classifier."""
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
