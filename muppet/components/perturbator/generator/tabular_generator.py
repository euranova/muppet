"""Tabular data generators for perturbation-based explanations.

This module provides generators specifically designed for tabular data perturbations in
the MUPPET XAI framework. These generators create realistic substitute values for masked
features during the perturbation process, enabling meaningful explanations for tabular
machine learning models.

Tabular data presents unique challenges for perturbation-based explanations due to mixed
data types (numerical and categorical), feature correlations, and distribution properties.
The generators in this module address these challenges by implementing different sampling
strategies tailored to tabular characteristics.

The module contains:
    GaussianSamplingGenerator: Simple statistical generator using Gaussian distributions
        estimated from historical data for time series or sequential tabular data
    StandardGaussianTabularGenerator: Advanced generator for mixed tabular data with
        separate handling of numerical and categorical features
    RandomSampleTabularGenerator: Frequency-based generator that samples from observed
        feature value distributions in training data

Key Features:
    - Handles mixed numerical and categorical features appropriately
    - Preserves feature distributions and correlations from training data
    - Supports instance-centered perturbations for local explanations
    - Configurable sampling strategies (statistical vs. frequency-based)
    - Deterministic sampling for reproducible explanations

These generators are typically used with tabular perturbators and are essential for
methods like LIME, SHAP, and other feature attribution techniques applied to structured
data, enabling realistic counterfactual analysis and feature importance discovery.
"""

import random
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from muppet import DEVICE
from muppet.components.perturbator.generator.base import Generator


class GaussianSamplingGenerator(Generator):
    """Simple Gaussian sampling generator for tabular data imputation.

    Generates replacement values for perturbed features by sampling from
    normal distributions. Provides basic statistical imputation without
    considering feature correlations or data distributions.

    """

    def __init__(self, seed: int | None = None) -> None:
        """A simple random sampling generator. Used for imputing missing values by a sampled ones from a Normal Distribution.

        Args:
            seed (int, optional): Seed to control reproducibility

        """
        self.seed = seed
        self.is_trained = True

        super().__init__()

    def generate(self, past, current, features_to_perturb) -> torch.Tensor:
        """Return sampled values from a Normal Distribution.

        past (torch.Tensor): past measurements from which Mean and Std will be estimated (b=1, f, t)
        current (torch.Tensor): current time step to perturbate (b=1, f, 1)
        features_to_perturb (torch.Tensor): features to perturb

        Returns:
            torch.Tensor: sampled values
        """
        if self.seed:
            torch.manual_seed(seed=self.seed)

        # Estimate mean and std for the the normal distribution
        mean = torch.mean(past, dim=-1).cpu().numpy()
        std = torch.std(past, dim=-1, unbiased=False).cpu().numpy()

        # Sample from the normal distribution
        res = torch.from_numpy(
            np.random.normal(loc=mean, scale=std, size=mean.shape)
        )

        # Take into account features to explain
        mask = torch.zeros_like(res, dtype=torch.bool)
        mask[:, list(set(range(mask.shape[-2])) - set(features_to_perturb))] = (
            True
        )

        return res


class StandardGaussianTabularGenerator(Generator):
    """Advanced generator for mixed tabular data with statistical modeling.

    Handles both numerical and categorical features by computing separate
    statistics and frequencies. Provides instance-centered perturbations
    for local explanations and maintains feature distributions from training data.

    """

    def __init__(
        self,
        train_data: torch.Tensor,
        categorical_features: List[int] = [],
        sample_around_instance: bool = True,
    ) -> None:
        """Initialize the StandardGaussianTabularGenerator for mixed data types.

        Sets up a generator that handles both numerical and categorical features by
        computing separate statistics and frequencies, enabling realistic perturbations
        for tabular machine learning explanations.

        Args:
            train_data (torch.Tensor): Training dataset tensor used to compute feature statistics
                and categorical frequencies. Shape: (n_samples, n_features).
            categorical_features (list[int]): List of column indices that contain categorical data.
                These features will be handled using frequency-based sampling.
            sample_around_instance (bool): If True, generates perturbations centered around
                the instance being explained. If False, samples from training data
                distribution. Useful for local vs. global explanation strategies.
        """
        self.means_tensor = None
        self.std_tensor = None
        self.sample_around_instance = sample_around_instance
        self.categorical_frequencies = []
        self.train_data = train_data
        self.categorical_features = categorical_features
        self.random_state = np.random.RandomState(
            seed=None
        )  # Initialization of the random state
        super().__init__()
        self.train_generator()

    """It is a generator that trains on a dataset containing
    the data rows to be explained. It calculates the mean and standard
    deviation per feature and the frequencies
    of values for categorical data, then generates
    values around the example to be explained.

        Args:
        - sample_around_instance (bool, optional): If True, the generator will sample values around the instance to be explained.
            Defaults to False.

        Attributes:
        - means_tensor (torch.Tensor or None): Tensor containing the means calculated per feature.
        - std_tensor (torch.Tensor or None): Tensor containing the standard deviations calculated per feature.
        - categorical_frequencies (dict or None): Frequencies of values for categorical data.

    """

    def train_generator(self) -> None:
        """Train the generator to compute summary statistics from the training data."""
        b, f = (
            self.train_data.shape
        )  # Get the shape of the training data (b: batch, f: number of features)
        numerical_features = list(
            set(range(f)) - set(self.categorical_features)
        )  # Identify numerical features

        # Fit StandardScaler to numerical features
        standard_scaler = StandardScaler()
        # Handle potential GPU tensor conversion
        try:
            standard_scaler.fit(self.train_data[:, numerical_features].numpy())
        except TypeError:  # Added handling for tensors on GPU
            standard_scaler.fit(
                self.train_data[:, numerical_features].cpu().numpy()
            )

        # Save means and standard deviations as tensors and ensure they are on the same device as train_data
        self.means_tensor = torch.tensor(standard_scaler.mean_).to(
            DEVICE
        )  # Added device handling
        self.std_tensor = torch.tensor(standard_scaler.scale_).to(
            DEVICE
        )  # Added device handling
        self.numerical_features = numerical_features

        if len(self.categorical_features) == 0:
            return

        # Calculate frequencies of each element for each categorical feature
        self.categorical_frequencies = {}
        for feat_idx in self.categorical_features:
            feat_values = (
                self.train_data[:, feat_idx].cpu().numpy()
            )  # Ensure it's on CPU for numpy operations
            unique, counts = np.unique(feat_values, return_counts=True)
            total_count = len(feat_values)
            freq_dict = dict(zip(unique, counts / total_count))
            self.categorical_frequencies[feat_idx] = freq_dict

    def generate(
        self, x_instance: torch.Tensor, data_scaled: torch.Tensor
    ) -> torch.Tensor:
        """Generate a perturbed sample based on the learned statistics.

        Args:
            x_instance (torch.Tensor): The instance to be explained, of shape (1, f).
            data_scaled (torch.Tensor): Pre-scaled data based on normal distribution, of shape (n, 1, f).

        Returns:
            torch.Tensor: Generated sample tensor with perturbations.
        """
        # Separate numerical and categorical indices
        numerical_indices = self.numerical_features

        # Initialize sampled_values_tensor with the same shape as data_scaled
        sampled_values_tensor = data_scaled.clone()

        # Extract the instance to explain (assume it's of shape (1, f))
        instance_to_explain = x_instance[0, :]

        # Generate perturbations for numerical features
        if len(numerical_indices) > 0:
            numerical_data = data_scaled[:, :, numerical_indices]
            if self.sample_around_instance:
                # Rescale the normal data using the instance's values plus some noise
                for i, num_idx in enumerate(numerical_indices):
                    sampled_values_tensor[:, :, num_idx] = (
                        instance_to_explain[num_idx]
                        + numerical_data[:, :, i] * self.std_tensor[i]
                    )
            else:
                # Rescale the normal data using the learned means and standard deviations
                for i, num_idx in enumerate(numerical_indices):
                    sampled_values_tensor[:, :, num_idx] = (
                        numerical_data[:, :, i] * self.std_tensor[i]
                        + self.means_tensor[i]
                    )

        # If categorical frequencies are available, sample categorical features
        if len(self.categorical_frequencies) > 0:
            for feat_idx, freq_dict in self.categorical_frequencies.items():
                unique_values = list(freq_dict.keys())
                probabilities = list(freq_dict.values())

                # Use random_state.choice to sample categorical values
                sampled_values = torch.tensor(
                    self.random_state.choice(
                        unique_values,
                        size=data_scaled.shape[0],
                        p=probabilities,
                    )
                )

                # Replace the values in the sampled tensor with the sampled categorical values
                sampled_values_tensor[:, :, feat_idx] = sampled_values.view(
                    -1, 1
                )  # Ensure correct shape
        return sampled_values_tensor


class RandomSampleTabularGenerator(Generator):
    """Generate random sample vectors based on feature values
    and frequencies from training data.


    Attributes:
        train_data (torch.Tensor): The training data from which feature values will be sampled.
        n_features (int): The number of features in the training data.
        feature_values (list[lists]): List of unique values for each feature.
        method (str): The method to generate samples, either 'freq' or 'mean'.

    Methods:
        train_generator: A static method reserved for future use; currently does nothing.

        generate(n_samples): Generates a specified number of random sample vectors from the feature values in the training data
            based on the specified method ('freq' or 'mean').
    """

    def __init__(self, train_data, method="freq", seed=None):
        """Initializes the RandomSampleTabularGenerator with training data.

        Args:
            train_data (torch.Tensor): The training data used to fit samplers.
                Expected shape: (num_train_samples, num_features).
            method (str, optional): The method to generate samples, either
                'freq' or 'mean'. Default is 'freq'.
            seed (int, optional): The seed for random number generation.
                Default is None, which means no fixed seed.
        """
        self.train_data = train_data
        self.n_features = train_data.shape[1]
        self.method = method
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Extract unique feature values and their abundances for each feature
        super().__init__()
        self.train_generator()

    def train_generator(self):
        """This method can be extended or implemented in the future if additional training logic is required."""
        self.feature_values = []
        self.feature_frequencies = []

        for i in range(self.n_features):
            feature_column = self.train_data[:, i]
            unique_values, counts = torch.unique(
                feature_column, return_counts=True
            )
            self.feature_values.append(unique_values.tolist())
            self.feature_frequencies.append(counts.tolist())

        # Calculate the mean of each feature
        self.feature_means = torch.mean(self.train_data, dim=0).tolist()

    def generate(self, n_samples):
        """Generates random samples by frequency based sampling feature values
        or by using the mean values for each feature.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor containing the generated random samples
                with shape (n_samples, 1, n_features).
        """
        if self.method == "freq":
            generated_tensor = torch.zeros((n_samples, 1, self.n_features))

            for i in range(n_samples):
                sample = []
                for j in range(self.n_features):
                    values = self.feature_values[j]
                    frequencies = self.feature_frequencies[j]

                    # Generate a random sample based on frequencies
                    sampled_value = random.choices(
                        values, weights=frequencies, k=1
                    )[0]
                    sample.append(sampled_value)

                # Convert the sample to a tensor and reshape
                sample_tensor = torch.tensor(sample).float().unsqueeze(0)
                # Store the sample in the generated tensor
                generated_tensor[i] = sample_tensor

        elif self.method == "mean":
            # Generate samples using the mean values of each feature
            generated_tensor = (
                torch.tensor(self.feature_means)
                .float()
                .unsqueeze(0)
                .repeat(n_samples, 1, 1)
            )

        else:
            raise ValueError("Invalid method. Choose either 'freq' or 'mean'.")

        return generated_tensor
