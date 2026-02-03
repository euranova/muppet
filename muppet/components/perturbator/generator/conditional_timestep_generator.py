"""Conditional Gaussian generator for time series perturbations using RNN-based VAE.

This module implements a sophisticated conditional generator for time series data that
learns to impute missing values by modeling the conditional distribution P(X_t|X_0:t-1).
The generator uses a variational autoencoder architecture with RNN encoder and Gaussian
decoder to generate contextually appropriate perturbations for temporal explanations.

As part of the MUPPET perturbation framework, this generator enables advanced time series
explanation methods by producing realistic substitute values that maintain temporal
dependencies and feature correlations. This is essential for explaining models that
depend on sequential patterns and temporal dynamics.

The module contains:
    ConditionalGaussianFeatureGenerator: Main trainable generator combining encoder-decoder
        with conditional sampling capabilities for multivariate time series
    GaussianRNNEncoder: RNN-based encoder that maps time series to latent Gaussian parameters
    GaussianDecoder: Decoder that generates likelihood distributions from latent representations
    check_cov_pd: Utility function ensuring positive definite covariance matrices

Key Technical Features:
    - Variational autoencoder with RNN encoder for temporal modeling
    - Conditional sampling P(X_S'|X_S) for feature subsets
    - Multivariate Gaussian distributions with learned covariances
    - Positive definite covariance correction with noise injection
    - Support for both univariate and multivariate time series
    - Deterministic sampling for reproducible explanations

The generator is designed for use with time series explanation methods like temporal LIME,
SHAP for sequences, or custom perturbation-based attributions that require realistic
temporal imputations rather than simple masking strategies.
"""

#
# Created on Wed May 24 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import itertools
from typing import Tuple

import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader

from muppet import logger
from muppet.components.perturbator.generator.base import TrainableGenerator


class ConditionalGaussianFeatureGenerator(TrainableGenerator):
    """Conditional Gaussian generator for time series perturbations.

    Implements a variational autoencoder with RNN encoder and Gaussian
    decoder for learning conditional distributions P(X_t|X_{0:t-1}).
    Enables sophisticated temporal perturbations that preserve realistic
    time series patterns and feature dependencies.

    """

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        latent_size: int,
        mid_layer_size: int,
        prediction_size: int,
        num_samples: int,
        cov_noise_level: float,
        max_noise_correction: int,
        lr: float,
        num_epochs: int,
        timesteps_divide_num: int,
        seed: int | None = None,
    ) -> None:
        """Conditional generator model to predict perturbed values.

        Args:
            feature_size (int): Number of features in the input (f)
            hidden_size (int): The encoder's hidden layer size
            latent_size (int): The encoder's latent space size
            mid_layer_size (int): The mid-layer size used in Encoder and Decoder
            prediction_size (int): The number of predictions to make. The prediction window [t:t+p] (p)
            num_samples (int): Number of Zs to sample from the latent distribution (n)
            cov_noise_level (float): The noise to add to the covariance to make it positive definite (PD)
            max_noise_correction (int): Maximum number of covariance PD correction iterations
            lr (float): Training learning rate used with Adam optimizer
            num_epochs (int): Training number of epochs
            timesteps_divide_num (int): Used to divide the time series. E.g, when set to 1, it means predict only at time t=T using X0:T-1
            seed: the seed to used for reproducible sampling at inference time. If not provided the sampling is nondeterministic

        """
        # general parameters
        self.seed = seed
        if self.seed:
            torch.manual_seed(seed=self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # init untrained generator
        self.is_trained = False

        # architectures parameters
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.mid_layer_size = mid_layer_size
        self.prediction_size = prediction_size
        self.output_size = self.feature_size * self.prediction_size

        # decoder's parameters
        self.num_samples = num_samples
        self.cov_noise_level = cov_noise_level
        self.max_noise_correction = max_noise_correction

        # training specific parameters (used only by this generator)
        self.timesteps_divide_num = timesteps_divide_num

        # global training parameters (used by all generators)
        super().__init__(lr=lr, num_epochs=num_epochs)

        # initiate encoder and decoder
        self.rnn_encoder = GaussianRNNEncoder(
            feature_size=self.feature_size,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            mid_layer_size=self.mid_layer_size,
            device=self.device,
        )

        self.decoder = GaussianDecoder(
            feature_size=self.feature_size,
            output_size=self.output_size,
            latent_size=self.latent_size,
            mid_layer_size=self.mid_layer_size,
            device=self.device,
        )

    def likelihood_distribution(
        self,
        past: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate the mean and (co)variance of the joint distribution P(X_t|X_0:t-1).

        Args:
            past (torch.Tensor): Batch of past data of the shape (b, f, t).

        Returns: mean and (co)variance

        """
        # estimate the past's Gaussian distribution
        mu, std = self.rnn_encoder.latent_distribution(X=past)
        mean, covariance = self.decoder.likelihood_distribution(
            mu=mu,
            std=std,
            num_samples=self.num_samples,
            cov_noise_level=self.cov_noise_level,
            max_noise_correction=self.max_noise_correction,
        )

        # multivar: (n, p*f), (n, p*f, p*f) | univar: (n, p), (n, p)
        return mean, covariance

    def joint_sample(
        self,
        past: torch.Tensor,
    ) -> torch.Tensor:
        """Generate the missing measurements at time current(=t) based on past (X0:t-1) through sampling from the joint distribution P(X_t|X_0:t-1).

        Args:
            past (torch.Tensor): Batch of previous data measurements (b, f, t).

        Returns:
            torch.Tensor: A sample from the the Gaussian distribution of P(X_t|X_0:t-1).

        """
        # multivar: (n, p*f), (n, p*f, p*f) | univar:(n, p), (n, p)
        mean, covariance = self.likelihood_distribution(past=past)

        # univariate case
        if self.feature_size == 1:
            std = torch.sqrt(covariance).squeeze(dim=-1)
            sample = torch.normal(mean=mean, std=std)  # (n, p)
            return sample

        likelihood = MultivariateNormal(loc=mean, covariance_matrix=covariance)

        return likelihood.rsample()  # (n, p*f)

    def generate(
        self,
        past: torch.Tensor,
        current: torch.Tensor,
        features_to_perturb: set,
    ) -> torch.Tensor:
        """Generate values for the features_to_perturb at time current(=t) based on past (historical data) through conditional sampling from P(X_{S^,t}|X_{S,t}).

        Takes 'current' the measurements at time t, and returns same 'current' at time t with features in S^ being replaced by values estimated from the Gaussian distribution.

        Args:
            past (torch.Tensor): Batch of previous data measurements (b, f, t)
            current (torch.Tensor): Batch of measurements at time t (b, f)
            features_to_perturb (set): Set of features' indices that are not known/measured. We sample on these features

        Returns:
            full_sample (torch.Tensor): The imputed sample at time t with the generated values for missing measurements (S^). (b, f, t)

        """
        conditioning_features = sorted(
            set(range(current.shape[-1])) - set(features_to_perturb)
        )
        # when len(S)=0
        # TODO check if it could work fine with no feature kept unchanged, only perturbed feature (ex 1channel timeseries)
        # TODO see univariate case
        assert len(conditioning_features) > 0, (
            "ConditionalGaussianFeatureGenerator should be conditionned on at least one feature."
        )

        # when len(S)=feature_size: when 'don't perturb any feature' ==> return current
        if len(conditioning_features) == self.feature_size:
            return current

        # (b, f)
        assert len(current.shape) == 2, (
            f"The passed data at time t 'current' has different shape than what is expected! Expected: (batch, features), but received: {current.shape}"
        )

        # estimate mean and covariance of P(X_t|X_0:t-1) or P(X_{t}|past) if univariate case
        mean, covariance = self.likelihood_distribution(
            past
        )  # (n, p*f), multivar:(n, p*f, p*f), univar:(n, p)

        # UNIVARIATE CASE 1: f=1 => len(S)=1
        if self.feature_size == 1:
            assert len(conditioning_features) == self.feature_size, (
                "For univariate case, the features to explain must match the initialized feature size!"
            )
            # P(x_{t}|past)
            std = torch.sqrt(covariance).squeeze(dim=-1)  # (n, p)
            sample = torch.normal(mean=mean, std=std)
            return sample, mean

        conditioning_inds = [
            list(
                range(i * self.prediction_size, (i + 1) * self.prediction_size)
            )
            for i in conditioning_features
        ]
        # e.g [0, 1, 2, 3, ..., len(S)*p-1]
        conditioning_inds = list(
            itertools.chain.from_iterable(conditioning_inds)
        )

        perturb_inds = list(
            set(range(self.output_size)) - set(conditioning_inds)
        )

        conditioning_len = len(conditioning_inds)
        perturb_len = len(perturb_inds)

        cov_1_2 = covariance[:, perturb_inds, :][:, :, conditioning_inds].view(
            -1, perturb_len, conditioning_len
        )  # (n, s^, s)
        cov_2_2 = covariance[:, conditioning_inds, :][
            :, :, conditioning_inds
        ].view(-1, conditioning_len, conditioning_len)  # (n, s, s)
        cov_1_1 = covariance[:, perturb_inds, :][:, :, perturb_inds].view(
            -1, perturb_len, perturb_len
        )  # (n, s^, s^)

        # make full sample of the same shape as the mean by repeating the batch num_samples times,
        #  and the duplicating the features values prediction_size times (f1,f2)==> (f1,f1,f2,f2)
        # shape (n, p*f)
        full_sample = (
            current.unsqueeze(0)
            .repeat(self.num_samples, 1, 1)
            .reshape(-1, self.feature_size)
            .float()
        )
        full_sample = (
            full_sample.unsqueeze(2)
            .repeat(1, 1, self.prediction_size)
            .reshape(full_sample.shape[0], -1)
        ).to(self.device)

        conditioning_fs_mean = mean[:, conditioning_inds].view(
            -1, conditioning_len
        )
        conditioning_fs_vals = full_sample[:, conditioning_inds].view(
            -1, conditioning_len
        )
        conditioning_fs_vals_mean_diff = (
            conditioning_fs_vals - conditioning_fs_mean
        ).view(-1, conditioning_len, 1)  # (n, s, 1)

        temp = torch.bmm(cov_1_2, torch.inverse(cov_2_2))  # (n, s^, s)
        conditioning_fs_comp_mean = mean[:, perturb_inds].view(
            -1, perturb_len
        )  # (n, s^)

        mean_conditional = conditioning_fs_comp_mean + torch.bmm(
            temp, conditioning_fs_vals_mean_diff
        ).squeeze(-1)  # (n,s^,1)=>(n,s^)

        cov_conditional = cov_1_1 - torch.bmm(
            temp, torch.transpose(cov_1_2, 2, 1)
        )  # (n, s^, s^)

        # P(x_{s^,t}|x_{s,t})
        likelihood = MultivariateNormal(
            loc=mean_conditional, covariance_matrix=cov_conditional
        )
        # (n, s^)
        sample = likelihood.rsample()
        full_sample[:, perturb_inds] = sample  # (n, p*f)

        return full_sample.detach()

    def run_epoch(
        self,
        dataloader: DataLoader,
        in_train: bool,
    ) -> float:
        """Run one training epoch

        Args:
            dataloader (DataLoader): The train loader
            in_train (bool, optional): Either if training or evaluating. E.g set to True ==> training mode.

        Returns:
            float: the epoch loss.

        """
        if in_train:
            self.train()

        else:
            self.eval()

        # divide the timesteps
        try:
            signal_length = dataloader.dataset.shape[-1]  # (b, f, t)
        except AttributeError:
            signal_length = dataloader.dataset.dataset.features.shape[-1]
            # dataloader.dataset is a torch subset

        if self.timesteps_divide_num == 1:
            # when only predicting at time t=T
            timepoints = [signal_length - self.prediction_size]
        else:
            assert self.timesteps_divide_num < signal_length + 1, (
                f"If the time series needs to be devided, it must respect its lenght. Provided timesteps_divide_num excceded the signal length: {signal_length}!"
            )
            timepoints = [
                int(tt)
                for tt in np.logspace(
                    1.0,
                    np.log10(signal_length - self.prediction_size),
                    num=self.timesteps_divide_num,
                )
            ]
        epoch_loss = 0
        for _, (signals, true_label) in enumerate(dataloader):
            for t in timepoints:
                if in_train:
                    self.optimizer.zero_grad()
                # the label is the future measures t:t+p (#Xt:t+p)
                label = signals[:, :, t : t + self.prediction_size].reshape(
                    signals.shape[0], -1
                )
                # match label to number of generated samples (num_samples) ==> (n, p*f)
                label = (
                    label.unsqueeze(0)
                    .repeat(self.num_samples, 1, 1)
                    .reshape(-1, self.feature_size * self.prediction_size)
                    .to(self.device)
                )

                prediction = self.joint_sample(
                    past=signals[:, :, :t]
                )  # (n, p*f), X0:t-1
                reconstruction_loss = torch.nn.MSELoss(reduction="none")(
                    prediction.float(), label.float()
                )
                reconstruction_loss = reconstruction_loss.mean().float()

                epoch_loss = epoch_loss + reconstruction_loss.item()

                if in_train:
                    reconstruction_loss.backward(retain_graph=True)
                    self.optimizer.step()

        return float(epoch_loss) / len(dataloader)


class GaussianRNNEncoder(torch.nn.Module):
    """RNN encoder for mapping input sequences to Gaussian latent spaces.

    Encodes time series data using a GRU and maps it to latent Gaussian
    parameters (mean and standard deviation). Used in variational approaches
    for conditional time series generation.
    """

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        latent_size: int,
        mid_layer_size: int,
        device,
    ) -> None:
        """An RNN encoder that is responsible of transforming the input x into an encoding space (hidden) using one-layer GRU, and then
        maps it to the latent space the represent the Gaussian parameters for every input sample.


        Args:
            feature_size (int): The number of input features
            hidden_size (int): The RNN/GRU hidden space size.
            latent_size (int): The latent space size. It is multiplied by two as it represents the mean and covariance of the latent
                representation Z of x.
            mid_layer_size (int): The size of a mid layer between the hidden space (RNN encoding) and the latent space (Z).
            device (str): The device to use.

        """
        super().__init__()
        self.device = device
        # RNN masker
        self.rnn = torch.nn.GRU(
            input_size=feature_size, hidden_size=hidden_size, num_layers=1
        ).to(self.device)  # 1-layer GRU
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if "weight" in p:
                    torch.nn.init.normal_(self.rnn.__getattr__(p), 0.0, 0.02)

        # latent space masker (Z)
        self.dist_predictor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_size, out_features=mid_layer_size
            ),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(num_features=mid_layer_size),
            torch.nn.Linear(
                in_features=mid_layer_size, out_features=latent_size * 2
            ),
        ).to(self.device)

    def latent_distribution(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate mean and the std of the distribution of the latent representation Z of X.

        Args:
            X: Input time series (b, f, t)

        Returns:
            A tuple of mu and std. Each of shape (b, latent_size)
        """
        X = X.permute(2, 0, 1).float()  # reshape to (t, b, f)

        # _: encoding/mapping of every t to the hidden_size space (t, b, hidden_size),
        # final_h_state: the last layers hidden state (num_layers=1, b, hidden_size)

        _, final_h_state = self.rnn(
            X.to(self.device)
        )  # (num_layers=1, b, hidden_size)

        # maps the batch inputs from hidden_size space to latent_size space (latent variable Z)
        mu_std = self.dist_predictor(
            final_h_state[0, :, :]
        )  # passing only (b, hidden_size), ignore the dim=0 bcz we use 1-layer GRU (num_layers=1)

        # semantic meaning of mean and std
        mu = mu_std[:, : mu_std.shape[1] // 2]  # (b, latent_size)
        std = mu_std[:, mu_std.shape[1] // 2 :]  # (b, latent_size)

        return mu, std


class GaussianDecoder(torch.nn.Module):
    """Gaussian decoder for generating distributions from latent representations.

    Decodes latent variables into likelihood distributions over the output
    space. Supports both univariate (with variance) and multivariate (with
    covariance) Gaussian distributions for flexible time series generation.
    """

    def __init__(
        self,
        feature_size: int,
        output_size: int,
        latent_size: int,
        mid_layer_size: int,
        device,
    ) -> None:
        """A Gaussian decoder that estimate the likelihood distribution of the latent representation Z (encoding) of X.

        Args:
            feature_size (int): The number of input features
            output_size (int): The expected output size (something like number of predictions to make * number of input features)
            latent_size (int, optional): The latent representation size (output size of encoder).
            mid_layer_size (int, optional): The size of a mid-layer between the latent space and final output mapping.
            device (str, optional): The device to use.

        """
        super().__init__()

        self.feature_size = feature_size
        self.output_size = output_size
        self.device = device

        # Gaussian mean generator network from the latent space Z. The output_size is proportional to the number of input features as we are estimating
        #  the mean of every feature.
        self.mean_generator = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=latent_size, out_features=mid_layer_size
            ),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(num_features=mid_layer_size),
            torch.nn.Linear(
                in_features=mid_layer_size, out_features=self.output_size
            ),
        ).to(self.device)
        # UNIVARIATE CASE
        if feature_size == 1:
            # Gaussian variance generator network from the latent space Z. This is used for the univariate time series.
            # The output_size is proportional to the number of input features as we are estimating the variance of every feature.
            self.var_generator = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=latent_size, out_features=mid_layer_size
                ),
                torch.nn.Tanh(),
                torch.nn.BatchNorm1d(num_features=mid_layer_size),
                torch.nn.Linear(
                    in_features=mid_layer_size, out_features=self.output_size
                ),
                torch.nn.ReLU(),
            ).to(self.device)
        # MULTIVARIATE CASE
        else:
            # Gaussian covariance generator network from Z. The output_size is proportional to the number of input features as we are estimating
            #  the covariance of every feature. Because it's the covariance matrix we generate output_size*output_size values.
            self.cov_generator = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=latent_size, out_features=mid_layer_size
                ),
                torch.nn.Tanh(),
                torch.nn.BatchNorm1d(num_features=mid_layer_size),
                torch.nn.Linear(
                    in_features=mid_layer_size,
                    out_features=self.output_size * self.output_size,
                ),
                torch.nn.ReLU(),
            ).to(self.device)

    def likelihood_distribution(
        self,
        mu: torch.Tensor,
        std: torch.Tensor,
        num_samples: int,
        cov_noise_level: float,
        max_noise_correction: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate the likelihood Gaussian distribution of the output (proportional to number of features and needed predictions) given the latent
        representation Z (encoding) of X.

        Args:
            mu (torch.Tensor): The mean of the latent distribution. Shape = (b, latent_size)
            std (torch.Tensor): The std of the latent distribution. Shape = (b, latent_size)
            num_samples (int, optional): Number of Zs to sample from the latent distribution. In case multi-sampling is needed!
            cov_noise_level (float, optional): The noise to add to the covariance to make it positive definite (PD).
            max_noise_correction (int, optional): Maximum number of covariance PD correction iterations.

        Returns:
            tuple: estimated mean and covariance or variance if univariate case

        Note: n = b*num_samples, output_size = p*f (number of prediction to make * input features) where p=prediction_size the prediction window.
        """
        # sample Z from the distribution
        if num_samples == 1:
            Z = mu + std * torch.randn_like(mu).to(
                self.device
            )  # (b, latent_size)
        else:
            rand = torch.randn((num_samples, *mu.shape))
            Z = mu.unsqueeze(0) + std.unsqueeze(0) * rand
            Z = Z.reshape(-1, Z.shape[-1]).to(self.device)  # (n, latent_size)

        # Generate the distribution P(X|H,Z)
        mean = self.mean_generator(Z)  # (n, output_size)

        # UNIVARIATE CASE:
        if self.feature_size == 1:
            variance = self.var_generator(Z)  # (n, p)
            return mean, variance

        # MULTIVARIATE CASE:
        # make len(Z)=n=b*num_samples of identity matrix of shape (p*f, p*f)
        cov_noise = (
            torch.eye(self.output_size).unsqueeze(0).repeat(len(Z), 1, 1)
            * cov_noise_level
        )
        cov_noise = cov_noise.to(self.device)

        # self.cov_generator(Z): (n, output_size*output_size) output_size=p*f
        A = self.cov_generator(Z).view(
            -1, self.output_size, self.output_size
        )  # (n, p*f, p*f)
        A_transpose = torch.transpose(A, 1, 2)  # transpose A on dim 1 and 2

        # perform batch matrix-multi
        # torch.use_deterministic_algorithms(True)
        covariance = torch.bmm(A, A_transpose) + cov_noise  # (n, p*f, p*f)

        # check if cov is positive definite and try to add noise to it max_noise_correction of times, if no success log the problem and use identity matrix for cov
        covariance = check_cov_pd(
            covariance_matrix=covariance,
            cov_noise_level=cov_noise_level,
            device=self.device,
            max_noise_correction=max_noise_correction,
        )

        return mean, covariance


# Helpful functions


def check_cov_pd(
    covariance_matrix: torch.Tensor,
    cov_noise_level,
    device,
    max_noise_correction: int = 20,
) -> torch.Tensor:
    """Check if a covariance matrix is Positive Definite (PD) if not keep adding noise to it till it becomes PD. If max_noise_correction is exceeded,
    return the identity matrix.

    Args:
        covariance_matrix (torch.Tensor): A matrix of shape (n, k, k)
        cov_noise_level (_type_): A noise value to be added to make the cov PD
        max_noise_correction (int, optional): Number of tries to correct the matrix, if exceeded return I.
        device (str, optional): The device to use.

    Returns:
        torch.Tensor: A PD covariance matrix with noise added to the original one or the identity matrix I of same shape.

    """
    cov_noise = (
        torch.eye(*covariance_matrix[0].size())
        .unsqueeze(0)
        .repeat(len(covariance_matrix), 1, 1)
        * cov_noise_level
    )  # (n, output_size, output_size)
    cov_noise = cov_noise.to(device)

    count_loop = 0
    while True:
        valid = constraints.positive_definite.check(covariance_matrix)
        if valid.all():
            return covariance_matrix
        else:
            error_index = torch.where(~valid)[0]
            covariance_matrix[error_index, :, :] = (
                covariance_matrix[error_index, :, :]
                + cov_noise * cov_noise_level
            )
            logger.warning(
                f"Covariance matrix is not positive definite at {len(error_index)} indices."
            )
            logger.warning(
                f"Adding {cov_noise_level}I to the matrix at those indices"
            )

            count_loop += 1
            logger.info(f"count_loop={count_loop}")
            if count_loop > max_noise_correction:
                covariance_matrix[error_index, :, :] = cov_noise
                logger.warning(
                    "Attempt to add more noise failed. Setting that covariance to I"
                )
                valid_loop = constraints.positive_definite.check(
                    covariance_matrix
                )
                np.save(
                    f"debug.array.{error_index}.npy",
                    covariance_matrix[error_index, :, :].detach().cpu().numpy(),
                )
                if valid_loop.all():
                    return covariance_matrix
                else:
                    logger.warning("Should not be here.")
                    return covariance_matrix
