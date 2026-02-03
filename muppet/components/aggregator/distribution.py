"""Distribution-based aggregator for time series classification explanations.

This module provides aggregators that work with probability distributions over class
predictions in time series classification tasks. It implements Monte Carlo aggregation
methods to compute feature importance scores from KL divergence measurements between
original and perturbed predictions.

The aggregator groups Monte Carlo samples by time steps and features, calculating
final attribution scores through statistical aggregation of the KL divergences.

Classes:
    MonteCarloKLAggregator: Aggregates attributions using Monte Carlo sampling and
                           KL divergence for time series classification tasks.
"""
#
# Created on Tue Apr 18 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import torch

from muppet.components.aggregator.base import Aggregator
from muppet.components.memory.base import PremiseList


class MonteCarloKLAggregator(Aggregator):
    """Monte Carlo aggregator using KL divergence for time series classification.

    This aggregator works on probability distributions over classes for classification tasks.
    It aggregates attributions using Monte Carlo sampling and KL divergence measurements
    to compute feature importance scores for time series data.

    The aggregator groups Monte Carlo samples by time steps and features, calculating
    final attribution scores through statistical aggregation of KL divergences between
    original and perturbed predictions.

    """

    def __init__(
        self,
        num_sampling: int,
    ) -> None:
        """Initialize the Monte Carlo KL divergence aggregator.

        Args:
            num_sampling: The number of Monte-Carlo sampling iterations.
        """
        self.num_sampling = num_sampling
        # check that it is the case
        self.convention = "destructive"
        super().__init__()

    def get_explanation(
        self,
        memory: PremiseList,
    ) -> torch.Tensor:
        """Calculates the final heatmap by grouping the Monte-Carlo samples premises of the same time-step,
        then aggregating over their attributions in order to calculate the final time-step's score.


        Args:
            memory (Premiselist): List of premises.

        Returns:
            torch.Tensor: The final heatmap where at every timestep the score $score(t, S)$ is calculated. Shape (b, f, t)

        """
        temp_premise = memory.get_premises()[0]

        nb_features, signal_length = temp_premise.key[1]  # (L, (f, t)) => t

        heatmap = torch.zeros((nb_features, signal_length))
        for monte_carlo_premises in self._splitter(
            memory.get_premises(), self.num_sampling
        ):
            kl_divs = torch.stack(
                [premise.attribution for premise in monte_carlo_premises]
            ).to(self.device)  # (num_sampling, b)
            E_div = torch.mean(kl_divs, axis=0)  # (b, t)

            score_timestep = 2.0 / (1 + torch.exp(-5 * E_div)) - 1

            time_step = monte_carlo_premises[0].key[0]["timestep"]
            feature = monte_carlo_premises[0].key[0]["feature"]

            heatmap[feature, time_step] = score_timestep

        return heatmap.unsqueeze(dim=0)  # (b, f, t)

    # TODO should be moved to the memory object
    def _splitter(self, iter, n):
        for i in range(0, len(iter), n):
            yield iter[i : i + n]
