"""Timestep-based Explorer Components for Sequential Data Explanations.

Provides explorer implementations for explaining sequential and time series
models in the MUPPET XAI framework. Focuses on **temporal relationships**
and **feature interactions** across time steps using perturbation masks.

Strategy: Systematically perturbs **(timestep, feature)** pairs to assess
individual contributions in sequential models (e.g., RNNs, Transformers).

Classes:
    RepeatedTimestepExplorer: Generates premises for each (timestep, feature)
        combination using **Monte Carlo sampling** (num_sampling).

Technical Summary:
    - Explores timesteps $t \in [1, \text{signal\_length}-1]$. Timestep 0 is excluded.
    - Total Premises $= (\text{T}-1) \times \text{F} \times \text{S}$, where $\text{S} = \text{num\_sampling}$.
    - Input Format Assumption: (batch, features, timesteps).
"""
#
# Created on Tue Apr 18 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

import itertools
from typing import List

from muppet.components.explorer.base import Explorer
from muppet.components.memory import TimeStepPremise


class RepeatedTimestepExplorer(Explorer):
    """Timestep explorer for temporal sequence explanations.

    Generates premises for explaining time series by perturbing different
    **(timestep, feature)** pairs. Uses **Monte Carlo sampling** ($\text{num\_sampling}$)
    for statistical robustness.

    Crucial for understanding how sequential models (RNNs, Transformers) rely on
    **temporal dependencies** and **feature interactions** over time.

    Technical Details:
        - **Coverage**: Timesteps $t \in [1, \text{signal\_length}-1]$ and all features.
        - **Total number of generated premises**: $(\text{T}-1) \times \text{F} \times \text{num\_sampling}$.
    """

    def __init__(
        self,
        num_sampling: int = 100,
    ) -> None:
        """Initialize the RepeatedTimestepExplorer.

        Args:
            num_sampling (int): Number of Monte Carlo samples per timestep-feature pair. Defaults to 100.
        """
        self.num_sampling = num_sampling
        super().__init__()

    def get_premises_to_explore(self) -> List[TimeStepPremise]:
        """Generates all TimeStepPremise objects for exploration in a single pass.

        The premises cover all combinations of timesteps $t \in [1, \text{signal\_length}-1]$
        and features, each repeated $\text{num\_sampling}$ times.

        Returns:
            List[TimeStepPremise]: List of premises.
        """
        feature_size = self.example.shape[1]
        signal_length = self.example.shape[2]
        premises = []
        # Iterate over timesteps, features, and samples
        for i, j, _ in itertools.product(
            range(signal_length - 1, 0, -1),  # Timesteps 1 to T-1
            set(range(self.example.shape[1])),  # Features
            range(self.num_sampling),  # Samples
        ):
            # TODO enventually other exploring strategy for masks
            key = (
                {"timestep": i, "feature": j},
                (feature_size, signal_length),
            )
            premises.append(TimeStepPremise(key=key, **self.premise_kwargs))

        # Stop exploration after this single pass
        self.stop = True

        return premises
