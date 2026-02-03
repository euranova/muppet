"""Base aggregator component for MUPPET XAI.

This module defines the abstract base class for all aggregators in the MUPPET XAI
framework. Aggregators are responsible for combining individual feature attributions
calculated by attributors to produce the final local explanations (heatmaps, feature
importance scores, etc.).

The aggregation process is the final step in the four-step perturbation-based XAI
approach: generate masks → apply perturbations → calculate attributions → aggregate results.

Classes:
    Aggregator: Abstract base class defining the interface for all aggregator components.
"""
#
# Created on Wed Jun 14 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

from abc import ABC, abstractmethod

import torch

from muppet import DEVICE, logger
from muppet.components.convention import AttributionConvention
from muppet.components.memory.base import Memory


class Aggregator(ABC):
    """Abstract base class for aggregator components in MUPPET XAI.

    A global aggregator component that is responsible for aggregating the attribution
    calculated by the attributor in order to generate the final heatmap.

    Attributes:
        device: The used device. Will get updated from the main explainer after initialization.
        convention: The attribution convention (constructive or destructive).
        allowed_conventions: The allowed convention types from AttributionConvention.

    Example:
        Typical usage involves subclassing the Aggregator base class:

        ```python
        class CustomAggregator(Aggregator):
            def __init__(self, convention="destructive"):
                self.convention = convention
                super().__init__()

            def get_explanation(self, memory):
                # Custom aggregation logic
                return final_heatmap
        ```
    """

    allowed_conventions = AttributionConvention

    def __init__(self) -> None:
        """Initialize the aggregator component.

        Sets up the default device and attribution convention for the aggregator.
        If no convention is set, defaults to 'destructive' with a warning.
        """
        self.device = DEVICE

        # Check that the convention has been set, if not set it with warning
        if not hasattr(self, "convention"):
            logger.warning(
                "Convention should be set for the Aggregator, by default let set it to 'destructive'"
            )
            self.convention = "destructive"

        super().__init__()

    def reinitialize(self):
        """Reset the aggregator to its initial state.

        This method restores the aggregator to its original configuration,
        clearing any internal state or accumulated data that may affect
        subsequent aggregation operations.
        """
        pass

    @abstractmethod
    def get_explanation(
        self,
        memory: Memory,
    ) -> torch.Tensor:
        """A custom method that calculates the final explanations as a heatmap of the same shape as the input. To do so, it uses the stored attributions in memory.

        Args:
            memory (Memory, optional): The used memory structure where premises are stored.

        Returns:
            torch.Tensor: The final explanation in the form of a heatmap of shape input.

        """
        raise NotImplementedError

    @property
    def convention(self):
        """Get the aggregation convention."""
        return self._convention

    @convention.setter
    def convention(self, value):
        try:
            value = self.allowed_conventions(value)
        except ValueError:
            raise ValueError(
                "Attributor convention should be either constructive or destructive"
            )

        self._convention = value
