"""Base Explorer Component for MUPPET XAI Framework.

This module provides the abstract base class for all exploration strategies in the MUPPET framework.
Explorers are the first component in the four-step perturbation-based XAI process,
responsible for generating masks and exploration strategies that define how input
data will be systematically perturbed to understand model behavior.

The Explorer component serves as foundation for all perturbation-based explanation
methods by defining the logic for mask generation strategies. Different exploration
approaches can be implemented by extending this base class with specific mask generation
logic suitable for various data modalities (images, tabular data, time series) and
explanation requirements.

Classes:
    Explorer: Abstract base class defining the exploration interface for generating
        perturbation premises. Provides iteration protocol, state management, and
        premise generation framework.

The four-step MUPPET process:
    1. **Explorer** (this module): Generate masks and exploration strategies
    2. **Perturbator**: Apply masks to create perturbed inputs
    3. **Attributor**: Calculate feature scores from model predictions on perturbed data
    4. **Aggregator**: Combine attributions to produce final explanations


Note:
    All Explorer implementations must implement the `get_premises_to_explore()` method
    and properly manage the `stop` flag to indicate when exploration is complete.
    The explorer follows an iterator protocol and maintains state across exploration
    iterations through the `current_iteration` counter.
"""

#
# Created on Wed Jun 14 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#
from abc import ABC, abstractmethod
from typing import Iterable

import torch

from muppet.components.memory.base import Memory, Premise


class Explorer(ABC):
    """Base class for exploration strategies in XAI perturbation-based methods.

    An Explorer generates perturbation premises that define how input examples
    should be perturbed for explanation purposes. It follows the four-step
    perturbation approach: generate masks, apply masks, calculate scores,
    and aggregate attributions.

    The Explorer serves as the foundation for all perturbation-based explanation
    methods by defining the interface for mask generation strategies. Different
    exploration approaches can be implemented by extending this base class with
    specific mask generation logic suitable for various data modalities (images,
    tabular data, time series) and explanation requirements.

    """

    def __init__(
        self, example=None, memory=None, model=None, device=None
    ) -> None:
        """Initialize the Explorer base class.

        Args:
            example (torch.Tensor, optional): The example to be explained. Will be set at runtime.
            memory (Memory, optional): The memory where premises are saved.
            model (optional): The model being explained.
            device (optional): The device to use. Will be updated from the main explainer.
        """
        self._stop = False
        self._current_iteration = 0
        self._premise_kwargs = {}

        # Initial arguments. These will usually be None
        self.example = example
        self.memory = memory
        self.model = model
        self.device = device

        super().__init__()

    def reinitialize(self):
        """Reset the explorer to its initial state.

        Clears the stop flag and resets the iteration counter to prepare
        for a new exploration session.
        """
        self.stop = False
        self.current_iteration = 0

    @property
    def stop(self):
        """Get the exploration stop flag."""
        return self._stop

    @stop.setter
    def stop(self, value: bool):
        """Set the exploration stop flag.

        Args:
            value (bool): True to stop exploration, False to continue.
        """
        self._stop = value

    @property
    def current_iteration(self):
        """Get the current iteration counter."""
        return self._current_iteration

    @current_iteration.setter
    def current_iteration(self, value: int):
        """Set the current iteration counter.

        Args:
            value (int): The iteration number to set.
        """
        self._current_iteration = value

    @property
    def premise_kwargs(self):
        """Get the premise keyword arguments."""
        return self._premise_kwargs

    @premise_kwargs.setter
    def premise_kwargs(self, premise_kwargs: dict):
        """Set the premise keyword arguments.

        Args:
            premise_kwargs (dict): Dictionary of keyword arguments to pass to premises.
        """
        self._premise_kwargs = premise_kwargs

    def __call__(self, example: torch.Tensor, memory: Memory, model=None):
        """Generate masks for the given example using the exploration strategy."""
        # if example not set, set it. Same for model.
        if example is not None:
            self.example = example
        if model is not None:
            self.model = model

        # always re-set it with the received memory
        self.memory = memory
        return self

    def __iter__(self):
        """Make the explorer iterable.

        Returns:
            Explorer: Self as an iterator.
        """
        return self

    def __next__(self):
        """Get the next set of premises to explore.

        Returns:
            Iterable[Premise]: The next batch of premises.

        Raises:
            StopIteration: When exploration should stop.
        """
        if self.stop:
            raise StopIteration()
        return self.next()

    def next(self):
        """Generate the next batch of premises for exploration.

        Increments the iteration counter, generates premises, and sets
        their device before returning them.

        Returns:
            Iterable[Premise]: The premises to explore in this iteration.
        """
        # increase iteration
        self._current_iteration += 1
        premises_to_explore = self.get_premises_to_explore()

        # Set device for all premises
        for p in premises_to_explore:
            p.device = self.device

        return premises_to_explore

    @abstractmethod
    def get_premises_to_explore(self) -> Iterable[Premise]:
        """The premises' generator. Creates the premises objects and sends them back to main explainer.

        Returns:
            Iterable[Premise]: An iterable over the created premises.

        """
        raise NotImplementedError
