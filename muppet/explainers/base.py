"""Base explainer module for the MUPPET XAI framework.

This module provides the fundamental `MuppetExplainer` class that serves as the
foundation for all explainable AI (XAI) methods in the MUPPET library. The base
explainer implements the core four-step perturbation-based workflow that all
MUPPET explainers follow:

1. **Explorer**: Manage exploration strategies for perturbation by generating masks
2. **Perturbator**: Applies perturbations to input data using the generated masks
3. **Attributor**: Calculates attribution scores from model predictions on perturbed data
4. **Aggregator**: Combines individual attributions into final explanations

The MUPPET framework decomposes XAI methods into these modular components that can
be mixed and matched to create different explanation techniques. This modular design
enables systematic comparison of XAI methods and facilitates the development of new
explainers by combining existing components.

Classes:
    MuppetExplainer: The parent class that all specific explainers must inherit from.
"""
#
# Created on Wed Jun 14 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

from typing import Any

import torch

# muppet standard components
from muppet import DEVICE, logger
from muppet.components.aggregator.base import Aggregator
from muppet.components.attributor.base import Attributor
from muppet.components.explorer.base import Explorer
from muppet.components.memory.base import Memory, PremiseList
from muppet.components.perturbator.base import Perturbator


class MuppetExplainer:
    """Base explainer class that orchestrates the four-component MUPPET framework.

    This class serves as the foundation for all explainable AI (XAI) methods in the MUPPET
    library, coordinating Explorer, Perturbator, Attributor, and Aggregator components
    to generate explanations for PyTorch models using perturbation-based methods.
    Supports multimodal data including images, tabular data, and time series.

    The base explainer handles device management, component coordination, and the
    standard explanation workflow. Specific XAI methods like RISE, LIME, and SHAP
    inherit from this class and customize the four components to implement their
    respective algorithms.

    The MUPPET framework decomposes XAI methods into modular components that can
    be mixed and matched to create different explanation techniques. This modular design
    enables systematic comparison of XAI methods and facilitates the development of new
    explainers by combining existing components.

    Example:
        Creating a custom explainer by combining components:

        ```python
        from muppet.components.explorer.mask import RandomMasksExplorer
        from muppet.components.perturbator.simple import SetToZeroPerturbator
        from muppet.components.attributor import ClassScoreAttributor
        from muppet.components.aggregator.mask import WeightedSumAggregator
        from muppet.explainers.base import MuppetExplainer

        # Define custom explainer by combining components
        class CustomExplainer(MuppetExplainer):
            def __init__(self, model, nmasks=100):
                explorer = RandomMasksExplorer(nmasks=nmasks)
                perturbator = SetToZeroPerturbator()
                attributor = ClassScoreAttributor()
                aggregator = WeightedSumAggregator()
                super().__init__(model, explorer, perturbator, attributor, aggregator)
        ```

    Note:
        This base class handles device management, component coordination, and the
        standard explanation workflow by implemeting the main __call__ for all explainers.
        This method handles the main logic of the MUPPET XAI framework,
        it should not be overrided.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        explorer: Explorer,
        perturbator: Perturbator,
        attributor: Attributor,
        aggregator: Aggregator,
        memory: Memory | None = None,
    ) -> None:
        """Initialize the base MUPPET explainer with the four core components.

        Args:
            model (torch.nn.Module): The black-box torch model to be explained.
            explorer (Explorer): It generates what is called "Premise" which will represent the perturbation-element.
            perturbator (Perturbator): Responsible for perturbing the input example.
            attributor (Attributor): Responsible for calculating the attribution which could be the model output directly or anything else
                that will be aggregated to calculate the final explanation.
            aggregator (Aggregator): Responsible for aggregating the calculated attribution and providing the final explanation.
            memory (Memory): A memory class where the "premises" are stored. E.g Tree, List, Set, ...
        """
        # register the custom components
        self.model = model
        self.explorer = explorer
        self.perturbator = perturbator
        self.attributor = attributor
        self.aggregator = aggregator
        if memory is None:
            self.memory = PremiseList()
        else:
            self.memory = memory

        # use cuda device otherwise use cpu
        self.device = DEVICE
        self.model.to(self.device)

        # share used device with all components
        self.explorer.device = self.device
        self.perturbator.device = self.device
        self.attributor.device = self.device
        self.aggregator.device = self.device
        self.memory.device = self.device

        # Prepare a spot for the premise_kwargs dict
        # This dict can be updated to have elements passed to Premises at their
        # creation : typically, by overcharging the __call__ function to compute
        # it before calling the original __call__ function. But it could also
        # have been modified by overcharging the __init__ instead to directly
        # store parameters : hence, we only create it if it does not already exist.
        if not hasattr(self, "premise_kwargs"):
            self.premise_kwargs = dict()

    def reinitialize(self):
        """Return the explainer to its original state."""
        self.memory.reinitialize()
        self.explorer.reinitialize()
        self.perturbator.reinitialize()
        self.attributor.reinitialize()
        self.aggregator.reinitialize()

    def __call__(
        self,
        example: torch.Tensor,
        premise_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """The explaining method that generates the explanation for a given example(s).

        Args:
            example (torch.Tensor): Batch of examples to be explained. Shape (b, *dims) where b is the number of input examples
            premise_kwargs (dict[str, Any]): Optional. A dictionary that will be passed to Premises at their creation ; generally used to store precomputed values and parameters.

        Returns:
            explanation (torch.Tensor): The final explanation holding the example(s)' explanations. Same shape as input example (b, *dims)

        """
        return self.__main_explainer_logic(
            example=example,
            premise_kwargs=premise_kwargs,
        )

    def __main_explainer_logic(
        self,
        example: torch.Tensor,
        premise_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """The explaining method that generates the explanation for a given example(s).

        Args:
            example (torch.Tensor): Batch of examples to be explained. Shape (b, *dims) where b is the number of input examples
            premise_kwargs (dict[str, Any]): Optional. A dictionary that will be passed to Premises at their creation ; generally used to store precomputed values and parameters.

        Returns:
            explanation (torch.Tensor): The final explanation holding the example(s)' explanations. Same shape as input example (b, *dims)

        """
        assert isinstance(example, torch.Tensor), (
            "Example must be a torch Tensor"
        )
        self.reinitialize()

        self.example = example
        self.example = self.example.type(torch.get_default_dtype())
        self.example = self.example.to(self.device)

        # If a new premise_kwargs was passed, over-ride the current one (which will usually be None,
        # as it is at __init__, unless __init__ was overcharging to precompute some parameters)
        if premise_kwargs is not None:
            self.premise_kwargs = premise_kwargs
        # Register it with the Explorer, since the Explorer will be the one to use it since it it the creator of Premises
        # We register it again to ensure the current one is passed
        self.explorer.premise_kwargs = self.premise_kwargs

        assert self.attributor.convention == self.aggregator.convention, (
            "Attribution convention between Attributor and Aggregator should correspond"
        )

        logger.debug("Default tensor type: " + str(torch.get_default_dtype()))
        logger.debug("Input example type:  " + str(self.example.type()))

        # Explorer: generates the premises to be explored
        for new_premises in self.explorer(
            example=self.example, memory=self.memory, model=self.model
        ):
            # Create the perturbation masks from generated premises
            masks = torch.stack([premise.mask for premise in new_premises]).to(
                self.device
            )  # Shape (N, *mask_shape) where N is the number of masks
            # and mask_shape is of the same dimensions as example.shape (len(mask_shape)==example.dim()). E.g (N, b=1, c, w, h)

            logger.debug(
                f"The number of generated masks at iteration {self.explorer.current_iteration} is "
                f"{len(masks)} of shape {masks[0].shape}"
            )

            # Register the premises into memory
            self.memory.register_premises(premises=new_premises)

            # PERTURBATE
            perturbed_inputs = self.perturbator(
                x=self.example, masks=masks
            )  # (N, *example.shape) where example.shape=(b=1, *dims)
            # dims depends on the data modality.
            # E.g
            #   images dims=(channels, width, height)
            #   timeseries dims=(nb_features, length)
            #   tabular dims=(nb_features)

            # Attributor: Calculate the attributions corresponding to every perturbation and then store them into memory
            self.attributor(
                x=self.example,
                perturbed_inputs=perturbed_inputs,
                model=self.model,
                memory=self.memory,
            )

            # Aggregator: Get current explanation
            # NOTE By default it is not stored, but a smart aggregator may
            # store in the memory.
            explanation = self.aggregator.get_explanation(memory=self.memory)

        # Return the latest calculated explanation
        return explanation
