"""Base attributor component for MUPPET XAI.

This module defines the abstract base class for all attributors in the MUPPET XAI
framework. Attributors are responsible for calculating attribution scores from model
predictions on perturbed inputs. These attributions quantify how much each perturbation
affects the model's output and serve as the basis for feature importance calculations.

The attribution process is the third step in the four-step perturbation-based XAI
approach: generate masks → apply perturbations → calculate attributions → aggregate results.

Classes:
    Attributor: Abstract base class defining the interface for all attributor components.
"""

from abc import ABC, abstractmethod

import torch

from muppet import DEVICE, logger
from muppet.components.convention import AttributionConvention
from muppet.components.memory.base import Memory


class Attributor(ABC):
    """Abstract base class for attributor components in MUPPET XAI.

    A global component that defines the 'calculate_attribution' method which is responsible
    for filling-up the premises' attribution. An attribution could be the model's output
    or something else that will be used by the aggregator to find the final heatmap.

    Attributes:
        device: The used device. It gets updated from the main explainer after initialization.
        convention: The attribution convention (constructive or destructive).
        allowed_conventions: The allowed convention types from AttributionConvention.

    Example:
        Typical usage involves subclassing the Attributor base class:

        ```python
        class CustomAttributor(Attributor):
            def __init__(self, convention="destructive"):
                self.convention = convention
                super().__init__()

            def calculate_attribution(self, x, perturbed_inputs, model, memory):
                # Calculate attributions and store in memory
                predictions = model(perturbed_inputs)
                # Custom attribution logic here
                pass
        ```
    """

    allowed_conventions = AttributionConvention

    def __init__(self) -> None:
        """Initialize the attributor component.

        Sets up the default device and attribution convention for the attributor.
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
        """Reset the attributor to its initial state.

        This method restores the attributor to its original configuration,
        clearing any internal state or cached data that may affect
        subsequent attribution calculations.
        """
        pass

    @abstractmethod
    def calculate_attribution(
        self,
        x: torch.Tensor,
        perturbed_inputs: torch.Tensor,
        model: torch.nn.Module,
        memory: Memory,
    ) -> None:
        """Calculates the attribution based on the example's perturbations (x'), the model and the original example (x) if needed.

        It is generally expected that the most impactful masks will get the highest attribution score : indeed, at the end of the pipeline,
        most Aggregators will use a mask's attribution as a direct proxy for its importance.

        Args:
            x (torch.Tensor): The original input example. Shape (b=1, *x.shape[1:]):
                - b is the batch size
                - x.shape[1:] is the input data dimensions. E.g images (c, w, h) channels, width and height.

            perturbed_inputs (torch.Tensor): The perturbations calculated by the Perturbator.

            model (torch.nn.Module): The black-box model.

            memory (Memory): The used memory structure.

        Returns:
            None: It fills up the memory in place.

        """
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        perturbed_inputs: torch.Tensor,
        model: torch.nn.Module,
        memory: Memory,
    ) -> None:
        """Execute attribution calculation by calling the calculate_attribution method.

        This method provides a convenient callable interface for the attributor
        and delegates to the abstract calculate_attribution method.

        Args:
            x (torch.Tensor): The original input example.
            perturbed_inputs (torch.Tensor): The perturbed input variations.
            model (torch.nn.Module): The target model for attribution.
            memory (Memory): Memory structure to store attribution results.
        """
        logger.debug("Computing attribution for x.shape[0] perturbations")
        return self.calculate_attribution(
            x=x, perturbed_inputs=perturbed_inputs, model=model, memory=memory
        )

    @property
    def convention(self):
        """Get the attribution convention."""
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
