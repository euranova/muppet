"""Base Memory Components for MUPPET XAI Framework.

Provides the abstract base classes for memory management in the
MUPPET XAI framework.
Memory components **store and manage premises**, which represent perturbation
data (masks, keys, attribution results) throughout the explanation process.

Memory bridges the four-step process (exploration, perturbation, attribution,
aggregation) by maintaining state via **Premise** objects.

Classes:
    Premise: Abstract base class for a single perturbation point. Manages key,
        mask, attribution, and provides **lazy evaluation with caching**.
    Memory: Abstract base class defining the interface for storing and
        retrieving **Premise** collections.
    PremiseList: Simple, list-based implementation of **Memory**.

Mask Convention (Consistent Binary Format):
    - 0: Preserve the original input value (no perturbation)
    - 1: Perturb the input value (apply perturbation strategy)

Technical Note:
    Premises implement **key-changed detection** to trigger dynamic recomputation
    of cached masks and explanations, supporting both static and trainable methods.
"""

from abc import ABC, abstractmethod
from typing import Iterable, List

import torch


class Premise(ABC):
    """Abstract base class for a single perturbation premise.

    A premise is the fundamental unit of exploration, containing the key for
    deterministic mask generation, the mask itself, and attribution results.
    It utilizes **lazy evaluation and caching** for efficiency.

    The **key-changed** mechanism ensures recomputation when parameters are modified.
    Mask Convention: 0 = preserve input, 1 = perturb input.
    """

    def __init__(
        self,
        key: object,
        **kwargs,
    ) -> None:
        """Initialize the Premise with a perturbation key.

        Args:
            key (object): The key used to generate the mask deterministically.
            kwargs (dict[str, Any]): Additional premise-specific arguments.
        """
        self._key = key
        # Store a copy of the initial key for change detection
        if isinstance(key, torch.Tensor):
            self.last_key = key.detach().clone()
        else:
            self.last_key = key
        self._attribution = None
        self._mask = None
        self._heatmap = None
        self.device = None

        super().__init__()

    @property
    def key(self):
        """Get the premise key."""
        return self._key

    @property
    def attribution(self):
        """Get or set the premise attribution result."""
        return self._attribution

    @attribution.setter
    def attribution(self, value: object):
        """Set the attribution value for this premise."""
        self._attribution = value

    def key_changed(self):
        """Check if the key has changed since last computation."""
        if isinstance(self.last_key, torch.Tensor):
            return torch.any(self._key != self.last_key)
        else:
            return self._key != self.last_key

    @property
    def heatmap(self):
        """Get the heatmap explanation (cached and recomputed if key changed)."""
        if self._heatmap is None:
            self._heatmap = self.get_explanation()
        elif self.key_changed():
            if isinstance(self._key, torch.Tensor):
                self.last_key = self.key.detach().clone()
            else:
                self.last_key = self.key
            self._heatmap = self.get_explanation()
        return self._heatmap

    def get_explanation(self) -> torch.Tensor:
        """Generate explanation from the premise key (default: returns mask)."""
        # Note: Call self.mask to use the cached/lazy-evaluated mask
        return self.mask

    @property
    def mask(self):
        """Retrieve mask from premise

        Returns:
            torch.key: mask following mask perturbation convention
                1 for pertrubed features and 0 for non perturbed features
        """
        if self._mask is None:
            self._mask = self.get_mask()
        elif self.key_changed():
            if isinstance(self._key, torch.Tensor):
                self.last_key = self.key.detach().clone()
            else:
                self.last_key = self.key
            self._mask = self.get_mask()
        if self._mask is not None:
            self._mask = self._mask.to(self.device)
        return self._mask

    @abstractmethod
    def get_mask(self):
        """The abstract method to map the key to a mask."""
        raise NotImplementedError


class Memory(ABC):
    """Abstract base class for memory structures in XAI exploration.

    Manages the storage and retrieval of Premise objects, bridging the
    perturbation process by maintaining the state (premises) between phases.
    """

    def __init__(self) -> None:
        """Initialize the basic Memory structure."""
        self.device = None
        super().__init__()

    def reinitialize(self):
        """Reset the memory to its initial state."""
        raise NotImplementedError

    @abstractmethod
    def register_premises(
        self,
        premises: Iterable[Premise],
    ) -> None:
        """Receives and stores an iterable of premises."""
        raise NotImplementedError

    @abstractmethod
    def get_premises(self) -> Iterable[Premise]:
        """Returns the stored premises from memory."""
        raise NotImplementedError


class PremiseList(Memory):
    """Simple list-based memory implementation for storing premises.

    Provides basic, in-memory sequential storage with efficient retrieval.
    Premises are replaced on new registration.
    """

    def __init__(self) -> None:
        """Initialize the PremiseList memory structure."""
        self._premises = []

        super().__init__()

    def reinitialize(self):
        """Return the Premiselist to its original state."""
        self._premises = []

    def register_premises(self, premises: Iterable[Premise]) -> None:
        """Register a collection of premises, replacing any existing premises.

        Args:
            premises (Iterable[Premise]): The premises to store in memory.
        """
        self._premises = list(premises)

    def get_premises(self) -> List[Premise]:
        """Return the list of stored premises

        Returns:
            List[Premise]: The stored premises.
        """
        return self._premises
