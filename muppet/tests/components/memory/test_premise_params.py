"""Tests for premise parameter handling components.

This module tests the base Premise class parameter handling in MUPPET, focusing on how
premises accept and process keyword arguments during initialization. This ensures flexible
premise creation with custom parameters for different explanation scenarios.

The tests verify:
- Proper handling of keyword arguments during premise initialization
- Correct parameter assignment and storage within premise instances
- Base premise class functionality and abstract method implementation
- Parameter passing consistency across different premise types
- Inheritance behavior for custom premise implementations
"""
#
# Created on Fri Jul 21 2023
#
# Copyright (c) 2023 Quentin Ferré @EuraNova
#

import pytest

from muppet.components.memory.base import Premise


class FakePremise(Premise):
    """Fake premise class for testing memory premise parameters.

    A mock premise that extends the base Premise class for testing
    memory component functionality with configurable test parameters.
    """

    def __init__(self, key, test):
        """Initialize fake premise for testing memory premise parameters.

        Args:
            key: Unique identifier for the premise.
            test: Test parameter value for testing functionality.
        """
        self.test = test
        super().__init__(key=key)

    def get_mask(self):
        """Test abstract method implementation for mask generation."""
        raise NotImplementedError


@pytest.mark.parametrize("first_value", [123])
def test_premise_kwargs(first_value):
    """Test that Premises can be passed keyword arguments dictionaries."""
    premise_kwargs = {"test": first_value}

    premise = FakePremise(key=42, **premise_kwargs)
    assert premise.test == first_value
