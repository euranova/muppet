"""Attribution conventions for MUPPET XAI framework.

This module defines the attribution conventions used throughout the MUPPET XAI framework
to standardize how perturbations attributions are interpreted across different components.
Attribution conventions determine whether high attribution values indicate features that
are important when present (constructive) or when absent (destructive).

These conventions ensure consistency between explorers, perturbators, attributors, and
aggregators when generating and interpreting explanations. The choice of convention
affects how perturbation effects are measured and how final explanations are presented
to users. Some components (like RISE components) accept desctructive and constructive
conventions but not all. To ensure API coherence a third convention has been introduced
for local explanable surrogate model attributors and aggregators (LIME like explainers):
PERTURBED_INPUT_SIMILARITY.


Classes:
    AttributionConvention: Enumeration of supported attribution conventions.

Note:
    The convention choice should align with the underlying XAI method's philosophy.
    Most methods use DESTRUCTIVE convention as the default, measuring how much
    performance degrades when features are removed or perturbed.
"""

import enum


class AttributionConvention(str, enum.Enum):
    """Enumeration defining attribution conventions for perturbation-based explanations.

    This class defines the different conventions used to interpret attribution scores
    in the context of perturbation-based explainable AI methods. The convention affects
    how attribution values are calculated and interpreted across the framework.

    Attributes:
        CONSTRUCTIVE: Attribution is high when non-perturbed features retrieve
            the model's behavior starting from a completely perturbed input.
        DESTRUCTIVE: Attribution is high when the perturbation destroys
            the model's behavior from the reference input. This is the main default.
        PERTURBED_INPUT_SIMILARITY: For local surrogate models, high attribution
            occurs when the perturbed input is similar to the reference input.
    """

    CONSTRUCTIVE = "constructive"
    # Attribution is high when the non perturbed features retrieves
    # the model's behaviour starting from a completely perturbed input"

    DESTRUCTIVE = "destructive"  # main default param
    # Attribution is high when the perturbation destroys
    # the model's behaviour from the reference input"

    PERTURBED_INPUT_SIMILARITY = "perturbed_input_similarity"
    # For local surrogate model we have a specific convention
    # which relies on high attribution when
    # the perturbed input is similar to the reference input
