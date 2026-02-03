"""Feature-based Explorer Components for Convolutional Neural Networks.

This module provides explorer implementations that leverage convolutional feature maps
to generate perturbation masks in the MUPPET XAI framework. These explorers are specifically
designed for explaining convolutional neural networks by using the spatial feature activations
from the last convolutional layer to guide the perturbation process.

The feature-based exploration strategy generates masks based on individual feature maps
(channels) from the final convolutional layer, creating targeted perturbations that help
understand which spatial regions and features contribute most to model predictions. This
approach is particularly effective for image classification tasks where spatial locality
and feature hierarchy are important.

Classes:
    CAMExplorer: Generates one premise per feature map in the last convolutional layer,
        creating class activation map (CAM) style explanations through feature-guided
        perturbations.

The feature exploration process:
    1. **Extract Features**: Hook into the last convolutional layer during forward pass
    2. **Generate Premises**: Create one premise per feature channel with spatial upsampling
    3. **Enable Perturbation**: Each premise contains upsampled feature activations as masks
    4. **Single-shot Explanation**: Completes exploration in one iteration
"""
#
# Created on Wed Nov 29 2023
#
# Copyright (c) 2023 Léo Beaumont @Euranova
#

from typing import List

import torch.nn

from muppet.components.explorer.base import Explorer
from muppet.components.memory import ConvolutionalFeaturePremise


class CAMExplorer(Explorer):
    """Class Activation Map (CAM) explorer for convolutional neural networks.

        Generates perturbation premises based on feature maps from the last
        convolutional layer. Each feature map is used to create a mask premise
        for explaining the contribution of that particular feature to the model's
        predictions.

        This explorer leverages convolutional feature maps to generate perturbation masks
        in the MUPPET XAI framework. It is specifically designed for explaining convolutional
        neural networks by using the spatial feature activations from the last convolutional
        layer to guide the perturbation process.

        The feature-based exploration strategy generates masks based on individual feature maps
        (channels) from the final convolutional layer, creating targeted perturbations that help
        understand which spatial regions and features contribute most to model predictions. This
        approach is particularly effective for image classification tasks where spatial locality
        and feature hierarchy are important.

    Technical Details:
        - Works with any PyTorch model containing Conv2d layers
        - Automatically finds the last convolutional layer in the model architecture
        - Uses bilinear interpolation to upscale feature maps to input resolution
        - Generates exactly k premises where k is the number of output channels
        - Each premise represents one feature channel's spatial contribution
        - Suitable for models like ResNet, VGG, DenseNet, etc.

    Note:
        This explorer requires the model to contain at least one Conv2d layer.
        The exploration completes in a single iteration, making it computationally
        efficient for generating pixel-importance-based explanations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        """Initialize the CAM explorer.

        Args:
            model (torch.nn.Module): The model to explain.
        """
        self.example = None
        self.conv_layer_activation = None
        super().__init__(model=model)

    def get_premises_to_explore(self) -> List[ConvolutionalFeaturePremise]:
        """Generate a `k * b` number of premises where every one corresponds to the perturbation of the input example.
        Expects 4D input example. Shape (b=1, c, h, w).

         where
            - k is the amount of features in last convolutional layer of the model,
            - b is batch dimension, expected to be set to 1 as only one example is being explained for the moment,
            - c is the channel dimensions,
            - w is the width,
            - h is the height,

        Returns:
            List[ConvolutionalFeaturesPremise]: Every premise includes the necessary information to generate its masks from the key attributes.

        """
        upscaled_mask_shape = self.example.size()[-2:]

        # Finding last Conv2d layer of the model
        last_conv_layer = None

        for layer in reversed(list(self.model.modules())):
            if isinstance(layer, torch.nn.Conv2d):
                last_conv_layer = layer
                break

        if last_conv_layer is None:
            raise TypeError("given model has no Conv2d layer")

        # Placing a hook on the last Conv2d layer, to register layer activity during forward pass
        def hook(model, input, output):
            """Hook function to register convolutional layer activations.

            Args:
                model: The PyTorch model being hooked.
                input: Input tensor to the convolutional layer.
                output: Output tensor from the convolutional layer.
            """
            self.conv_layer_activation = output.detach()

        h = last_conv_layer.register_forward_hook(hook)

        # Forwarding input in model
        with torch.no_grad():
            self.model(self.example)

        # Removing hook
        h.remove()

        upsampled_activations = torch.nn.functional.interpolate(
            self.conv_layer_activation,
            size=upscaled_mask_shape,
            mode="bilinear",
        ).to(self.device)

        premises = []
        for channel in range(self.conv_layer_activation.size()[1]):
            premise = ConvolutionalFeaturePremise(
                key=(upsampled_activations, channel)
            )
            premises.append(premise)

        # tell the main explainer to stop the exploration
        self.stop = True

        return premises
