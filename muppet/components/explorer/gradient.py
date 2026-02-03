"""Gradient-based Explorer Components for Optimization-driven exploration.

This module implements explorer components that use gradient descent optimization to
iteratively refine perturbation masks in the MUPPET XAI framework. These explorers
represent approaches where masks are not randomly generated but instead
optimized through backpropagation to maximize or minimize specific attribution objectives.

The gradient-based exploration strategy starts with initialized mask parameters and
iteratively updates them using gradient information from the model's predictions and
attributed loss. This allows for finding optimal perturbation patterns that reveal
the most informative aspects of the model's decision-making process.

These Explorers are aimed to be used in association with Differentiable attributors.

Classes:
    GradientExplorer: Base gradient-based explorer that optimizes mask grid using
        Adam optimizer over multiple iterations to find optimal perturbation grid,
        the grid is upscaled to the input image shape for perturbation.
    GradientCAMExplorer: Extension of GradientExplorer that incorporates Class Activation
        Maps (CAM) from convolutional layers to guide the optimization process with
        spatial feature information. Weights associated to each feature maps
        are optimized over the iterations.

The gradient exploration process:
    1. **Initialize**: Create learnable mask parameters with random or zero initialization
    2. **Forward**: Generate premises with current mask parameters
    3. **Backward**: Compute gradients from attribution scores via backpropagation
    4. **Optimize**: Update mask parameters using gradient descent (Adam optimizer)
    5. **Iterate**: Repeat until convergence or maximum iterations reached
    6. **Clamp**: Normalize mask values to [0,1] range after each update


"""
#
# Created on Fri Jun 30 2023
#
# Copyright (c) 2023 Ismail Bachchar @Euranova
#

from typing import List, Type

import torch

from muppet.components.explorer.base import Explorer
from muppet.components.memory import GradientPremise
from muppet.components.memory.base import Premise


class GradientExplorer(Explorer):
    """Gradient-based explorer for iterative mask grid optimization.

    This explorer implements optimization-driven grid exploration using an Adam
    optimizer to iteratively refine perturbation masks. The perturbation mask
    is derived from the grid by upscaling it to the input image shape.

    The gradient-based exploration strategy starts with initialized grid parameters and
    iteratively updates them using gradient information from the model's predictions and
    attribution loss. This allows for finding optimal perturbation patterns that reveal
    the most informative aspects of the model's decision-making process.

    Technical Details:
        - Uses Adam optimizer for stable and efficient mask parameter updates
        - Supports both spatial masks (for images) and feature masks (for tabular data)
        - Gradient flow enabled through premise.attribution.backward() calls
        - Automatic gradient accumulation and parameter clamping to valid ranges
        - Reinitializable for multiple explanation sessions

    Optimization Strategy:
        The gradient-based approach transforms the explanation problem into an optimization
        objective where the goal is to find mask parameters that maximize the loss usually
        derived from the model predictions of the perturbed inputs in order to reveal the most
        influential input regions or features. This behaviour is controled by the chosen
        attribution loss (see Differentiable attributors).

    """

    def __init__(
        self,
        max_iter: int = 100,
        lr: float = 0.2,
        mask_shape: tuple = (28, 28),
        premise_class: Type[Premise] = GradientPremise,
        nb_premises_at_startup: int = 1,
    ) -> None:
        """Initialize the gradient-based explorer.

        Args:
            max_iter (int, optional): The number of iterations for optimization. Defaults to 100.
            lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.2.
            mask_shape (tuple, optional): The learning mask shape. Defaults to (28, 28).
            premise_class (Type[Premise]): Premise class to create. Defaults to GradientPremise.
            nb_premises_at_startup (int, optional): Number of premises at startup. Defaults to 1.
        """
        self.max_iter = max_iter
        self.lr = lr
        self.learning_mask_shape = mask_shape
        self.premise_class = premise_class
        self.nb_premises_at_startup = nb_premises_at_startup

        self.optimizers = []

        super().__init__()

    def get_premises_to_explore(self) -> List[GradientPremise]:
        """Responsible for, at first, initializing the premises and at every subsequent call, calculating the gradients and
            doing one optimization step associated to each premise.

        The memory is list of premises corresponding to the number of input examples.

        Returns:
            List[GradientPremise]: List of created/updated premises.

        """
        # get premises
        premises = self.memory.get_premises()
        key_shape = (1, *self.learning_mask_shape)  # (1, w, h)

        # at first iteration when memory is still empty, generate the premises
        if len(premises) == 0:
            for _ in range(self.nb_premises_at_startup):
                key = torch.zeros(
                    key_shape,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=self.device,
                )

                premise = self.premise_class(
                    key=key,
                    upscaled_mask_shape=self.example.shape[2:],
                    **self.premise_kwargs,
                )
                premises.append(premise)
                self.optimizers.append(
                    torch.optim.Adam([premise.key], lr=self.lr)
                )

        # subsequent calls, update the premise
        else:
            for premise, optimizer in zip(premises, self.optimizers):
                optimizer.zero_grad()
                premise.attribution.backward(
                    retain_graph=True
                )  # retain_graph allows optimization on multiple premises which still share some elements of the compute graph (ie. the model)
                optimizer.step()

                # normalize the learning mask
                premise.key.data.clamp_(0, 1)

        # tell the main explainer to stop the exploration
        if self.current_iteration == self.max_iter:
            self.stop = True

        return premises

    def reinitialize(self):
        """Reset the gradient explorer to its initial state.

        Clears the list of optimizers and calls the parent reinitialize method.
        """
        self.optimizers = []
        return super().reinitialize()


class GradientCAMExplorer(GradientExplorer):
    """Gradient explorer with Class Activation Maps integration.

    Extends GradientExplorer by incorporating CAM (Class Activation Maps) from
    convolutional layers to guide mask optimization with spatial feature information.
    The CAM maps are stored in premise_kwargs and can be accessed by premises after creation.
    Here the optimized parameters are the weights associated to each CAM maps.

    Technical Details:
        - CAM variant extracts feature maps for initialization and guidance
        - Automatic gradient accumulation and parameter clamping to valid ranges
        - Reinitializable for multiple explanation sessions
    """

    def __init__(
        self,
        max_iter: int = 100,
        lr: float = 0.2,
        mask_shape: tuple[int, int] = (28, 28),
        premise_class: Type[Premise] = GradientPremise,
        nb_premises_at_startup: int = 1,
    ) -> None:
        """Initialize the GradientCAMExplorer with Class Activation Maps integration.

        Extends GradientExplorer by incorporating CAM (Class Activation Maps) from
        convolutional layers to guide mask optimization with spatial feature information.

        Args:
            max_iter (int): Number of optimization iterations for the Adam optimizer.
            lr (float): Learning rate for Adam optimizer.
            mask_shape (tuple[int, int]): Shape of the learnable mask parameters.
            premise_class (Type[Premise]): Class of premises to create,
                should have a backward attribute (usually a torch function computation
                for auto-différentiability) (default: GradientPremise).
            nb_premises_at_startup (int): Number of premises to generate at initialization.
        """
        self.cam_maps_were_obtained = False
        super().__init__(
            max_iter, lr, mask_shape, premise_class, nb_premises_at_startup
        )

    def get_cam_maps(self):
        """Extract CAM (Class Activation Maps) from the model.

        Registers a forward hook on the last convolutional layer to capture
        activations and stores them in premise_kwargs for use by premises.
        """
        if self.cam_maps_were_obtained is False:
            last_conv_layer = None

            for layer in reversed(list(self.model.modules())):
                if isinstance(layer, torch.nn.Conv2d):
                    last_conv_layer = layer
                    break

            if last_conv_layer is None:
                raise TypeError("given model has no Conv2d layer")

            # Placing a hook on the last Conv2d layer, to register layer activity during forward pass
            def hook(model, input, output):
                """Hook function to register layer activations during forward pass.

                Args:
                    model: The PyTorch model being hooked.
                    input: Input tensor to the layer.
                    output: Output tensor from the layer.
                """
                self.premise_kwargs["activations"] = output.detach()

            h = last_conv_layer.register_forward_hook(hook)

            # Forwarding input in model
            with torch.no_grad():
                self.model(self.example)

            # Removing hook
            h.remove()

            self.learning_mask_shape = (
                self.premise_kwargs["activations"].size()[1],
            )  # tuple

        self.cam_maps_were_obtained = True

    def reinitialize(self):
        """Reset the GradientCAM explorer to its initial state.

        Resets the CAM maps flag and calls the parent reinitialize method.
        """
        self.cam_maps_were_obtained = False
        return super().reinitialize()

    def get_premises_to_explore(self) -> List[GradientPremise]:
        """Get premises with CAM maps included.

        First extracts CAM maps, then calls the parent method to generate premises.

        Returns:
            List[GradientPremise]: List of premises with CAM data available.
        """
        self.get_cam_maps()
        return super().get_premises_to_explore()
