"""Concrete Premise Implementations for MUPPET XAI Framework.

Provides specialized **Premise** classes for different perturbation strategies
and data modalities in the MUPPET framework. Each class generates a concrete
perturbation **mask** from an abstract **key**, supporting various XAI methods
(e.g., LIME, SHAP, gradient-based).

Classes:
    TimeStepPremise: Generates masks for time series data by masking a specific features
        at a specific timesteps, keep future timesteps in the mask as NaN (to enable
        different padding strategy).
    BinaryRandomPremise: Creates random binary masks with configurable preservation
        probability, supporting both image and tabular data modalities.
    SegmentedBinaryImagePremise: Generates masks based on image segmentation, allowing
        segment-wise perturbation using pre-computed segmentation maps.
    GradientPremise: Provides optimizable masks for optimization-based perturbation strategies.
    ConvolutionalFeaturePremise: Creates masks from convolutional neural network feature
        activations, enabling feature-maps based explanations.
    FeaturesCombinationPremise: Generates masks from linear combinations of CNN features,
        supporting learnable feature selection and combination.
    KeyBasedMaskPremise: Direct key-to-mask mapping for cases where keys are already
        in mask format, providing minimal processing overhead.

Supported Modalities:
    - **Time Series**: Timestep-based perturbations with temporal masking
    - **Images**: Pixel-wise, segment-wise, and feature-based perturbations
    - **Tabular Data**: Feature-wise random perturbations
    - **CNN Features**: Activation-based and gradient-optimized perturbations

Mask Convention:
    All premise types follow the MUPPET binary mask convention:
    - 0: Preserve the original input value (no perturbation)
    - 1: Perturb the input value (apply perturbation strategy)


Technical Implementation Details:
    - Lazy evaluation with caching for computational efficiency
    - Device-aware tensor management for GPU acceleration
    - Deterministic mask generation from keys for reproducibility
    - Support for both static, trainable and optimizable perturbation strategies
    - Automatic mask resizing and interpolation for different input sizes
    - Integration with scikit-image for advanced image processing operations

The premise implementations support the full MUPPET four-step process:
    1. **Explorer**: Generates diverse premise keys for systematic perturbation
    2. **Perturbator**: Uses premise masks to create perturbed input variations
    3. **Attributor**: Stores model predictions and attribution results in premises
    4. **Aggregator**: Combines premise attributions into final explanations

"""

import numpy as np
import torch
from skimage.transform import resize

from muppet.components.memory.base import Premise


class TimeStepPremise(Premise):
    """Premise for temporal perturbations in time series data.

    Creates masks that perturb time series data by masking the current time step (mask=1)
    for perturbation. It sets future values to Nan to enable various padding strategie.
    Useful for explaining sequential models where temporal dependencies are important.
    """

    def __init__(
        self,
        key: tuple,
    ) -> None:
        """A timestep premise object.

        Attributes:
            key (tuple): Holds the timestep index, and the mask shape. Enough to generate the corresponding mask.
                         Expected structure: ({"timestep": int, "feature": int}, (num_features, sequence_length))

        """
        super().__init__(key=key)

    def get_mask(self) -> torch.Tensor:
        """Generates a mask for a specific feature at a specific timestep.

        The mask is a 2D tensor with an added batch dimension.
        - The value at `mask[feature, time_step]` is set to 1.
        - All values for timesteps greater than `time_step` are set to `torch.nan`.
        - All other values are 0.

        The key attribute is expected to be a tuple:
        - `self.key[0]` (dict): A dictionary containing 'timestep' and 'feature' indices.
        - `self.key[1]` (tuple): The shape of the mask (e.g., (number_of_features, sequence_length)).

        Returns:
            torch.Tensor: A 3D tensor representing the mask, with shape (1, num_features, sequence_length).
                          The first dimension is the batch size (1).
        """
        # Extract timestep and feature index from the key
        time_step = self.key[0]["timestep"]
        feature = self.key[0].get(
            "feature", 0
        )  # default to zero for single channel time series
        # Extract the desired mask shape (e.g., [num_features, sequence_length])
        mask_shape = self.key[1]

        # Initialize a mask of zeros with the given shape
        mask = torch.zeros(mask_shape)  # Shape: (num_features, sequence_length)
        # Set the specific (feature, time_step) to 1, indicating the point of interest
        mask[feature, time_step] = 1
        # Set all subsequent time steps to NaN, indicating they are not considered in this premise
        # This applies to all features from time_step + 1 onwards.
        mask[:, time_step + 1 :] = torch.nan
        # Add a batch dimension at the beginning (unsqueeze dim=0)
        # Resulting shape: (1, num_features, sequence_length)
        return mask.unsqueeze(dim=0)


class BinaryRandomPremise(Premise):
    """Premise for generating random binary perturbation masks.

    Creates binary masks with random patterns based on specified
    probability distributions. Supports both image and tabular data
    modalities with appropriate dimension handling.
    """

    def __init__(
        self,
        key: tuple,
        seed: int,
        modality="image",
    ) -> None:
        """Responsible for creating random binary masks with a probability $p$ of *not*perturbing the corresponding pixel.
        When $p$ is set to 1, the masks will have more 0 which means we preserve input example, when $p$ is close to 0 the mask will be fuller and we will perturb more of the input.

        Mask Convention:
            0: Preserve the input value
            1: Perturb the input value

        Args:
            key (tuple): Holds enough information in order to generate the mask.
            modality (bool): Should we unsqueeze one more dimension at the beginning ? Default to True.

        Interpretation:
            key[0]: mask dim before up-sampling.
            key[1]: probability $p$ of preserving the input pixels: binary mask will have entry of 0 with probability $p$.
            key[2]: the final mask shape (after up-scaling) (w, h) as the input example without batch nor channel dimensions.

        """
        self.seed = seed
        self.modality = modality

        super().__init__(key=key)

    def get_mask(self) -> torch.Tensor:
        """Create the premise's random mask.

        Returns:
            torch.tensor: The final random mask of shape (b=1, c=1, w, h)
        """
        np.random.seed(self.seed)
        mask_dim = self.key[0]
        mask_proba = self.key[1]
        mask_shape = self.key[2]

        # binary grid
        if isinstance(mask_dim, tuple):
            mask_dim_x, mask_dim_y = mask_dim
        else:
            mask_dim_x = mask_dim_y = mask_dim

        cell_size = np.ceil(np.array(mask_shape) / mask_dim)
        up_size_x = (mask_dim_x) * cell_size[0]
        up_size_y = (mask_dim_y + 1) * cell_size[1]
        if self.modality == "image":
            up_size_x += cell_size[0]

        grid = np.random.rand(mask_dim_x, mask_dim_y) < mask_proba
        grid = 1 - grid.astype("float32")

        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])

        mask = resize(
            grid,
            (up_size_x, up_size_y),
            order=1,
            mode="reflect",
            anti_aliasing=False,
        )

        mask = mask[x : x + mask_shape[0], y : y + mask_shape[1]]

        mask = torch.from_numpy(mask)
        # add channels dimension
        if self.modality == "image":
            mask = mask.unsqueeze(dim=0)

        # add batch dimension
        return mask.unsqueeze(dim=0)  # (b=1, c=1, w, h) = input_shape


class SegmentedBinaryImagePremise(Premise):
    """Premise for segment-based binary masks on images.

    Generates binary masks based on image segmentation, where entire
    segments (superpixels) are either preserved or perturbed together.
    Provides more meaningful perturbations for image explanations.
    """

    def __init__(
        self,
        key: torch.Tensor,
        segmented_example: torch.Tensor,
        modality="image",
    ) -> None:
        """Creates random binary masks with a probability $p$ of keeping the corresponding pixel.
        When $p$ is set to 1, it means mask everything in the input example,
        when 0 keeps everything in the input.

        Mask Convention:
            0: Preserve the input value
            1: Perturb the input value

        Args:
            key (torch.Tensor): Holds enough information in order to generate the mask.
            segmented_example (torch.Tensor) : the segmented tensor of
                the explained example of shape (s, *example.shape).
            modality (bool): modality of the data

        """
        self.segmented_example = segmented_example
        self.modality = modality
        super().__init__(key=key)

    def get_mask(self) -> torch.Tensor:
        """Creates the premise's random binary mask.

        Returns:
            torch.tensor: The final random mask of shape (b=1, c=1, h, w)
        """
        binary_vector = self.key.to(self.device)
        self.segmented_example = self.segmented_example.to(self.device)

        s, h, w = self.segmented_example.shape
        segmented_example = self.segmented_example.view(s, h * w)

        if self.modality:
            mask = torch.matmul(
                binary_vector.float(), segmented_example.float()
            ).view(1, 1, h, w)
        else:
            mask = torch.matmul(
                binary_vector.float(), segmented_example.float()
            ).view(1, h, w)
        return mask


class GradientPremise(Premise):
    """Premise for gradient-based trainable masks.

    Provides learnable mask parameters that can be optimized through
    gradient descent. Useful for iterative mask refinement methods
    that learn optimal perturbation patterns.
    """

    def __init__(
        self,
        key: torch.Tensor,
        upscaled_mask_shape: tuple,
    ) -> None:
        """Provides a trainable mask.

        Mask Convention:
            0: Preserve the input value
            1: Perturb the input value

        Args:
            key (torch.Tensor): The premise's key. The key is expected to have gradients enabled.
            upscaled_mask_shape: The dimension to which the key is upscaled when transformed into a mask. 2-dimensional.

        Interpretation:
            key: Representing the down-scaled trainable mask. Shape (1, small_w, small_h),
            mask: This maps between the down-scaled mask to its final shape (=x.shape) (1, small_w, small_h) => (1, w, h).

        """
        self.upscaled_mask_shape = upscaled_mask_shape

        super().__init__(key=key)

        # Ensure mask is trainable
        self.key.requires_grad_()
        self.key.retain_grad()

    @property
    def mask(self):
        """Get the upscaled mask for this premise."""
        return self.get_mask()

    def get_mask(self) -> torch.Tensor:
        """Upscale the key's mask from (1, small_w, small_h) to the shape (1, w, h) where w=224 and h==224 are the width and hight of x, respectively.

        Returns:
            torch.Tensor: The up-scaled mask

        """
        key = self.key.unsqueeze(dim=0)  # interpolate requires 4d input
        upscaled_mask = torch.nn.functional.interpolate(
            key,
            size=self.upscaled_mask_shape,
            mode="bilinear",
            align_corners=False,
        ).to(self.device)

        return upscaled_mask  # (b=1, 1, w, h)


class ConvolutionalFeaturePremise(Premise):
    """Premise for creating masks from convolutional feature maps.

    Generates discrete masks based on convolutional neural network
    feature activations. Each feature map is converted into a binary
    mask for targeted perturbations of specific learned features.
    """

    def __init__(
        self,
        key: tuple,
    ) -> None:
        """Responsible for creating discrete masks from convolutional features of the example.

        Mask Convention:
            0: Preserve the input value
            1: Perturb the input value

        Args:
            key (tuple): Holds enough information in order to generate the mask.

        Interpretation:
            key[0]: up-sampled activation of the convolutional layer (torch.Tensor)
            key[1]: channel (int)

        """
        super().__init__(key=key)

    def get_explanation(self) -> torch.Tensor:
        """Create the premise's heatmap from convolutional features.

        Extracts a specific channel from the upsampled activations to create
        a visual explanation heatmap.

        Returns:
            torch.Tensor: The up-sampled activation of shape (b=1, c=1, w, h) as a heatmap.
        """
        upsampled_activations = self.key[0]
        channel = self.key[1]

        return upsampled_activations[:, channel].unsqueeze(dim=1)

    def get_mask(self) -> torch.Tensor:
        """Creates the premise's mask.

        Returns:
            torch.Tensor: The final feature mask of shape (b=1, c=1, w, h)

        """
        upsampled_activations = self.key[0]
        channel = self.key[1]

        # Extracting min and max values of the channel
        min_value = upsampled_activations[:, channel].min()
        max_value = upsampled_activations[:, channel].max()

        # When activation is constant, set it to 0 everywhere
        if min_value == max_value:
            mask = upsampled_activations[:, channel] * 0

        # Else, normalise its values
        else:
            mask = (upsampled_activations[:, channel] - min_value) / (
                max_value - min_value
            )

        # To respect MUPPET mask convention
        mask = 1 - mask

        return mask.unsqueeze(dim=1)  # (b=1, c=1, w, h)


class FeaturesCombinationPremise(GradientPremise):
    """Premise for linear combinations of convolutional features.

    Creates masks from weighted combinations of multiple convolutional
    feature maps. Allows for more complex perturbation strategies by
    combining different learned feature representations.
    """

    def __init__(
        self,
        key: torch.Tensor,
        activations: torch.Tensor,
        upscaled_mask_shape: tuple[int, int],
    ) -> None:
        """Responsible for creating discrete masks from a linear combination of convolutional features of the example.

        Mask Convention:
            0: Preserve the input value
            1: Perturb the input value

        Args:
            key (torch.Tensor): linear combination coefficients.

            activations (torch.Tensor): activation of the last convolutional layer of the model.

            upscaled_mask_shape (tuple[int]): tuple of the form (w, h) with w the width and h the height of the input.
        """
        self.activations = activations

        super().__init__(key=key, upscaled_mask_shape=upscaled_mask_shape)

    def get_mask(self) -> torch.Tensor:
        """Creates the premise's mask.
        This is done by multiplying the previously-acquired feature maps in a
        combination based on the premise's key.

        Recall that the key is the vector that is being optimized by the Explorer.

        Returns:
            torch.tensor: The final feature mask of shape (b=1, c=1, w, h)
        """
        activations = self.activations
        upscaled_mask_shape = self.upscaled_mask_shape

        b, c, u, v = activations.size()

        soft_coefs = torch.nn.functional.softmax(
            self.key, dim=1
        )  # The key is the coefficients of the linear combination of features
        activations = activations.view(c, u * v)  # (b=1, c, u, v) => (c, u * v)

        features_combination = torch.matmul(
            soft_coefs, activations
        )  # (1, c) @ (c, u * v) => (1, u * v)
        features_combination = features_combination.view(
            1, 1, u, v
        )  # (u * v) => (1, 1, u, v)

        upsampled_fc = torch.nn.functional.interpolate(
            features_combination, size=upscaled_mask_shape, mode="bilinear"
        ).to(self.device)

        min_value = upsampled_fc.min()
        max_value = upsampled_fc.max()

        # When activation is constant, set it to 0 everywhere
        if min_value == max_value:
            mask = upsampled_fc * 0

        else:
            mask = (upsampled_fc - min_value) / (max_value - min_value)

        # To respect MUPPET mask convention
        mask = 1 - mask

        return mask  # (b=1, c=1, w, h)


class KeyBasedMaskPremise(Premise):
    """Premise for generating masks based on tensor keys.

    Creates perturbation masks using tensor keys as the basis for
    random mask generation. Provides reproducible mask patterns
    based on the key values and optional random seed.
    """

    def __init__(self, key: torch.Tensor, seed: int | None) -> None:
        """Create the premise's random mask based on the key.
        Mask Convention:
            0: Preserve the input value

            1: Perturb the input value

        Args:
            - key (torch.Tensor): Tensor serving as the base for generating masks.
            - seed (int): Seed for the random number generator.
        """
        super().__init__(key=key)
        self.seed = seed

    def get_mask(self) -> torch.Tensor:
        """Create the premise's random mask based on the key.

        Returns:
            torch.Tensor: The final random mask, which is a copy of the key.
        """
        mask = self.key.clone()  # Create a copy of the key
        return mask
