"""Faithfulness metrics for segmentation tasks in the MUPPET benchmark framework.

This module extends the Quantus faithfulness metrics to support segmentation tasks,
providing specialized implementations for evaluating explanation quality in
computer vision segmentation models. It includes utilities for converting
probability outputs to one-hot encodings and custom faithfulness correlation
metrics adapted for pixel-wise predictions.

Functions:
    probs2one_hot: Convert probability tensors to one-hot encoded arrays

Classes:
    FaithfulnessCorrelationSeg: Faithfulness correlation metric for segmentation
"""

from typing import Any, Callable, Dict, Optional  # noqa: F811

import numpy as np
import torch
import torch.nn.functional as F
from quantus import ModelInterface
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface  # noqa: F811
from quantus.metrics.faithfulness.faithfulness_correlation import (
    FaithfulnessCorrelation,
)
from tqdm import tqdm


def probs2one_hot(probs: torch.Tensor) -> torch.Tensor:
    """_summary_: function that transforms probability values into one hot arrays
    of dimensions (n_classes, x, y)

    Args:
        probs (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
        TO DO: Consider dimensions and include assertions
    """
    return (
        F.one_hot(
            torch.tensor(probs).argmax(axis=0), num_classes=probs.shape[0]
        )
        .permute(2, 0, 1)
        .numpy()
    )


class FaithfulnessCorrelationSeg(FaithfulnessCorrelation):
    """Faithfulness correlation metric specialized for segmentation tasks.

    Extends the base FaithfulnessCorrelation metric with specific handling
    and default parameters optimized for image segmentation explanations.
    """

    def __init__(
        self,
        similarity_func: Optional[Callable] = None,
        nr_runs: int = 100,
        subset_size: int = 224,
        abs: bool = False,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_func: Optional[Callable] = None,
        perturb_baseline: str = "black",
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = True,
        aggregate_func: Optional[Callable] = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """Initialize the FaithfulnessCorrelationSeg metric for segmentation tasks.

        Sets up the faithfulness correlation metric with parameters optimized for
        evaluating explanations on image segmentation models, extending the base
        metric with segmentation-specific handling.

        Args:
            similarity_func: Function to compute correlation between attribution sums
                and prediction changes. If None, uses Pearson correlation.
            nr_runs: Number of random perturbation runs for correlation calculation.
            subset_size: Number of pixels to randomly select for each perturbation.
            abs: If True, uses absolute values for correlation computation.
            normalise: If True, normalizes attribution values before evaluation.
            normalise_func: Custom normalization function for attributions.
            normalise_func_kwargs: Additional arguments for normalization function.
            perturb_func: Function to apply perturbations to input images.
            perturb_baseline: Baseline value for perturbations (e.g., "black").
            perturb_func_kwargs: Additional arguments for perturbation function.
            return_aggregate: If True, returns aggregated results across classes.
            aggregate_func: Function to aggregate results (default: mean).
            default_plot_func: Default plotting function for visualization.
            disable_warnings: If True, suppresses warning messages.
            display_progressbar: If True, shows progress bar during evaluation.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(
            abs=abs,
            similarity_func=similarity_func,
            nr_runs=nr_runs,
            subset_size=subset_size,
            perturb_func=perturb_func,
            perturb_baseline=perturb_baseline,
            perturb_func_kwargs=perturb_func_kwargs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )
        self.a_axes = (1, 2)

    def evaluate_instance(
        self,
        model: ModelInterface,
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
    ) -> float:
        """Evaluate the quality of an explanation for a single image instance using faithfulness correlation
        adapted for segmentation tasks.

        This method perturbs the input image based on random subsets of the attribution map,
        then measures how much the segmentation prediction changes in response. The change
        is correlated with the sum of attributions in the perturbed region to assess whether
        high-attribution pixels truly affect the model's output.

        Args:
            model (ModelInterface): A model interface object wrapping a segmentation model to be evaluated.
            x (np.ndarray): The original input image (typically 3D: C×H×W or H×W×C).
            y (np.ndarray): The model's predicted probabilities for the original input (before masking),
                expected to have the same shape as the output segmentation map.
            a (np.ndarray): The attribution map (explanation) for the input image. Typically same shape as `y`.

        Returns:
            float: The average correlation (e.g., Pearson/Spearman) between the magnitude of prediction
                change and the attribution sum across perturbed regions, computed across all classes.
        """
        # Flatten the attributions.
        # a = a.flatten()
        similarity = []
        # Predict on input.
        x_input = model.shape_input(x, x.shape, channel_first=True)
        one_hot_predicted_maps = probs2one_hot(y)
        y_pred = model.predict(x_input).squeeze() * one_hot_predicted_maps

        pred_deltas = []
        att_sums = []

        # For each test data point, execute a couple of runs.
        for i_ix in tqdm(range(self.nr_runs)):
            # Randomly mask by subset size.
            a_ix = np.random.choice(
                a.flatten().shape[0], self.subset_size, replace=False
            )
            x_perturbed = self.perturb_func(
                arr=x,
                indices=a_ix,
                indexed_axes=self.a_axes,
            )
            warn.warn_perturbation_caused_no_change(
                x=x, x_perturbed=x_perturbed
            )

            # Predict on perturbed input x.
            x_input = model.shape_input(
                x_perturbed, x.shape, channel_first=True
            )
            y_pred_perturb = (
                model.predict(x_input).squeeze() * one_hot_predicted_maps
            )
            pred_deltas.append(np.abs(y_pred - y_pred_perturb).sum(axis=(1, 2)))

            # Sum attributions of the random subset.
            att_sums.append(np.sum(a.flatten()[a_ix]))

        for class_ in range(0, y_pred.shape[0]):
            similarity.append(
                self.similarity_func(
                    a=att_sums, b=np.array(pred_deltas)[:, class_].tolist()
                )
            )

        similarity = np.array(similarity)
        return similarity[~np.isnan(similarity)].mean()
