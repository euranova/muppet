"""Wrapper utilities for integrating MUPPET explainers with the Quantus evaluation framework.

This module provides wrapper functions that adapt MUPPET explainer interfaces to be
compatible with the Quantus library's evaluation pipeline. It handles batch processing,
explanation visualization, and format conversion between different frameworks.

Functions:
    quantus_explainer_wrapper: Wrap MuppetExplainer instances for Quantus compatibility
"""

from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from muppet import DEVICE
from muppet.benchmark.plot_explanation import (
    plot_explanation_image,
    plot_explanation_multichannel_time_series,
    plot_explanation_tabular,
)
from muppet.explainers.base import MuppetExplainer


def quantus_explainer_wrapper(explainer: MuppetExplainer) -> Callable:
    """Wraps a MuppetExplainer instance into a callable function compatible with the Quantus library.

    This wrapper formats the explainer's interface so it matches Quantus' expectation:
    a callable taking model, inputs, and targets as arguments, and returning explanations as
    a NumPy array.

    Args:
        explainer (MuppetExplainer): An instance of a MuppetExplainer or similar wrapper
                                     exposing a callable `__call__` method that takes
                                     an input sample and returns an explanation.

    Returns:
        Callable: A function with the Quantus-compatible signature for explanation.
    """

    def explain_func(model, inputs, targets, **kwargs):
        """Applies the explainer to each input sample individually and aggregates the results.

        Args:
            model (torch.nn.Module): The model to explain (unused here but required by Quantus).
            inputs (Union[np.ndarray, torch.Tensor]): Input batch to be explained, shape (B, ...).
            targets (Any): Target labels (unused here but required by Quantus).
            **kwargs: Additional arguments passed by Quantus (ignored here).

        Returns:
            np.ndarray: The aggregated explanation results with shape (B, ...).
        """
        input_tensor = torch.tensor(inputs, device=DEVICE)
        input_batches = torch.split(input_tensor, 1)

        plot_explanation = kwargs.get("plot_explanation", False)
        explanation_savedir = Path(
            kwargs.get("explanation_savedir", Path.cwd())
        ) / kwargs.get("method", "heatmaps")
        explanation_savedir.mkdir(parents=True, exist_ok=True)
        num_plot_explanations = kwargs.get(
            "num_plot_explanations", len(input_batches)
        )
        labels = kwargs.get("labels")
        explanations = []
        with logging_redirect_tqdm():
            for k, single_input in enumerate(
                tqdm(
                    input_batches,
                    desc=f"Compute explanations: {kwargs['method']}",
                )
            ):
                explanation = explainer(example=single_input)
                explanations.append(explanation)
                if k >= num_plot_explanations:
                    continue
                if plot_explanation:
                    if kwargs.get("modality") == "image":
                        plot_explanation_image(
                            example=single_input[0],
                            explanation=explanation[0][0],
                            target=str(labels[k])
                            if labels is not None
                            else None,
                            figure_title=kwargs.get("method") or "",
                            save_path=explanation_savedir / f"{k:05d}.png",
                        )
                    elif kwargs.get("modality") == "tabular":
                        plot_explanation_tabular(
                            explanation=explanation[0],
                            feature_names=None,
                            save_path=explanation_savedir / f"{k:05d}.png",
                        )
                    elif kwargs.get("modality") == "timeseries":
                        plot_explanation_multichannel_time_series(
                            example=single_input,
                            explanation=explanation,
                            figure_title=kwargs.get("method") or "",
                            save_path=explanation_savedir / f"{k:05d}.png",
                        )
                    else:
                        raise ValueError(
                            "If plot heatmap is enable, `modality` must be given and must be 'image', 'tabular' or 'timeseries'."
                        )

        return torch.cat(explanations, dim=0).cpu().numpy()

    explain_func.name = explainer.__class__.__name__

    return explain_func
