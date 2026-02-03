"""Experiment result visualization callbacks for the MUPPET benchmark framework.

This module provides callback classes to be used with hydra multirun and monorun
for processing and visualizing experiment results after benchmark runs.
It handles aggregation of results and generates plots for performance analysis
and comparison of the various evaluated explainers.

Classes:
    PlotExperiencesResultCallback: Callback for plotting experiment results after run completion

Functions:
    get_dataframe_from_quantus_results: Convert Quantus results to DataFrame for visualization
"""

import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from muppet import logger
from muppet.benchmark.plot_explanation import plot_explanation_comparison_image
from muppet.benchmark.plot_metrics import plot_benchmark_results


class PlotExperiencesResultCallback:
    """Callback to process and visualize experiment results after a multirun execution.

    This class is intended to be used in experiment workflows where multiple runs
    are executed (MULTIRUN mode), and the results from those runs are collected,
    aggregated, and plotted automatically at the end.
    """

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Handle actions to be performed after a single run ends.

        This method delegates to `on_multirun_end` to provide consistent behavior
        regardless of whether it's called from a single run or multirun context.

        Args:
            config (DictConfig): The configuration object containing run details.
            **kwargs (Any): Additional keyword arguments passed through.
        """
        return self.on_multirun_end(config, **kwargs)

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Handle actions to be performed after all jobs complete in a multirun.

        This method reads all experiment log files generated during the multirun,
        aggregates the results, prepares the data, and creates a box plot
        (or violin plot, if modified) summarizing the outcomes.

        Args:
            config (DictConfig): The configuration object containing details
                such as model name and dataset name, used to locate the result logs.
            **kwargs (Any): Additional keyword arguments (unused).
        """
        logger.info(
            "Run Plot Experiences Result Callback at the end of multirun!"
        )
        result_path = Path(
            f"results/{config.model.name}/{config.dataset.name}/"
            f"{'/'.join(config.hydra.run.dir.split('/')[-2:])}"
        )

        # Plot explanations comparison
        if config.dataset.type == "image":
            explanations = defaultdict(dict)
            targets = {}
            for path in result_path.glob("**/*.npy"):
                idx, sample_type = path.stem.split("_")
                if sample_type == "image":
                    explanations[int(idx)]["Input"] = np.load(path)
                else:
                    explanations[int(idx)][path.parent.name] = np.load(path)
            for path in result_path.glob("**/*.txt"):
                idx, sample_type = path.stem.split("_")
                with open(path) as f:
                    targets[int(idx)] = f.read()
            plot_explanation_comparison_image(
                explanations,
                targets,
                save_path=result_path / "explanation_comparison_plot.png",
            )
        else:
            logger.warning(
                "At the moment only plot explanation comparison for modality `image` is supported."
            )

        # Plot benchmark results in box plot
        log_paths = [p.as_posix() for p in result_path.glob("*.log")]
        log_results = []
        for path in log_paths:
            with open(path, mode="r") as f:
                log_results.append(json.load(f))
        log_results = list(itertools.chain.from_iterable(log_results))
        df_benchmark_results = get_dataframe_from_quantus_results(log_results)

        plot_benchmark_results(
            df_benchmark_results,
            dataset_name=config.dataset.name,
            save_path=result_path / "benchmark_box_plot.png",
            violin_or_box="box",
            show_plot=False,
        )


def get_dataframe_from_quantus_results(data: list[dict]) -> pd.DataFrame:
    """Prepare data for visualization.

    Args:
        data (list): List of dictionaries containing the data.

    Returns:
        DataFrame: Prepared DataFrame for visualization:
            df (pd.DataFrame): A DataFrame containing the performance metrics of explainers.
                       The DataFrame should have the following columns:
                       - 'Dataset': The dataset used.
                       - 'Model': The model used.
                       - 'Explainer': The name of the explainer.
                       - 'Metric': The name of the metric.
                       - 'Value': The value of the metric.
                       - 'Category': The category of the metric.
                       - 'Score_direction': the direction that the score of a metric should go in for better results.
                           - HIGHER: Higher scores are better.
                           - LOWER: Lower scores are better.
    """
    metrics_data = []
    for dataset in data:
        dataset_name = dataset["dataset"]
        metric_attributs = dataset["metric_attributs"]
        for model, explainers in dataset["models"].items():
            for explainer, metrics in explainers.items():
                for met, values in metrics.items():
                    if values is not None:
                        # Determine category and score direction based on metric name
                        category, score_direction = metric_attributs[met]
                        for value in values:
                            metrics_data.append(
                                {
                                    "Dataset": dataset_name,
                                    "Model": model,
                                    "Explainer": explainer,
                                    "Metric": met,
                                    "Score_direction": score_direction,
                                    "Category": category,
                                    "Value": value,
                                }
                            )

    return pd.DataFrame(metrics_data)
