"""Metrics visualization utilities for the MUPPET benchmark framework.

This module provides functions for creating various plots to visualize benchmarking
results and metrics comparisons across different explainers and datasets. It supports
violin plots, box plots, and radar charts for comprehensive performance analysis.

Functions:
    plot_violin: Create violin/box plots for metric distributions across explainers
    plot_explainer_rankings: Generate radar charts for explainer performance rankings
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from muppet import logger

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")


def plot_benchmark_results(
    df: pd.DataFrame,
    dataset_name: str,
    save_path: str | Path | None = None,
    show_plot: bool = False,
    violin_or_box: Literal["violin", "box"] = "violin",
):
    """Plots a violin plot or box plot showing the distribution of metric values for different explainers.

    This function generates a violin plot for different explainers across a set of metrics, grouped by
    metric categories. Each explainer's performance is visualized across multiple metrics, with the
    metric categories color-coded.

    Args:
        df (pd.DataFrame): DataFrame containing the explainer data. It should include the following columns:
                       - 'Value': The metric values.
                       - 'Metric': The name of the metric.
                       - 'Explainer': The name of the explainer.
                       - 'Category': The metric category (e.g., Faithfulness, Robustness).
                       - 'Score_direction': Indicates whether higher or lower values are better.
                       - 'Model': The name of the model to which the explainers were applied.
                       If any of these columns don't exist, the plotting still proceeds,
                       but that particular piece of information is omitted.
        dataset_name (str): name of dataset displayed in the plot title
        save_path (str, optional): If provided, saves the plot to the specified file path. Defaults to None.
        show_plot (bool, optional): If True, displays the plot window. If False, the plot is not shown.
                                    Defaults to True.
        violin_or_box (Literal["violin", "box"]): type of plot

    Returns:
        None: Displays the violin plot and optionally saves it to a file.
    """
    # Define colors for each metric category
    category_colors = {
        "FAITHFULNESS": "#1f77b4",  # Blue
        "ROBUSTNESS": "#ff7f0e",  # Orange
        "COMPLEXITY": "#2ca02c",  # Green
        "RANDOMISATION": "#d62728",  # Red
        "Time(s)": "#9467bd",  # Purple
        "AXIOMATIC": "#8c564b",  # Brown
    }
    score_direction_symbols = {"HIGHER": "↑", "LOWER": "↓"}

    # Make a copy to avoid modifying the original DataFrame
    df_tmp = df.copy()

    # If 'Category' is missing, create a dummy category so plotting doesn't break
    if "Category" not in df_tmp.columns:
        df_tmp["Category"] = "UNKNOWN"

    # Map category to colors (unknown categories will default to black)
    df_tmp["Color"] = df_tmp["Category"].map(category_colors).fillna("black")

    num_samples_explained = int(
        df.groupby(["Dataset", "Explainer", "Metric"]).count()["Value"].min()
    )

    # Safely build the new 'Metric' label
    # Only proceed if 'Metric' and 'Score_direction' exist
    if "Metric" in df_tmp.columns and "Score_direction" in df_tmp.columns:
        df_tmp["Metric"] = (
            df_tmp["Category"].str.capitalize()
            + ":\n"
            + df_tmp["Metric"]
            + df_tmp["Score_direction"].map(score_direction_symbols).fillna("?")
        )

    # Create the violin plot only if we have the necessary columns
    if {"Metric", "Value", "Explainer"}.issubset(df_tmp.columns):
        g = sns.catplot(
            y="Value",
            kind=violin_or_box,
            col="Metric",
            legend_out=True,
            hue="Explainer",
            palette=sns.color_palette("colorblind"),
            hue_order=sorted(df_tmp["Explainer"].unique().tolist()),
            data=df_tmp,
            col_wrap=3,
            height=4,
            aspect=1.2,
            sharex=False,
            sharey=False,
            linewidth=0.8,
        )
        g.set_titles(template="{col_name}")
        sns.move_legend(
            g, loc="lower left", bbox_to_anchor=(0.2, -0.25), ncols=3
        )

        model_name = df_tmp["Model"].iloc[0]
        g.figure.suptitle(
            f"Metric Distribution for {model_name} on {dataset_name} \n {num_samples_explained} samples",
            fontsize=18,
            fontweight="bold",
            y=1.10,
            x=0.45,
        )
    else:
        logger.warning(
            "DataFrame does not contain the required columns for violin plot."
        )
        return

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f'Summary plot saved: "{save_path}"')

    # Show the plot if show_plot is True
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_explainer_rankings(df, dataset_name, path_to_save=None):
    """Plots a radar chart of explainer rankings across various metric categories.

    This function generates a radar plot that visualizes the rankings of different explainers
    for a set of metrics. Each explainer's performance is represented by connecting points that
    correspond to their ranks in each metric category.

    Args:
        df (pd.DataFrame): DataFrame containing the rankings of explainers. The index should represent
                        explainer names, and the columns should represent metric categories.
                        The DataFrame's values are the ranks of the explainers.
        path_to_save (str, optional): If provided, saves the plot to the specified file path. Defaults to None.

    Returns:
        None: Displays the radar plot and optionally saves it to a file.
    """
    try:
        # Extract categories and number of variables
        categories = df.columns.tolist()
        num_vars = len(categories)

        # Split the circle into even parts and append the start angle to close the circle
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        categories += categories[:1]

        # Create radar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Set plot rotation and direction
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Set category labels
        plt.xticks(angles[:-1], categories[:-1])

        min_rank = df.min().min()
        max_rank = df.max().max()

        # Invert ranks for plotting (so higher ranks are closer to the center)
        df_inverted = max_rank - df + min_rank

        # Set radial limits and ticks
        ax.set_ylim(min_rank - 0.5, max_rank)
        ax.set_yticks(np.arange(min_rank, max_rank + 0.5, 1))
        ax.set_yticklabels([])
        ax.grid(True, linestyle="--", alpha=0.7)

        # Plot each explainer
        for explainer in df.index:
            values = df_inverted.loc[explainer].tolist()
            values += values[:1]  # close the polygon
            ax.plot(angles, values, linewidth=2, label=explainer)
            ax.fill(angles, values, alpha=0.25)

        # Add legend and title
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.title(
            f"dataset_name: {dataset_name}, Radar Plot of Explainer Rankings"
        )

        # Save the plot if path_to_save is provided
        if path_to_save:
            plt.savefig(path_to_save, bbox_inches="tight")

        plt.show()
    except Exception as e:
        logger.info(f"Plotting failed: {e}")
        return
