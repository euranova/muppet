"""Main benchmarking script for the MUPPET XAI evaluation framework.

This module provides the main entry point for running comprehensive XAI benchmarking
experiments. It orchestrates model training if needed, dataset loading, explainer execution,
and metrics evaluation using the Quantus framework. The script supports both single
runs and multi-run experiments with automatic result aggregation and persistence.

Functions:
    main: Main function for running XAI benchmarking experiments
"""

import functools
import json
from datetime import datetime
from pathlib import Path

import hydra
import hydra.types
import numpy as np
from omegaconf import DictConfig
from quantus.evaluation import evaluate

from muppet import DEVICE, logger
from muppet.benchmark.wrappers import quantus_explainer_wrapper


@hydra.main(config_path="conf", config_name="image_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for running XAI benchmarking experiments.

    This function orchestrates the complete benchmarking pipeline including:
    - Model instantiation and training (if required)
    - Dataset loading and preparation
    - Explainer initialization and execution
    - Metrics evaluation using Quantus framework
    - Results aggregation and persistence

    Args:
        cfg: Hydra configuration object containing all experiment parameters
            including model, dataset, explainers, metrics, and execution settings.

    Returns:
        None: Results are saved to disk as JSON files with timestamps.
    """
    if not hydra.core.hydra_config.HydraConfig.initialized():
        # Initialize HydraConfig if not yet done
        # Useful when use Compose API of hydra to run this main function
        # Inside a Jupyter notebook, a unit test...
        hydra.core.hydra_config.HydraConfig.instance().set_config(cfg)

    model_to_explain = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    datamodule.prepare_data()
    train_loader, test_loader = (
        datamodule.train_loader,
        datamodule.test_loader,
    )

    if train_loader is not None and not model_to_explain.pretrained:
        # train model here
        model_to_explain.fit(train_loader)

    # infer model
    model_to_explain.eval()
    inputs_array_to_explain: np.ndarray
    predictions_array: np.ndarray
    inputs_array_to_explain, predictions_array = model_to_explain.infer_model(
        test_loader
    )

    if (
        runtime_cfg := hydra.core.hydra_config.HydraConfig.get()
    ).mode is hydra.types.RunMode.MULTIRUN:
        # In MULTIRUN mode each script execution evaluate only one explainer identified by cfg.run_explainer
        logger.info(
            f"MULTIRUN mode: '{cfg.run_explainer}' will be run in the job "
            f"'{runtime_cfg.job.id}' simultaneously alongside over other explainer(s)"
        )
        explainers = {
            cfg.run_explainer: hydra.utils.instantiate(
                cfg.explainers[cfg.run_explainer],
                model=model_to_explain,
            )
        }
    else:
        logger.info(
            f"MONORUN mode: {', '.join(cfg.explainers.keys())} will be run sequentially"
        )
        explainers = {
            expl_name: hydra.utils.instantiate(
                explainer, model=model_to_explain
            )
            for expl_name, explainer in cfg.explainers.items()
        }

    for expl_name, explainer in explainers.items():
        # Case of trainable explainers (like LIME, FIT)
        if isinstance(explainer, functools.partial):
            explainers[expl_name] = explainer(train_loader=train_loader)

    metrics = {}
    for metric_name, metric_cfg in cfg.metrics.items():
        # Convert OmegaConf config to a regular dictionary to allow modification
        metric_cfg = dict(metric_cfg)
        # Some metric configurations define function arguments ending with "_func"
        # In the .yaml config, these are strings pointing to the function path
        # We need to resolve these strings into actual Python functions
        # using `hydra.utils.get_method` before instantiating metric instance object
        for arg_name, arg_val in metric_cfg.items():
            if arg_name.endswith("_func"):
                metric_cfg[arg_name] = hydra.utils.get_method(arg_val)
        metrics[metric_name] = hydra.utils.instantiate(metric_cfg)

    results_dir = (
        f"results/{cfg.model.name}/{cfg.dataset.name}/"
        f"{'/'.join(hydra.core.hydra_config.HydraConfig.get().run.dir.split('/')[-2:])}"
    )
    results = evaluate(
        metrics=metrics,
        xai_methods={
            expl_name: quantus_explainer_wrapper(expl)
            for expl_name, expl in explainers.items()
        },
        model=model_to_explain,
        x_batch=inputs_array_to_explain,
        y_batch=predictions_array,
        explain_func_kwargs={
            "explanation_savedir": results_dir,
            "modality": cfg.dataset.type,
            "labels": [
                f"{int(idx)}\n{datamodule.labels_mapping[int(idx)]}"
                for idx in predictions_array
            ]
            if datamodule.labels_mapping is not None
            and np.ndim(predictions_array) == 1
            else None,
            **cfg.explain_func_kwargs,
        },
        call_kwargs={
            "metrics_call_kwargs": {
                "batch_size": 1,
                "device": DEVICE,
                "softmax": False,
            }
        },
        agg_func=lambda x: [
            float(e) for e in x
        ],  # ensure explanations in native float instead of np.float for better dumps results
        verbose=True,
    )
    metric_attributs = {
        metric_name: (
            metric.evaluation_category.name,
            metric.score_direction.name,
        )
        for metric_name, metric in metrics.items()
    }
    results = [
        {
            "dataset": cfg.dataset.name,
            "models": {cfg.model.name: results},
            "metric_attributs": metric_attributs,
        }
    ]
    now = datetime.now()
    timestamp = (
        now.strftime("%Y-%m-%d_%H-%M-%S") + f".{now.microsecond // 1000:03d}"
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/{timestamp}.log", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
