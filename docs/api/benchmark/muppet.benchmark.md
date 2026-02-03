The `muppet.benchmark` module provides a comprehensive and flexible framework for evaluating and comparing perturbation-based eXplanation (PXAI) methods within the **MUPPET XAI** library. It's designed to automate the process of running various explainers on different models and datasets, calculating a wide range of evaluation metrics, and aggregating the results for easy analysis and visualization. The module's design is centered on a configuration-driven approach using [Hydra](https://hydra.cc/), which allows users to define and run complex benchmarking experiments via simple YAML files.

## Key Features & Components

### Configuration-Driven Benchmarking ⚙️

The entire benchmarking process is controlled through Hydra-based configurations, enabling users to specify every aspect of an experiment. This includes:

- **Models**: Defining the model to be explained (e.g., `torchvision` models, `scikit-learn` classifiers, or custom PyTorch models).

- **Datasets**: Selecting from a variety of pre-configured datasets for image, tabular, and time series data, with the flexibility to add custom ones.

- **Explainers**: Choosing which **MUPPET** explainers to evaluate (e.g., LIME, SHAP, RISE), with support for the four-block decomposition framework (Exploration, Perturbation, Attribution, Aggregation).

- **Metrics**: Specifying the evaluation metrics to be used, primarily leveraging the [Quantus](https://quantus.readthedocs.io/en/stable/index.html) library, along with custom metrics like Sparseness and Faithfulness.

This approach ensures reproducibility and simplifies the management of complex experimental setups.

### Multi-Modality Support & Extensibility 🧩

The module is built to handle different data modalities, making it suitable for benchmarking explainers on a diverse range of tasks.

- **Datasets**: It includes built-in support for standard datasets across different modalities, with a [`muppet.benchmark.datamodule`](muppet.benchmark.datamodule.md) abstraction that simplifies data loading and preprocessing.

- **Models**: Wrappers [`muppet.benchmark.models`](muppet.benchmark.models.md) are provided for popular model types from libraries like `torchvision`, `scikit-learn`, and `aeon`, and the system is designed to easily accommodate new custom models.

- **Explainers**: Explainers from the MUPPET library are integrated seamlessly, allowing for easy comparison.

- **Metrics**: The integration with the `Quantus` library provides access to a comprehensive suite of XAI evaluation metrics, covering key aspects like faithfulness, robustness, and complexity.

### Automated Workflow & Analysis 📊

The `benchmark` module automates the end-to-end evaluation process, from training models (if needed) and executing explainers to computing metrics and generating results. It then aggregates these results and provides tools for visualization, such as heatmaps and bar plots, which are essential for comparing the performance of different explainers. This streamlined workflow allows researchers and developers to efficiently analyze and understand the strengths and weaknesses of various PXAI methods.
