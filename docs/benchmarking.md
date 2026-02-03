# Benchmarking

The `Muppet` library includes a comprehensive benchmarking module located in the `muppet/benchmark/` directory. This tool is designed to evaluate and compare various Perturbation-based eXplanation (PXAI) methods across different models, datasets, and evaluation metrics.

## Features

*   **Configuration-Driven:** Powered by Hydra, allowing for flexible and extensive configuration of experiments via YAML files.
*   **Multi-Modality Support:** Easily benchmark explainers on different data types:
    *   Image, Tabular, Time Series
*   **Extensible Components:**
    *   **Datasets:** Pre-configured for standard datasets (e.g., ImageNet sample, UCI Dry Beans, AEON time series, synthetic spike data) and easily extendable for new ones.
    *   **Models:** Includes wrappers for various model types (e.g., `torchvision` models, `sklearn` models, `aeon` classifiers).
    *   **Explainers:** Integrates explainers implemented in `Muppet` (e.g., LIME, SHAP, RISE, MP, OptiCAM, ScoreCAM, FIT).
    *   **Metrics:** Leverages the [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) library for a wide range of XAI evaluation metrics.
*   **Automated Evaluation:** Runs experiments, computes predictions, and evaluates explanations systematically.
*   **Result Aggregation & Visualization:** Generates aggregated results and visualizations.

## Running Benchmarks (for pip users)

To run a benchmark, use the `muppet-benchmark` command-line utility (a wrapper around the `run_benchmark.py` script).
This tool uses **Hydra**, allowing you to specify a main configuration file and override parameters dynamically.

This section assumes that you have installed `muppet-xai` via pip in your python environment. You can run benchmarks using both the built-in configuration
from the `muppet` library and your own custom configuration files without cloning the repository.
This allows you to leverage the library's built-in components (datasets, explainers, metrics) while easily overriding specific parameters
(like switching models or changing hyperparameters).

### Reproducing the library's benchmarks
If you want to reproduce the library's benchmarks exactly as they are defined in the package (without creating any local configuration files),
you can instruct Hydra to look directly inside the installed muppet package.

This method is useful for quick sanity checks or reproducing baseline results.
Instead of providing a local path, you pass the package path to `hydra.searchpath`.

**Note**: This method allows you to run benchmarks using only the datasets, models and explainers that already exist in our benchmark configuration.

```bash
muppet-benchmark --config-name=image_config hydra.searchpath=[pkg://muppet.benchmark.conf]
```

- `--config-name`: The name of the built-in configuration file you want to run.

- `hydra.searchpath=[pkg://muppet.benchmark.conf]`: This override tells Hydra to add the installed package's configuration directory to its search list.

### Overriding Configuration

You can override any parameter from the configuration files directly from the command line. For example, to change the test set size for the image benchmark:

```bash
muppet-benchmark --config-name=image_config hydra.searchpath=[pkg://muppet.benchmark.conf] dataset.test_size=50
```

### Execution Modes

The benchmarking module supports two execution modes for comparing explainers: a default sequential mode for full-suite evaluation and a Multirun mode for
selective, parallel comparison.

By default (when `hydra.mode` is not specified), the benchmark runs sequentially on all explainers defined in your configuration file's `explainers` dictionary. This is ideal for running a complete evaluation suite in one go.

To compare specific explainers or leverage multiprocessing (via `joblib`), you can invoke Hydra's `MULTIRUN` mode. This allows you to sweep over specific
explainers using the `+run_explainer` argument. When enabled, the script loops through the list provided to `+run_explainer` and looks up their corresponding
configurations in your `explainers` dictionary.

Example: To run the benchmark only on the `RISE` and `MP` explainers:

```bash
muppet-benchmark --config-name=image_config hydra.searchpath=[pkg://muppet.benchmark.conf] hydra.mode=MULTIRUN +run_explainer=RISEExplainer,MPExplainer
```

### Examples

1.  **Run image benchmark with default settings:**

    ```bash
    muppet-benchmark --config-name=image_config hydra.searchpath=[pkg://muppet.benchmark.conf]
    ```

2.  **Run tabular benchmark with MULTIRUN settings:**

    ```bash
    muppet-benchmark --config-name=tabular_config hydra.searchpath=[pkg://muppet.benchmark.conf] hydra.mode=MULTIRUN
    ```

3.  **Run time series benchmark for a specific dataset:**

    ```bash
    muppet-benchmark --config-name=timeseries_config hydra.searchpath=[pkg://muppet.benchmark.conf] dataset.name=EthanolConcentration dataset.loader.name=EthanolConcentration hydra.mode=MULTIRUN
    ```

4. **Run image benchmark for image segmentation:**

   ```bash
   muppet-benchmark --config-name=image_seg_config hydra.searchpath=[pkg://muppet.benchmark.conf]
   ```

### Benchmark Outputs

Benchmark results, including raw scores (JSON/CSV) and visualizations, are saved in the `results` directory, organized by date and time.
The `outputs/` or `multirun/` directory contains the Hydra execution logs.


### Running Custom Benchmarks

If you need to perform complex experiments that cannot be handled by simple CLI overrides, you can create a local configuration file.

To do this, we use `--config-path` alongside the **Hydra's Search Path** feature to tell the script: "Look for configs in my local folder first, but also look inside the installed `muppet` library."

### Step 1: Create a Local Configuration File

First, create a directory for your experiment and add a YAML file. For this example, we'll create `my_experiment/custom_benchmark.yaml`.

The key is to use `hydra.searchpath` directly in your configuration to add the `muppet` library's configurations to Hydra's search path.
Then, you can inherit from a default configuration and override specific values.

**Example: `my_experiment/custom_benchmark.yaml`**

This example runs the image benchmark, but changes the model from the default `resnet` to `vgg16` and reduces the test set size for a quicker run.

```yaml
# 1. Add the installed library's config path to Hydra's search path.
#    This makes 'image_config.yaml' and other packaged configs discoverable.
hydra:
  searchpath:
    - pkg://muppet.benchmark.conf

# 2. Inherit from the default image benchmark configuration.
#    '_self_' allows this file to override the defaults.
defaults:
  - image_config
  - _self_

# 3. Override specific parameters.
#    Hydra will merge these values into the loaded 'image_config'.

# Change the model. Hydra will now load new model configuration into `model` key.
model:
  _target_: muppet.benchmark.models.classifier.Classifier
  name: vgg16
  model:
      # load from torchvision
      _target_: torchvision.models.get_model
      name: vgg16
      weights: IMAGENET1K_V1
  pretrained: True
  random_state: 167

# Tweak a dataset parameter for a quicker test run.
dataset:
  test_size: 10

# You can also completely replace a configuration group.
# For example, to run ONLY the RISE explainer:
# explainers:
#   RISEExplainer:
#     _target_: muppet.explainers.RISEExplainer
#     nb_masks: 2000
```

### Step 2: Run the Benchmark

From your terminal, execute the `muppet-benchmark` command, pointing to your custom configuration.

**Important**: Again, you must provide the **absolute path** to your configuration folder so the globally installed script can locate it.

```bash
muppet-benchmark --config-path $(pwd)/my_experiment --config-name custom_benchmark
```

- `--config-path`: The absolute path to the directory containing your YAML file.
- `--config-name`: The name of your YAML file (without the `.yaml` extension).

**How it works**

When you run this command, Hydra performs the following steps:

- Locates your local folder via `--config-path`.

- Reads `custom_benchmark.yaml`.

- Detects the searchpath directive and conceptually "mounts" the installed `muppet` package's config folder.

- Composes the final configuration by merging the library's `image_config` with your local overrides.

- Executes the benchmark.

**Best Practice**: Organize like a Library

We strongly recommend structuring your local folder with the same subdirectories as the library (e.g., `model/`, `dataset/`, `explainer/`...).

If you create a file `my_experiment/model/vgg16.yaml`, you can reference it simply by name instead of writing the full definition in your main file as the above example.

Benefit:

- In YAML: You can use `- override model: vgg16` in the `defaults` list.

- In CLI: You can swap models easily:

```bash
muppet-benchmark --config-path $(pwd)/my_experiment --config-name custom_benchmark model=vgg16
```

**Important: Using Absolute Paths**:

When using the `muppet-benchmark` wrapper, you **must provide the absolute path** to your configuration folder.
Because the `muppet-benchmark` script is installed globally in your Python environment (e.g., in `site-packages`), not in your local folder.
Therefore, it does not automatically know that relative paths (like `./conf`) refer to the directory you are currently standing in.
Using an absolute path ensures Hydra can locate your configuration files regardless of where the command is executed.

**Tip:** On Linux/macOS, use `$(pwd)` to automatically insert the current directory.

## Running Benchmarks from Source (Git Clone)

If you have cloned the repository and are working directly with the source code, the workflow is simplified.
You do not need to use the hydra.searchpath argument because the configuration files are available locally.

### Basic Execution

To run benchmarks, simply point the script to the local configuration directory using the `--config-path` argument.

**Important**: You must use the absolute path to the repository's `conf` directory.

For example:

```bash
# Assuming you are in the root of the cloned repository
muppet-benchmark --config-path=$(pwd)/muppet/benchmark/conf --config-name=image_config
```

### Adding Custom Configurations

Since you have direct access to the source code, you can integrate custom configurations by adding files directly into the existing directory structure.

Navigate to `muppet/benchmark/conf` and place your new YAML files in the corresponding subdirectories:

- `dataset/`: for new dataset definitions.

- `model/`: for new model architectures.

- `explainer/`: for specific explainer settings.

Once added, these files are immediately available to Hydra and can be selected by name (e.g., `dataset=my_custom_dataset`).
