"""SHAP (SHapley Additive exPlanations) explainer for tabular data.

This module implements kernel SHAP, a game-theory based approach to explain machine learning
model predictions.

MUPPET Component Integration:
    - **Explorer**: `BinaryFeaturePermutationsExplorer` - generates all 2^n feature combinations (coalitions)
    - **Perturbator**: `RandomSamplePerturbator` - replaces missing features with samples from training distribution
    - **Attributor**: `SimilarityAttributor` - applies kernel weighting based on coalition size
    - **Aggregator**: `ModelAggregator` - fits weighted Ridge regression to compute SHAP feature importance

Classes:
    ShapTabularExplainer: Implementation of KernelSHAP for tabular data.

References:
    Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model
    predictions." Advances in neural information processing systems 30 (2017).
    https://arxiv.org/abs/1705.07874
"""

import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from muppet.components.aggregator.model import (
    ModelAggregator,
)
from muppet.components.attributor.similarity import (
    SimilarityAttributor,
    kernel_shap_similarity,
)
from muppet.components.explorer.mask import BinaryFeaturePermutationsExplorer
from muppet.components.perturbator.generator.tabular_generator import (
    RandomSampleTabularGenerator,
)
from muppet.components.perturbator.scale_feature_generator import (
    RandomSamplePerturbator,
)
from muppet.explainers.mp import MuppetExplainer


class ShapTabularExplainer(MuppetExplainer):
    """Implementation of kernel SHAP (SHapley Additive exPlanations) for tabular data.

    Implements KernelSHAP that computes Shapley values by evaluating all possible feature
    coalitions and fitting a weighted surrogate model. SHAP is grounded in cooperative game
    theory and provides the only explanation method that satisfies four desirable properties:
    efficiency, symmetry, dummy, and additivity.

    Key principles of SHAP:
    - **Efficiency**: Feature attributions sum to the prediction difference
    - **Symmetry**: Equal contributions from equally important features
    - **Dummy**: Zero attribution for irrelevant features
    - **Additivity**: Consistent attributions across different models

    The Shapley values sum to the difference between the prediction for the instance and
    the average prediction, providing a complete attribution of the model's decision.
    The implementation uses KernelSHAP, which approximates Shapley values by fitting a
    weighted linear regression on all possible feature coalitions.

    SHAP perturbs the input data by masking or altering features to generate a set of samples,
    then evaluates the model on these perturbed inputs. It fits a weighted surrogate model to
    approximate the behavior of the original model locally, around the specific instance
    being explained.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        generator=RandomSampleTabularGenerator,
        surrogate_model=Ridge(alpha=1, fit_intercept=True),
        n_repeats: int = 20,
        seed: int = 1,
        similarity_fun=kernel_shap_similarity,
    ) -> None:
        """Initialize the SHAP Tabular explainer.

        Args:
            model: The black-box model whose predictions are to be explained.
            train_loader: The train dataloader used to fit the generator for perturbing the input data.
            generator: The generator used to create random masks for perturbing the input data.
                Defaults to RandomSampleTabularGenerator.
            surrogate_model: The regressor model used to fit the surrogate model on the perturbed data.
                Defaults to Ridge(alpha=1, fit_intercept=True).
            n_repeats: Number of times to repeat each permutation. Default is 20.
            seed: The random seed for reproducibility. Default is 1.
            similarity_fun: The similarity function for SHAP kernel weighting.
        """
        self.n_repeats = n_repeats
        self.similarity_fun = similarity_fun
        # Instantiate modules
        explorer = BinaryFeaturePermutationsExplorer(
            n_repeats=n_repeats, seed=seed
        )

        # re-initialize the generator if not provided
        train_data = torch.concat([data[0] for data in train_loader])
        generator = RandomSampleTabularGenerator(
            train_data=train_data,
            method="freq",
            seed=seed,
        )

        # perturbator initialization & generator training if required
        perturbator = RandomSamplePerturbator(generator=generator)
        attributor = SimilarityAttributor(similarity_fun=kernel_shap_similarity)
        aggregator = ModelAggregator(surrogate_model=surrogate_model)

        # Initiate the explainer with these modules
        super().__init__(
            model=model,
            explorer=explorer,
            perturbator=perturbator,
            attributor=attributor,
            aggregator=aggregator,
        )
