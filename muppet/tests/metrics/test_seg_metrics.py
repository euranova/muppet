"""Tests for segmentation metrics components.

This module tests segmentation-specific metrics in MUPPET, including faithfulness correlation
metrics adapted for segmentation tasks. It validates proper metric calculation for
segmentation models and compares behavior with classification metrics.

The tests verify:
- Proper probability to one-hot conversion for segmentation outputs
- FaithfulnessCorrelationSeg metric calculation and validation
- Comparison between segmentation and classification faithfulness metrics
- Expected correlation values within valid ranges [-1, 1]
- Integration with dummy segmentation and classification models
"""

import numpy as np
import torch
from quantus.metrics.faithfulness.faithfulness_correlation import (
    FaithfulnessCorrelation,
)

from muppet.benchmark.metrics.faithfulness import (
    FaithfulnessCorrelationSeg,
    probs2one_hot,
)


def test_probs2one_hot_shape():
    """Test probability tensor to one-hot conversion shape preservation.

    Validates that the probs2one_hot function maintains tensor dimensions
    when converting probability distributions to one-hot encoded tensors
    for segmentation tasks.

    Returns:
        None: Test passes if output shape matches input shape.
    """
    probs = torch.rand(3, 224, 224)
    one_hot = probs2one_hot(probs)
    assert one_hot.shape == (3, 224, 224)


def test_FaithfulnessCorrelationSeg_Metric(
    DummySegmentationModel, create_dummy_inputs
):
    """_summary_: Test the FaithfulnessCorrelationSeg metric for segmentation tasks.
    This test checks if the metric can handle segmentation inputs and outputs,
    and if it produces a valid correlation score.
    It uses a dummy segmentation model and creates dummy inputs to simulate a segmentation task.
    The test ensures that the metric can evaluate the faithfulness of the segmentation model's predictions
    against the provided attributions.
    It checks that the result is a float, not NaN, and within the expected range [-1, 1].
    This is important to ensure that the metric is functioning correctly and can be used for evaluating segmentation
    models in terms of faithfulness to the provided attributions.
    """
    n_classes = 1
    np.random.seed(42)
    model = DummySegmentationModel(n_classes)
    # Example usage in your test
    x, y, a = create_dummy_inputs(n_classes)
    print(x.shape, y.shape, a.shape)
    metric = metric = FaithfulnessCorrelationSeg()
    result = metric.evaluate_instance(model, x, y, a)

    # Assert
    assert isinstance(result, float)
    assert ~np.isnan(result)
    assert -1 <= result <= 1


def identity_perturb_func(arr, **kwargs):
    """Identity perturbation function that returns the input as-is.
    This function is used to test the FaithfulnessCorrelationSeg metric without any perturbation.
    It simulates a scenario where the input is not altered, allowing us to check if the
    metric behaves correctly when no perturbation is applied.

    Args:
        x (np.ndarray): Input data to be perturbed.
        **kwargs: Additional keyword arguments (not used in this function).

    Returns:
            np.ndarray: The input data unchanged.
    """
    # Identity perturbation: returns input as-is
    return arr


def test_segmentation_vs_classification_faithfulness(
    DummySegmentationModel, dummy_classification_model, create_dummy_inputs
):
    """_summary_: Test the faithfulness correlation between segmentation and classification models.
    This test checks if the faithfulness correlation metric produces similar results for both segmentation and classification models.
    It uses dummy segmentation and classification models to simulate a segmentation task and a classification task.
    The test ensures that the metric can evaluate the faithfulness of both models' predictions against the provided attributions.
    It checks that the results for both models are valid floats, not NaN, and reasonably close to each other.
    This is important to ensure that the metric can be used for evaluating both segmentation and classification models
    in terms of faithfulness to the provided attributions.

    Args:
        dummy_seg_model (_type_):   _dummy_segmentation_model_: A dummy segmentation model that outputs random predictions for testing purposes.
        dummy_classification_model (_type_): _dummy_classification_model_: A dummy classification model that outputs the mean of the segmentation model outputs.
        create_dummy_inputs (_type_): _create_dummy_inputs_: A function that creates dummy inputs for testing purposes.
    """
    n_classes = 2

    # creating the classification & Segmentation models:
    model = DummySegmentationModel(n_classes)
    model_classification = dummy_classification_model(n_classes)

    # Creating the inputs:
    x, y, a = create_dummy_inputs(n_classes)
    x_input = model.shape_input(x)

    # Get segmentation predictions (batch, classes, H, W)
    class_preds = model_classification.predict(x_input)

    # Instantiate metrics
    seg_metric = FaithfulnessCorrelationSeg()
    class_metric = FaithfulnessCorrelation()

    # bypassing the model's initialization - bug issue to resolve later:
    class_metric.a_axes = (1, 2)

    # Run segmentation faithfulness on full heatmaps
    seg_result = seg_metric.evaluate_instance(model, x, y, a)

    # For classification faithfulness metric, create dummy class label (scalar_preds) and attribution (flattened)
    # The classification metric expects: model, x, label, attribution

    # Attribution heatmap must be (1, H, W) for classification metric —
    # so sum/average attribution across channels or just take channel 0 attribution from a
    attribution_cls = a[0]  # (224,224) assuming a shape (1,224,224)

    # Classification metric expects 2D attribution: (H,W)
    # and scalar label per class — so test for each class independently
    class_results = []
    for c in range(n_classes):
        # Create dummy attribution for class c as flat heatmap (could just reuse attribution_cls)
        # In real test, you might want attribution per class if available; here just reuse a for simplicit
        result = class_metric.evaluate_instance(
            model_classification,
            x,
            class_preds.argmax(),
            attribution_cls.reshape(
                1, attribution_cls.shape[0], attribution_cls.shape[1]
            ),
        )
        class_results.append(result)

    # Average classification faithfulness result over classes
    avg_class_result = np.mean(class_results)

    print(f"Segmentation faithfulness: {seg_result}")
    print(
        f"Classification faithfulness averaged over classes: {avg_class_result}"
    )

    # Check if results are correlated or reasonably close
    # You can loosen the threshold depending on metric scale
    diff = abs(seg_result - avg_class_result)
    print(f"Difference between seg and classification faithfulness: {diff}")

    assert isinstance(seg_result, float)
    assert isinstance(avg_class_result, float)
    assert not np.isnan(seg_result)
    assert not np.isnan(avg_class_result)
    assert diff < 0.3, (
        "Segmentation and classification faithfulness should be reasonably close"
    )
