"""Tests for LIME image explainer components.

This module tests the LIMEImageExplainer in MUPPET, comparing its implementation against
the official LIME implementation for image data. It validates that MUPPET's LIME produces
similar explanations to the reference implementation for image classification tasks.

The tests verify:
- Consistency with official LIME image implementation from GitHub
- Proper heatmap generation for image classification models
- Correct prediction function wrapping and preprocessing
- Expected explanation quality and similarity metrics
- Integration with VGG models and standard image preprocessing
"""
#
# Created on Thu Dec 07 2023
#
# Copyright (c) 2023 Jérémy Rozier @Euranova
#

import numpy as np
import torch
from github_version import lime_image as lime_image_github
from sklearn.linear_model import Ridge

from muppet import DEVICE
from muppet.explainers import LIMEImageExplainer


def predict_fn(model_vgg):
    """Create a prediction function wrapper for LIME image testing.

    Args:
        model_vgg: The VGG model to use for predictions.

    Returns:
        function: A prediction function that takes images and returns scores.
    """

    def predict(images):
        """Predict function for LIME image explanation testing.

        Args:
            images: Array of images to predict on.

        Returns:
            np.ndarray: Stacked array of prediction scores.
        """
        list_scores = []
        images = torch.tensor(images)
        for image in images:
            input_data = image.permute(2, 0, 1)
            input_data = input_data.unsqueeze(dim=0)
            input_data = input_data.to(DEVICE)
            score = model_vgg(input_data).squeeze(dim=0)
            list_scores.append(score.detach().cpu().numpy())

        return np.stack(list_scores)

    return predict


def get_explanation_github(
    ret_exp: lime_image_github.ImageExplanation, predicted_class_index: int
):
    """Compute and returns the heatmap for a given ImageExplanation instance."""
    labels = torch.tensor(ret_exp.segments).cpu().numpy()
    exp = ret_exp.local_exp[predicted_class_index]
    exp = list(exp)
    exp = sorted(exp, key=lambda x: x[0])
    features, coefs = zip(*exp)
    coefs = torch.tensor(coefs).cpu().numpy()
    heatmap = torch.zeros(*labels.shape)
    for feature in features:
        heatmap[labels == feature] = coefs[feature]
    min_value = heatmap.min()
    max_value = heatmap.max()
    heatmap = (heatmap - min_value) / (max_value - min_value)
    heatmap = heatmap.view(1, *labels.shape)
    return heatmap.unsqueeze(dim=0)


def test_lime_image(model_vgg, cat_image_for_vgg):
    """The purpose is to compare our implementation of LIME for images with an implementation
    from github available at https://github.com/marcotcr/lime
    """
    SEED = 420
    image = cat_image_for_vgg
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    image = cat_image_for_vgg

    image_github = image[0].permute(
        1, 2, 0
    )  # TODO is it normal that the expected format is different ?
    image_github = image_github.cpu().numpy()
    predicted_class_index = (
        torch.nn.Softmax(dim=-1)(model_vgg(image).detach())
        .argmax(dim=-1)[0]
        .item()
    )
    clf = Ridge(alpha=1, fit_intercept=True)

    lime_explainer = LIMEImageExplainer(
        surrogate_model=clf,
        model=model_vgg,
        nmasks=100,
        masked_proba=0.5,
    )

    lime_explainer_github = lime_image_github.LimeImageExplainer()
    ret_exp = lime_explainer_github.explain_instance(
        image_github,
        predict_fn(model_vgg),
        distance_metric="cosine",
        random_seed=None,
        labels=(predicted_class_index,),
        top_labels=None,
        hide_color=0,
        num_samples=100,
    )
    predicted_class_index = (
        torch.nn.Softmax(dim=-1)(model_vgg(image).detach())
        .argmax(dim=-1)[0]
        .item()
    )

    lime_heatmap = lime_explainer(example=image)
    lime_heatmap = lime_heatmap.detach().cpu()
    lime_heatmap_github = get_explanation_github(ret_exp, predicted_class_index)

    distance_heatmap = torch.abs(lime_heatmap - lime_heatmap_github)

    assert torch.mean(distance_heatmap) <= 0.2
