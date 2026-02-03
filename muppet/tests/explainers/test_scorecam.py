"""Tests for ScoreCAM explainer components.

This module tests the ScoreCAMExplainer in MUPPET, comparing its implementation against
the official Score-CAM implementation. It validates that MUPPET's ScoreCAM produces
similar explanations to the reference implementation for image classification.

The tests verify:
- Element-wise similarity with official Score-CAM implementation
- Proper convolutional layer activation extraction and processing
- Correct upsampling and normalization of activation maps
- Expected heatmap generation with proper spatial dimensions
- Integration with VGG models and forward hook mechanisms
"""
#
# Created on Tue Dec 5 2023
#
# Copyright (c) 2023 Léo Beaumont @Euranova
#

# Score-CAM imports
import torch
import torch.nn.functional as F

# Muppet imports
from muppet.explainers import ScoreCAMExplainer


def score_cam(model_vgg, cat_image_for_vgg):
    """Official Score-CAM release https://github.com/haofanwang/Score-CAM/blob/master/cam/scorecam.py, slightly modified.

    Args:
        model (torch.Module): model to explain

        image (torch.Tensor): image on which model is used of size (b=1, c=3, h, w)

    Returns:
        heatmap (torch.Tensor): heatmap of size (b=1, c=1, h, w)
    """
    model = model_vgg
    image = cat_image_for_vgg

    b, c, h, w = image.size()

    # For test purposes, force these to the same device (CPU)
    # predication on raw input
    logit = model(image)

    predicted_class = logit.max(1)[-1]

    # From this point, code is modified
    last_conv_layer = None
    conv_layer_activation = dict()

    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise TypeError("given model has no Conv2d layer")

    # Placing a hook on the last Conv2d layer, to register layer activity during forward pass
    def hook(model, input, output):
        """Hook function to register convolutional layer activation for testing.

        Args:
            model: The PyTorch model being hooked.
            input: Input tensor to the layer.
            output: Output tensor from the convolutional layer.
        """
        conv_layer_activation["value"] = output.detach()

    hk = last_conv_layer.register_forward_hook(hook)

    # Forwarding input in model
    with torch.no_grad():
        model(image)

    # Removing hook
    hk.remove()
    # From this point, code is no longer modified

    activations = conv_layer_activation["value"]
    b, k, u, v = activations.size()

    score_saliency_map = torch.zeros((1, 1, h, w))

    with torch.no_grad():
        for i in range(k):
            # upsampling
            saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
            saliency_map = F.interpolate(
                saliency_map, size=(h, w), mode="bilinear", align_corners=False
            )

            if saliency_map.max() == saliency_map.min():
                continue

            # normalize to 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (
                saliency_map.max() - saliency_map.min()
            )

            # how much increase if keeping the highlighted region
            # prediction on masked input
            output = model(image * norm_saliency_map)
            output = F.softmax(output, dim=1)
            score = output[0][predicted_class]
            score = score.detach().cpu()
            saliency_map = saliency_map.cpu()

            score_saliency_map += score * saliency_map

    score_saliency_map = F.relu(score_saliency_map)
    score_saliency_map_min, score_saliency_map_max = (
        score_saliency_map.min(),
        score_saliency_map.max(),
    )

    if score_saliency_map_min == score_saliency_map_max:
        return 0 * score_saliency_map

    score_saliency_map = (
        (score_saliency_map - score_saliency_map_min)
        .div(score_saliency_map_max - score_saliency_map_min)
        .data
    )

    return score_saliency_map


def test_scorecam_explainer(model_vgg, cat_image_for_vgg):
    """Test if MUPPET's heatmap (with Score-CAM explainer) is similar to official Score-CAM's heatmap element wise.

    Args:
        model_vgg (torch.Module): model to explain

        image (torch.Tensor): image on which model is used

        margin (float): acceptable margin of error
    """
    margin = 10 ** (-4)
    explainer = ScoreCAMExplainer(model=model_vgg)

    heatmap_muppet = explainer(example=cat_image_for_vgg)
    heatmap_scorecam = score_cam(model_vgg, cat_image_for_vgg)

    heatmaps_similar = (
        torch.abs(heatmap_muppet.cpu() - heatmap_scorecam.cpu()) <= margin
    )

    assert torch.all(heatmaps_similar), (
        f"correct {torch.sum(heatmaps_similar)} of {heatmaps_similar.numel()}"
    )
