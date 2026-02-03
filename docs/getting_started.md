The following will give a short introduction to how to get started with MUPPET-XAI.

## Usage

```python

from muppet import FITExplainer

explainer = FITExplainer(model=my_model)

heatmap = explainer(example=x)

```

The example $x$ must be of shape `[b, **dim]` where b is the batch of example and dim is the data modality dimensions, E.g for one image (b=1, c=3, w=224, h=224).

## Working Example

Generate explanation heatmaps for images using the RISE method by referring to the [basic_explanation.ipynb](muppet/notebooks/basic_explanation.ipynb) notebook, or just run this piece of code from the root:

```python
from torchvision.models import get_model

from muppet import DEVICE
from muppet.benchmark.plot_explanation import plot_explanation_image
from muppet.benchmark.tools import load_imagenet_image
from muppet.explainers import RISEExplainer

# Prepare VGG-16 model,
model = get_model(
    name="vgg16",
    weights="IMAGENET1K_V1",
)
model.to(DEVICE)

# Prepare the image to explain its classification
image = load_imagenet_image(
    "muppet/tests/data/cat.jpg"
)  # (1, 3, 224, 224)

# Explain the image class prediction using RISE method
rise_explainer = RISEExplainer(model=model)
heatmap = rise_explainer(example=image)  # (b=1, c=1, w=224, h=224)

# plot heatmap
plot_explanation_image(
    example=image[0],
    explanation=heatmap[0][0],
    figure_title="Heatmap",
)
```
