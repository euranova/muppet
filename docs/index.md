## **Welcome to MUPPET-XAI documentation!**

**Mu**lti**p**le **Pe**r**t**urbation e**X**plainable **A**rtificial **I**ntelligence (**MUPPET-XAI**) is a multimodal python library for explaining and interpreting Pytorch models through perturbation-based XAI methods. It supports all data modalities (including images, tabular data, time series, ...).

### **Principle**

Given a black-box model $f$ that only provides inference functionality, regardless of its inner working, and a data point $x$. The goal is to understand the prediction $f(x)$ made by the model by perturbing the input data feature values $x'$ and observing the model $f$ prediction on those perturbations.

The perturbation-based methods follow four steps:

1. Generate the masks to use for perturbing the input data $x$,

2. Apply those masks on the input data to get the $x'$,

3. Calculate feature scores/attributions of every perturbation from the model prediction on perturbed data $f(x')$ and on original data,

4. Finally aggregate the attributions to find the final local explanation such as feature importance, heat-maps, ...

This documentation is complementary to the `README.md` in the MUPPET-XAI repository and provides documentation for how to install MUPPET-XAI, how to contribute and details on the API.

MUPPET-XAI can be installed from PyPI:

```bash
pip install muppet-xai
```

### **Contents**

**Installation**

- [`Quick installation`](installation.md)

**Getting Started**

- [`Getting Started`](getting_started.md)
- [`Benchmarking`](benchmarking.md)

**API Reference**

- [`muppet package`](api/index.md)

**Developer Documentation**

- [`Contribute to MUPPET-XAI`](dev_documentation.md)

**Citation**

If you find this toolkit or its companion paper [Paper-Name](paper-link) interesting or useful in your research, please use the following Bibtex annotation to cite us:

```bibtex
@article{hedstrom2025muppet,
  author  = {},
  title   = {Muppet: a modular and constructive decomposition for perturbation-based explanation methods},
  journal = {},
  year    = {2025},
  volume  = {24},
  number  = {34},
  pages   = {1--11},
  url     = {}
}
```

