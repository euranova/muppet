## Installing MUPPET-XAI from PyPI

To install Muppet-XAI with pip from [PyPI](https://pypi.org/):

```bash
pip install muppet-xai
```

### Installing MUPPET-XAI for Development

First clone or fork the repository.

MUPPET supports both standard and containerized environments for streamlined setup. Follow the instructions below depending on your preference.

#### Without Docker

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Run the following command to install development and documentation dependencies:

```bash
uv sync --locked
```

#### With Docker (Recommended)

1. Build the development Docker image:

If you develop on a host machine with an NVIDIA GPU and the required drivers installed, build Docker image that supports CUDA:

```bash
docker build -f Dockerfile.cuda -t muppet/muppet:latest .
```

Otherwise, on machines without a dedicated NVIDIA GPU:

```bash
docker build -t muppet/muppet:latest .
```

2. Run a development container and open a VS Code window using the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers):

```bash
docker run -it --rm --gpus all -v /app/.venv -v $(pwd):/app --name muppet_dev muppet/muppet:latest /bin/bash
```

Alternatively, you can open the project directly via VSCode, **this method is recommended
because it ensures that every developer has the same dev environement, including VSCode extensions, settings,...**, by using:

**Command Palette** → `Dev Containers: Open Folder in Container...`
