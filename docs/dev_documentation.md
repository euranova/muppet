Thank you for taking interest in contributions to **MUPPET-XAI**! We encourage you to contribute new explainers, new components, refactorings or report any bugs you may come across. In this guide, you will get an overview of the workflow and best practices for contributing to **MUPPET-XAI**.

Questions. If you have any developer-related questions, please open an issue or write us at [antoine.bonnefoy@euranova.eu](antoine.bonnefoy@euranova.eu).

## Reporting Bugs

If you discover a bug, as a first step please check the existing Issues to see if this bug has already been reported. In case the bug has not been reported yet, please do the following:

- Open an issue.

- Add a descriptive title to the issue and write a short summary of the problem.

- Adding more context, including reference to the problematic parts of the code, would be very helpful to us.

Once the bug is reported, our development team will try to address the issue as quickly as possible.

## General Guide to Making Changes

This is a general guide to contributing changes to **MUPPET-XAI**. If you would like to add a new explainer to **MUPPET-XAI**, please refer to next section. Before you start the development work, make sure to read our documentation first.

### Development Installation

Make sure to install the latest version of **MUPPET-XAI** from the main branch.

```bash
git clone git@code.euranova.eu:rd/taudos/muppet.git
cd muppet/
```

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

### Unit Tests

We use unit and integration tests to ensure code quality, stability, and maintainability. All contributions must maintain 100% test pass rate, and we strive for high code coverage.

**Tools:**

- Pytest: Our primary testing framework for writing and running tests.

- Coverage: Used to measure the percentage of code executed by the unit tests.

To run all tests locally (without measuring coverage), simply use the pytest command targeting the test directory:

```bash
pytest muppet/tests/
```

To measure which parts of the muppet package the tests are executing, we use coverage run with specific parameters to ensure accurate results and exclude non-essential files.

```bash
coverage run \
    --omit=config-3.py,config.py,*__init__*,muppet/tests/*,*/tmp/* \
    --source muppet \
    -m pytest $* muppet/tests/
```

To display a detailed, file-by-file summary directly in the terminal:

```bash
coverage report -m
```

To generate an interactive, navigable HTML report that shows exactly which lines of code were missed by tests:

```bash
coverage html
```

### Documentation

Make sure to add docstrings to every class, method and function that you add to the codebase. The docstring should include a description of all parameters and returns. Use the existing documentation as an example.

### Code Style

We use automated tools to enforce the following standards:

- Code Style: All Python code must strictly follow PEP 8.

- Docstrings: Docstrings must use the MkDocs-style format for compatibility with our automatic documentation generator.

- Line Length: A strict maximum width of 80 characters per line is enforced across all code and docstrings.

### Auto-Formatting with Ruff

We use Ruff for all linting and formatting. Ruff is a high-performance linter and formatter that unifies checks for tools like pycodestyle, Pyflakes, isort, and Pydocstyle. The the enabled rules are configured in the `[tool.ruff]` section of the main `pyproject.toml` file.

**Workflow:**

Before committing any changes, you must run the following commands to ensure your code is fully compliant:

1. **Check and Fix**: This command runs the linter and automatically corrects style errors and logical issues where possible.

```bash
ruff check --fix .
```

2. **Format**: This command applies the code formatter, ensuring imports are sorted and line length is respected.

```bash
ruff format .
```

Always run both commands before creating a Pull/Merge Request.

### Pull/Merge Requests

Once you are done with the changes:

- Create a pull/merge request

- Provide a summary of the changes you are introducing.

- In case you are resolving an issue, don’t forget to link it.

- Add us as a reviewer.

## Contributing a New Explainer
TODO

## Questions

If you have any developer-related questions, please open an issue or write us at xxx@euranova.eu.
