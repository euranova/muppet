import os

import torch

# RUN AT PACKAGE IMPORT

# Detect the default Torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set logger with appropriate default logger level
os.environ["LOGURU_LEVEL"] = os.getenv("LOGURU_LEVEL", "INFO")

from loguru import logger

__version__ = "0.1.3"
