"""General experiment utilities."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    """Set common random seeds for more reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Return CUDA device when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model_weights(model, path):
    """Save model weights to a .pth file."""

    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    print("Model saved to:", path)


def append_experiment_log(log_path, experiment_log):
    """Append one experiment record to a CSV log file."""

    log_path = Path(log_path)
    log_path.parent.mkdir(exist_ok=True)

    new_row = pd.DataFrame([experiment_log])

    if log_path.exists():
        old_rows = pd.read_csv(log_path)
        updated = pd.concat([old_rows, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(log_path, index=False)
    print("Experiment log saved to:", log_path)

    return updated
