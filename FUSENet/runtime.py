"""Device selection, reproducible seeds, and experiment artifact writing."""

import csv
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This implementation supports CPU and CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device does not exist")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, payload):
    _atomic_write(path, lambda p: p.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"))


def save_checkpoint(path, payload):
    _atomic_write(path, lambda p: torch.save(payload, p))


def write_predictions(path, ids, source_ids, targets, predictions):
    if not len(ids) == len(source_ids) == len(targets) == len(predictions):
        raise ValueError("Prediction columns have different lengths")
    def write(destination):
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("sample_id", "source_id", "target", "prediction"))
            writer.writerows(zip(ids, source_ids, map(float, targets), map(float, predictions)))
    _atomic_write(path, write)
