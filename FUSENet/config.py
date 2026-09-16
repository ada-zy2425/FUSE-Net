"""Configuration for the single FUSE-Net implementation."""

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    data: str = "mosi"
    data_dir: str = ""
    bert_dir: str = "roberta-large"
    revision: str = "main"
    batch_size: int = 64
    n_epoch: int = 100
    patience: int = 10
    learning_rate: float = 3e-5
    bert_learning_rate: float = 1.5e-5
    weight_decay: float = 0.01
    clip: float = 1.0
    hidden_size: int = 128
    dropout: float = 0.4
    temperature: float = 0.3
    task_weight: float = 1.0
    info_weight: float = 0.1
    info_gain_weight: float = 0.25
    cycle_weight: float = 0.02
    recon_weight: float = 0.015
    vib_beta: float = 0.01
    max_text_length: int = 512
    seed: int = 42
    device: str = "auto"
    visual_size: int = 0
    acoustic_size: int = 0

    def __post_init__(self):
        if self.data not in {"mosi", "mosei", "simsv2"}:
            raise ValueError("data must be mosi, mosei, or simsv2")
        if not self.data_dir:
            names = {"mosi": "MOSI", "mosei": "MOSEI", "simsv2": "SIMSv2"}
            self.data_dir = str(ROOT / "datasets" / names[self.data])
        self.data_dir = str(Path(self.data_dir).expanduser().resolve())
        for name in ("batch_size", "n_epoch", "patience", "hidden_size", "max_text_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("learning_rate", "bert_learning_rate", "clip", "temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("{} must be finite and positive".format(name))
        for name in ("weight_decay", "task_weight", "info_weight", "info_gain_weight", "cycle_weight", "recon_weight", "vib_beta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("{} must be finite and nonnegative".format(name))
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if not self.bert_dir or not self.revision:
            raise ValueError("Specify a text encoder and revision")

    def to_dict(self):
        return asdict(self)


def training_parser():
    parser = argparse.ArgumentParser(description="Train FUSE-Net on MOSI, MOSEI, or SIMSv2.")
    parser.add_argument("--data", choices=("mosi", "mosei", "simsv2"), required=True)
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--bert_dir", required=True, help="Explicit encoder choice: the paper names both RoBERTa and BERT-base.")
    parser.add_argument("--revision", default="main", help="Prefer the model/tokenizer revision used for your experiment.")
    defaults = Config()
    for name in ("batch_size", "n_epoch", "patience", "hidden_size", "max_text_length"):
        parser.add_argument("--" + name, type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "bert_learning_rate", "weight_decay", "clip", "dropout", "temperature", "task_weight", "info_weight", "info_gain_weight", "cycle_weight", "recon_weight", "vib_beta"):
        parser.add_argument("--" + name, type=float, default=getattr(defaults, name))
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser
