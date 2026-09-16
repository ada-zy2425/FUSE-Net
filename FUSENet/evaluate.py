"""Evaluate an existing FUSE-Net checkpoint using its saved encoder and tokenizer."""

import argparse
import json
from pathlib import Path


def load_model(checkpoint_path, device="cpu", data_dir=None):
    import torch
    from transformers import AutoConfig, AutoModel
    from .config import Config
    from .models import FUSENet
    from .losses import INFORMATION_OBJECTIVE_VERSION
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    version = checkpoint.get("format_version")
    if version not in (2, 3):
        raise ValueError("This evaluator expects FUSE-Net checkpoint format 2 or 3")
    if version == 3 and checkpoint.get("information_objective") != INFORMATION_OBJECTIVE_VERSION:
        raise ValueError("Unsupported information objective in checkpoint")
    config = Config(**checkpoint["config"])
    if data_dir is not None:
        config.data_dir = str(Path(data_dir).expanduser().resolve())
    config.device = str(device)
    values = dict(checkpoint["encoder_config"])
    model_type = values.pop("model_type")
    encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **values))
    model = FUSENet(config, text_encoder=encoder)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # Inference architecture is unchanged. Older checkpoints remain usable but
    # are never relabeled as having been trained with the corrected objective.
    model.training_objective_version = (checkpoint["information_objective"] if version == 3
                                        else "legacy_signed_mse_v0")
    return model.to(device).eval(), config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output_dir already exists; choose a new evaluation directory")
    from transformers import AutoTokenizer
    from .data_loader import make_loaders, file_identity
    from .runtime import select_device, write_json, write_predictions
    from .solver import evaluate
    device = select_device(args.device)
    model, config = load_model(args.checkpoint, device, args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint.parent / "tokenizer", local_files_only=True)
    loaders, _ = make_loaders(config, splits=(args.split,), tokenizer=tokenizer)
    metrics, predictions = evaluate(model, loaders[args.split], device, config.data)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "metrics.json", {"split": args.split, "metrics": metrics,
               "training_objective": model.training_objective_version,
               "checkpoint": file_identity(args.checkpoint), "dataset": file_identity(loaders[args.split].dataset.path)})
    write_predictions(args.output_dir / "predictions.csv", *predictions)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
