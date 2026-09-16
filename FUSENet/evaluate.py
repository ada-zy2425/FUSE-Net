"""Evaluate an existing FUSE-Net checkpoint using its saved encoder and tokenizer."""

import argparse
import json
from pathlib import Path


def load_model(checkpoint_path, device="cpu", data_dir=None):
    import torch
    from transformers import AutoConfig, AutoModel
    from .config import Config
    from .models import FUSENet
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("This evaluator expects the current FUSE-Net checkpoint format")
    config = Config(**checkpoint["config"])
    if data_dir is not None:
        config.data_dir = str(Path(data_dir).expanduser().resolve())
    config.device = str(device)
    values = dict(checkpoint["encoder_config"])
    model_type = values.pop("model_type")
    encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **values))
    model = FUSENet(config, text_encoder=encoder)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
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
               "checkpoint": file_identity(args.checkpoint), "dataset": file_identity(loaders[args.split].dataset.path)})
    write_predictions(args.output_dir / "predictions.csv", *predictions)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
