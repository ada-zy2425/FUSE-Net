"""Training, validation-only checkpoint selection, and saved test predictions."""

import importlib.metadata
import platform
import warnings
from pathlib import Path

import numpy as np
import torch

from .data_loader import file_identity
from .losses import objective
from .metrics import calculate_metrics
from .models import FUSENet
from .runtime import move_batch, save_checkpoint, select_device, write_json, write_predictions


@torch.no_grad()
def evaluate(model, loader, device, dataset):
    model.eval()
    predictions, targets, ids, source_ids = [], [], [], []
    for batch in loader:
        output, _ = model(move_batch(batch, device))
        predictions.append(output.detach().cpu().reshape(-1))
        targets.append(batch["labels"].cpu().reshape(-1))
        ids.extend(batch["ids"])
        source_ids.extend(batch["source_ids"])
    if not predictions:
        raise ValueError("Evaluation loader is empty")
    predictions, targets = torch.cat(predictions).numpy(), torch.cat(targets).numpy()
    return calculate_metrics(targets, predictions, dataset), (ids, source_ids, targets, predictions)


class Solver:
    def __init__(self, config, loaders, tokenizer, output_dir, model=None):
        self.config, self.loaders, self.tokenizer = config, loaders, tokenizer
        self.output_dir = Path(output_dir)
        self.device = select_device(config.device)
        self.model = (FUSENet(config) if model is None else model).to(self.device)
        bert, other = [], []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                (bert if name.startswith("bert.") else other).append(parameter)
        self.optimizer = torch.optim.AdamW([
            {"params": bert, "lr": config.bert_learning_rate},
            {"params": other, "lr": config.learning_rate},
        ], weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=config.patience)

    def train(self):
        # Reserve one directory per run so previous measurements are not overwritten.
        self.output_dir.mkdir(parents=True, exist_ok=False)
        if self.config.info_gain_weight:
            warnings.warn("Eq. (3) score definition remains unresolved: the uploaded -MSE(shared)-MSE(private)+MSE(noise) proxy is retained. See docs/paper_alignment.md.", RuntimeWarning)
        self.tokenizer.save_pretrained(self.output_dir / "tokenizer")
        paths = sorted({str(loader.dataset.path) for loader in self.loaders.values()})
        versions = {}
        for name in ("torch", "transformers", "numpy", "scikit-learn"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        metadata = {"config": self.config.to_dict(), "environment": {"python": platform.python_version(), **versions},
                    "device": str(self.device), "dataset_files": [file_identity(path) for path in paths],
                    "split_counts": {name: len(loader.dataset) for name, loader in self.loaders.items()},
                    "encoder_revision_resolved": getattr(self.model.bert.config, "_commit_hash", None),
                    "information_score_status": "unresolved_uploaded_mse_proxy"}
        write_json(self.output_dir / "run.json", metadata)
        best_mae, remaining, history = float("inf"), self.config.patience, []
        for epoch in range(1, self.config.n_epoch + 1):
            self.model.train()
            totals, count = {}, 0
            for batch in self.loaders["train"]:
                batch = move_batch(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                output, reps = self.model(batch)
                terms = objective(self.model, output, reps, batch["labels"], self.config)
                terms["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip, error_if_nonfinite=True)
                self.optimizer.step()
                size = len(batch["labels"])
                count += size
                for name, value in terms.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
            if count == 0:
                raise ValueError("Training loader is empty")
            dev_metrics, _ = evaluate(self.model, self.loaders["dev"], self.device, self.config.data)
            dev_mae = dev_metrics["mae"]
            self.scheduler.step(dev_mae)
            history.append({"epoch": epoch, "losses": {k: v / count for k, v in totals.items()}, "dev": dev_metrics,
                            "learning_rates": [group["lr"] for group in self.optimizer.param_groups]})
            write_json(self.output_dir / "history.json", history)
            print("Epoch {}/{} | validation MAE {:.6f}".format(epoch, self.config.n_epoch, dev_mae), flush=True)
            if dev_mae < best_mae:
                best_mae, remaining = dev_mae, self.config.patience
                save_checkpoint(self.output_dir / "best.pt", {
                    "format_version": 2, "epoch": epoch, "validation_mae": best_mae,
                    "config": self.config.to_dict(), "encoder_config": self.model.bert.config.to_dict(),
                    "state_dict": {name: value.detach().cpu() for name, value in self.model.state_dict().items()},
                })
            else:
                remaining -= 1
                if remaining <= 0:
                    break
        checkpoint = torch.load(self.output_dir / "best.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        metrics, prediction_data = evaluate(self.model, self.loaders["test"], self.device, self.config.data)
        write_json(self.output_dir / "test_metrics.json", {"selected_epoch": checkpoint["epoch"], "metrics": metrics})
        write_predictions(self.output_dir / "test_predictions.csv", *prediction_data)
        return metrics


def summarize_runs(results):
    if not results:
        raise ValueError("No runs to summarize")
    summary = {}
    for key in results[0]:
        values = [result[key] for result in results if result[key] is not None]
        summary[key] = {"defined_runs": len(values), "total_runs": len(results),
                        "mean": float(np.mean(values)) if values else None,
                        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return summary
