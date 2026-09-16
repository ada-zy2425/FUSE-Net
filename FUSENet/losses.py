"""Paper equations (2)--(7) and (17); see docs/paper_alignment.md for Eq. (3)."""

import torch
from torch.nn import functional as F

from .models import MODALITIES


def contrastive_loss(shared, private, temperature):
    terms = []
    for a in MODALITIES:
        negative = F.cosine_similarity(shared[a], private[a], dim=1) / temperature
        for b in MODALITIES:
            if a != b:
                positive = F.cosine_similarity(shared[a], shared[b], dim=1) / temperature
                terms.append(F.softplus(negative - positive).mean())
    return torch.stack(terms).mean()


def dual_loss(model, shared, private):
    # Eq. (4): squared L2 norms (sum over features), mean over batch and modalities.
    return torch.stack([
        ((shared[m] - model.G[m](private[m])).square().sum(1)
         + (private[m] - model.G_inv[m](shared[m])).square().sum(1)).mean()
        for m in MODALITIES
    ]).mean()


def reconstruction_loss(outputs, beta):
    # Eq. (7): sum over modalities; the per-example feature norm is not an elementwise mean.
    terms = []
    for m in MODALITIES:
        out = outputs[m]
        recon = (out["recon"] - out["target"]).square().sum(1).mean()
        kl = 0.5 * (out["mu"].square() + out["log_var"].exp() - 1 - out["log_var"]).sum(1).mean()
        terms.append(recon + beta * kl)
    return torch.stack(terms).sum()


def information_loss(model, reps, targets):
    """Preserve the uploaded MSE proxy and Eq. (3) signs pending author clarification.

    IMPORTANT: treating an information score as MSE reverses the stated semantic
    objective. No new adversarial or bounded score is invented here. This term
    is explicitly unresolved and is not evidence of complete paper equivalence.
    """
    terms = []
    for m in MODALITIES:
        shared_error = F.mse_loss(model.info_gain_head_s(reps["shared"][m]).reshape(-1), targets)
        private_error = F.mse_loss(model.info_gain_head_h(reps["private"][m]).reshape(-1), targets)
        noise_error = F.mse_loss(model.info_gain_head_n(reps["noise"][m]).reshape(-1), targets)
        terms.append(-shared_error - private_error + noise_error)
    return torch.stack(terms).mean()


def objective(model, predictions, reps, targets, config):
    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes differ")
    terms = {
        "task": F.mse_loss(predictions, targets),
        "contrastive": contrastive_loss(reps["shared"], reps["private"], config.temperature),
        "information": information_loss(model, reps, targets),
        "dual": dual_loss(model, reps["shared"], reps["private"]),
        "reconstruction": reconstruction_loss(reps["mrc"], config.vib_beta),
    }
    weights = {"task": config.task_weight, "contrastive": config.info_weight,
               "information": config.info_gain_weight, "dual": config.cycle_weight,
               "reconstruction": config.recon_weight}
    for name, value in terms.items():
        if not torch.isfinite(value):
            raise FloatingPointError("Non-finite {} loss".format(name))
    terms["total"] = sum(weights[name] * value for name, value in terms.items())
    return terms
