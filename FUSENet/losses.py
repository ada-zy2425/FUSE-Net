"""Paper equations (2)--(7) and (17); see docs/paper_alignment.md for Eq. (3)."""

import torch
from torch.nn import functional as F

from .models import MODALITIES


INFORMATION_OBJECTIVE_VERSION = "bounded_mse_grl_v1"


class _ReverseGradient(torch.autograd.Function):
    """Identity forward; reverse the gradient to the feature encoder only."""

    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def auxiliary_prediction(head, features, bound):
    """Smoothly bound auxiliary regression to the dataset's annotation range.

    This leaves the main sentiment prediction unchanged and prevents the
    adversarial feature objective from decreasing without bound.
    """
    logits = head(features).reshape(-1)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite auxiliary prediction logits")
    return bound * torch.tanh(logits / bound)


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


def information_loss(model, reps, targets, *, return_details=False):
    """Train predictive shared/specific factors and an adversarial noise factor.

    Define a branch score as NEGATIVE bounded-prediction MSE. Equation (3)
    then gives the encoder objective E_shared + E_private - E_noise. All
    auxiliary heads must instead MINIMIZE their own error, including noise.

    The positive error sum below, with gradient reversal on the noise input,
    implements these different player gradients in one backward pass. Its
    forward value is a training surrogate, not the encoder objective. Both
    values and each branch error are logged separately. The outer objective
    applies info_gain_weight once; no second scaling occurs in the GRL.
    """
    if targets.ndim > 2 or (targets.ndim == 2 and targets.size(1) != 1):
        raise ValueError("Information targets must be scalar, [batch], or [batch,1]")
    targets = targets.reshape(-1)
    bound = model.config.sentiment_bound
    if (targets.numel() == 0 or not torch.is_floating_point(targets)
            or not torch.isfinite(targets).all() or torch.any(targets.abs() > bound + 1e-6)):
        raise ValueError("Information targets must be finite floating labels in the dataset range")
    errors = {}
    for branch, head in (("shared", model.info_gain_head_s),
                         ("private", model.info_gain_head_h),
                         ("noise", model.info_gain_head_n)):
        per_modality = []
        for m in MODALITIES:
            features = reps[branch][m]
            if branch == "noise":
                features = _ReverseGradient.apply(features)
            prediction = auxiliary_prediction(head, features, bound)
            if prediction.shape != targets.shape:
                raise ValueError("Auxiliary prediction and target shapes differ")
            per_modality.append(F.mse_loss(prediction, targets))
        errors[branch] = torch.stack(per_modality).mean()
    training_loss = errors["shared"] + errors["private"] + errors["noise"]
    if not return_details:
        return training_loss
    details = {"information_{}_mse".format(branch): value.detach() for branch, value in errors.items()}
    details["information_encoder"] = (errors["shared"] + errors["private"] - errors["noise"]).detach()
    return training_loss, details


def objective(model, predictions, reps, targets, config):
    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)
    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes differ")
    terms = {
        "task": F.mse_loss(predictions, targets),
        "contrastive": contrastive_loss(reps["shared"], reps["private"], config.temperature),
        "dual": dual_loss(model, reps["shared"], reps["private"]),
        "reconstruction": reconstruction_loss(reps["mrc"], config.vib_beta),
    }
    if config.info_gain_weight:
        terms["information"], information_details = information_loss(model, reps, targets, return_details=True)
    else:
        # Avoid creating even zero gradients: AdamW would otherwise decay the
        # unused heads. Disabling this term must disable its parameter updates.
        terms["information"] = predictions.new_zeros(())
        information_details = {}
    weights = {"task": config.task_weight, "contrastive": config.info_weight,
               "information": config.info_gain_weight, "dual": config.cycle_weight,
               "reconstruction": config.recon_weight}
    for name, value in terms.items():
        if not torch.isfinite(value):
            raise FloatingPointError("Non-finite {} loss".format(name))
    terms["total"] = sum(weights[name] * value for name, value in terms.items())
    if not torch.isfinite(terms["total"]):
        raise FloatingPointError("Non-finite total loss")
    terms.update(information_details)
    return terms
