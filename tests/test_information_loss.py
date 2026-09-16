"""Behavioral checks for the encoder/predictor game, not benchmark claims."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from FUSENet.config import Config
from FUSENet.losses import auxiliary_prediction, information_loss
from FUSENet.models import MODALITIES


BRANCHES = ("shared", "private", "noise")
torch.set_num_threads(1)


class AuxiliaryModel(nn.Module):
    def __init__(self, data="mosi", width=3):
        super().__init__()
        self.config = Config(data=data)
        self.info_gain_head_s = nn.Linear(width, 1)
        self.info_gain_head_h = nn.Linear(width, 1)
        self.info_gain_head_n = nn.Linear(width, 1)

    def heads(self):
        return (self.info_gain_head_s, self.info_gain_head_h, self.info_gain_head_n)


def branch_errors(model, reps, targets):
    return {branch: torch.stack([
        F.mse_loss(auxiliary_prediction(head, reps[branch][m], model.config.sentiment_bound), targets)
        for m in MODALITIES]).mean() for branch, head in zip(BRANCHES, model.heads())}


def synthetic_training_check(seed):
    """Independent held-out probe distinguishes suppression from a broken head.

    Each factor is a learnable amount of label signal plus independent noise.
    A fresh least-squares probe is fitted on separate data, never on the test
    set. This controlled regression experiment is not a full FUSE-Net run.
    """
    torch.manual_seed(seed)
    model = AuxiliaryModel("simsv2", width=1)
    coefficients = nn.Parameter(torch.tensor([0.25, 0.25, 1.0]))
    for head in model.heads():
        nn.init.constant_(head.weight, 0.4)
        nn.init.zeros_(head.bias)
    optimizer = torch.optim.SGD([{"params": model.parameters(), "lr": 0.15},
                                 {"params": [coefficients], "lr": 0.1}])
    generator = torch.Generator().manual_seed(seed + 100)
    y_eval = 2 * torch.rand(4096, generator=generator) - 1
    nuisance_eval = torch.randn(4096, generator=generator)

    def probe_r2(coefficient):
        y_fit = 2 * torch.rand(4096, generator=torch.Generator().manual_seed(1000 + seed)) - 1
        nuisance_fit = torch.randn(4096, generator=torch.Generator().manual_seed(2000 + seed))
        fit = torch.stack((float(coefficient) * y_fit + nuisance_fit, torch.ones_like(y_fit)), dim=1)
        weight = torch.linalg.lstsq(fit, y_fit).solution
        test = torch.stack((float(coefficient) * y_eval + nuisance_eval, torch.ones_like(y_eval)), dim=1)
        return float(1 - (test @ weight - y_eval).square().mean() / y_eval.var(unbiased=False))

    def errors():
        reps = {branch: {m: (coefficients[i] * y_eval + nuisance_eval)[:, None] for m in MODALITIES}
                for i, branch in enumerate(BRANCHES)}
        with torch.no_grad():
            return {key: float(value) for key, value in branch_errors(model, reps, y_eval).items()}

    before, probe_before = errors(), probe_r2(coefficients[2].detach())
    for _ in range(400):
        targets, nuisance = 2 * torch.rand(256) - 1, torch.randn(256)
        reps = {branch: {m: (coefficients[i] * targets + nuisance)[:, None] for m in MODALITIES}
                for i, branch in enumerate(BRANCHES)}
        optimizer.zero_grad(set_to_none=True)
        information_loss(model, reps, targets).backward()
        optimizer.step()
    return {"seed": seed, "before_mse": before, "after_mse": errors(),
            "noise_probe_r2_before": probe_before, "noise_probe_r2_after": probe_r2(coefficients[2].detach())}


class InformationLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def make_case(self, data="mosi", count=4):
        model = AuxiliaryModel(data)
        reps = {branch: {m: torch.randn(count, 3, requires_grad=True) for m in MODALITIES}
                for branch in BRANCHES}
        targets = torch.linspace(-model.config.sentiment_bound, model.config.sentiment_bound, count)
        return model, reps, targets

    def test_all_head_gradients_fit_labels_and_only_noise_feature_gradient_reverses(self):
        for data in ("mosi", "mosei", "simsv2"):
            for count in (1, 4):
                with self.subTest(data=data, count=count):
                    model, reps, targets = self.make_case(data, count)
                    parameters = list(model.parameters())
                    features = [reps[branch][m] for branch in BRANCHES for m in MODALITIES]
                    ordinary = sum(branch_errors(model, reps, targets).values())
                    reference = torch.autograd.grad(ordinary, parameters + features)
                    actual = torch.autograd.grad(information_loss(model, reps, targets), parameters + features)
                    # Every auxiliary predictor minimizes its error, including noise.
                    for found, expected in zip(actual[:len(parameters)], reference[:len(parameters)]):
                        torch.testing.assert_close(found, expected)
                    for i, (found, expected) in enumerate(zip(actual[len(parameters):], reference[len(parameters):])):
                        torch.testing.assert_close(found, -expected if i >= 6 else expected)
                        self.assertGreater(float(expected.abs().sum()), 0)

    def test_one_step_improves_shared_specific_and_opposes_noise_predictability(self):
        model = AuxiliaryModel()
        for head in model.heads():
            nn.init.constant_(head.weight, 0.3)
            nn.init.constant_(head.bias, 0.5)
        reps = {branch: {m: torch.full((4, 3), 0.2, requires_grad=True) for m in MODALITIES}
                for branch in BRANCHES}
        targets = torch.zeros(4)
        before = {k: float(v.detach()) for k, v in branch_errors(model, reps, targets).items()}
        encoder_optimizer = torch.optim.SGD([value for branch in reps.values() for value in branch.values()], lr=0.1)
        information_loss(model, reps, targets).backward()
        encoder_optimizer.step()
        after = {k: float(v.detach()) for k, v in branch_errors(model, reps, targets).items()}
        self.assertLess(after["shared"], before["shared"])
        self.assertLess(after["private"], before["private"])
        self.assertGreater(after["noise"], before["noise"])
        # Hold features fixed and update heads: the noise predictor also improves.
        predictor_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        predictor_optimizer.zero_grad(set_to_none=True)
        detached = {b: {m: x.detach() for m, x in values.items()} for b, values in reps.items()}
        information_loss(model, detached, targets).backward()
        predictor_optimizer.step()
        fitted = branch_errors(model, detached, targets)
        for branch in BRANCHES:
            self.assertLess(float(fitted[branch].detach()), after[branch])

    def test_reversal_does_not_affect_other_consumers_of_noise_features(self):
        model, reps, targets = self.make_case()
        noise = [reps["noise"][m] for m in MODALITIES]
        # Represents task/reconstruction paths sharing these tensors.
        other_loss = sum((value - 0.25).square().mean() for value in noise)
        ordinary_noise_loss = branch_errors(model, reps, targets)["noise"]
        other_gradient = torch.autograd.grad(other_loss, noise, retain_graph=True)
        noise_gradient = torch.autograd.grad(ordinary_noise_loss, noise)
        combined = other_loss + 0.25 * information_loss(model, reps, targets)
        actual = torch.autograd.grad(combined, noise)
        for found, other, adversary in zip(actual, other_gradient, noise_gradient):
            torch.testing.assert_close(found, other - 0.25 * adversary)

    def test_bounded_predictions_prevent_unbounded_negative_encoder_loss(self):
        for data in ("mosi", "simsv2"):
            for extreme in (-1e6, 1e6):
                with self.subTest(data=data, extreme=extreme):
                    model, reps, targets = self.make_case(data)
                    for head in model.heads():
                        nn.init.zeros_(head.weight)
                        nn.init.constant_(head.bias, extreme)
                    training, details = information_loss(model, reps, targets, return_details=True)
                    bound = model.config.sentiment_bound
                    self.assertGreaterEqual(float(training), 0)
                    self.assertLessEqual(float(training), 12 * bound ** 2)
                    self.assertGreaterEqual(float(details["information_encoder"]), -4 * bound ** 2)
                    self.assertLessEqual(float(details["information_encoder"]), 8 * bound ** 2)
                    training.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_training_surrogate_and_encoder_objective_are_logged_separately(self):
        model, reps, targets = self.make_case()
        loss, details = information_loss(model, reps, targets, return_details=True)
        errors = branch_errors(model, reps, targets)
        torch.testing.assert_close(loss, sum(errors.values()))
        torch.testing.assert_close(details["information_encoder"], errors["shared"] + errors["private"] - errors["noise"])
        self.assertTrue(all(not value.requires_grad for value in details.values()))
        # Column labels and singleton labels retain sample alignment.
        torch.testing.assert_close(loss, information_loss(model, reps, targets[:, None]))
        singleton = {b: {m: x[:1] for m, x in values.items()} for b, values in reps.items()}
        self.assertTrue(torch.isfinite(information_loss(model, singleton, targets[0])))

    def test_invalid_labels_and_broadcasting_fail_explicitly(self):
        model, reps, _ = self.make_case()
        invalid = (torch.tensor([]), torch.ones(3), torch.ones(2, 2), torch.ones(4, dtype=torch.long),
                   torch.full((4,), float("nan")), torch.full((4,), float("inf")), torch.full((4,), 3.1))
        for targets in invalid:
            with self.subTest(shape=targets.shape, dtype=targets.dtype):
                with self.assertRaises(ValueError):
                    information_loss(model, reps, targets)

    def test_nonfinite_logits_are_not_hidden_by_tanh(self):
        model, reps, targets = self.make_case()
        with torch.no_grad():
            model.info_gain_head_n.bias.fill_(float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "logits"):
            information_loss(model, reps, targets)

    def test_training_reduces_noise_signal_for_a_fresh_held_out_probe(self):
        for seed in (7, 11, 42):
            with self.subTest(seed=seed):
                result = synthetic_training_check(seed)
                for branch in ("shared", "private"):
                    self.assertLess(result["after_mse"][branch], 0.5 * result["before_mse"][branch])
                self.assertGreater(result["noise_probe_r2_before"], 0.2)
                self.assertLess(result["noise_probe_r2_after"], 0.02)


if __name__ == "__main__":
    unittest.main()
