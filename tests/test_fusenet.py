"""Offline synthetic checks for formula semantics, masks, metrics, and run artifacts."""

import csv
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, RobertaConfig, RobertaModel

from FUSENet.config import Config
from FUSENet.data_loader import Collator, MSADataset, make_loaders
from FUSENet.evaluate import load_model
from FUSENet.losses import INFORMATION_OBJECTIVE_VERSION, contrastive_loss, objective, reconstruction_loss
from FUSENet.metrics import calculate_metrics, simsv2_classes
from FUSENet.models import FUSENet, MODALITIES
from FUSENet.runtime import select_device
from FUSENet.solver import summarize_runs


torch.set_num_threads(1)


def tiny_encoder(kind="bert"):
    kwargs = dict(vocab_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                  intermediate_size=24, max_position_embeddings=64, hidden_dropout_prob=0.0,
                  attention_probs_dropout_prob=0.0)
    return BertModel(BertConfig(**kwargs)) if kind == "bert" else RobertaModel(RobertaConfig(**kwargs))


def tiny_config(**kwargs):
    values = dict(hidden_size=8, visual_size=2, acoustic_size=3, dropout=0.0, max_text_length=16)
    return Config(**{**values, **kwargs})


def batch():
    torch.manual_seed(123)
    return {"input_ids": torch.tensor([[2, 6, 3, 0], [2, 7, 8, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            "visual": torch.randn(2, 4, 2), "acoustic": torch.randn(2, 7, 3),
            "visual_lengths": torch.tensor([2, 4]), "acoustic_lengths": torch.tensor([7, 3]),
            "labels": torch.tensor([-0.2, 0.8]), "ids": ["a", "b"], "source_ids": ["a", "b"]}


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = tiny_config()
        self.model = FUSENet(self.config, tiny_encoder())

    def test_unaligned_forward_backward_bert_and_roberta(self):
        for kind in ("bert", "roberta"):
            with self.subTest(kind=kind):
                model = FUSENet(self.config, tiny_encoder(kind))
                output, reps = model(batch())
                self.assertEqual(tuple(output.shape), (2,))
                losses = objective(model, output, reps, batch()["labels"], self.config)
                losses["total"].backward()
                for module in (model.vrnn, model.arnn, model.mrc_decoders, model.fuse_mlp):
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_padding_does_not_change_predictions(self):
        self.model.eval()
        original = batch()
        changed = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in original.items()}
        changed["visual"][0, 2:] = 10000
        changed["acoustic"][1, 3:] = -10000
        changed["input_ids"][0, 3] = 10
        with torch.no_grad():
            first, _ = self.model(original)
            second, _ = self.model(changed)
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    def test_mdf_normalizes_each_modality_and_uses_paper_order(self):
        self.model.eval()
        _, reps = self.model(batch())
        expected = []
        for index, factor in enumerate(("shared", "private", "noise")):
            parts = []
            for m in MODALITIES:
                torch.testing.assert_close(reps["weights"][m].sum(1), torch.ones(2))
                self.assertTrue(((reps["gate"][m] > 0) & (reps["gate"][m] < 1)).all())
                value = reps[factor][m]
                if factor == "noise":
                    value = value * reps["gate"][m]
                parts.append(reps["weights"][m][:, index:index+1] * value)
            expected.append(sum(parts))
        torch.testing.assert_close(reps["fused"], torch.cat(expected, dim=1))

    def test_latent_empirical_variance_matches_kl_parameterization(self):
        torch.manual_seed(5)
        mu = torch.full((40000,), 0.3)
        log_var = torch.full_like(mu, np.log(4.0))
        sampled = FUSENet.sample_latent(mu, log_var)
        self.assertAlmostEqual(float(sampled.mean()), 0.3, delta=0.04)
        self.assertAlmostEqual(float(sampled.var()), 4.0, delta=0.12)
        torch.testing.assert_close(FUSENet.sample_latent(mu, log_var, False), mu)

    def test_single_example_and_invalid_lengths(self):
        item = {key: value[:1] for key, value in batch().items()}
        output, _ = self.model(item)
        self.assertEqual(tuple(output.shape), (1,))
        for invalid in (0, 8):
            item["acoustic_lengths"] = torch.tensor([invalid])
            with self.assertRaisesRegex(ValueError, "lengths"):
                self.model(item)

    def test_reconstruction_sums_feature_norms_and_modalities(self):
        outputs = {m: {"target": torch.zeros(2, 5), "recon": torch.ones(2, 5),
                       "mu": torch.zeros(2, 5), "log_var": torch.zeros(2, 5)} for m in MODALITIES}
        self.assertEqual(float(reconstruction_loss(outputs, 0.01)), 15.0)

    def test_contrastive_alignment_improves_loss(self):
        shared = {m: torch.tensor([[1.0, 0.0]]) for m in MODALITIES}
        private = {m: torch.tensor([[0.0, 1.0]]) for m in MODALITIES}
        aligned = contrastive_loss(shared, private, 0.3)
        shared["a"] = torch.tensor([[-1.0, 0.0]])
        self.assertLess(float(aligned), float(contrastive_loss(shared, private, 0.3)))

    def test_disabled_information_term_does_not_update_auxiliary_heads(self):
        self.config.info_gain_weight = 0.0
        parameters = [p for head in (self.model.info_gain_head_s, self.model.info_gain_head_h,
                                    self.model.info_gain_head_n) for p in head.parameters()]
        before = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.1)
        output, reps = self.model(batch())
        terms = objective(self.model, output, reps, batch()["labels"], self.config)
        terms["total"].backward()
        self.assertEqual(float(terms["information"]), 0.0)
        self.assertTrue(all(p.grad is None for p in parameters))
        optimizer.step()
        for old, parameter in zip(before, parameters):
            torch.testing.assert_close(old, parameter, rtol=0, atol=0)


class MetricTests(unittest.TestCase):
    def test_binary_mappings_and_both_f1_values(self):
        result = calculate_metrics([-1, 0, 1], [-0.1, -0.2, 0.7], "mosi")
        self.assertAlmostEqual(result["acc2_nonneg_neg"], 2 / 3)
        self.assertAlmostEqual(result["f1_nonneg_neg"], 2 / 3)
        self.assertEqual(result["acc2_pos_neg"], 1.0)
        self.assertEqual(result["f1_pos_neg"], 1.0)

    def test_simsv2_boundary_conventions(self):
        for dtype in (np.float32, np.float64):
            values = np.array([-1, -0.7, -0.1, 0, 0.1, 0.7, 1], dtype=dtype)
            np.testing.assert_array_equal(simsv2_classes(values, 5), [0, 0, 1, 2, 2, 3, 4])
            np.testing.assert_array_equal(simsv2_classes(values, 3), [0, 0, 0, 1, 1, 2, 2])
            np.testing.assert_array_equal(simsv2_classes(values, 2), [0, 0, 0, 0, 1, 1, 1])

    def test_simsv2_clips_regression_like_official_evaluator(self):
        result = calculate_metrics([-1, 1], [-2, 2], "simsv2")
        self.assertEqual(result["mae"], 0)
        self.assertEqual(result["acc5"], 1)

    def test_undefined_and_invalid_metrics(self):
        result = calculate_metrics([0], [0.3], "mosei")
        self.assertIsNone(result["pearson"])
        self.assertIsNone(result["acc2_pos_neg"])
        self.assertIsNone(result["f1_pos_neg"])
        for truth, pred in (([], []), ([0], [0, 1]), ([0], [np.nan]), ([4], [0])):
            with self.assertRaises(ValueError):
                calculate_metrics(truth, pred, "mosi")
        summary = summarize_runs([result, result])
        self.assertEqual(summary["pearson"]["defined_runs"], 0)
        self.assertIsNone(summary["pearson"]["mean"])


class DataAndRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        vocab = self.root / "vocab.txt"
        vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\npositive\nnegative\ntext\n")
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab), model_max_length=16)

    def tearDown(self):
        self.temporary.cleanup()

    def sims_payload(self):
        return {"train": {"vision": np.ones((2, 4, 2), dtype=np.float32), "audio": np.ones((2, 7, 3), dtype=np.float32),
                "vision_lengths": [2, 4], "audio_lengths": [6, 1], "raw_text": ["a positive", "a negative"],
                "regression_labels": np.array([0.5, -0.5]), "id": ["train1", "train2"]}}

    def test_simsv2_preserves_independent_lengths(self):
        config = tiny_config(data="simsv2", data_dir=str(self.root))
        data = MSADataset(config, "train", self.sims_payload())
        result = Collator(self.tokenizer, 16)(data.samples)
        self.assertEqual(result["visual_lengths"].tolist(), [2, 4])
        self.assertEqual(result["acoustic_lengths"].tolist(), [6, 1])
        self.assertEqual(tuple(result["visual"].shape), (2, 4, 2))
        self.assertEqual(tuple(result["acoustic"].shape), (2, 6, 3))

    def test_missing_and_invalid_simsv2_lengths_fail_explicitly(self):
        config = tiny_config(data="simsv2", data_dir=str(self.root))
        payload = self.sims_payload()
        del payload["train"]["audio_lengths"]
        with self.assertRaisesRegex(ValueError, "audio_lengths"):
            MSADataset(config, "train", payload)
        for length in (0, 8, 2.5, float("nan")):
            payload = self.sims_payload()
            payload["train"]["audio_lengths"][0] = length
            with self.assertRaises(ValueError):
                MSADataset(config, "train", payload)

    def make_mosi(self):
        data_dir = self.root / "mosi"
        data_dir.mkdir()
        rng = np.random.default_rng(9)
        for split, count in (("train", 3), ("dev", 1), ("test", 2)):
            samples = []
            for i in range(count):
                length = i + 1
                samples.append(((None, rng.normal(size=(length, 2)).astype(np.float32),
                                 rng.normal(size=(length, 3)).astype(np.float32), ["text"] * length),
                                np.array([i - 1], dtype=np.float32), split + str(i)))
            with (data_dir / (split + ".pkl")).open("wb") as stream:
                pickle.dump(samples, stream)
        return data_dir

    def test_split_overlap_is_rejected(self):
        data_dir = self.make_mosi()
        (data_dir / "dev.pkl").write_bytes((data_dir / "train.pkl").read_bytes())
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_loaders(tiny_config(data_dir=str(data_dir)), tokenizer=self.tokenizer)

    def test_real_cli_two_seeds_and_saved_checkpoint_evaluation(self):
        data_dir = self.make_mosi()
        encoder_dir = self.root / "tiny_bert"
        tiny_encoder().save_pretrained(encoder_dir)
        self.tokenizer.save_pretrained(encoder_dir)
        output = self.root / "runs"
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        train = [sys.executable, "-m", "FUSENet.train", "--data", "mosi", "--data_dir", str(data_dir),
                 "--bert_dir", str(encoder_dir), "--output_dir", str(output), "--seeds", "7", "11",
                 "--batch_size", "2", "--n_epoch", "1", "--hidden_size", "8", "--max_text_length", "16", "--device", "cpu"]
        result = subprocess.run(train, text=True, capture_output=True, env=environment, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["seeds"], [7, 11])
        self.assertEqual(summary["metrics"]["mae"]["defined_runs"], 2)
        checkpoint = output / "seed_7/best.pt"
        model, config = load_model(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(saved["format_version"], 3)
        self.assertEqual(model.training_objective_version, INFORMATION_OBJECTIVE_VERSION)
        run = json.loads((output / "seed_7/run.json").read_text())
        self.assertEqual(run["information_objective"]["version"], INFORMATION_OBJECTIVE_VERSION)
        history = json.loads((output / "seed_7/history.json").read_text())
        losses = history[0]["losses"]
        self.assertAlmostEqual(losses["information_encoder"], losses["information_shared_mse"]
                               + losses["information_private_mse"] - losses["information_noise_mse"], places=5)
        # Version 2 has the same inference architecture and explicit legacy provenance.
        legacy = {**saved, "format_version": 2}
        legacy.pop("information_objective")
        legacy_path = self.root / "legacy.pt"
        torch.save(legacy, legacy_path)
        legacy_model, _ = load_model(legacy_path)
        self.assertEqual(legacy_model.training_objective_version, "legacy_signed_mse_v0")
        with torch.no_grad():
            torch.testing.assert_close(model(batch())[0], legacy_model(batch())[0], rtol=0, atol=0)
        invalid = {**saved, "information_objective": "unknown_future_objective"}
        invalid_path = self.root / "invalid.pt"
        torch.save(invalid, invalid_path)
        with self.assertRaisesRegex(ValueError, "objective"):
            load_model(invalid_path)
        self.assertEqual(config.seed, 7)
        self.assertEqual(str(next(model.parameters()).device), "cpu")
        evaluation = self.root / "evaluation"
        command = [sys.executable, "-m", "FUSENet.evaluate", "--checkpoint", str(checkpoint),
                   "--data_dir", str(data_dir), "--device", "cpu", "--output_dir", str(evaluation)]
        result = subprocess.run(command, text=True, capture_output=True, env=environment, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        before = json.loads((output / "seed_7/test_metrics.json").read_text())["metrics"]
        after = json.loads((evaluation / "metrics.json").read_text())["metrics"]
        self.assertEqual(json.loads((evaluation / "metrics.json").read_text())["training_objective"],
                         INFORMATION_OBJECTIVE_VERSION)
        self.assertEqual(before, after)
        with (evaluation / "predictions.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)
        # Existing output directories must be protected from accidental overwrite.
        result = subprocess.run(train, text=True, capture_output=True, env=environment, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)

    def test_configuration_and_cpu_selection(self):
        self.assertEqual(str(select_device("cpu")), "cpu")
        for kwargs in ({"temperature": 0}, {"batch_size": 0}, {"dropout": 1}, {"seed": -1}):
            with self.assertRaises(ValueError):
                tiny_config(**kwargs)


if __name__ == "__main__":
    unittest.main()
