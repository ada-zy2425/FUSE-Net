# FUSE-Net

**Factorize, Reconstruct, Enhance: A Unified Framework for Multimodal Sentiment Analysis**

FUSE-Net predicts sentiment from text, audio, and visual signals using hierarchical modality factorization (**HMF**), a variational modality reconstruction channel (**MRC**), and multi-factor dynamic fusion (**MDF**).

[Paper](paper/FUSE-Net.pdf) · [Paper ↔ code](docs/paper_alignment.md) · [Data preparation](docs/data.md)

![Framework from Figure 1 of the supplied manuscript](docs/assets/framework.png)

## Implementation

This repository contains one FUSE-Net implementation. The paper-to-code mapping is:

| Paper | Code |
| --- | --- |
| Modality encoders and HMF | [`FUSENet/models.py`](FUSENet/models.py) |
| Contrastive separation, information term, dual consistency | [`FUSENet/losses.py`](FUSENet/losses.py), equations (2)–(5) |
| MRC: Gaussian sampling, reconstruction, KL | `models.py::sample_latent`, `losses.py::reconstruction_loss`, equations (6)–(7) |
| MDF: modulation, branch softmax, noise gate, aggregation | `models.py::forward`, equations (8)–(16) |
| Optimization, validation selection, checkpoints | [`FUSENet/solver.py`](FUSENet/solver.py), equation (17) |
| MOSI/MOSEI and SIMSv2 evaluation | [`FUSENet/metrics.py`](FUSENet/metrics.py) |

**Implementation status:** the information-loss direction is corrected. Shared/specific factors and all auxiliary predictors minimize prediction error; the noise representation receives an opposing gradient through a gradient reversal layer. Auxiliary predictions are bounded to the dataset's sentiment range. [The correction and its checks](docs/information_loss_fix.md) document the exact optimization rule. The main task remains ordinary MSE regression. This revision has synthetic validation; the paper's benchmark tables have not been rerun.

## Setup

```bash
git clone https://github.com/ada-zy2425/FUSE-Net.git
cd FUSE-Net
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.12, PyTorch 2.5.1 CPU, Transformers 4.46.3, NumPy 2.3.5, and scikit-learn 1.8.0 were used for the synthetic tests. Install the appropriate [PyTorch build](https://pytorch.org/get-started/previous-versions/#v251) for the intended hardware. The historical paper environment and model weights were not supplied.

## Prepare data

See [data.md](docs/data.md) for the exact cache schemas and preprocessing commands. Data lives under `datasets/MOSI`, `datasets/MOSEI`, or `datasets/SIMSv2`, or a directory passed with `--data_dir`.

- MOSI/MOSEI use word-aligned visual and acoustic features plus raw words.
- SIMSv2 uses its original temporal sequences and **independent visual/audio lengths**.
- Training reads existing data; raw-data preparation is a separate command.

## Train

From the repository root:

```bash
python -m FUSENet.train \
  --data mosi \
  --bert_dir roberta-large \
  --seeds 42 \
  --output_dir runs/mosi_trial
```

This command uses the text backbone and seed found in the uploaded source. It is a development run, **not a verified recipe for the paper's table**. Choose `mosi`, `mosei`, or `simsv2` with `--data`; provide the actual text model using `--bert_dir`, and pin its revision with `--revision` when known. The Chinese model used for SIMSv2 needs confirmation.

Multiple explicit seeds are supported with `--seeds`. For example, `--seeds 42 43 44 45 46` runs five independent seeds and aggregates their metrics; this example does not assert the original paper's seed set.

Each seed writes `run.json`, `history.json`, `best.pt`, the tokenizer, `test_metrics.json`, and per-example `test_predictions.csv`. The best epoch is selected using **validation MAE**. Test evaluation follows checkpoint selection. `summary.json` reports the mean, sample standard deviation, and number of defined runs for every metric. Existing output directories are not overwritten.

The information constraint is enabled by default (`--info_gain_weight 0.25`). `history.json` records its training surrogate, encoder objective, and separate shared/private/noise MSE values. `run.json` and checkpoints identify the corrected objective as `bounded_mse_grl_v1`. Setting the weight to zero disables auxiliary-head updates as well as their gradients to the factors.

## Evaluate a checkpoint

```bash
python -m FUSENet.evaluate \
  --checkpoint runs/mosi_trial/seed_42/best.pt \
  --data_dir datasets/MOSI \
  --output_dir runs/mosi_evaluation
```

Evaluation restores the saved encoder configuration, trained weights, and tokenizer. It does not download a new pretrained encoder. New checkpoints use format 3 and identify the corrected training objective. Format 2 checkpoints remain supported for inference and are explicitly marked as using the legacy information loss; the architecture is unchanged by this fix. Weights predating the shared/specific/noise fusion-order correction still require migration.

## Verify

```bash
python -m unittest discover -s tests -v
```

The offline synthetic suite checks BERT/RoBERTa forward and backward passes, unequal audio/visual lengths, padding invariance, Gaussian sampling, MDF normalization and fusion order, metric boundaries, singleton evaluation, split overlap, two-seed CLI training, and checkpoint evaluation. Information-loss checks verify the predictor/encoder gradients, boundedness, isolation from other objectives, disabled-head behavior, and three controlled training seeds with fresh held-out probes. Passing these checks does not reproduce the benchmark tables.

## Paper-reported results

These values are transcribed from Tables 1–2 of the [supplied manuscript](paper/FUSE-Net.pdf); full benchmark training has not been rerun for this revision. Accuracy and F1 below are percentages. The evaluator writes them as fractions.

| Dataset | MAE ↓ | Corr ↑ | Acc7 ↑ | Acc2 ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMU-MOSI | 0.688 | 0.798 | 49.27 | 83.53 / 85.83 | 83.75 / 85.96 |
| CMU-MOSEI | 0.527 | 0.772 | 54.32 | 84.52 / 85.88 | 84.50 / 86.14 |

Paired values use non-negative/negative on the left and positive/negative with neutral targets excluded on the right.

| Dataset | MAE ↓ | Corr ↑ | Acc2 ↑ | Acc3 ↑ | Acc5 ↑ | F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SIMSv2 | 0.297 | 0.712 | 79.59 | 73.69 | 55.51 | 79.53 |

## Paper

**Zhilu Yang and Mingcheng Li.** *Factorize, Reconstruct, Enhance: A Unified Framework for Multimodal Sentiment Analysis.* Author-provided manuscript. The [PDF](paper/FUSE-Net.pdf) is preserved unchanged; final publication metadata is not asserted here.

The repository remains private. No project-wide license has been selected. The dataset tools and pretrained models are obtained separately from their respective providers.
