# Data preparation and input contracts

This document describes the **supplied source**, not a newly validated preprocessing pipeline. Datasets and derived features are not included.

Run `python FUSENet/train.py ...` from the repository root. Both `FUSENet/config.py` and `FUSENet/create_dataset.py` resolve their root to the parent of `FUSENet/`.

| Location under the repository root | Expected content |
| --- | --- |
| `CMU-MultimodalSDK/` | Local SDK checkout, installed in the Python environment. |
| `datasets/MOSI/train.pkl`, `dev.pkl`, `test.pkl` | Cached MOSI samples, or an output directory for SDK preprocessing. |
| `datasets/MOSEI/train.pkl`, `dev.pkl`, `test.pkl` | Cached MOSEI samples, or an output directory for SDK preprocessing. |
| `datasets/CH-SIMS/unaligned.pkl` | The schema described below; exact dataset version is not established. |

The SDK is imported before dataset selection, so even loading an existing pickle currently requires `mmsdk`. Use pickle files only from a trusted source: Python pickle can execute code when loaded.

## CMU-MOSI and CMU-MOSEI

Each cache is a list of samples with this structure:

```python
((None, visual, acoustic, words), label, video_id)
```

- `visual`: numeric array of shape `(T, visual_dim)`.
- `acoustic`: numeric array of shape `(T, acoustic_dim)`.
- `words`: list of decoded strings used to build the tokenizer input.
- `label`: a NumPy array containing the utterance-level regression target; retain the SDK-produced shape accepted by the collator.
- `video_id`: source video identifier used for split assignment.

The model uses visual sequence lengths for **both** recurrent encoders. Consequently, the current path requires equal, positive audio and visual sequence lengths for every sample, consistent feature dimensions, nonempty splits, and finite features and targets. The source does not enforce all of these conditions.

If any of the three caches is missing, `_create_data` requests SDK features and labels, aligns features to word timestamps, then aligns to labels. It removes `sp` tokens, standardizes visual/audio features within each sample, applies `nan_to_num`, and writes all three caches. Keep a copy of an established preprocessing output before rebuilding it.

| Dataset | Words | Visual | Audio | Labels |
| --- | --- | --- | --- | --- |
| MOSI | `CMU_MOSI_TimestampedWords` | `CMU_MOSI_VisualFacet_4.1` | `CMU_MOSI_COVAREP` | `CMU_MOSI_Opinion_Labels` |
| MOSEI | `CMU_MOSEI_TimestampedWords` | `CMU_MOSEI_VisualFacet42` | `CMU_MOSEI_COVAREP` | `CMU_MOSEI_LabelsSentiment` |

Use the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) to obtain authorized data and inspect feature recipes. The uploaded source does not request a separate raw-word recipe, although it requires timestamped-word files. Verify those files exist for the chosen SDK revision; do not assume a high-level feature download supplies every required sequence.

The code drops segments with mismatched preprocessed lengths. The paper's nominal split sizes therefore do not prove the sizes of the resulting caches. Record final sample counts, excluded segments, video IDs, SDK revision, and cache hashes for a reproducible run.

## CH-SIMS / SIMSv2 reconciliation

The `ch-sims` loader expects `unaligned.pkl` to contain `train`, `valid`, and `test` dictionaries. Each split must provide:

| Key | Used for |
| --- | --- |
| `vision` | Per-sample visual arrays. |
| `audio` | Per-sample acoustic arrays. |
| `raw_text` | Text strings for tokenization. |
| `regression_labels` | Continuous sentiment labels. |
| `id` | Sample identifiers. |

The internal `dev` mode maps to `valid`. The loader does not establish whether the file is CH-SIMS v1 or CH-SIMS v2, and it does not read independent modality length fields or trim padding based on them.

**Do not treat the current loader as a verified SIMSv2 recipe.** The paper uses original unaligned temporal sequences, whereas the model forwards the visual length to the acoustic encoder as well. Unequal lengths can truncate audio or fail during packed-sequence processing. Five-class and three-class SIMSv2 discretization also need a dataset-specific evaluator; the current generic rounding metric is insufficient.

Before a SIMSv2 run, establish the exact feature release, split membership, actual audio/visual lengths, Chinese language model/tokenizer, label range, and evaluation thresholds. Dataset renaming or arbitrary padding to equal lengths would not resolve the protocol mismatch.

## What to retain with experiments

For each actual run, retain the full command and resolved configuration, seed, package versions, model/tokenizer revisions, dataset hashes and split counts, best checkpoint selected on validation data, per-sample test predictions and IDs, and metric definitions. These records were not present in the upload.
