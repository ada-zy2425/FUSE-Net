# Data and evaluation protocols

The implementation accepts `mosi`, `mosei`, and `simsv2`. The dataset release used in the original runs still needs to be established; a filename alone does not identify it. Datasets and pretrained assets are not included.

## MOSI / MOSEI

Each dataset directory contains `train.pkl`, `dev.pkl`, and `test.pkl`. Each file is a list of:

```python
((unused, visual, acoustic, words), label, video_id)
```

`visual` and `acoustic` are finite numeric arrays of shape `[T, Dv]` and `[T, Da]`. `words` is a list of `T` strings. `label` contains one finite scalar in `[-3, 3]`. Empty examples, mismatched word alignment, inconsistent feature dimensions, and overlapping source video IDs between splits are rejected with an explicit error.

To construct caches from raw CMU sequences, install the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) and run:

```bash
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
python -m pip install -e ./CMU-MultimodalSDK

python -m FUSENet.create_dataset --data mosi --data_dir datasets/MOSI
python -m FUSENet.create_dataset --data mosei --data_dir datasets/MOSEI
```

The preparation command requests missing raw-word, visual, acoustic, and label files, aggregates features over word intervals, aligns utterances to labels, removes silence tokens, and applies the uploaded per-utterance visual/acoustic standardization. It writes split caches plus `preprocessing.json` with retained segment IDs, exclusions, and counts. Existing caches require an explicit `--overwrite` to rebuild. Record the SDK revision separately. SDK downloading and full-dataset preprocessing have not been executed in this revision's synthetic validation.

Invalid targets are excluded rather than converted to neutral labels, and empty post-silence examples are excluded explicitly. Consequently, new cache counts must be compared with the actual experimental splits; the paper's nominal counts are not assumed to match automatically.

Only load trusted pickle files, since loading pickle can execute Python code. Cached training does not import or require the SDK.

## SIMSv2

Place the actual **CH-SIMS v2 supervised feature release** at `datasets/SIMSv2/unaligned.pkl`, or use `--data_dir` to point to its parent directory. The original CH-SIMS v1 dataset is not treated as interchangeable with v2.

The top-level keys are `train`, `valid`, and `test`. Each split provides:

| Field | Meaning |
| --- | --- |
| `vision` | Visual sequences `[N, Tv, Dv]`, or a list of unpadded `[Ti, Dv]` arrays. |
| `audio` | Audio sequences `[N, Ta, Da]`, or a list of unpadded `[Ti, Da]` arrays. |
| `vision_lengths` | Valid visual length for each sample; required for dense/padded arrays. |
| `audio_lengths` | Valid audio length for each sample; required for dense/padded arrays. |
| `raw_text` | Raw utterance strings for the selected tokenizer. |
| `regression_labels` | One fused regression label in `[-1, 1]` per utterance. |
| `id` | Source sample identifiers. |

Padded sequences are trimmed using their own recorded lengths before batching. The model packs each stream independently; it never reuses visual lengths for audio or pads two modalities to pretend they are aligned. Ragged inputs without length fields must already be unpadded. Non-finite values within valid regions are rejected rather than silently changing the feature protocol.

The [official SIMSv2 loader](https://github.com/thuiar/ch-sims-v2/blob/6eb09e5e9d17355cec436c59f4d3fdb41f3b5891/data/load_data.py) records separate visual/audio lengths. Some feature releases contain pretokenized `text_bert` without raw strings. This implementation follows the uploaded raw-text encoder path: recover matching raw transcripts and the exact tokenizer/model provenance for such a release rather than decoding arbitrary token IDs with a different tokenizer.

## Metrics

MOSI/MOSEI use raw regression scores for MAE/correlation, rounded clipped `[-3, 3]` scores for Acc7, and both binary mappings for accuracy and weighted F1:

- Non-negative/negative: `y >= 0`, including neutral examples.
- Positive/negative: remove targets equal to zero, then classify using `y > 0`.

SIMSv2 follows the [official evaluator](https://github.com/thuiar/ch-sims-v2/blob/6eb09e5e9d17355cec436c59f4d3fdb41f3b5891/utils/metricsTop.py), including clipping predictions and targets to `[-1, 1]` **before both regression and classification metrics**:

| Metric | Bins, from negative to positive |
| --- | --- |
| Acc2 / weighted F1 | `[-1, 0]`, `(0, 1]` |
| Acc3 | `[-1, -0.1]`, `(-0.1, 0.1]`, `(0.1, 1]` |
| Acc5 | `[-1, -0.7]`, `(-0.7, -0.1]`, `(-0.1, 0.1]`, `(0.1, 0.7]`, `(0.7, 1]` |

All accuracy/F1 outputs are fractions in `[0, 1]`. Undefined Pearson correlation or an empty nonzero-target subset produces JSON `null`; no artificial zero or NaN is reported. Empty evaluations and non-finite predictions raise errors. The official thresholds and clipping protocol are implemented, but the original paper's saved predictions are needed to establish that its table used exactly this evaluator.
