# Implementation review

Review date: **2026-09-16**. Scope: the ten Python files in the uploaded FUSENet ZIP and the accompanying ten-page manuscript. The pre-existing `fusenetpp/` variant is retained but is not covered by this review or used to substantiate the manuscript results.

## Release assessment

**Suitable for a private research archive; not yet a verified public reproduction release.** Core HMF, MRC, MDF, training, and dataset code is present. The supplied files do not establish that this exact implementation generated the manuscript tables.

The packaging change restores the complete uploaded source under `FUSENet/`, establishes one documented entry point, and adds documentation and the original PDF. It does not change the scientific implementation. All ten uploaded source files and the PDF are preserved byte for byte, with checksums in [source_manifest.json](source_manifest.json).

## Findings that affect experiments

| Priority | Location | Observation and consequence | Required follow-up |
| --- | --- | --- | --- |
| High | `solver.py::compute_info_gain` | The code minimizes `-MSE(shared) - MSE(private) + MSE(noise)` with positive weight `0.25`. Ordinary gradient descent therefore increases shared/private auxiliary prediction errors and decreases noise prediction error. This conflicts with concentrating sentiment information in shared/private factors and suppressing it in noise. The negative MSE terms are unbounded below through their auxiliary heads. | Recover the source/configuration behind the reported runs. Define an explicit information objective and its optimization roles; validate gradient direction and boundedness. Simply reversing every sign would instead leave a negative noise-head MSE unbounded. |
| High | `models.py::forward`, MRC branch | Sampling uses `z = mu + eps * exp(log_var)`, whereas the KL formula treats `log_var` as log variance. For that interpretation, standard deviation is `exp(0.5 * log_var)`. The sampled posterior and KL penalty are inconsistent. | Reconcile the parameterization, document any change, and rerun affected experiments. |
| High | `data_loader.py`, `models.py::_extract_features` | A single length vector derived from vision is used to pack both vision and audio. The `ch-sims` input is explicitly unaligned and no separate valid lengths are read. | Carry independent modality lengths through the batch and model, with padding-aware validation. |
| High | `solver.py::calc_metrics` | SIMSv2 uses labels in `[-1, 1]`, but `acc5` rounds values after clipping to `[-2, 2]`. Ground-truth labels then occupy only three integer values. `Acc3` is absent. The non-negative/negative F1 reported for MOSI/MOSEI is also absent from the returned metrics. | Implement the established dataset-specific discretization and both binary mappings, then recompute metrics from saved predictions. |
| Medium | `utils/convert.py::to_gpu` | `.cuda()` is unconditional, although model placement checks CUDA availability. CPU-only and MPS-only training fail. | Use one explicit device consistently across model and batches. |
| Medium | `solver.py::train`, `train.py` | The best state is kept only in memory. There is no saved checkpoint, predictions file, full run record, or evaluation CLI. The seed is fixed to `42`; five-seed results cannot be reconstructed from provided artifacts. | Add persistent run outputs and a seed-controlled evaluation workflow. |
| Medium | `solver.py::calc_metrics` | Unrestricted `squeeze()` makes a one-example evaluation scalar and `len(preds)` fails. Constant prediction/target vectors can yield undefined Pearson correlation; empty splits also fail. | Validate shapes, define degenerate-metric behavior, and handle singleton/empty cases explicitly. |
| Medium | `create_dataset.py` | Some invalid or mismatched segments are skipped without an exclusion manifest; empty post-filter sequences are not rejected explicitly. Missing caches start preprocessing and overwrite the cache set. | Validate samples, record exclusions and final split sizes, and make cache rebuilds explicit. |
| Medium | `create_dataset.py` | The SDK is imported globally. Raw timestamped words are required but only high-level and label recipes are requested. | Verify the pinned SDK recipe and all required files; make SDK loading conditional if cached-only operation is supported. |
| Low | `config.py`, `data_loader.py`, `models.py` | `warmup_steps` is unused. Non-transformer text inputs are dummy values and a usable vocabulary is not supplied. `ur_funny` and `iemocap` appear in the path dictionary without a complete loader/task configuration. | Limit supported flags to implemented paths or complete those paths before advertising them. |

Paths in the table are relative to `FUSENet/`. Findings are based on source inspection unless otherwise stated; they are not full-training observations.

## Manuscript and code reconciliation

1. **Text backbone:** Section 4.2 names RoBERTa under feature extraction but BERT-base under experimental setup. The source defaults to `roberta-large`. The actual model/tokenizer revision for each dataset needs confirmation, particularly for Chinese data.
2. **Information score versus error:** Equation (3) uses the same sign pattern as the code but describes its terms as information scores. The implementation substitutes MSE prediction errors. Information scores and prediction errors have opposite interpretations, so matching the printed signs is not enough to establish the stated mechanism.
3. **Reconstruction normalization:** Equation (7) sums over modalities. The implementation averages the reconstruction and KL terms over three modalities, and uses elementwise mean MSE. Loss coefficients must be tied to the implemented reductions.
4. **Dynamic fusion:** The code uses scalar MLP attention, three-way softmax **within each modality**, and a sigmoid gate on the noise branch. Figure 1 labels a sub-block as self-attention, but no Transformer self-attention layer is used in this fusion module. The final concatenation order in code is specific/shared/noise, while the equation lists shared/specific/noise.
5. **Data protocol:** The paper explicitly distinguishes word-aligned MOSI/MOSEI from unaligned SIMSv2. Its feature-extraction paragraph also makes a general word-alignment statement. The code's shared length vector does not implement the full unaligned case.
6. **Experimental evidence:** The manuscript states five-seed averages and ablation results. No per-seed outputs, checkpoints, ablation runner, or original environment lockfile were included. The supplied default training entry point alone does not substantiate those claims.
7. **Table interpretation:** The manuscript's prose says DTN does not report the relevant binary metrics, while Table 1 contains positive/negative DTN entries. Avoid repeating unsupported blanket superiority claims from that prose.
8. **Publication metadata:** The PDF contains a CVPR open-access template header and proceedings-like page numbers. These are preserved as supplied, but are not independent evidence of acceptance. Reference [32] still contains a literal TODO placeholder. Final venue/citation metadata and the bibliography need confirmation before public distribution.

## Validation performed for this packaging change

- Compared every existing source file retrieved from GitHub with its Git blob hash.
- Compared all ten canonical source files and the attached PDF with the uploaded bytes.
- Parsed and compiled repository Python source without importing the training stack.
- Checked relative Markdown file links and the source manifest.
- Inspected rendered manuscript pages and the extracted framework figure; the PDF itself was not edited.
- Screened tracked source/document additions for common credential patterns and excluded macOS archive metadata, caches, data, and weights.

**Not performed:** dependency installation validation, model download, CPU/GPU forward/backward smoke testing, dataset preprocessing, full training, benchmark evaluation, or confirmation of paper results. The review environment did not have PyTorch, Transformers, or the datasets installed. Syntax/integrity checks do not establish runtime correctness.

The current Python interpreter also emits `SyntaxWarning` for two non-raw regex strings containing `\[` in each copy of `create_dataset.py`. They still parse and compile; the source bytes were retained and the warnings were not treated as proof of runtime failure.

## Next experimental step

First identify the exact source, environment, feature releases, and checkpoints used to obtain the manuscript numbers. Then resolve each mechanism or protocol discrepancy in a separately recorded change, run focused synthetic checks, and rerun the original protocol with retained predictions and multiple seeds. Keep the manuscript numbers labeled as reported until those runs provide reproducible evidence.
