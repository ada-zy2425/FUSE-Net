# Paper-to-code correspondence

This revision contains one FUSE-Net implementation derived from the uploaded code. The unrelated `fusenetpp` variant, duplicate root-level Python entry points, unused legacy utilities, and archival scaffolding were removed from the current tree. Prior versions remain available in Git history. The only retained dotfile is `.gitignore`, which excludes datasets, weights, secrets, and generated files.

## Formula mapping

| Paper component | Executable counterpart | Status |
| --- | --- | --- |
| Text + visual + acoustic encoders | `models.FUSENet`: masked token mean, one-layer BiGRUs, ReLU projections, LayerNorm | Architecture implemented; exact text backbone still needs confirmation. |
| HMF shared/specific/noise factors | `shared`, `private_t/v/a`, `noise_t/v/a` | Retains the uploaded decomposition. |
| Eq. (2), contrastive separation | `losses.contrastive_loss` | Equivalent log ratio computed with stable softplus; mean over six ordered cross-modal pairs. |
| Eq. (3), information constraint | `losses.information_loss` | Corrected using negative auxiliary MSE scores, bounded predictions, and noise feature gradient reversal; see [details](information_loss_fix.md). |
| Eq. (4), dual consistency | `losses.dual_loss` | Squared feature-vector L2 norms, averaged over samples and modalities. |
| Eq. (6), MRC input | Concatenation of shared, specific, and noise factors | Implemented in the stated order. |
| Eq. (7), variational reconstruction | `sample_latent`, `reconstruction_loss` | `std = exp(log_var / 2)`; reconstruction squared L2 and Gaussian KL, summed over modalities and averaged over samples. |
| Eqs. (8)–(10), dynamic logits | `alpha_mlp`, learnable beta scalars, `attn_mlp` | Sample modulation multiplied by factor-type coefficient and scalar branch attention. |
| Eq. (11), normalization | `softmax(logits, dim=1)` per modality | Three factor weights sum to one separately for each modality/sample. |
| Eqs. (12)–(14), aggregation | Weighted sums over modalities; sigmoid elementwise gate on noise | Implemented; the gate scales coordinates without changing their signs. |
| Eqs. (15)–(16), prediction | Concatenate shared/specific/noise, then two-layer MLP | Order corrected to match the equation. |
| Eq. (17), optimization | `losses.objective`, `solver.Solver` | Task MSE plus the mapped weighted objectives, with distinct feature/predictor gradients for the information term. |

## Information-loss correction

**1. Information score.** The manuscript calls each `ell` in equation (3) an information score but does not define its conversion from the auxiliary predictions. The uploaded code substitutes MSE, yielding:

```text
L_info = mean_modalities(-MSE(shared) - MSE(specific) + MSE(noise))
```

With positive coefficient and joint gradient descent, the shared/specific prediction errors are rewarded for increasing and the noise error is rewarded for decreasing. Negative MSE through freely trained auxiliary heads is unbounded below. This contradicts the stated goal of making shared/specific factors informative and noise factors uninformative.

That proxy has been replaced. A branch information score is now defined as **negative** MSE, with auxiliary predictions bounded to the dataset's annotation range. Shared/specific features minimize their prediction errors, while noise features receive a reversed error gradient. All auxiliary predictors, including noise, minimize their own prediction errors. The training surrogate and encoder objective are recorded separately. [The implementation note](information_loss_fix.md) gives the equations, gradient allocation, bounds, and verification.

This is an explicit correction consistent with the manuscript's stated factorization goal. The bounded auxiliary prediction and adversarial optimization details are newly specified here; the supplied manuscript did not define them, and they are not claimed to be the settings used for its historical tables. The PDF is unchanged.

## Historical experiment details

**2. Text backbone.** Section 4.2 names RoBERTa in one paragraph and BERT-base in another. The uploaded source selects `roberta-large`. The training command now requires `--bert_dir`, and records the selected model, revision, configuration, and tokenizer. No Chinese backbone is invented for SIMSv2. Confirm the exact encoder/tokenizer per dataset from the original runs.

**3. Experimental recipes.** The manuscript reports five-seed results and component/modality ablations, while the upload supplies one fixed seed, no ablation definitions, no checkpoints, and no prediction logs. Multi-seed execution and saved evaluation artifacts are now implemented, but example seeds are not labeled as historical seeds. Ablation variants are not invented under the labels in Table 3: the original intervention definitions and run configurations are needed to reproduce them.

The supplied framework image also labels a block as self-attention, while the accompanying code uses scalar MLP attention; the equation specifies an abstract `Attn` function. MRC in the executable equations reconstructs each modality separately. The original PDF/figure are retained as manuscript material, not silently redrawn to conceal these wording issues.

## Changes that affect numerical results

Compared with the uploaded source, this implementation corrects the information-loss gradient allocation, bounds its auxiliary predictions, corrects the MRC standard deviation, replaces featurewise mean losses with the squared L2/summed reductions shown in equations (4) and (7), uses the equation (15) concatenation order, preserves independent audio/visual lengths, and implements the dataset-specific evaluation protocols. Those changes require new benchmark runs. Original loss coefficients are retained as configurable defaults; they are not claimed to be tuned for the corrected objectives.

The non-transformer text path, unsupported dataset names, unused warmup argument, cosine scheduler alternative, LSTM alternative, and inactive Siamese regularizer were removed from the active implementation. The paper states that the Siamese coefficient was zero in every experiment. The active model uses the paper's BiGRU, AdamW, and ReduceLROnPlateau path.

Training now writes a best validation checkpoint, tokenizer, full configuration, data hashes, package versions, learning curves, per-sample test predictions, and metric JSON. Independent evaluation rebuilds the encoder from its saved configuration and loads all trained weights strictly. Format 3 identifies the corrected information objective; format 2 can still be evaluated with its legacy training provenance retained.

## Verification of this revision

Validation used Python 3.12, PyTorch 2.5.1 CPU, Transformers 4.46.3, NumPy 2.3.5, and scikit-learn 1.8.0.

- **25 offline synthetic tests**, including real tiny BERT and RoBERTa modules, forward/backward gradients, independent sequence masks, empirical latent variance, per-modality softmax, fusion order, metric boundaries, singleton/error cases, and the corrected information-loss behavior. The original wrong-direction diagnostic has been replaced by checks of the intended directions.
- Three controlled training seeds test information removal with a fresh predictor fitted on separate probe-training samples and evaluated on held-out samples; see [recorded results](information_loss_fix.md).
- A real CLI run trained two synthetic seeds, saved checkpoints/tokenizers, and independently reevaluated a checkpoint. The reevaluated metrics exactly matched the saved test metrics.
- SIMSv2 evaluation matched the official source on **12 metric/dtype comparisons** covering all six reported metrics, out-of-range predictions, and exact boundaries in float32/float64.
- The manuscript PDF is unchanged. Source imports, CLI help, Python compilation, and relative documentation links were checked.

No full MOSI, MOSEI, or SIMSv2 benchmark training, GPU execution, raw SDK download/alignment, or Table 3 ablation reproduction was performed. The original datasets and checkpoints were not provided. This revision improves executable correspondence but does **not** claim complete paper equivalence or revalidated paper results.
