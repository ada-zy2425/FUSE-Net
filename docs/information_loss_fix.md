# Information-loss correction

The uploaded implementation minimized `-MSE(shared) - MSE(private) + MSE(noise)` with all predictors and features updated together. That rewards increasing shared/private errors and decreasing noise errors, opposite to the stated HMF goal. This revision fixes both the score sign and the predictor/feature gradient allocation.

## Exact objective

For auxiliary predictor $f_b$ and factor $H_b$, define

$$
\widehat y_b = R\tanh(f_b(H_b)/R),\qquad
E_b = \operatorname{MSE}(\widehat y_b,y),\qquad
\ell_b=-E_b.
$$

Here $R=3$ for MOSI/MOSEI and $R=1$ for SIMSv2. The division by $R$ keeps the auxiliary mapping approximately equal to its original linear output near zero. This bounding applies only to the auxiliary predictors. The main prediction and its MSE are unchanged.

Substituting the negative-error scores into paper equation (3) gives the feature-encoder objective, averaged over modalities:

$$
L_{\mathrm{info,encoder}}=E_s+E_h-E_n.
$$

The auxiliary predictors all minimize their own error. In particular, the noise predictor minimizes $E_n$ while the noise encoder maximizes $E_n$: a predictor must try to recover the label for its error to provide a useful adversarial signal.

| Parameters | Information-term update |
| --- | --- |
| Shared/specific features | Minimize their auxiliary MSE |
| Shared/specific predictors | Minimize their auxiliary MSE |
| Noise features | Maximize the noise predictor's MSE |
| Noise predictor | Minimize its MSE |

Implementation uses a gradient reversal layer on the noise features **only when they are passed to the noise auxiliary predictor**. The forward training surrogate is the positive sum $E_s+E_h+E_n$. Its custom backward gives the encoder objective above and ordinary error-minimizing gradients for every predictor. All parameters can therefore use the existing AdamW step. The outer `info_gain_weight` scales the term once. Gradients from the main task, MRC, and other consumers of noise features are unaffected by reversal.

The gradient-routing pattern follows [Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 2016](https://jmlr.org/papers/v17/15-239.html). Its application to these sentiment auxiliary regressors and the bounded prediction are the implementation choices in this correction; the cited work does not establish FUSE-Net's empirical performance.

## Stability and interpretation

For labels in $[-R,R]$, each auxiliary MSE lies in $[0,4R^2]$. Thus the encoder's information term lies in $[-4R^2,8R^2]$ after modality averaging: it cannot tend to negative infinity by increasing an auxiliary bias or a feature magnitude. Non-finite logits are rejected before `tanh`; saturation cannot silently hide an overflow.

Bounding does not guarantee convergence of the adversarial game. MSE predictability is a proxy for label information within the auxiliary predictor family, not a mutual-information estimator or a proof of statistical independence. Other losses also influence the shared encoders. Real-data validation should inspect main-task performance and fresh probes as well as the in-training adversary.

This is a documented correction to match the manuscript's stated objective. The uploaded paper did not specify this score conversion, output bound, or adversarial training rule. These details are not presented as a recovered historical experiment recipe, and its PDF and reported tables are unchanged.

## Logging and checkpoints

- `information`: the positive training surrogate used in backpropagation.
- `information_encoder`: the detached value $E_s+E_h-E_n$ for interpreting the encoder objective.
- `information_shared_mse`, `information_private_mse`, `information_noise_mse`: individual branch errors, averaged over modalities and samples.
- `total`: the weighted training surrogate, including the main task and remaining objectives. It is not a scalar objective jointly minimized by both adversarial players.

`run.json` records the bound, enablement, score, and gradient rule. New checkpoints use format 3 and objective ID `bounded_mse_grl_v1`. Format 2 checkpoints can still be evaluated, but are marked `legacy_signed_mse_v0`. The inference architecture and state-dict keys are unchanged by this fix. Setting `--info_gain_weight 0` skips the auxiliary loss and leaves the auxiliary heads without gradients, preventing AdamW weight decay from updating disabled heads.

## Verification

Run the full offline suite:

```bash
python -m unittest discover -s tests -v
```

The focused tests in `tests/test_information_loss.py` compare feature and predictor gradients against ordinary bounded MSE on all three datasets and singleton/multiple-example batches. They also check separate feature/head steps, isolation of other gradients, finite bounds under extreme predictions, invalid labels, and diagnostic values. The integration suite checks two-seed training, artifact metadata, current and legacy checkpoint loading, and identical metrics after independent checkpoint evaluation.

The controlled training check uses factors of the form $a_b y+\epsilon$ with trainable $a_b$, $y\sim U[-1,1]$, and independent Gaussian $\epsilon$. It runs 400 minibatches of 256 samples. The predictor learning rate is 0.15 and the factor-coefficient learning rate is 0.1 under SGD; this experiment calls the same `information_loss` as the full trainer. A **fresh least-squares probe** is fitted on 4,096 separate samples and evaluated on another 4,096 samples. Probe-test outcomes do not enter training or checkpoint selection.

| Seed | Shared/private MSE before → after | Fresh noise probe $R^2$ before → after |
| --- | --- | --- |
| 7 | 0.39677 → 0.06828 | 0.25104 → −0.00143 |
| 11 | 0.39923 → 0.06982 | 0.24812 → −0.00059 |
| 42 | 0.40669 → 0.06820 | 0.24757 → −0.00013 |

Shared and private errors coincide in this deliberately symmetric experiment. Near-zero held-out $R^2$ indicates that the independent probe cannot improve on a constant baseline in this controlled setup. These results test the mechanism beyond the training head becoming inaccurate. They are **not** MOSI/MOSEI/SIMSv2 benchmark results and do not establish information removal in the complete model.
