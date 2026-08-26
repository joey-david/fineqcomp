# Relative information candidates

**Status: complete.** Two-cell smoke job 1295402 and the 78-cell H100 panel
1295432 completed. Twelve cells with stale prepared-test metadata were restaged
without weakening validation and completed in retry job 1295538. Every cell has
ten finite measures and a bracketed likelihood-based R*(0.90).

## Hypothesis

The behavior-preserving adapter rate is set by information in the learned
dataset relative to the frozen model. The past gzip, base-loss, divergence and
weight-error results reject those estimators; they do not reject the existence
of a better relative measure.

One frozen-model pass now produces representation, output-residual and
target-gradient sketches. Ten candidates are reduced from that shared pass:

| key | proposed meaning |
|---|---|
| `base_codelength_bits_per_token` | how many bits the receiver needs to encode the taught tokens before learning |
| `surprisal_variance_bits2` | whether the unknown content is spread evenly or concentrated in hard examples |
| `margin_deficit_bits` | how far the target lies behind the receiver's preferred output |
| `hidden_effective_rank` | how many independent directions the examples occupy in the receiver's representation |
| `surprise_weighted_hidden_logdet` | representation volume after costly examples receive more weight |
| `residual_effective_rank` | how many independent output corrections the labels request |
| `fisher_trace` | total size of the requested parameter correction |
| `fisher_effective_rank` | number of independent correction-gradient directions |
| `fisher_logdet` | joint size and volume of those correction directions |
| `unit_gain_ntk_cost` | linearized parameter norm needed to improve every example by one unit |

The last five use a random sketch of the output-head gradient. For one scored
token that gradient factors into the receiver's hidden state and its output
probability residual, so it can be measured without a backward pass through
the 7B model. The fixed sketch seed and dimensions make cells comparable.

## Panel

| source | cells | role |
|---|---:|---|
| five corpora × two models × three seeds | 30 | test cross-task and cross-receiver ordering |
| SQL and XBRL high-gain arms | 12 | test whether a measure handles large learned gains |
| SQL/XBRL fixed-row diversity panels | 36 | test paired within-task changes |

Each cell measures 256 prepared training rows, up to 32 response tokens per
row, a 64-dimensional hidden sketch and a 32-dimensional residual sketch. The
array allows 36 concurrent one-H100 tasks and caps every task at eight hours.

## Result

The correction-gradient spectrum is the first model-relative proxy in this
campaign that passes the screen. For each example, the output-head gradient
factors into its frozen-model hidden state and target probability residual.
`fisher_logdet` sums `log(1 + eigenvalue)` over a fixed random sketch of those
gradients. It measures both the number and size of independent corrections
that the dataset asks this receiver to make.

The reduction treats three seeds as repeats, averages them into model-dataset
arms, and merges four exact data reuses between the high-gain and diversity
panels. This leaves 22 distinct arms in seven task families.

| candidate | mean within-model Spearman | adjusted p | task-held-out RMSE | ratio to model-only | pooled arm Spearman |
|---|---:|---:|---:|---:|---:|
| Fisher log-volume | **0.764** | **0.00034** | 0.187 | 0.707 | 0.584 |
| Fisher effective rank | 0.609 | 0.0112 | **0.181** | **0.683** | **0.814** |
| surprise-weighted hidden log-volume | 0.614 | 0.0103 | 0.204 | 0.771 | 0.441 |
| base-model code length | 0.200 | 0.524 | 0.261 | 0.990 | 0.269 |

The p-values come from 50,000 within-model shuffles and use the maximum result
over all ten candidates, so they account for choosing the best candidate on
this panel. Task-held-out fits leave one whole family out and include only a
model intercept plus one logged measure. The model-only RMSE is 0.264. Base
code length does not improve it; Fisher log-volume lowers it by 29%, and Fisher
effective rank lowers it by 32%.

The controlled diversity check gives the same qualitative split as R*. Fisher
log-volume stays nearly flat as SQL groups rise and increases on XBRL for both
models. Its log slopes per doubling are -0.0008 and +0.0031 on SQL, versus
+0.0364 and +0.0353 on XBRL. Effective rank gives the best pooled ordering and
retains a positive residual relation after model and task intercepts are
removed (`r=0.566`); log-volume gives the strongest within-model ordering and
the cleanest diversity slopes.

## Strongest current claim

The evidence no longer supports saying that the bit budget is unrelated to the
learned dataset. In this two-model discovery panel, R*(0.90) tracks the spectral
volume of the corrections that the dataset requests from the frozen model far
better than it tracks gzip-like size or base-model code length. The relevant
quantity is therefore consistent with information in the learned dataset
relative to the receiver, not information intrinsic to the dataset alone.

This remains a screened claim, not a finished law. The measure is a fixed
output-head gradient sketch, not the full LoRA Fisher; the same 22 arms both
selected and assessed the candidates; and only two 7B receivers and seven task
families are present. A prospective model/task panel must now test the fixed
`fisher_logdet` definition without choosing it again.

## Preregistered gates and outcome

1. **Passed in internal task-held-out fits:** Fisher log-volume and effective
   rank beat base code length and the model-only baseline.
2. **Passed:** both retain a positive relation after model and task intercepts.
3. **Passed most clearly by log-volume:** positive XBRL slopes on both models
   and near-zero SQL slopes.
4. **Passed:** median seed coefficient of variation is 0.018 for log-volume and
   0.045 for effective rank.
5. **Passed as a screen:** log-volume correlates negatively with learned gain
   (`rho=-0.275`) and weakly with base code length, so it is not either measure
   under a new name.

The first four gates use the discovery panel. A locked prospective replication
is still needed before the measure can support a scaling law.

## Artifacts

| file | contents |
|---|---|
| `cells.csv` | all 78 joined measure and R* records |
| `arms.csv` | 22 seed-averaged, deduplicated model-dataset arms |
| `candidate_ranking.csv` | all ten screens and adjusted p-values |
| `task_heldout_folds.csv` | leave-one-task-family-out errors |
| `diversity_slopes.csv` | SQL/XBRL slopes for R* and all candidates |
| `summary.json` | headline counts and strongest candidate |
