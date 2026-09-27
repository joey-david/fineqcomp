# Correction-spectrum rate: locked analysis plan

Recorded before reading jobs 1295787 and 1295788.

## Question

Can a scale-free description of the corrections requested by a supervised
dataset predict the smallest adapter rate that preserves 90% of learned
behavior?

For one example, the frozen-model output-head gradient sketch is a row of
`G`. Four spectra are saved: raw, mean-centered, equal-row-norm (angular), and
centered angular. For eigenvalues `lambda` of `G.T @ G / n`, the main new
family uses the Gaussian rate-distortion function at the same 90% target as
R*:

```
sum(min(lambda_j, theta)) = 0.10 * sum(lambda_j)
R_corr(.90) = 0.5 * sum_{lambda_j > theta} log2(lambda_j / theta)
```

Because `theta` rescales with the spectrum, this quantity has no fitted norm or
projection scale. Raw, centered, angular, and centered-angular variants are
screened. Energy-rank and trace-normalized log-volume variants are controls.

## Discovery selection

The discovery set is fixed at the original 78 cells: five corpora, SQL/XBRL
high-gain arms, and SQL/XBRL diversity arms on Mistral-7B and Qwen2.5-7B. Seeds
are repeats. Exact data reuses are merged, leaving 22 arms in seven task
families.

All original and spectral candidates enter one 50,000-shuffle maximum-statistic
test. A candidate is eligible only if it:

1. has adjusted `p < 0.05` for mean within-model Spearman correlation;
2. beats both model-only and base-code-length task-held-out RMSE;
3. has a positive residual relation after model and task intercepts;
4. rises on XBRL for both receivers and stays near flat on SQL;
5. is finite on every cell.

Among eligible spectral candidates, choose the lowest task-family-held-out RMSE.
If two differ by less than 0.02 bits/value, choose the one with higher
within-model Spearman correlation. If both remain tied, choose the simpler
scale-free 90%-rate-distortion form. Lock that one definition before opening
the prospective result root.

## Prospective panel

Job 1295788 measures 126 untouched cells drawn from existing finished
campaigns. Only cells with bracketed R* and the fixed rank-16 adapter contract
enter the test. Seeds are averaged into 24 prospective arms:

| block | arms | intervention |
|---|---:|---|
| CoT span, Mistral | 3 | full response, reasoning, answer |
| CoT span, Qwen | 3 | full response, reasoning, answer |
| response form | 10 | five deterministic forms x broad/narrow content |
| fixed-row content | 3 | 400, 1,200, 3,600 source problems |
| fixed-compute rows | 5 | 2k, 4k, 8k, 16k, 32k distinct rows |

The primary statistic ranks the chosen measure and R* within each block, pools
the centered ranks, and tests them with 100,000 blockwise permutations. Success
requires `rho >= 0.60` and `p < 0.01`. No alternative candidate may replace a
failure on this panel.

Secondary gates:

- a discovery-fitted model-intercept plus logged measure must lower prospective
  RMSE by at least 20% against model-only and base-code-length fits;
- the sign of every three-or-more-arm block slope must match R*;
- the result must remain after removing any one block.

## Measurement stability

After the prospective test, rerun twelve arms at 64, 128, and 256 sampled rows
and at three fixed sketch seeds. The locked measure must have median within-arm
coefficient of variation below 5% and arm-rank Spearman above 0.95 against the
256-row, seed-1729 reference. The earlier draft also asked this twelve-arm
subset to reproduce the full 24-arm pass/fail test; that statistic is not
defined on the subset and is removed before the stability outputs are read.

## Final external test

Only a measure that clears the prospective and stability gates advances to a
third frozen base model. That model uses the same five-corpus, rank-16, four-
epoch contract as the discovery panel. The measure is computed before its R*
results are analyzed. Paper-ready status requires the third-model arm ordering
to have Spearman `rho >= 0.70` and the prefit absolute prediction RMSE to beat
base code length.

Until all gates pass, `fisher_logdet` remains a discovery result and no new law
is claimed.

## Amendment after the first locked test

The first pass locked `angular_correction_logdet` and failed the primary
prospective rank gate (`rho=-0.110`, blockwise `p=0.640`). It nevertheless cut
absolute RMSE from 0.268 model-only and 0.324 base-code-length to 0.150. The
failure exposed a data-contract error: the measurement command took
`train[:256]`. The fixed-compute and fixed-row-content arms share that prefix,
so all eight arms received exactly the same spectrum despite having different
full training sets. Broad/narrow response pairs had the same problem.

That result is retained as a failed diagnostic and will not count as
prospective evidence. The corrected contract, fixed before jobs 1296347 and
1296348 finish, is:

1. sample 256 rows uniformly with fixed seed 271828 from the full staged train
   split, never from its prefix;
2. record total and exact-distinct prompt-response row counts;
3. add `dataset_correction_log_volume = logdet(I + N_unique F_angular)`, where
   every sampled correction gradient has unit norm;
4. rerun every discovery and diagnostic cell in new output roots;
5. use the same discovery gates to lock one corrected measure;
6. treat all existing Mistral/Qwen campaigns as development data from this
   point onward; only an untouched third-model panel can supply the final
   prospective result.

The third-model gate remains unchanged and gains one safeguard: its config,
measure key, sampling rule, and analysis command must be written before any
third-model R* result is opened.

## Fixed-rank correction

The full-dataset Fisher volume passed the five-block development test but its
first 2,048-dimensional sketch failed the row-count stability gate. With fewer
sampled rows than sketch coordinates, each added row created a new covariance
mode, so the estimated volume grew with sample rank. A fixed 32-dimensional
correction sketch (8 hidden by 4 output-residual dimensions) removes that
fault: even the 64-row setting has two observations per coordinate. This choice
was fixed before the Llama R* results were read. Its nine-setting stability
grid passed with median arm CV 0.013 and minimum arm-rank Spearman 0.951.

The low-dimensional form then failed two development gates: it reversed the
Mistral full/answer CoT ordering and did not cut RMSE by 20% against model-only.
That failure showed that the tail modes carry signal. The final candidate uses
the full sketch but treats row count as the axis of an information-growth
curve rather than a nuisance. Its scalar is the trapezoidal mean over equal
log2 intervals, `[I(64) + 2 I(128) + I(256)] / 4`. Across the 12-arm, three-
seed pilot, this area has median CV 0.0024 and identical ranks. Its definition
was fixed before the full area panels ran or any Llama R* result was read.

The area form retained the primary rank and RMSE result but again reversed the
three-arm Mistral CoT block, so it was rejected. The surviving estimator fixes
the calibration budget at 256 uniform rows and calls the resulting quantity a
Nyström correction-information volume. On 12 arms, three row-sampling seeds
give median CV 0.0018 and minimum rank Spearman 0.979; three sketch seeds give
median CV 0.0020 and identical ranks. The Llama external test therefore uses
the 256-row `dataset_fisher_log_volume` already computed before any Llama R*
result was read.
