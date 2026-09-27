# Predicting R* from the trained adapter's spectrum

## Attenuation predictor experiment (25 Sep 2026, before the new probes)

**Current main result:** after the two additional seeds, the unchanged rule
scores **0.0682 RMSE over 42 checkpoints** on the five test receivers, versus
0.0931 for spectrum plus log token count and 0.1749 for text redundancy.
R-squared is 0.945; averaging each model/corpus pair over three seeds gives
0.0571 RMSE across 14 pairs. The original six seed-11 development checkpoints
remain the only data used by the secondary affine calibration. The primary
rule still fits no coefficients. A bootstrap over five receiver clusters gives
an RMSE-improvement interval of [0.0016, 0.0695] against spectrum plus tokens;
five receiver clusters still give limited precision. The full per-seed errors
are 0.0492, 0.0743 and 0.0776, so the first seed was favorable. Qwen2.5-Math is
the hardest receiver (0.1052 RMSE across nine checkpoints).
`attenuation_predictor/replication/` records all 60 profiles, including the
18 checkpoints on the two development receivers. The history below preserves
the order in which the candidate and checks were chosen.

The new candidate measures the loss of the *uncompressed* update at scales
0, 1/8, 1/4, 1/2 and 1 on 64 training rows sampled with seed 271828. It combines
this response with the actual one-bit quantizer's projection onto the full
update. A scalar attenuation model predicts that the fraction of the one-bit
rank pairs needed is the scale retaining 90% of the raw gain divided by that
projection. This is a hypothesis, not a validated model of compression noise.
The profile also records the orthogonal residual, which can refute that model.

The first panel is Mistral-7B and Qwen2.5-7B on math, code and XBRL, seed 11.
The remaining five receivers in the transposed panel are reserved for testing
the chosen form. Their historical rates are already known: this is retrospective
validation with new features, not a prospective rate experiment. No coded
model is evaluated to form the features. The implementation reuses
`scripts/analysis/adapter_spectrum_profile.py`, `causal_nll`, the row quantizer and the factor Gram products.
Start with four rows on one adapter; expand only after the smoke completes.

Score against each seed's recorded interpolated, best-decoded-gain R*(0.90),
and report raw-gain and first-feasible-rung targets separately. Compare against
the constant, text-redundancy and spectrum baselines on exactly the same cells.
Any fitted calibration uses development receivers only. Do not use a corpus
fixed effect in the primary predictor: it requires rates from other receivers
on the target corpus. Report both held-out receiver and corpus-family errors.
The initial useful-result gate is RMSE below 0.10 bits/value and at least 25%
below the strongest eligible baseline; report all receiver errors, even if
the pooled gate passes.

Development amendment, before reading the six new profiles: repeat the same
scales on 64 calibration rows. An offline control showed that a fixed half-bit
probe transfers much worse when measured on training data (RMSE 0.153 versus
0.081 on calibration data). The attenuation probe must therefore distinguish
compression tolerance on the training distribution from held-out tolerance.
Calibration rows overlap the historical target's evaluation sample; this is
not an independent-example validation. Both variants remain development work.

The six calibration profiles give direct RMSE 0.0741 bits/value (training
profiles: 0.5958). The frozen rule is
`R_hat = alpha_90 / binary_projection`, where `alpha_90` uses the same linear
crossing of a running maximum as the target, with the *raw* gain as reference.
No coefficient is fitted in the primary rule. The secondary affine calibration
fits only those six development profiles. `attenuation_predictor/lock.json`
records the split and frozen form before the five other receivers are profiled.
Values above one bit extrapolate the pair-dropping approximation; they do not
establish that it describes the higher-bit quantizer.

The frozen test completed on 14 adapters from five other receivers (one seed,
three corpora). Direct RMSE is **0.0492**, against 0.0758 for the audit-fitted
spectrum-plus-log-token-count baseline and 0.1933 for text redundancy. Each
receiver's RMSE lies between 0.0252 and 0.0666. The affine calibration is worse
(0.0706). Using a fixed projection, averaged from development weights alone,
raises error to 0.0766; the adapter's actual projection matters. This is a small
receiver sample: a receiver-cluster bootstrap interval for the RMSE improvement
over spectrum-plus-tokens includes zero (-0.0064 to 0.0637).

The raw-gain target gives RMSE 0.0539. These are interpolated-rate errors; the
rule does not guarantee that its chosen file retains 90% of the gain. Rounding
each test prediction up to the next measured rung succeeds on 13/14 adapters,
with minimum retention 88.5%. Its mean file size is 1.079 times the measured
first-feasible file. The six development adapters have only 3/6 such successes.
`attenuation_predictor/` retains profiles, targets, per-cell predictions and
the complete score tables. Reproduce with:

```
uv run python scripts/analysis/adapter_spectrum_score.py \
  --attenuation results/rate/adapter_spectrum/attenuation_predictor/profiles.csv \
  results/rate/adapter_spectrum/attenuation_predictor/targets.csv \
  results/rate/adapter_spectrum/attenuation_predictor
```

Two further checks use the same rule: 64 archived audit adapters across more
corpora, and a fresh dense target curve on calibration examples excluded from
the 64-row predictor sample. The latter calls the existing quantizer, decoder,
loss evaluator and R* estimator. It excludes duplicate prompt/response content,
as well as the sampled rows themselves. The validation sweep is not a predictor
input. Neither check changes the frozen formula.

The 64-adapter audit completed. The direct rule has RMSE 0.1202 if all recorded
rates are treated as exact. Six records already meet the target at their lowest
stored rate, so those values are upper bounds. On the 58 bracketed targets,
spanning 34 corpora, direct RMSE is **0.1045** (R-squared 0.840), against 0.1766
for spectrum plus log token count and 0.2115 for text redundancy under
held-out-receiver fits. The raw-gain target gives 0.0936. The frozen affine
calibration gives 0.0924 against the best-decoded target. The direct rule thus
misses the original 0.10 gate narrowly on this broader panel; it still lowers
error by 41% against the stronger baseline. `attenuation_predictor/audit/`
contains every profile, target, prediction and score, including the censored
records. `scripts/analysis/adapter_spectrum_score.py --attenuation-audit PROFILES TARGETS OUT` reproduces it.

### Why this predicts a rate

Let Delta be the full effective update, and Delta_1 the update reconstructed
from one-bit **LoRA factors in their stored basis**. The weight projection is
`gamma = <Delta_1, Delta> / ||Delta||^2`, summed over all adapter sites. It is
computed from small factor Gram matrices. It is not the norm ratio, and it
does not assume Gaussian weights.

Below one bit, the codec keeps paired rank directions in A and B. Averaging
over uniform masks with fraction p retained gives the exact identity
`E[Delta_p] = p * Delta_1`. Projecting onto Delta therefore gives expected
scale `p * gamma`. The empirical step is to approximate the gain of the coded
update by the gain of the full-precision update at that scale:

```
G(alpha) = L(base) - L(base + alpha * Delta)
G(coded at p) approximately equals G(p * gamma)
alpha_90 = first scale retaining 90% of G(1)
R_hat = alpha_90 / gamma
```

The last line uses nominal sub-bit rate p as an approximation to file bits per
value. Headers, scales, interpolation, mask fluctuations and the part of the
decoded update orthogonal to Delta can all cause error. Orthogonality in
weight space alone does not justify neglecting its effect on loss. Predictions
above one bit also extend past the masking identity's domain. These are
measurable approximation errors, not an information-theoretic optimum.

The gain-plus-distortion decomposition has a standard signal-processing
precedent: [Demir and Bjornson, The Bussgang Decomposition of Non-Linear
Systems](https://arxiv.org/abs/2005.01597). Here gamma is a deterministic
projection measured on each adapter; we do not invoke a Gaussian-input theorem
or claim the decomposition itself is new. The result to establish is that
this projection and a short uncompressed loss curve predict behavioral rate
on other adapters and receivers.

As a further retrospective check, the same rule predicts 50%, 75% and 95%
retention by changing only the crossing threshold. On the original 14 test
adapters, errors against the best-decoded target are 0.0378, 0.0462 and 0.1039,
respectively (90%: 0.0492). At 95%, eight predictions extend above one bit,
where the paired-dropping identity no longer applies. This check uses no new
fit. `scripts/analysis/adapter_spectrum_score.py --attenuation-checks PREDICTIONS RUNS_ROOT OUT` also reproduces
the full sub-bit curve predictions and the measured-file selection check.

## Scope of the earlier spectrum results

All new primary scores use the estimator actually implemented in
`fineqcomp.rstar.from_run`: a linearly interpolated crossing against 90% of
the larger of raw gain and best decoded gain. This differs from the current
manuscript's wording of the smallest measured file retaining 90% of raw gain.
The manuscript needs to distinguish those definitions when incorporating the
result. The tables also report raw-gain targets and first-feasible file rates;
interpolated-rate accuracy is not a guarantee about an actual file.

The historical tables below include a fitted corpus term. They ask whether
adapter features help when other receivers have already been compressed on the
same corpus. Their numbers do not measure transfer to an unseen corpus.
The new attenuation and fixed-probe comparisons omit that term.

`scripts/analysis/adapter_spectrum_score.py --probe RUNS_ROOT OUTPUT_DIR` fits a single half-bit probe on the
67-arm audit, then transfers the coefficients unchanged to the 20-arm receiver
panel. Its feature divides probe gain by raw-adapter gain, never by the best
decoded gain. Removing the probe from both the target curve and its ceiling
checks direct coupling, but not shared evaluation-example noise. This method
uses one coded-model evaluation and must be kept separate from an adapter-only
predictor. The audit-to-panel error is 0.0809 bits/value, or 0.1085 on the nine
arms whose receivers do not appear in the audit. The training-data probe is
weaker (0.1529 and 0.1917, respectively).

Every earlier candidate for R* measured the corpus, or the corpus against the
frozen model. None read the trained adapter itself, which is the one object
that already carries the corpus-by-receiver interaction the rate-law audit
showed a corpus measure cannot. `fineqcomp.studies.generalisation.spectral_profile`
takes each LoRA site's exact singular values (from the r x r core of the two
thin QR factors) and pools, energy-weighted: the top-direction and top-4 energy
shares, effective and stable rank, LoRA-pair overlap (the energy the stored
pairs carry one by one over the true energy -- what random pair dropping below
one bit acts on), row kurtosis, total energy, and cross-site concentration.

`scripts/analysis/adapter_spectrum_extract.py` streams the archived run tars on prepost; `scripts/analysis/adapter_spectrum_score.py` scores each
feature with the transposed receiver panel's own protocol (fit R* ~ corpus +
feature on six receivers, predict the seventh) and reproduces the recorded
baselines exactly (`corpus_only` 0.1919, `dataset_fisher_log_volume` 0.1878 /
0.386) before scoring anything new.

## Development result (transposed panel: 20 arms, 7 receivers, 3 corpora)

| feature | held-out-receiver RMSE | receiver-pair sign accuracy |
|---|---:|---:|
| spectrum_log_energy | 0.154 | 0.561 |
| spectrum_pair_overlap | 0.159 | 0.684 |
| spectrum_top1 | 0.163 | 0.684 |
| spectrum_effective_rank | 0.168 | 0.667 |
| dataset_fisher_log_volume (best recorded) | 0.188 | 0.386 |
| corpus_only | 0.192 | 0.000 |

The concentration measures are the first candidates to order two receivers on
the same corpus better than chance. Ten features were tried on 20 arms, so
this is development evidence only. `top1` and `pair_overlap` are fixed here,
before any other panel is scored, as the two candidates for confirmation on
the rate-law audit's 67 arms (4 receivers, 31 corpora), whose adapters are
being profiled from the archive.

## Confirmation on the rate-law audit (pre-registered: top1, pair_overlap)

`adapter_spectrum_archive.csv` profiles all 420 archived adapters;
`run_map.csv` maps each run to its receiver and corpus. The audit's 67 arms
have adapters for 64 (4 receivers, 34 corpora); 44 of them have a corpus that
another receiver also trained on, which held-out-receiver prediction needs.
Every candidate below is scored on those same 44 arms:

| predictor | held-out-receiver RMSE | receiver-pair sign accuracy |
|---|---:|---:|
| **spectrum_top1** (pre-registered) | **0.129** | **0.731** |
| **spectrum_pair_overlap** (pre-registered) | 0.130 | 0.654 |
| correction_channel_bits (best recorded RMSE) | 0.138 | 0.481 |
| dataset_fisher_log_volume | 0.145 | 0.558 |
| text_cross_row_redundancy (two zlib calls) | 0.167 | 0.000 |
| corpus_only | 0.167 | 0.000 |

Both pre-registered features replicate on a panel they were not chosen on.
The gain in absolute error is modest (0.129 against 0.138); the gain in
ordering receivers is not (0.73 against 0.48). `spectrum_log_energy`, best on
the development panel and not pre-registered, collapses here (sign 0.288):
it was tracking model size, not the corpus-receiver interaction.

These features read the adapter alone. The version that joins adapter, base
model and training data (`functional_profile`, `scripts/analysis/adapter_spectrum_profile.py`) is running
on the panel next.

## Adding the base model and the data: activation-weighted spectrum (negative)

`functional_profile` runs each adapter's own base model over 128 of its
training rows and weights the update's directions by how the base
activations excite them (spectrum of C^1/2 B^T B C^1/2, C the bottleneck
covariance), plus the update's output energy relative to the frozen layer's.
`scripts/analysis/adapter_spectrum_profile.py`, `functional_transposed.csv`. On the development panel it is
worse than the weight spectrum it refines:

| feature | RMSE | sign |
|---|---:|---:|
| spectrum_pair_overlap (weights only) | 0.159 | 0.684 |
| functional_pair_overlap | 0.182 | 0.649 |
| functional_top1 | 0.189 | 0.596 |
| functional_log_relative_energy | 0.265 | 0.246 |
| best two-feature (weight + functional) | 0.162 | 0.684 |

Activation weighting barely reorders adapters (Spearman 0.96 between the
weight and functional top-direction shares), and the relative-energy term
tracks model scale, not bits. The codec acts on the weights -- it quantizes A
and B and drops random LoRA pairs -- so weight structure is what decides what
survives; activation energy is close to a monotone rescaling of it. Not
carried to the audit panel.

What activation energy does not measure is which directions carry the
*gain*. The next candidate removes each singular direction in turn and
measures the training-data likelihood gain over base that is lost (adapter,
base model and data jointly, 18 forward passes per adapter).

## Adding the base model and the data: gain carried per direction

`gain_profile` removes each of the 16 singular directions in turn and measures
the training-data likelihood gain over base that is lost (128 training rows,
the adapter's own base model; `scripts/analysis/adapter_spectrum_profile.py ... gain`, `gain_transposed.csv`).
Development panel:

| feature | RMSE | sign |
|---|---:|---:|
| **gain_directions_90** (directions carrying 90% of the gain) | **0.145** | 0.632 |
| gain_top1_share | 0.166 | 0.632 |
| spectrum_pair_overlap (weights only) | 0.159 | 0.684 |
| gain_bits_per_token (size of the gain) | 0.209 | 0.193 |

`gain_directions_90` has the lowest error of any candidate so far; it
correlates -0.65 with the weight-space top-direction share, so both measure
how concentrated the update is, and adding one to the other overfits 20 arms.
The size of the gain predicts nothing (as the rate-law audit found for bits
saved); how it is spread across directions does. `gain_directions_90` is fixed
here as the candidate for the audit panel, run on its 64 adapters next.

## Confirmation of gain_directions_90 on the rate-law audit (failed)

`gain_audit.csv`: the gain profile of all 64 audit adapters (one seed per arm;
Mistral runs mix rank-16 and rank-64 adapters, so the runner re-attaches per
adapter). Scored on the same 44 predictable arms as above:

| predictor | RMSE | sign |
|---|---:|---:|
| spectrum_top1 (weights only) | 0.129 | 0.731 |
| spectrum_pair_overlap (weights only) | 0.130 | 0.654 |
| correction_channel_bits | 0.138 | 0.481 |
| corpus_only | 0.167 | 0.000 |
| **gain_directions_90** (pre-registered) | 0.179 | 0.173 |
| gain_top1_share | 0.172 | 0.173 |

The pre-registered three-input feature fails the held-out-receiver test: worse
than knowing only the corpus. It is not noise: within each receiver it tracks
R* across corpora (correlation 0.68 Llama, 0.45 Mistral, 0.36 Qwen2.5, 0.54
Qwen3). But ordering receivers on a fixed corpus is exactly the axis every
corpus-side measure has failed on, and the gain profile fails there too; only
the weight spectrum carries it. As things stand, the base model and the data
add ranking power inside a receiver and nothing across receivers.
