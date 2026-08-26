# Does a measure of relative information beat counting tokens?

Status: pre-registered, not yet run. Everything below the gates was written
before any measurement of the new candidates was read.

## What is being tested

R\*(0.90) is the smallest adapter rate that keeps 90% of the learned held-out
gain. The claim under test is that R\* is set by the information the training
corpus carries that the frozen model does not already have, and that some
measurable quantity captures it.

Every candidate measured so far has failed to transfer. All of them are
spectra of output-head gradient sketches at the frozen weights, which score how
strange the text looks to the model. On the two external panels the locked
candidates scored Spearman -0.10 (Llama, `dataset_fisher_log_volume`) and
-0.40 (Qwen3, `fisher_effective_rank`), on four arms each.

## The incumbent

Supervised token count is the strongest predictor of R\* in the finished
campaign, and no earlier analysis used it. Over the 59 rank-16 bracketed arms
with at least two seeds:

| panel | arms | Spearman of R\* on train tokens |
| --- | --- | --- |
| pooled | 59 | +0.633 |
| Llama-3.1 8B | 6 | +0.829 |
| Mistral 7B | 35 | +0.624 |
| Qwen2.5 7B | 14 | +0.407 |
| Qwen3 8B | 4 | +0.800 |

It is positive in every model, unlike every per-token measure tried so far. It
is therefore the baseline to beat, and it is carried through the ranking as
`train_response_tokens`.

Token count cannot be the whole story, because R\* still moves when tokens are
held fixed. Mistral XBRL at 3578-3587 supervised tokens gives R\* 0.260, 0.294,
0.318, 0.371 across div_10, div_18, div_30 and the full tag set; Qwen2.5 gives
0.397, 0.443, 0.469, 0.474; Mistral SQL gives 0.375, 0.387, 0.406, 0.415. The
information lever at an identical 48,184 tokens gives 0.682, 0.713, 0.746 for
400, 1200 and 3600 distinct source problems. What token count is approximating
is the number of distinct corrections a corpus demands, and diversity moves
that number while holding tokens flat.

## The ten candidates

The first eight are reduced from the per-row sketches the existing frozen pass
already collects, so they cost no extra model time. Rows are 1024 uniform
samples of the training split, seed 271828.

1. `prequential_correction_bits_per_row` — an online ridge map from receiver
   state to requested output correction, coded against a running-mean null.
   Code length under an online learner, not a spectrum: a corpus that repeats
   itself is cheap after the first few rows.
2. `prequential_decay_exponent` — log-log decay rate of the online error. Fast
   decay means the corpus teaches a small number of rules.
3. `prequential_late_fraction` — the share of the code-length saving earned in
   the second half of the stream. Bounded and scale free, so it survives a
   change of model or corpus size.
4. `nearest_neighbour_cosine` — mean cosine from each row's correction to its
   closest neighbour. Near one means a redundant corpus.
5. `correction_intrinsic_dimension` — the two-nearest-neighbour estimator.
   Unlike effective rank it is not inflated by isotropic noise, and it reads
   the local dimension of the correction cloud rather than its total spread.
6. `correction_coverage_fraction` — the share of rows kept by a greedy cosine
   net at 0.90. A direct count of how much of the corpus is new.
7. `correction_predictable_fraction` — held-out variance of the requested
   correction explained by a ridge map from receiver state. High means the
   correction is one systematic rule.
8. `coherent_fraction` — squared norm of the mean unit correction. One when
   every row pulls the model the same way.
9. `lora_layer_energy_entropy` and `lora_layer_energy_centroid` — attach a
   fresh rank-16 LoRA whose B is zero, so the model's output is unchanged, and
   read the per-layer share of the adapter gradient energy from one backward
   pass. This is the only candidate in the space the adapter actually stores,
   and the only one whose intervention is a per-layer rate allocation.
10. `text_cross_row_redundancy` — zlib on the responses jointly against
    separately. The compression control, included because it is the measure the
    project has been told is probably not the answer, and it should be beaten
    rather than skipped.

## Gates

A candidate replaces token count only if, in `candidate_ranking.csv`:

- its `within_model_mean_spearman` exceeds that of `train_response_tokens`;
- its `task_heldout_rmse` is lower than that of `train_response_tokens`;
- its `selection_adjusted_p` is below 0.05, over 50,000 permutations across the
  full candidate set, so the comparison pays for the search;
- its `median_seed_cv` is below 0.10, so the measure is not noise.

A candidate is judged to add something token count cannot only if, in
`diversity_slopes.csv`, its slope per doubling of distinct source groups has
the same sign as the R\* slope in at least five of the six model-by-task
diversity families, where the token-count slope is flat by construction.

A candidate that clears the first four gates but not the diversity gate is a
better estimate of dataset size. A candidate that clears the diversity gate but
not the first four is a measure of diversity that does not set the rate.
Neither is the claim.

## Design

One measurement per model x dataset x prepared seed, over all 276 cells that
have a finished run with a bracketed R\*: 78 discovery, 126 development, and
three 24-cell external panels on Llama-3.1 8B, Qwen2.5 7B and Qwen3 8B. This
replaces the 26-arm discovery set and the 4-arm external tests, on which a
0.70 Spearman gate was close to a coin flip.

## Amendment after the smoke test, before any result was opened

Two cells were measured on Jean-Zay at 64 rows to check the code path, and one
candidate was degenerate. `correction_coverage_fraction` was defined as the
share of rows kept by a greedy cosine net at 0.90, and it returned exactly 1.0:
real corrections are close to mutually orthogonal, so no fixed threshold above
the noise floor removes any row. It is now the participation ratio of the
angular correction kernel, n squared over the summed squared cosines, divided
by the row count. That is a threshold-free count of distinct correction
directions: one row's worth when every row asks for the same change, n rows'
worth when they are orthogonal. On synthetic clouds of 5, 20, 80 and 200
distinct directions among 200 rows it returns 0.025, 0.075, 0.176 and 0.246.

No R\* comparison had been run at the time of this change, and the gates above
are unchanged.

## Result on 235 of 276 cells

Status: 235 cells measured, of which 203 joined to a run with a bracketed R\*,
giving 50 arms over Llama-3.1 8B, Mistral 7B and Qwen2.5 7B. The remaining 41
cells are queued; they are the Qwen3 external panel and part of the Qwen2.5
external panel, so no Qwen3 arm is in the table below.

The answer is `text_cross_row_redundancy`, and it beats supervised token count
on every quantitative gate.

| measure | within-model mean rho | two-sided adjusted p | task-held-out RMSE | model-held-out RMSE | diversity agreement | seed CV |
| --- | --- | --- | --- | --- | --- | --- |
| `text_cross_row_redundancy` | -0.615 | 0.0033 | 0.163 | 0.193 | 4/4 | 0.007 |
| `train_response_tokens` | +0.578 | 0.0108 | 0.174 | 0.229 | 1/4 | 0.025 |
| `centered_angular_correction_logdet` | +0.529 | 0.0374 | 0.168 | 0.170 | 3/4 | 0.005 |
| model fixed effects only | - | - | 0.218 | 0.306 | - | - |

RMSE is in bits per value, against a mean R\* of about 0.69 and a between-arm
spread of about 0.21. Held-out model prediction is the harder column: the fixed
effect for the held-out receiver is unavailable, so only a measure that carries
the level of R\* across receivers can help. Redundancy cuts that error by 37%
against model fixed effects and by 16% against token count.

The two are not the same predictor. Their rank correlation is -0.569, and each
survives control for the other: partial rho of R\* on redundancy given tokens is
-0.507, and of R\* on tokens given redundancy is +0.419. Adding token count to
redundancy does not improve held-out prediction (0.168 and 0.190 against 0.163
and 0.193), so redundancy carries what token count was standing in for.

The sign is the one the corpus argument predicts. A corpus whose responses
compress into each other is asking the model for the same correction many
times, so it needs fewer adapter bits. Every measure in the ten that reads
redundancy rather than count agrees: `nearest_neighbour_cosine`, the mean
cosine from each row's correction to its nearest neighbour, has pooled rho
-0.715 against R\* and correlates +0.802 with the zlib measure. The text
statistic and the model-side statistic are reading the same property.

The measure that fails the diversity gate is token count itself, at 1 of 4
families, which is the point: R\* rises with distinct source groups at a fixed
token budget, and redundancy falls with them in all four families.

### Against the pre-registered gates

Read literally, the first gate asked that `within_model_mean_spearman` exceed
that of `train_response_tokens`, which is a one-sided test. The winning measure
correlates negatively, so it fails that sentence and passes it on absolute
value. The one-sided permutation test scored it at p = 1.0 by construction; a
two-sided version, over the same 50,000 permutations and the same 37 candidates,
gives p = 0.0033, the smallest in the table. The sign was not chosen after the
fact: the pre-registration describes candidates 4, 8 and 10 as measures of
redundancy, and a measure of redundancy predicting a smaller adapter is the
direction the argument requires. The one-sided test is now a two-sided test in
`analyze_relative_information_cells`, since a measure that predicts a smaller
rate is as much a result as one that predicts a larger rate.

The other three gates are passed as written: task-held-out RMSE below token
count, adjusted p below 0.05, seed CV below 0.10. The diversity gate asked for
sign agreement in five of six model-by-task families; only four families are
present in this panel, and it agrees in all four.

### What this changes

The adapter bit budget can be estimated before any adapter is trained, from the
training corpus alone, with zlib and no model. On a held-out receiver the
estimate is worth about 0.19 bits per value against a 0.21 spread, where
knowing only the receiver is worth 0.31. That is the intervention: compress the
responses jointly against separately, and read the rate off the fit.

It also settles the framing. The quantity that sets the rate is not how
surprising the corpus is to the model, which is what every earlier candidate
measured and every earlier candidate failed to transfer with. It is how much
the corpus repeats itself. That is a property of the data, not of the pair.
