# Predicting the bit budget of a fine-tune before running it

Status: **pre-registered; one section measured, the prediction not yet run.**
The gates in `configs/bit_budget.yaml` were written before any probe existed
and nothing may re-choose them. The one thing already measured is what changing
the target does to numbers the campaign has published, which is reported below
because it is the reason this study exists; it needed no new GPU time and no
probe. The rate ladder was made finer after reading it, which refines where a
crossing is bracketed and touches no gate. This file will be updated with the
outcome whether or not it confirms.

## The claim

The information needed to keep a behaviour learned by fine-tuning is not a
property of the dataset and not a property of the adapter. It is a property of
the base model and of the corrections the data asks that model to make, and it
can be predicted before the fine-tune runs.

The campaign has the correlational half of this and has failed twice to get the
predictive half. This study changes what is measured on both sides.

## What went wrong before, in two sentences each

**The target was not a bit budget.** Below one bit the shipped codec ladder is
not a precision code: `codec.blend_widths` at zero bits keeps a random fraction
of the LoRA rank directions and drops the rest, keyed on the pair name so that
it cannot look at the weights. Almost every R\* this project has recorded lies
between 0.2 and 1.2 bits per value, so almost every one of them measures how
much *random rank pruning* an adapter survives; informed truncation beats that
mask by a median of 28 retained-gain points on the very same files
(`../1_rate_behaviour_frontier/budget_matched_rank/`).

**The predictors were the wrong kind of object.** Forty-six candidates have been
screened, every one a scalar summary of a gradient sketch, every one scored with
a per-receiver intercept. The locked one failed its prospective test
(`../1_rate_behaviour_frontier/correction_channel_rate/prospective_validation/`),
and the audit found the reason: a corpus statistic returns the same number on
every receiver, so it can only be graded on the corpus axis, and the sign of its
relation to R\* is opposite on the two axes
(`../1_rate_behaviour_frontier/rate_law_audit/`).

## The target

**B\***: the smallest adapter file, in bits, that still keeps 90% of the
held-out likelihood gain the finished fine-tune reached, minimised over the
whole rank-by-precision surface with informed truncation. Rank and precision are
commensurable because the file shrinks in proportion to both, so the envelope
over the two is a single question — at this many bytes, how many directions at
what precision? — and B\* is where that envelope crosses 90%.

Bits per value against the full-rank container is reported alongside it, so the
new numbers can be read next to every R\* already published.

A cell whose sweep never brackets the crossing is dropped rather than
extrapolated.

## What changing the target already does

`target_change/` reads both crossings off the twelve adapters the rank sweep
had already scored, at no GPU cost:

```bash
python -m fineqcomp bit-budget-report --sweeps reports/rank_frontier \
  --runs-root runs --config configs/bit_budget.yaml \
  --out results/5_predicting_the_bit_budget/target_change
```

| receiver | corpus | shipped ladder | B\*/N | overstated by | winning rank | shipped MB | budget MB |
|---|---|---:|---:|---:|---:|---:|---:|
| Mistral-7B | code | 0.891 | 0.249 | 3.6x | 2 | 4.67 | 1.31 |
| Mistral-7B | math | 0.737 | 0.353 | 2.1x | 2.7 | 3.86 | 1.85 |
| Qwen2.5-7B | code | 0.668 | 0.090 | 7.5x | 1 | 3.37 | 0.45 |
| Qwen2.5-7B | math | 0.411 | 0.074 | 5.5x | 1 | 2.08 | 0.38 |

Three things follow, and none of them needed a new run.

The campaign's rate overstates the bit budget by two to eight times, and by how
much is itself a property of the receiver and the corpus rather than a constant:
Qwen2.5 is overstated twice as badly as Mistral. A predictor fitted to the left
column was not being fitted to a bit budget.

**The corpus ordering inside a receiver reverses.** On the shipped ladder
Mistral finds mathematics cheaper than code, 0.737 against 0.891. On the budget
it finds mathematics dearer, 0.353 against 0.249. This is the same ordering the
campaign's within-receiver Spearman of 0.74 was scored against.

**The container that wins is rank one or two, never sixteen.** At the crossing
the file holds one or two directions per projection at between one and two bits
each, not sixteen directions at a fraction of a bit. Rank-16 all-linear was a
convention the whole campaign inherited; measured against a budget it is about
an order of magnitude more container than the behaviour needs. That is why the
rank axis below trains at rank 4 and 64 rather than assuming the answer.

These twelve adapters were swept on the older nine-rung rate ladder, which
brackets the crossing inside a factor of 1.7 in file size. The grid used from
here adds quarter-bit rungs between one and two bits, where the crossings
actually fall.

## The predictor

The same measurement, on an adapter that was never trained.

1. Attach the container at the run's own seed. Cap the optimizer at `k` updates
   over a 256-row batch. At `k = 1` this is one accumulated gradient at the
   frozen weights: the correction the corpus asks this model for before
   anything has been learned.
2. The update direction is `theta_k - theta_0`, and `theta_0` is not stored
   because attaching with the same seed reproduces it exactly.
3. Search a single scale along that direction on the held-out split. This is
   the only free quantity in the construction; without it the probe's magnitude
   is the learning rate, which is a property of the config rather than of the
   corpus.
4. Sweep the resulting adapter over the identical rank-by-precision surface.

The prediction is **B\*_probe = B\*_trained**, on the identity line. No fitted
constant, no receiver intercept, no ranking step — which is what lets the same
prediction be made for a receiver and a corpus that nothing was fitted on.

`k` runs over 1, 8 and 64, so the result is predictive accuracy against probe
cost rather than a single point. Sixty-four updates is 3% of the 2,000 the full
contract spends.

There is one known reason the shortest probe might fail, and the ladder exists
to find out whether it does. At `k = 1` the gradient with respect to LoRA-A is
zero, so only B has moved and the update is `B @ A0` with A0 the random init.
A random A0 is well conditioned, so the probe's update spreads its energy over
all sixteen directions, while a trained pair co-adapts and usually concentrates
it in a few. Truncation acts on exactly that concentration, so a one-update
probe could over-predict the budget for a reason that has nothing to do with
the corpus. The codec removes half of the concern by itself -- every cell in
the sweep is refactored through a balanced SVD of the product before coding, so
both sides are scored on the spectrum of the update rather than on how it
happens to be split between the factors -- and the remaining half is what
`k = 8` and `k = 64` measure.

## The design

Nothing here trains at full length. Every target the panel predicts is an
adapter the campaign has already paid for, so the budget buys probes and
forward-only sweeps.

| stage | what runs | count |
|---|---|---:|
| probes | capped runs at k=1, one update over a 256-row batch | 30 |
| probe ladder | the same cells at k=8, on one corpus | 3 |
| targets | finished adapters, swept, never retrained | 30 |

The panel is seven receivers by six corpora. Three corpora — code, mathematics
and XBRL tag extraction — come from the transposed receiver panel and carry all
seven receivers, including the two pairs a corpus statistic cannot tell apart
at all: Qwen2.5-7B against Qwen2.5-Math-7B, and Gemma-2-9B base against its
instruct model, each pair sharing a tokenizer and an architecture exactly.
Three more — summarisation, dialogue preference and text-to-SQL — carry Mistral
and Qwen2.5, so the corpus axis is six task shapes wide rather than three.
Three cells are repeated at a second seed, because a budget that does not
reproduce is not a measurement.

A sweep is 28 cells, four ranks by seven rungs, and a cell is one held-out code
length over 256 rows plus an encode and a decode. The grid is narrower than the
sweep's default because the twelve adapters already scored say where the
crossing falls: rank one or two, between one and two and a bit bits per value.
The array unit is a receiver, since loading a 9B model in NF4 costs more than
sweeping one of its adapters. The whole study is about ten GPU-hours.

There is no separate prospective stage and none is needed to keep the test
honest: the predictor has no fitted parameter, no receiver term and no ranking
step, so there is nothing for a development set to leak into it. What has to
stay locked is `configs/bit_budget.yaml`, and it is locked before any probe
exists.

What this budget does **not** buy is a receiver nobody has fine-tuned, or a
container other than rank-16 all-linear. Both need full-length training runs.
They are the next thing to spend on, and only if the gates below hold.

## First read, and the repair it forced

Status of this section: **partial.** `kind_code` is the one corpus whose seven
receivers have all been swept; the other five were still sweeping when this was
written, and nothing here is the panel's verdict.

The k=1, rank-16 probe **ranks receivers and misprices them**.

| gate | value | threshold | |
|---|---|---|---|
| P2 Spearman of B\*_probe against B\*_trained | 0.75 | 0.60 | pass |
| P4 receiver-pair sign accuracy, 21 pairs | 0.81 | 0.70 | pass |
| P3 identity RMSE against the corpus mean | 0.227 vs 0.095 | 0.75x | **fail** |

That is the case the pre-registration named in advance: the probe ranks but
does not predict. It over-predicts the budget everywhere, by 1.39, 1.58, 1.60,
1.62, 1.69 and 1.87 on six receivers and by 4.37 on Qwen2.5-Math.

The cause is in the construction and is visible in the cells. While B is zero
the gradient with respect to A is zero, so a one-update probe's correction is
`B @ A0` with A0 the random initialisation -- a rank-dimensional *random
sketch* of the correction the corpus actually asks for. A random sketch is well
conditioned exactly where a trained pair is concentrated, and truncation is the
operation that punishes being well conditioned. Retained gain at rank 1 and 3
bits, probe against the target it predicts:

| receiver | target | probe | over-prediction |
|---|---:|---:|---:|
| Qwen2.5-Math-7B | 0.943 | 0.783 | 4.37x |
| Gemma-2-9B | 0.841 | 0.790 | 1.87x |
| Llama-3.1-8B | 0.877 | 0.823 | 1.69x |
| Gemma-2-9B-it | 0.890 | 0.831 | 1.62x |
| Mistral-7B | 0.856 | 0.834 | 1.58x |
| Qwen2.5-7B | 0.968 | 0.989 | 1.39x |

A few points of retention become a large factor in file size because the 90%
crossing sits on a steep part of the curve. Qwen2.5-Math is the clearest case:
its target already clears 0.90 at rank 1, its probe needs rank 4, and rank 4 is
four times the file.

## The repair, as a factorial

Two causes are available and they are not the same cause, so both are tested at
once on `kind_code`, whose targets are already swept.

| arm | container | updates | what it removes |
|---|---|---|---|
| `probe1_core` | 16 | 1 | nothing; the measurement above |
| `repair_wide_k1` | 64, cut to 16 | 1 | the sketch is 64-dimensional, so its leading directions estimate the real ones |
| `repair_narrow_k8` | 16 | 8 | A moves, so the pair begins to co-adapt |
| `repair_wide_k8` | 64, cut to 16 | 8 | both |

Cutting back to 16 goes through the same balanced SVD the sweep already uses,
and the sweep no longer forces a container rank into a grid that stops below
it, so a wide probe is scored on exactly the files its target is scored on.

Both repairs are parameter-free. **Rescaling by the observed factor is the
repair deliberately not attempted**: it would put back the fitted receiver term
this design exists to avoid, and it could not work anyway, because six
receivers cluster near 1.6 and one sits at 4.4. That the factor is not constant
is the reason P3 fails rather than a rescaled probe passing.

Reading the factorial: if width alone closes it, the failure was a sketching
artifact and the k=1 probe is salvageable at no extra training cost. If only
steps close it, the budget of a fine-tune is not visible until the factors
co-adapt, and the honest claim becomes "eight updates predict two thousand". If
neither closes it, a one-step direction is not the object that sets the budget,
and the study reports that.

## What the factorial said: the repair was wrong, the ladder was right

Status: **complete for k=1 and k=8 on the panel below.** 53 probes, 82 swept
adapters, 80 bracketed, no cell missing.

On the five cells all four arms cover -- the only comparison that is not
confounded by which corpora an arm happens to reach:

| arm | identity RMSE | mean over-prediction |
|---|---:|---:|
| k=1, rank 16 | 0.222 | 2.13x |
| k=1, rank 64 cut to 16 | 0.510 | 3.44x |
| k=8, rank 16 | **0.187** | **1.68x** |
| k=8, rank 64 cut to 16 | 0.317 | 2.52x |
| corpus mean baseline | 0.076 | -- |

**Sketching wider makes it worse, at both update counts.** The hypothesis was
that a rank-16 sketch of the correction is too narrow, so its leading
directions are a poor estimate and truncation punishes the spread. Sketching
into 64 and cutting back doubles the error instead of halving it. Whatever
makes a one-step direction hard to truncate, it is not the width of the
container it was sketched into, and the mechanism written above is refuted.

**More updates help, monotonically and not enough.** Eight updates cut the
error from 0.222 to 0.187 and the over-prediction from 2.13x to 1.68x, and on
the wider k=8 panel reach Spearman 0.867 and pair sign accuracy 0.833. But
0.187 is still 2.5 times the corpus-mean baseline, so P3 fails.

Over the whole panel: P1, P2 and P4 pass, P3 fails.

| arm | arms | corpora | identity RMSE | corpus mean | Spearman | sign accuracy |
|---|---:|---:|---:|---:|---:|---:|
| k=1, rank 16 | 26 | 6 | 0.420 | 0.078 | 0.515 | 0.733 |
| k=1, wide | 6 | 1 | 0.963 | 0.102 | 0.543 | 0.733 |
| k=8, rank 16 | 10 | 2 | 0.142 | 0.101 | 0.867 | 0.833 |
| k=8, wide | 6 | 1 | 0.999 | 0.079 | 0.829 | 0.867 |

The k=1 row spans six corpora and the k=8 row two, so those two RMSEs are not
comparable to each other; only the matched table above is.

What this leaves is a probe that orders receivers well and prices them badly,
with the only lever that moves it being optimizer steps. The open question is
whether the error keeps falling with k -- and if it does, where it crosses the
baseline, because that crossing is the price of a usable prediction.

## Gates

| gate | statement |
|---|---|
| P1 | at least 90% of swept adapters have a bracketed budget |
| P2 | Spearman of B\*_probe against B\*_trained is at least 0.60 at some probe budget |
| P3 | the probe's identity-line RMSE is at most 0.75 of the error made by predicting each arm from the mean of the other receivers on the same corpus |
| P4 | the probe orders receivers inside a corpus with pair sign accuracy at least 0.70 |

P1 failing means the sweep grid is wrong and nothing else can be read. P2
failing kills the construction. P2 passing with P3 failing means the probe ranks
but does not predict, which is the failure the correction-spectrum measures
already had and would be reported as such. P4 is the claim: it is the axis a
corpus statistic cannot score on at all.

A passing panel says the construction predicts the budget on these seven
receivers and six corpora. It does not say it predicts on a receiver nobody has
fine-tuned; that needs training runs this budget does not have, and is what a
pass would justify spending on next.

## Running it

```bash
sbatch --array=0-0   --export=ALL,MODE=manifest scripts/jean_zay_bit_budget.sbatch
sbatch --array=0-6%7 --export=ALL,MODE=probes   scripts/jean_zay_bit_budget.sbatch
sbatch --array=0-6%7 --export=ALL,MODE=sweep    scripts/jean_zay_bit_budget.sbatch
sbatch --array=0-0   --export=ALL,MODE=report   scripts/jean_zay_bit_budget.sbatch
```

Each stage is one array task per receiver. `MODE=manifest` needs no GPU and no
network: every corpus this panel touches is already prepared, which is the same
fact that lets it predict adapters that already exist.

## Files

| file | contents |
|---|---|
| `budgets.csv` | one row per swept adapter: its budget, its container, whether it was a probe |
| `probe_vs_trained.csv` | one row per cell: the probe's budget and the finished run's |
| `probe_scores.csv` | identity-line error and ordering accuracy at each probe budget |
| `summary.json` | the gates |
