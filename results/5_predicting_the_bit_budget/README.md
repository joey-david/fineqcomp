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
